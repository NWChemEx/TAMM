// Unit tests for the TAMM RMM-derived memory pool (src/tamm/mr/).
//
// These exercise the host pool only, so the test runs without a GPU. The device pool
// (gpu_memory_resource) shares all of its suballocation logic with the host pool via
// pool_memory_resource / stream_ordered_memory_resource, so the behavior covered here is
// the same behavior the device pool relies on.
//
// Coverage:
//  - pool accounting: pool_size(), free_bytes(), free_summary()
//  - allocation/deallocation round-trip returns the pool to its baseline (leak detection)
//  - exact-fit allocation
//  - coalescing: fragmented pool recovers full contiguity once holes are freed
//  - best-fit block selection
//  - alignment guarantees
//  - concurrent alloc/free (guards the pool mutex)
//
// Not covered here: the over-sized-request failure path. The pool is deliberately
// fixed-size (never grown from upstream), so an over-sized request is fatal by design --
// tamm_terminate() calls exit(), which cannot be asserted on in-process. That path is
// exercised manually; see the diagnostics it emits in
// stream_ordered_memory_resource::do_allocate / get_block.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <tamm/mr/host_memory_resource.hpp>
#include <tamm/mr/new_delete_resource.hpp>
#include <tamm/mr/pool_memory_resource.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <complex>
#include <cstring>
#include <memory>
#include <set>
#include <thread>
#include <utility>
#include <vector>

using pool_mr = tamm::rmm::mr::pool_memory_resource<tamm::rmm::mr::host_memory_resource>;

namespace {

constexpr std::size_t kAlign = tamm::rmm::detail::RMM_ALLOCATION_ALIGNMENT;

/// A pool of `bytes`, freshly constructed for each test.
std::unique_ptr<pool_mr> make_pool(std::size_t bytes) {
  return std::make_unique<pool_mr>(new tamm::rmm::mr::new_delete_resource, bytes);
}

std::size_t aligned(std::size_t n) { return tamm::rmm::detail::align_up(n, kAlign); }

} // namespace

TEST_CASE("MemoryPool: reports its own size") {
  constexpr std::size_t kPool = 1u << 20; // 1 MiB, already aligned
  auto                  pool  = make_pool(kPool);

  CHECK(pool->pool_size() == kPool);
  CHECK(pool->free_bytes() == kPool);

  auto const [largest, total] = pool->free_summary();
  CHECK(largest == kPool); // one contiguous block
  CHECK(total == kPool);
}

TEST_CASE("MemoryPool: allocate then deallocate returns to baseline") {
  constexpr std::size_t kPool = 1u << 20;
  auto                  pool  = make_pool(kPool);

  std::size_t const baseline = pool->free_bytes();

  void* ptr = pool->allocate(4096);
  REQUIRE(ptr != nullptr);
  CHECK(pool->free_bytes() == baseline - aligned(4096));

  pool->deallocate(ptr, 4096);
  CHECK(pool->free_bytes() == baseline);

  // and the pool is one contiguous block again
  auto const [largest, total] = pool->free_summary();
  CHECK(largest == kPool);
  CHECK(total == kPool);
}

TEST_CASE("MemoryPool: repeated alloc/free cycles do not leak") {
  // This is the regression guard for the aliasing leaks in kernels/multiply.hpp:
  // a buffer allocated from the pool must be returned with the size it was allocated
  // with. If any iteration leaks, free_bytes() ratchets downward.
  constexpr std::size_t kPool = 1u << 20;
  auto                  pool  = make_pool(kPool);

  std::size_t const baseline = pool->free_bytes();

  for(int i = 0; i < 1000; ++i) {
    void* a = pool->allocate(1024);
    void* b = pool->allocate(2048);
    void* c = pool->allocate(512);
    pool->deallocate(b, 2048);
    pool->deallocate(a, 1024);
    pool->deallocate(c, 512);
  }

  CHECK(pool->free_bytes() == baseline);
  // fully coalesced back into a single block
  CHECK(pool->free_summary().first == kPool);
}

TEST_CASE("MemoryPool: zero-size allocation is a no-op") {
  auto pool = make_pool(1u << 20);

  std::size_t const baseline = pool->free_bytes();
  CHECK(pool->allocate(0) == nullptr);
  CHECK(pool->free_bytes() == baseline);

  // deallocating null / zero must not corrupt the free list
  pool->deallocate(nullptr, 0);
  pool->deallocate(nullptr, 128);
  CHECK(pool->free_bytes() == baseline);
}

TEST_CASE("MemoryPool: allocations are correctly aligned and non-overlapping") {
  constexpr std::size_t kPool = 1u << 20;
  auto                  pool  = make_pool(kPool);

  std::vector<std::pair<char*, std::size_t>> allocs;
  for(std::size_t sz: {1u, 7u, 64u, 100u, 255u, 256u, 1000u}) {
    void* p = pool->allocate(sz);
    REQUIRE(p != nullptr);
    CHECK(reinterpret_cast<std::uintptr_t>(p) % kAlign == 0);
    allocs.emplace_back(static_cast<char*>(p), aligned(sz));
  }

  // no two live allocations may overlap
  for(size_t i = 0; i < allocs.size(); ++i) {
    for(size_t j = i + 1; j < allocs.size(); ++j) {
      auto const& [pi, si] = allocs[i];
      auto const& [pj, sj] = allocs[j];
      bool const disjoint  = (pi + si <= pj) || (pj + sj <= pi);
      CHECK(disjoint);
    }
  }

  for(auto const& [p, s]: allocs) { pool->deallocate(p, s); }
  CHECK(pool->free_bytes() == kPool);
}

TEST_CASE("MemoryPool: exact-fit allocation consumes the whole pool") {
  constexpr std::size_t kPool = 64u * 1024u;
  auto                  pool  = make_pool(kPool);

  void* p = pool->allocate(kPool);
  REQUIRE(p != nullptr);
  CHECK(pool->free_bytes() == 0);
  CHECK(pool->free_summary().first == 0);

  pool->deallocate(p, kPool);
  CHECK(pool->free_bytes() == kPool);
}

TEST_CASE("MemoryPool: fragmentation is recovered by coalescing") {
  constexpr std::size_t kPool  = 1u << 20;
  constexpr std::size_t kChunk = 4096;
  auto                  pool   = make_pool(kPool);

  // carve the pool into many chunks
  std::vector<void*> chunks;
  for(int i = 0; i < 64; ++i) { chunks.push_back(pool->allocate(kChunk)); }

  // free every other one -> heavily fragmented, lots of free bytes but small blocks
  for(size_t i = 0; i < chunks.size(); i += 2) { pool->deallocate(chunks[i], kChunk); }

  // 32 of the 64 chunks are free, plus whatever tail was never carved.
  auto const [frag_largest, frag_total] = pool->free_summary();
  CHECK(frag_total == kPool - 32 * kChunk);
  // The interleaved holes are each exactly one chunk and cannot merge with each other,
  // so the largest contiguous run is the uncarved tail -- far smaller than the total.
  CHECK(frag_largest == kPool - 64 * kChunk);
  CHECK(frag_largest < frag_total);

  // free the rest -> everything must coalesce back into one block
  for(size_t i = 1; i < chunks.size(); i += 2) { pool->deallocate(chunks[i], kChunk); }

  CHECK(pool->free_bytes() == kPool);
  CHECK(pool->free_summary().first == kPool);
}

TEST_CASE("MemoryPool: best-fit reuses the tightest hole") {
  constexpr std::size_t kPool = 1u << 20;
  auto                  pool  = make_pool(kPool);

  void* a = pool->allocate(8192);
  void* b = pool->allocate(1024); // small hole to be
  void* c = pool->allocate(8192);
  void* d = pool->allocate(4096); // larger hole to be
  void* e = pool->allocate(8192); // keeps d's hole from merging with the tail

  pool->deallocate(b, 1024);
  pool->deallocate(d, 4096);

  // a 1024 request should take the 1024 hole, not split the 4096 one
  void* reused = pool->allocate(1024);
  CHECK(reused == b);

  pool->deallocate(reused, 1024);
  pool->deallocate(a, 8192);
  pool->deallocate(c, 8192);
  pool->deallocate(e, 8192);
  CHECK(pool->free_bytes() == kPool);
}

TEST_CASE("MemoryPool: pool size is aligned down at construction") {
  // An unaligned pool size (as produced by a percentage of free memory) must be
  // rounded down, never up -- rounding up would over-commit the upstream allocation.
  constexpr std::size_t kUnaligned = (1u << 20) + 17;
  auto                  pool       = make_pool(kUnaligned);

  // kAlign is platform-dependent (4 on DPC++, 128 on HIP, 256 on CUDA,
  // alignof(max_align_t) on host-only builds), so state the property rather than a literal.
  CHECK(pool->pool_size() % kAlign == 0);
  // must round DOWN -- rounding up would over-commit the upstream reservation
  CHECK(pool->pool_size() <= kUnaligned);
  // and it must drop strictly less than one alignment unit
  CHECK(kUnaligned - pool->pool_size() < kAlign);
}

TEST_CASE("MemoryPool: concurrent allocate/deallocate is safe") {
  // Guards the pool mutex. The (T) code path issues pool traffic from a CUDA driver
  // thread via cudaLaunchHostFunc, so the free list must tolerate concurrent access.
  constexpr std::size_t kPool = 8u << 20;
  auto                  pool  = make_pool(kPool);

  std::size_t const baseline = pool->free_bytes();

  constexpr int            kThreads = 8;
  constexpr int            kIters   = 500;
  std::vector<std::thread> threads;

  // Vary the sizes per thread so the split/coalesce paths -- where an unsynchronized free
  // list actually corrupts -- are exercised, not just same-size reuse of one hole.
  for(int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&pool, t]() {
      std::size_t const sz = 256u << (t % 4); // 256..2048 B
      for(int i = 0; i < kIters; ++i) {
        void* p = pool->allocate(sz);
        // allocate() only returns null for a zero-size request; a genuine failure
        // terminates the process, so this is a hard requirement rather than a tally.
        REQUIRE(p != nullptr);
        // touch the memory so overlapping handouts show up under TSan/ASan
        std::memset(p, 0xAB, sz);
        pool->deallocate(p, sz);
      }
    });
  }
  for(auto& th: threads) { th.join(); }

  // every thread returned everything it took, and it all coalesced back
  CHECK(pool->free_bytes() == baseline);
  CHECK(pool->free_summary().first == pool->pool_size());
}

TEST_CASE("MemoryPool: distinct live allocations never alias") {
  constexpr std::size_t kPool = 1u << 20;
  auto                  pool  = make_pool(kPool);

  std::set<void*>    seen;
  std::vector<void*> live;
  for(int i = 0; i < 200; ++i) {
    void* p = pool->allocate(512);
    REQUIRE(p != nullptr);
    CHECK(seen.insert(p).second); // must not hand out a pointer that is already live
    live.push_back(p);
  }
  for(void* p: live) { pool->deallocate(p, 512); }
  CHECK(pool->free_bytes() == kPool);
}

// ---------------------------------------------------------------------------
// Dual-index consistency.
//
// coalescing_free_list keeps two views of the same blocks: an address-ordered set
// (for neighbour lookup during coalescing) and a size-ordered set (for O(log n)
// best-fit). If the two ever disagree, the pool either loses blocks or hands out a
// block it also believes is free. free_bytes()/free_summary() read the address index
// while allocate() consumes the size index, so exercising both against a known
// byte-total is what catches a desync.
// ---------------------------------------------------------------------------

TEST_CASE("MemoryPool: indices stay consistent under interleaved alloc/free") {
  constexpr std::size_t kPool = 1u << 20;
  auto                  pool  = make_pool(kPool);

  std::vector<std::pair<void*, std::size_t>> live;
  std::size_t                                live_bytes = 0;

  // Deterministic pseudo-random interleaving: allocations of many different sizes,
  // freed out of order, so blocks are repeatedly split and coalesced.
  std::size_t seed = 12345;
  auto        next = [&seed]() { return seed = seed * 1103515245 + 12345; };

  // Keep the working set far below capacity. A fixed-size pool legitimately fails once
  // fragmentation bites, and that is not what this test is probing -- it is probing index
  // agreement, so it must never actually run out.
  constexpr std::size_t kLiveCap = kPool / 8;

  for(int i = 0; i < 3000; ++i) {
    bool const do_alloc =
      live.empty() || (live_bytes < kLiveCap && ((next() >> 16) % 3 != 0));

    if(do_alloc) {
      std::size_t const sz = 64 + ((next() >> 16) % 4096);
      void*             p  = pool->allocate(sz);
      REQUIRE(p != nullptr);
      live.emplace_back(p, sz);
      live_bytes += aligned(sz);
    }
    else {
      std::size_t const idx    = (next() >> 16) % live.size();
      auto const [ptr, sz]     = live[idx];
      live[idx]                = live.back();
      live.pop_back();
      pool->deallocate(ptr, sz);
      live_bytes -= aligned(sz);
    }

    // The address index must always account for exactly the bytes not handed out.
    REQUIRE(pool->free_bytes() == kPool - live_bytes);
  }

  for(auto const& [ptr, sz]: live) { pool->deallocate(ptr, sz); }

  // Everything returned, everything coalesced back into one block.
  CHECK(pool->free_bytes() == kPool);
  CHECK(pool->free_summary().first == kPool);
}

TEST_CASE("MemoryPool: size index survives coalescing of equal-sized blocks") {
  // Blocks of identical size must all remain addressable in the size index. Ordering
  // the index by size alone would collapse them into one entry and lose the rest.
  constexpr std::size_t kPool  = 1u << 20;
  constexpr std::size_t kChunk = 8192;
  auto                  pool   = make_pool(kPool);

  std::vector<void*> chunks;
  for(int i = 0; i < 16; ++i) { chunks.push_back(pool->allocate(kChunk)); }

  // Free alternating chunks -> several free blocks all of exactly kChunk bytes.
  for(size_t i = 0; i < chunks.size(); i += 2) { pool->deallocate(chunks[i], kChunk); }

  // Each equal-sized hole must be independently reusable.
  std::vector<void*> reused;
  for(int i = 0; i < 8; ++i) {
    void* p = pool->allocate(kChunk);
    REQUIRE(p != nullptr);
    reused.push_back(p);
  }

  for(void* p: reused) { pool->deallocate(p, kChunk); }
  for(size_t i = 1; i < chunks.size(); i += 2) { pool->deallocate(chunks[i], kChunk); }

  CHECK(pool->free_bytes() == kPool);
  CHECK(pool->free_summary().first == kPool);
}

TEST_CASE("MemoryPool: full carve and release recoalesces completely") {
  // The property that actually matters for the reported failure: after the pool has been
  // chopped into many small pieces and everything is returned, a single allocation of the
  // entire pool must succeed again.
  constexpr std::size_t kPool  = 1u << 20;
  constexpr std::size_t kChunk = 1024;
  auto                  pool   = make_pool(kPool);

  std::vector<void*> chunks;
  while(pool->free_bytes() >= kChunk) { chunks.push_back(pool->allocate(kChunk)); }
  CHECK(chunks.size() == kPool / kChunk);

  // Release in a scattered order so coalescing has to merge from both directions.
  for(size_t i = 0; i < chunks.size(); i += 3) { pool->deallocate(chunks[i], kChunk); }
  for(size_t i = 1; i < chunks.size(); i += 3) { pool->deallocate(chunks[i], kChunk); }
  for(size_t i = 2; i < chunks.size(); i += 3) { pool->deallocate(chunks[i], kChunk); }

  CHECK(pool->free_bytes() == kPool);
  CHECK(pool->free_summary().first == kPool);

  // The whole pool must be allocatable as one contiguous block again.
  void* whole = pool->allocate(kPool);
  REQUIRE(whole != nullptr);
  pool->deallocate(whole, kPool);
}

// ---------------------------------------------------------------------------
// std::span API.
//
// The span carries its own length, so deallocate() derives the byte count instead of
// the caller recomputing count*sizeof(T) at a distant site. These tests pin that the
// derived size matches the allocation exactly for a range of element types, including
// ones whose size is not a divisor of the pool alignment.
// ---------------------------------------------------------------------------

TEST_CASE("MemoryPool: allocate_span round-trips exactly for several element types") {
  constexpr std::size_t kPool = 1u << 20;
  auto                  pool  = make_pool(kPool);
  std::size_t const     base  = pool->free_bytes();

  SUBCASE("double") {
    auto s = pool->allocate_span<double>(512);
    CHECK(s.size() == 512);
    CHECK(s.size_bytes() == 512 * sizeof(double));
    CHECK(pool->free_bytes() == base - aligned(s.size_bytes()));
    pool->deallocate(s);
    CHECK(pool->free_bytes() == base);
  }
  SUBCASE("char - size not a multiple of alignment") {
    auto s = pool->allocate_span<char>(7);
    CHECK(s.size() == 7);
    CHECK(s.size_bytes() == 7);
    pool->deallocate(s);
    CHECK(pool->free_bytes() == base);
  }
  SUBCASE("complex<double>") {
    auto s = pool->allocate_span<std::complex<double>>(100);
    CHECK(s.size_bytes() == 100 * sizeof(std::complex<double>));
    pool->deallocate(s);
    CHECK(pool->free_bytes() == base);
  }
  CHECK(pool->free_summary().first == kPool); // fully coalesced again
}

TEST_CASE("MemoryPool: allocate_span(0) yields an empty span and frees as a no-op") {
  auto              pool = make_pool(1u << 20);
  std::size_t const base = pool->free_bytes();

  auto s = pool->allocate_span<double>(0);
  CHECK(s.empty());
  CHECK(s.data() == nullptr);
  CHECK(pool->free_bytes() == base);

  pool->deallocate(s); // must not corrupt the free list
  CHECK(pool->free_bytes() == base);
  CHECK(pool->free_summary().first == base);
}

TEST_CASE("MemoryPool: span data is writable across its full extent") {
  auto pool = make_pool(1u << 20);
  auto s    = pool->allocate_span<std::uint64_t>(1024);
  REQUIRE(s.size() == 1024);

  // Touch every element; ASan would flag a short allocation here.
  for(std::size_t i = 0; i < s.size(); ++i) { s[i] = i; }
  std::uint64_t sum = 0;
  for(auto v: s) { sum += v; }
  CHECK(sum == (1023ull * 1024ull) / 2);

  pool->deallocate(s);
}

TEST_CASE("MemoryPool: span round-trip under churn returns to baseline") {
  // The property the migration is meant to guarantee: because the span carries the size,
  // no sequence of allocate/deallocate pairs can leak or mismatch.
  constexpr std::size_t kPool = 1u << 20;
  auto                  pool  = make_pool(kPool);
  std::size_t const     base  = pool->free_bytes();

  std::vector<std::span<double>> live;
  std::size_t                    seed = 999;
  auto                           next = [&seed]() { return seed = seed * 6364136223846793005ull + 1; };

  for(int i = 0; i < 2000; ++i) {
    if(live.empty() || ((next() >> 33) % 3)) {
      if(pool->free_bytes() > (kPool / 4)) { live.push_back(pool->allocate_span<double>(1 + ((next() >> 33) % 256))); }
    }
    else {
      std::size_t const k = (next() >> 33) % live.size();
      pool->deallocate(live[k]);
      live[k] = live.back();
      live.pop_back();
    }
  }
  for(auto s: live) { pool->deallocate(s); }

  CHECK(pool->free_bytes() == base);
  CHECK(pool->free_summary().first == kPool);
}


// ---------------------------------------------------------------------------
// GEMM batch/reduction stride arithmetic (kernels/multiply.hpp gemm_wrapper).
//
// gemm_wrapper previously computed its batch/reduction strides in `int`:
//
//   int bbatch_ld  = K * N;
//   int breduce_ld = B * bbatch_ld;
//
// The *offset* accumulation was already 64-bit, because the loop counters were
// size_t and promoted the int strides. The defect is in the strides themselves --
// int*int is evaluated in int before any promotion, so B*K*N overflows. At
// B=256, K=N=4096 it wraps to exactly 0, and with BR>1 every reduction iteration
// then re-reads batch 0 instead of advancing: silently wrong numbers, no crash.
//
// These tests compare the current int64 arithmetic against the old int arithmetic
// on the shapes that distinguish them, so they fail if the widening is reverted.
// They mirror gemm_wrapper's expressions rather than calling it, since invoking it
// needs a GPU/BLAS backend.
// ---------------------------------------------------------------------------

namespace {
/// Mirrors the (fixed) stride computation in kernels::gemm_wrapper.
struct GemmStrides {
  std::int64_t cbatch_ld, abatch_ld, bbatch_ld, areduce_ld, breduce_ld;
  GemmStrides(int B, int M, int N, int K):
    cbatch_ld{static_cast<std::int64_t>(M) * N},
    abatch_ld{static_cast<std::int64_t>(M) * K},
    bbatch_ld{static_cast<std::int64_t>(K) * N},
    areduce_ld{static_cast<std::int64_t>(B) * abatch_ld},
    breduce_ld{static_cast<std::int64_t>(B) * bbatch_ld} {}
};

/// Reproduces the *old* int-based stride computation, for contrast.
///
/// The old code performed these products in `int`, which is signed overflow (UB) on the
/// very shapes this test is about. Reproducing that literally would trip UBSan, so the
/// wrap is emulated exactly using unsigned arithmetic -- which is well-defined and, on
/// every platform TAMM targets (two's complement, 32-bit int), yields the identical bit
/// pattern the old signed code produced.
struct GemmStridesInt {
  int cbatch_ld, abatch_ld, bbatch_ld, areduce_ld, breduce_ld;

  static int wrap_mul(int lhs, int rhs) noexcept {
    return static_cast<int>(static_cast<std::uint32_t>(lhs) * static_cast<std::uint32_t>(rhs));
  }

  GemmStridesInt(int B, int M, int N, int K):
    cbatch_ld(wrap_mul(M, N)), abatch_ld(wrap_mul(M, K)), bbatch_ld(wrap_mul(K, N)),
    areduce_ld(wrap_mul(B, abatch_ld)), breduce_ld(wrap_mul(B, bbatch_ld)) {}
};
} // namespace

TEST_CASE("gemm strides: B*K*N wrapping to zero is the real regression") {
  // The shape that silently broke: B=256, K=N=4096 -> B*K*N == 2^32.
  constexpr int B = 256, M = 1024, N = 4096, K = 4096;

  GemmStrides const fixed{B, M, N, K};
  CHECK(fixed.breduce_ld == 4294967296LL); // 2^32, exact

  // The old arithmetic wrapped this to exactly 0 -- the pathological value, because
  // it makes every reduction step read the same batch instead of advancing.
  GemmStridesInt const old{B, M, N, K};
  CHECK(old.breduce_ld == 0);
  CHECK(fixed.breduce_ld != static_cast<std::int64_t>(old.breduce_ld));
}

TEST_CASE("gemm strides: with BR>1 the old arithmetic produced wrong offsets") {
  // Reproduce both offset computations end to end. This is the assertion that would
  // fail if the int64 widening were reverted.
  constexpr int BR = 4, B = 256, M = 1024, N = 4096, K = 4096;

  GemmStrides const    fixed{B, M, N, K};
  GemmStridesInt const old{B, M, N, K};

  std::int64_t worst_fixed = 0;
  std::size_t  worst_old   = 0; // old loop counters were size_t
  for(std::int64_t bri = 0; bri < BR; ++bri) {
    for(std::int64_t i = 0; i < B; ++i) {
      worst_fixed = std::max(worst_fixed, bri * fixed.breduce_ld + i * fixed.bbatch_ld);
    }
  }
  for(std::size_t bri = 0; bri < static_cast<std::size_t>(BR); ++bri) {
    for(std::size_t i = 0; i < static_cast<std::size_t>(B); ++i) {
      worst_old = std::max(worst_old, bri * old.breduce_ld + i * old.bbatch_ld);
    }
  }

  CHECK(worst_fixed == 17163091968LL);
  CHECK(static_cast<std::int64_t>(worst_old) == 4278190080LL); // wrong: BR steps collapsed
  CHECK(static_cast<std::int64_t>(worst_old) != worst_fixed);
}

TEST_CASE("gemm strides: a single stride above INT_MAX is exact") {
  // M*K alone exceeds INT_MAX. Arithmetic-only check: these dimensions imply a ~17 GB
  // single block, which TAMM tiling does not produce -- included to pin the widening,
  // not to describe a reachable configuration.
  GemmStrides const    fixed{1, 46341, 46341, 46341};
  GemmStridesInt const old{1, 46341, 46341, 46341};

  CHECK(fixed.abatch_ld == 2147488281LL);
  CHECK(fixed.abatch_ld > static_cast<std::int64_t>(std::numeric_limits<int>::max()));
  CHECK(old.abatch_ld < 0); // wrapped negative
}

TEST_CASE("gemm strides: shapes that already worked are unchanged") {
  // Regression guard: the widening must not perturb any case that was already correct.
  struct Shape { int B, M, N, K; std::int64_t abatch, bbatch, areduce; };
  constexpr Shape shapes[] = {
    {1, 1, 1, 1, 1LL, 1LL, 1LL},
    {1, 512, 512, 512, 262144LL, 262144LL, 262144LL},
    {64, 2048, 2048, 2048, 4194304LL, 4194304LL, 268435456LL},
    {32, 4096, 4096, 4096, 16777216LL, 16777216LL, 536870912LL},
  };
  for(auto const& sh: shapes) {
    GemmStrides const    fixed{sh.B, sh.M, sh.N, sh.K};
    GemmStridesInt const old{sh.B, sh.M, sh.N, sh.K};
    // Hardcoded expectations, not a re-derivation of the implementation.
    CHECK(fixed.abatch_ld == sh.abatch);
    CHECK(fixed.bbatch_ld == sh.bbatch);
    CHECK(fixed.areduce_ld == sh.areduce);
    // and the old arithmetic agreed on exactly these shapes
    CHECK(static_cast<std::int64_t>(old.abatch_ld) == sh.abatch);
    CHECK(static_cast<std::int64_t>(old.areduce_ld) == sh.areduce);
  }
}
