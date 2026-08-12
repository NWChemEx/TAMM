#pragma once

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>

namespace tamm::rmm::detail {

/**
 * @brief Default alignment used for CPU/GPU memory allocation.
 *
 */
#if defined(USE_DPCPP)
// currently L0 uses 4 bytes alignment (default)
static constexpr std::size_t RMM_ALLOCATION_ALIGNMENT{4};
#elif defined(USE_HIP)
static constexpr std::size_t RMM_ALLOCATION_ALIGNMENT{128};
#elif defined(USE_CUDA)
static constexpr std::size_t RMM_ALLOCATION_ALIGNMENT{256};
#else
// Default alignment used for host memory allocated by RMM.
static constexpr std::size_t RMM_ALLOCATION_ALIGNMENT{alignof(std::max_align_t)};
#endif

/**
 * @brief Returns whether or not `n` is a power of 2.
 *
 * Uses std::has_single_bit (C++20). The previous hand-rolled `(v & (v-1)) == 0` claimed
 * that 0 is a power of two, which made `is_supported_alignment(0)` pass and
 * `align_up(v, 0)` return SIZE_MAX for every nonzero v.
 */
[[nodiscard]] constexpr bool is_pow2(std::size_t value) noexcept {
  return std::has_single_bit(value);
}

/**
 * @brief Returns whether or not `alignment` is a valid memory alignment.
 *
 */
[[nodiscard]] constexpr bool is_supported_alignment(std::size_t alignment) noexcept {
  return is_pow2(alignment);
}

static_assert(!is_pow2(0), "0 is not a power of two");
static_assert(is_pow2(1) && is_pow2(2) && is_pow2(256));
static_assert(!is_pow2(3) && !is_pow2(255));

/**
 * @brief Align up to nearest multiple of specified power of 2
 *
 * @param[in] v value to align
 * @param[in] alignment amount, in bytes, must be a power of 2
 *
 * @return Return the aligned value, as one would expect
 */
[[nodiscard]] constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
  assert(is_supported_alignment(alignment));
  // Saturate rather than wrap: (value + alignment - 1) overflows to a small number for
  // values near SIZE_MAX, which would turn an absurd request into a tiny one that then
  // succeeds and hands back a buffer far smaller than the caller asked for.
  constexpr std::size_t max_v = std::numeric_limits<std::size_t>::max();
  if(value > max_v - (alignment - 1)) { return max_v; }
  return (value + (alignment - 1)) & ~(alignment - 1);
}

/**
 * @brief Align down to the nearest multiple of specified power of 2
 *
 * @param[in] v value to align
 * @param[in] alignment amount, in bytes, must be a power of 2
 *
 * @return Return the aligned value, as one would expect
 */
[[nodiscard]] constexpr std::size_t align_down(std::size_t value, std::size_t alignment) noexcept {
  assert(is_supported_alignment(alignment));
  return value & ~(alignment - 1);
}

/**
 * @brief Checks whether a value is aligned to a multiple of a specified power of 2
 *
 * @param[in] v value to check for alignment
 * @param[in] alignment amount, in bytes, must be a power of 2
 *
 * @return true if aligned
 */
[[nodiscard]] constexpr bool is_aligned(std::size_t value, std::size_t alignment) noexcept {
  assert(is_supported_alignment(alignment));
  return value == align_down(value, alignment);
}

// Compile-time proofs of the alignment invariants the pool depends on. These cost nothing
// at runtime and fail the build rather than corrupting a free list.
static_assert(align_up(0, 256) == 0);
static_assert(align_up(1, 256) == 256);
static_assert(align_up(256, 256) == 256, "already-aligned values must not be advanced");
static_assert(align_up(257, 256) == 512);
static_assert(align_down(255, 256) == 0);
static_assert(align_down(256, 256) == 256);
static_assert(align_down(511, 256) == 256);
// align_up must saturate, never wrap, near the top of the range.
static_assert(align_up(std::numeric_limits<std::size_t>::max(), 256) ==
                std::numeric_limits<std::size_t>::max(),
              "align_up must saturate rather than wrap to a small value");
// align_up is idempotent and always >= its input.
static_assert(align_up(align_up(1000, 256), 256) == align_up(1000, 256));
static_assert(align_up(1000, 256) >= 1000);
static_assert(is_aligned(align_up(1000, 256), 256));
static_assert(is_aligned(align_down(1000, 256), 256));

inline bool is_pointer_aligned(void* ptr, std::size_t alignment = RMM_ALLOCATION_ALIGNMENT) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return rmm::detail::is_aligned(reinterpret_cast<ptrdiff_t>(ptr), alignment);
}

/**
 * @brief The true number of bytes requested from the underlying allocator for a
 * `bytes`-sized allocation with `alignment` alignment.
 *
 * `aligned_allocate` over-allocates to leave room for alignment correction and for the
 * stored offset. Size-aware deallocators (notably `numa_free`, which unmaps the length it
 * is given) must be handed *this* value, not the caller's `bytes` -- otherwise the tail of
 * every allocation is never returned to the OS.
 *
 * Single definition so allocate and deallocate can never disagree.
 */
[[nodiscard]] constexpr std::size_t detail_padded_size(std::size_t bytes,
                                                       std::size_t alignment) noexcept {
  return bytes + alignment + sizeof(std::ptrdiff_t);
}

/**
 * @brief Normalize a caller-supplied alignment to one this allocator supports.
 *
 * Both `aligned_allocate` and `aligned_deallocate` must agree on the alignment, since it
 * feeds `detail_padded_size()` and therefore the byte count handed to size-aware
 * deallocators. Normalizing in one shared place is what makes them agree by construction;
 * doing it only on the allocate side silently mismatched the padded size.
 */
[[nodiscard]] constexpr std::size_t supported_alignment_or_default(std::size_t alignment) noexcept {
  return is_supported_alignment(alignment) ? alignment : RMM_ALLOCATION_ALIGNMENT;
}

// The allocate and deallocate sides must derive identical padded sizes, including for
// alignments that get normalized. This is the invariant whose violation leaked memory
// through numa_free().
static_assert(detail_padded_size(1000, supported_alignment_or_default(256)) ==
              detail_padded_size(1000, supported_alignment_or_default(256)));
static_assert(supported_alignment_or_default(0) == RMM_ALLOCATION_ALIGNMENT,
              "alignment 0 must normalize, not propagate");
static_assert(supported_alignment_or_default(3) == RMM_ALLOCATION_ALIGNMENT,
              "non-power-of-two alignment must normalize");
static_assert(supported_alignment_or_default(256) == 256, "valid alignment must pass through");

/**
 * @brief Allocates sufficient memory to satisfy the requested size `bytes` with
 * alignment `alignment` using the unary callable `alloc` to allocate memory.
 *
 * Given a pointer `p` to an allocation of size `n` returned from the unary
 * callable `alloc`, the pointer `q` returned from `aligned_alloc` points to a
 * location within the `n` bytes with sufficient space for `bytes` that
 * satisfies `alignment`.
 *
 * In order to retrieve the original allocation pointer `p`, the offset
 * between `p` and `q` is stored at `q - sizeof(std::ptrdiff_t)`.
 *
 * Allocations returned from `aligned_allocate` *MUST* be freed by calling
 * `aligned_deallocate` with the same arguments for `bytes` and `alignment` with
 * a compatible unary `dealloc` callable capable of freeing the memory returned
 * from `alloc`.
 *
 * If `alignment` is not a power of 2, behavior is undefined.
 *
 * @param bytes The desired size of the allocation
 * @param alignment Desired alignment of allocation
 * @param alloc Unary callable given a size `n` will allocate at least `n` bytes
 * of host memory.
 * @tparam Alloc a unary callable type that allocates memory.
 * @return void* Pointer into allocation of at least `bytes` with desired
 * `alignment`.
 */
template<typename Alloc>
void* aligned_allocate(std::size_t bytes, std::size_t alignment, Alloc alloc) {
  // Normalized identically in aligned_deallocate, so the padded size always matches.
  alignment = supported_alignment_or_default(alignment);

  // allocate memory for bytes, plus potential alignment correction,
  // plus store of the correction offset.
  // Must match detail_padded_size() exactly -- aligned_deallocate reconstructs this value
  // to hand a correct byte count to size-aware deallocators such as numa_free().
  std::size_t const padded_allocation_size = detail_padded_size(bytes, alignment);

  char* const original = static_cast<char*>(alloc(padded_allocation_size));

  // Not every allocator throws on failure: numa_alloc_onnode() returns NULL. Without this
  // check the offset store below would write through a null pointer, and callers that
  // (correctly) expect std::bad_alloc -- e.g. pool_memory_resource::block_from_upstream --
  // would never see the failure.
  if(original == nullptr) { throw std::bad_alloc{}; }

  // std::align takes its space parameter by mutable reference and shrinks it, so it needs
  // its own copy -- padded_allocation_size must stay intact to match aligned_deallocate.
  std::size_t space_remaining = padded_allocation_size;

  // account for storage of offset immediately prior to the aligned pointer
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  void* aligned{original + sizeof(std::ptrdiff_t)};

  // std::align modifies `aligned` to point to the first aligned location
  std::align(alignment, bytes, aligned, space_remaining);

  // Compute the offset between the original and aligned pointers
  std::ptrdiff_t offset = static_cast<char*>(aligned) - original;

  // Store the offset immediately before the aligned pointer
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  *(static_cast<std::ptrdiff_t*>(aligned) - 1) = offset;

  return aligned;
}

/**
 * @brief Frees an allocation returned from `aligned_allocate`.
 *
 * Allocations returned from `aligned_allocate` *MUST* be freed by calling
 * `aligned_deallocate` with the same arguments for `bytes` and `alignment`
 * with a compatible `dealloc` callable capable of freeing the memory
 * returned from `alloc`.
 *
 * The callable is *binary*: it receives both the original (unaligned) pointer and the
 * padded allocation size actually requested from the allocator. Size-aware deallocators
 * such as `numa_free` need the padded size; size-agnostic ones (`::operator delete`) can
 * ignore the second argument.
 *
 * @param ptr The aligned pointer to deallocate
 * @param bytes The number of bytes requested from `aligned_allocate`
 * @param alignment The alignment required from `aligned_allocate`
 * @param dealloc A binary callable `(void* original, std::size_t padded)` capable of
 * freeing memory returned from `alloc` in `aligned_allocate`.
 * @tparam Dealloc A binary callable type that deallocates memory.
 */
template<typename Dealloc>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void aligned_deallocate(void* ptr, std::size_t bytes, std::size_t alignment, Dealloc dealloc) {
  // Must apply the same normalization aligned_allocate did, or the reconstructed padded
  // size will not match what was actually requested from the allocator.
  alignment = supported_alignment_or_default(alignment);

  // Get offset from the location immediately prior to the aligned pointer
  // NOLINTNEXTLINE
  std::ptrdiff_t const offset = *(reinterpret_cast<std::ptrdiff_t*>(ptr) - 1);

  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  void* const original = static_cast<char*>(ptr) - offset;

  // Reconstruct exactly what aligned_allocate asked the allocator for.
  dealloc(original, detail_padded_size(bytes, alignment));
}
} // namespace tamm::rmm::detail
