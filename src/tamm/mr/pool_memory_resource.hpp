#pragma once

#include "aligned.hpp"
#include "coalescing_free_list.hpp"
#include "device_memory_resource.hpp"
#include "stream_ordered_memory_resource.hpp"

#include "tamm/errors.hpp" // tamm_terminate

#include <optional>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tamm::rmm::mr {

namespace detail {
// TAMM_RMM_TRACK = 0(default), 1
// When set, the pool records every outstanding allocation and validates that each block is
// returned with the size it was handed out with. A size mismatch silently corrupts the free
// list (freeing short orphans a sliver forever; freeing long overlaps the next block), and
// is otherwise almost impossible to attribute to a call site after the fact.
inline bool rmm_track_allocations() {
  static bool const enabled = [] {
    const char* raw = std::getenv("TAMM_RMM_TRACK");
    return (raw != nullptr) && (std::atoi(raw) != 0);
  }();
  return enabled;
}
} // namespace detail

/**
 * @brief A coalescing best-fit suballocator which uses a pool of memory allocated from
 *        an upstream memory_resource.
 *
 * Allocation (do_allocate()) and deallocation (do_deallocate()) are thread-safe. Also,
 * this class is compatible with CUDA per-thread default stream.
 *
 * @tparam UpstreamResource memory_resource to use for allocating the pool. Implements
 *                          rmm::mr::device_memory_resource interface.
 */
template<typename Upstream>
class pool_memory_resource final:
  public detail::stream_ordered_memory_resource<pool_memory_resource<Upstream>,
                                                detail::coalescing_free_list> {
public:
  friend class detail::stream_ordered_memory_resource<pool_memory_resource<Upstream>,
                                                      detail::coalescing_free_list>;

  /**
   * @brief Construct a `pool_memory_resource` and allocate the initial device memory pool using
   * `upstream_mr`.
   *
   * @throws rmm::logic_error if `upstream_mr == nullptr`
   * @throws rmm::logic_error if `initial_pool_size` is neither the default nor aligned to a
   * multiple of pool_memory_resource::allocation_alignment bytes.
   * @throws rmm::logic_error if `maximum_pool_size` is neither the default nor aligned to a
   * multiple of pool_memory_resource::allocation_alignment bytes.
   *
   * @param upstream_mr The memory_resource from which to allocate blocks for the pool.
   * @param initial_pool_size Minimum size, in bytes, of the initial pool. Defaults to half of the
   * available memory on the current device.
   * @param maximum_pool_size Maximum size, in bytes, that the pool can grow to. Defaults to all
   * of the available memory on the current device.
   */
  explicit pool_memory_resource(Upstream* upstream_mr, std::size_t maximum_pool_size):
    upstream_mr_{[upstream_mr]() {
      // NOTE: these were previously discarded temporaries (`std::logic_error(...)` with no
      // `throw`), so neither check had any effect. They are now actually enforced.
      if(upstream_mr == nullptr) {
        throw std::logic_error("Unexpected null upstream pointer.");
      }
      return upstream_mr;
    }()} {
    // Round down rather than reject: callers derive the pool size from a runtime
    // percentage of free memory, which is essentially never aligned.
    maximum_pool_size =
      rmm::detail::align_down(maximum_pool_size, rmm::detail::RMM_ALLOCATION_ALIGNMENT);

    initialize_pool(maximum_pool_size);
  }

  /**
   * @brief Destroy the `pool_memory_resource` and deallocate all memory it allocated using
   * the upstream resource.
   */
  ~pool_memory_resource() override {
    release();
    delete upstream_mr_;
  }

  pool_memory_resource()                                       = delete;
  pool_memory_resource(pool_memory_resource const&)            = delete;
  pool_memory_resource(pool_memory_resource&&)                 = delete;
  pool_memory_resource& operator=(pool_memory_resource const&) = delete;
  pool_memory_resource& operator=(pool_memory_resource&&)      = delete;

  /**
   * @brief Get the upstream memory_resource object.
   *
   * @return UpstreamResource* the upstream memory resource.
   */
  Upstream* get_upstream() const noexcept { return upstream_mr_; }

  /**
   * @brief The total size of the pool, allocated plus free.
   *
   * The pool is fixed-size: this never changes after construction.
   *
   * @return std::size_t total pool size in bytes
   */
  [[nodiscard]] std::size_t pool_size() const noexcept { return current_pool_size_; }

  /**
   * @brief Largest free block and total free bytes currently in the pool.
   *
   * Intended for diagnostics and leak tests. Takes the pool lock.
   *
   * @return Pair of {largest free block, total free bytes}.
   */
  [[nodiscard]] std::pair<std::size_t, std::size_t> free_summary() {
    std::lock_guard<std::mutex> lock(this->get_mutex());
    return this->free_list_summary();
  }

  /**
   * @brief Total free bytes currently in the pool.
   *
   * @return std::size_t free bytes
   */
  [[nodiscard]] std::size_t free_bytes() { return free_summary().second; }

protected:
  using free_list  = detail::coalescing_free_list;
  using block_type = free_list::block_type;
  using typename detail::stream_ordered_memory_resource<pool_memory_resource<Upstream>,
                                                        detail::coalescing_free_list>::split_block;

  /**
   * @brief Get the maximum size of allocations supported by this memory resource
   *
   * This pool is fixed-size and is never grown from upstream after construction, so no single
   * allocation can ever exceed the pool itself. Returning the true ceiling (rather than
   * `SIZE_MAX`) lets an over-sized request be reported as such, instead of falling through to
   * the generic "no block large enough" path.
   *
   * @return std::size_t The maximum size of a single allocation supported by this memory resource
   */
  [[nodiscard]] std::size_t get_maximum_allocation_size() const { return current_pool_size_; }

  /**
   * @brief Allocate initial memory for the pool
   *
   * If initial_size is unset, then queries the upstream memory resource for available memory if
   * upstream supports `get_mem_info`, or queries the device (using GPU API) for available memory
   * if not. Then attempts to initialize to half the available memory.
   *
   * @param maximum_size The optional maximum size for the pool
   */
  void initialize_pool(std::size_t maximum_size) {
    current_pool_size_ = 0;

    if(maximum_size == 0) {
      std::ostringstream os;
      os << "[TAMM ERROR] RMM initialize_pool() called with a zero-sized pool.\n"
         << "  Check TAMM_GPU_POOL / TAMM_CPU_POOL and the detected ranks-per-GPU.\n"
         << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }

    auto const block = block_from_upstream(maximum_size);
    if(block.has_value()) {
      current_pool_size_ = block.value().size();
      this->insert_block(block.value());
    }
    else {
      std::ostringstream os;
      os << "[TAMM ERROR] RMM initialize_pool() failed to reserve "
         << maximum_size << " B from the upstream resource.\n"
         << "  The upstream allocation was rejected -- typically too many processes per node,\n"
         << "  or another allocator (e.g. GA) already holds the memory.\n"
         << "  Lower TAMM_GPU_POOL / TAMM_CPU_POOL, or reduce ranks per node.\n"
         << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }
  }

  /**
   * @brief Allocate a block from upstream to expand the suballocation pool.
   *
   * @param size The size in bytes to allocate from the upstream resource
   * @return block_type The allocated block
   */
  std::optional<block_type> block_from_upstream(std::size_t size) {
    if(size == 0) { return {}; }

    try {
      void* ptr = get_upstream()->allocate(size);
      return std::optional<block_type>{
        *upstream_blocks_.emplace(static_cast<char*>(ptr), size, true).first};
    } catch(std::exception const& e) { return std::nullopt; }
  }

  /**
   * @brief Splits `block` if necessary to return a pointer to memory of `size` bytes.
   *
   * If the block is split, the remainder is returned to the pool.
   *
   * @param block The block to allocate from.
   * @param size The size in bytes of the requested allocation.
   * @return A pair comprising the allocated pointer and any unallocated remainder of the input
   * block.
   */
  split_block allocate_from_block(block_type const& block, std::size_t size) {
    block_type const alloc{block.pointer(), size, block.is_head()};

    if(detail::rmm_track_allocations()) { outstanding_[block.pointer()] = size; }

    auto rest = (block.size() > size)
                  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                  ? block_type{block.pointer() + size, block.size() - size, false}
                  : block_type{};
    return {alloc, rest};
  }

  /**
   * @brief Finds, frees and returns the block associated with pointer `ptr`.
   *
   * @param ptr The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @return The (now freed) block associated with `p`. The caller is expected to return the block
   * to the pool.
   */
  block_type free_block(void* ptr, std::size_t size, std::string& err) noexcept {
    if(detail::rmm_track_allocations()) { validate_free(static_cast<char*>(ptr), size, err); }

    auto const iter = upstream_blocks_.find(static_cast<char*>(ptr));
    return block_type{static_cast<char*>(ptr), size, (iter != upstream_blocks_.end())};
  }

  /**
   * @brief Check a deallocation against the recorded allocation (TAMM_RMM_TRACK=1).
   *
   * Reports, and does not attempt to repair, two classes of error:
   *  - freeing a pointer the pool never handed out (double free, or a foreign pointer)
   *  - freeing with a size different from the allocation size
   *
   * Both silently corrupt the free list, so it is far better to fail loudly at the
   * offending call than to hit an inexplicable allocation failure thousands of blocks later.
   *
   * @note Writes the diagnostic to `err` rather than terminating here. The caller holds the
   * pool mutex, and tamm_terminate() calls exit(), which runs static destructors that
   * re-enter release() on that same non-recursive mutex. Termination must happen only after
   * the lock has been released -- see do_deallocate().
   */
  void validate_free(char* ptr, std::size_t size, std::string& err) noexcept {
    auto const iter = outstanding_.find(ptr);

    if(iter == outstanding_.end()) {
      std::ostringstream os;
      os << "[TAMM ERROR] Pool deallocation of an untracked pointer.\n"
         << "  pointer     : " << static_cast<void*>(ptr) << "\n"
         << "  size        : " << size << " B\n"
         << "  This pointer was not handed out by this pool, or has already been freed\n"
         << "  (double free). Freeing it corrupts the free list.\n"
         << __FILE__ << ":L" << __LINE__;
      err = os.str();
      return;
    }

    if(iter->second != size) {
      std::ostringstream os;
      os << "[TAMM ERROR] Pool deallocation size does not match allocation size.\n"
         << "  pointer      : " << static_cast<void*>(ptr) << "\n"
         << "  allocated as : " << iter->second << " B\n"
         << "  freed as     : " << size << " B\n"
         << "  Freeing short orphans the remainder permanently; freeing long overlaps the\n"
         << "  next block. Either way the free list is corrupted and later allocations will\n"
         << "  fail with no obvious cause. Fix the call site so both sizes agree.\n"
         << __FILE__ << ":L" << __LINE__;
      err = os.str();
      return;
    }

    outstanding_.erase(iter);
  }

public:
  /**
   * @brief Number of allocations, and total bytes, still outstanding (TAMM_RMM_TRACK=1).
   *
   * A nonzero count once the application has released everything it believes it owns is a
   * leak. Returns {0,0} when tracking is disabled.
   *
   * @return Pair of {allocation count, total bytes}.
   */
  [[nodiscard]] std::pair<std::size_t, std::size_t> outstanding_summary() {
    std::lock_guard<std::mutex> lock(this->get_mutex());
    std::size_t                 bytes{0};
    for(auto const& [ptr, sz]: outstanding_) { bytes += sz; }
    return {outstanding_.size(), bytes};
  }

protected:

  /**
   * @brief Free all memory allocated from the upstream memory_resource.
   *
   */
  void release() {
    std::lock_guard<std::mutex> lock(this->get_mutex());

    // Report leaks before tearing the pool down. Deliberately does not terminate: this runs
    // from the destructor, and aborting during static destruction is worse than a warning.
    // On a fatal abort the outstanding blocks are whatever was legitimately live when the
    // process was killed, so they are reported without calling them leaks.
    if(detail::rmm_track_allocations() && !outstanding_.empty()) {
      std::size_t bytes{0};
      for(auto const& [ptr, sz]: outstanding_) { bytes += sz; }
      if(tamm::tamm_terminating()) {
        std::cerr << "[TAMM RMM] " << outstanding_.size()
                  << " allocation(s) live at abort (not necessarily leaks), totalling " << bytes
                  << " B.\n";
      }
      else {
        std::cerr << "[TAMM RMM] LEAK: " << outstanding_.size()
                  << " allocation(s) still outstanding at pool teardown, totalling " << bytes
                  << " B.\n";
      }

      // Group by size: a repeated size points straight at one call site.
      std::map<std::size_t, std::size_t> by_size;
      for(auto const& [ptr, sz]: outstanding_) { ++by_size[sz]; }
      std::cerr << "[TAMM RMM]   outstanding blocks by size (size B x count):\n";
      for(auto const& [sz, count]: by_size) {
        std::cerr << "[TAMM RMM]     " << sz << " x " << count << "\n";
      }
      std::cerr.flush();
    }
    outstanding_.clear();

    for(auto block: upstream_blocks_) { get_upstream()->deallocate(block.pointer(), block.size()); }
    upstream_blocks_.clear();
    current_pool_size_ = 0;
  }

private:
  Upstream* upstream_mr_; // The "heap" to allocate the pool from
  // The pool is fixed-size, so this is both the current and the maximum size. It is the
  // ceiling reported by get_maximum_allocation_size().
  std::size_t current_pool_size_{};

  // blocks allocated from upstream
  std::set<block_type, rmm::mr::detail::compare_blocks<block_type>> upstream_blocks_;

  // Outstanding allocations, keyed by the pointer handed to the caller. Populated only when
  // TAMM_RMM_TRACK=1; empty and untouched otherwise, so the default path pays nothing beyond
  // one predictable branch.
  //
  // unordered_map, not map: measured on this workload, the ordered map costs ~114 ns per
  // alloc/free pair (+71% against a ~160 ns pair) versus ~61 ns (+38%) for the hash map.
  // Ordering is never needed here -- lookups are always by exact pointer.
  std::unordered_map<char*, std::size_t> outstanding_;
}; // namespace mr

} // namespace tamm::rmm::mr
