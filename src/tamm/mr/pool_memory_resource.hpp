#pragma once

#include "aligned.hpp"
#include "coalescing_free_list.hpp"
#include "cuda_stream_view.hpp"
#include "device_memory_resource.hpp"
#include "stream_ordered_memory_resource.hpp"

#include <optional>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rmm::mr {

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
  explicit pool_memory_resource(Upstream*                  upstream_mr,
                                std::optional<std::size_t> initial_pool_size = std::nullopt,
                                std::optional<std::size_t> maximum_pool_size = std::nullopt):
    upstream_mr_{[upstream_mr]() {
      if(upstream_mr == nullptr) { std::logic_error("Unexpected null upstream pointer."); }
      return upstream_mr;
    }()} {
    if(!rmm::detail::is_aligned(initial_pool_size.value_or(0),
                                rmm::detail::CUDA_ALLOCATION_ALIGNMENT)) {
      std::logic_error("Error, Initial pool size required to be a multiple of 256 bytes");
    }
    if(!rmm::detail::is_aligned(maximum_pool_size.value_or(0),
                                rmm::detail::CUDA_ALLOCATION_ALIGNMENT)) {
      std::logic_error("Error, Maximum pool size required to be a multiple of 256 bytes");
    }

    initialize_pool(initial_pool_size, maximum_pool_size);
  }

  /**
   * @brief Destroy the `pool_memory_resource` and deallocate all memory it allocated using
   * the upstream resource.
   */
  ~pool_memory_resource() override { release(); }

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
   * @brief Computes the size of the current pool
   *
   * Includes allocated as well as free memory.
   *
   * @return std::size_t The total size of the currently allocated pool.
   */
  [[nodiscard]] std::size_t pool_size() const noexcept { return current_pool_size_; }

protected:
  using free_list  = detail::coalescing_free_list;
  using block_type = free_list::block_type;
  using typename detail::stream_ordered_memory_resource<pool_memory_resource<Upstream>,
                                                        detail::coalescing_free_list>::split_block;
  using lock_guard = std::lock_guard<std::mutex>;

  /**
   * @brief Get the maximum size of allocations supported by this memory resource
   *
   * Note this does not depend on the memory size of the device. It simply returns the maximum
   * value of `std::size_t`
   *
   * @return std::size_t The maximum size of a single allocation supported by this memory resource
   */
  [[nodiscard]] std::size_t get_maximum_allocation_size() const {
    return std::numeric_limits<std::size_t>::max();
  }

  /**
   * @brief Try to expand the pool by allocating a block of at least `min_size` bytes from
   * upstream
   *
   * Attempts to allocate `try_size` bytes from upstream. If it fails, it iteratively reduces the
   * attempted size by half until `min_size`, returning the allocated block once it succeeds.
   *
   * @throws rmm::bad_alloc if `min_size` bytes cannot be allocated from upstream or maximum pool
   * size is exceeded.
   *
   * @param try_size The initial requested size to try allocating.
   * @param min_size The minimum requested size to try allocating.
   * @param stream The stream on which the memory is to be used.
   * @return block_type a block of at least `min_size` bytes
   */
  block_type try_to_expand(std::size_t try_size, std::size_t min_size, cuda_stream_view stream) {
    while(try_size >= min_size) {
      auto block = block_from_upstream(try_size, stream);
      if(block.has_value()) {
        current_pool_size_ += block.value().size();
        return block.value();
      }
      if(try_size == min_size) {
        break; // only try `size` once
      }
      try_size = std::max(min_size, try_size / 2);
    }
  }

  /**
   * @brief Allocate initial memory for the pool
   *
   * If initial_size is unset, then queries the upstream memory resource for available memory if
   * upstream supports `get_mem_info`, or queries the device (using CUDA API) for available memory
   * if not. Then attempts to initialize to half the available memory.
   *
   * If initial_size is set, then tries to initialize the pool to that size.
   *
   * @param initial_size The optional initial size for the pool
   * @param maximum_size The optional maximum size for the pool
   */
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  void initialize_pool(std::optional<std::size_t> initial_size,
                       std::optional<std::size_t> maximum_size) {
    auto const try_size = [&]() {
      if(not initial_size.has_value()) {
        std::size_t free, total;
        cudaMemGetInfo(&free, &total);
        return rmm::detail::align_up(std::min(free, total / 2),
                                     rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
      }
      return initial_size.value();
    }();

    current_pool_size_ = 0; // try_to_expand will set this if it succeeds
    maximum_pool_size_ = maximum_size;

    if(!(try_size <= maximum_pool_size_.value_or(std::numeric_limits<std::size_t>::max()))) {
      std::logic_error("Initial pool size exceeds the maximum pool size!");
    }

    if(try_size > 0) {
      auto const block = try_to_expand(try_size, try_size, cuda_stream_legacy);
      this->insert_block(block, cuda_stream_legacy);
    }
  }

  /**
   * @brief Allocate space from upstream to supply the suballocation pool and return
   * a sufficiently sized block.
   *
   * @param size The minimum size to allocate
   * @param blocks The free list (ignored in this implementation)
   * @param stream The stream on which the memory is to be used.
   * @return block_type a block of at least `size` bytes
   */
  block_type expand_pool(std::size_t size, free_list& blocks, cuda_stream_view stream) {
    // Strategy: If maximum_pool_size_ is set, then grow geometrically, e.g. by halfway to the
    // limit each time. If it is not set, grow exponentially, e.g. by doubling the pool size each
    // time. Upon failure, attempt to back off exponentially, e.g. by half the attempted size,
    // until either success or the attempt is less than the requested size.
    return try_to_expand(size_to_grow(size), size, stream);
  }

  /**
   * @brief Given a minimum size, computes an appropriate size to grow the pool.
   *
   * Strategy is to try to grow the pool by half the difference between the configured maximum
   * pool size and the current pool size, if the maximum pool size is set. If it is not set, try
   * to double the current pool size.
   *
   * Returns 0 if the requested size cannot be satisfied.
   *
   * @param size The size of the minimum allocation immediately needed
   * @return std::size_t The computed size to grow the pool.
   */
  [[nodiscard]] std::size_t size_to_grow(std::size_t size) const {
    if(maximum_pool_size_.has_value()) {
      auto const unaligned_remaining = maximum_pool_size_.value() - pool_size();
      using rmm::detail::align_up;
      auto const remaining = align_up(unaligned_remaining, rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
      auto const aligned_size = align_up(size, rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
      return (aligned_size <= remaining) ? std::max(aligned_size, remaining / 2) : 0;
    }
    return std::max(size, pool_size());
  };

  /**
   * @brief Allocate a block from upstream to expand the suballocation pool.
   *
   * @param size The size in bytes to allocate from the upstream resource
   * @param stream The stream on which the memory is to be used.
   * @return block_type The allocated block
   */
  std::optional<block_type> block_from_upstream(std::size_t size, cuda_stream_view stream) {
    if(size == 0) { return {}; }

    try {
      void* ptr = get_upstream()->allocate(size, stream);
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
   * @param stream_event The stream and associated event on which the allocation will be used.
   * @return A pair comprising the allocated pointer and any unallocated remainder of the input
   * block.
   */
  split_block allocate_from_block(block_type const& block, std::size_t size) {
    block_type const alloc{block.pointer(), size, block.is_head()};
    auto             rest = (block.size() > size)
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
   * @param stream The stream-event pair for the stream on which the memory was last used.
   * @return The (now freed) block associated with `p`. The caller is expected to return the block
   * to the pool.
   */
  block_type free_block(void* ptr, std::size_t size) noexcept {
    auto const iter = upstream_blocks_.find(static_cast<char*>(ptr));
    return block_type{static_cast<char*>(ptr), size, (iter != upstream_blocks_.end())};
  }

  /**
   * @brief Free all memory allocated from the upstream memory_resource.
   *
   */
  void release() {
    lock_guard lock(this->get_mutex());

    for(auto block: upstream_blocks_) { get_upstream()->deallocate(block.pointer(), block.size()); }
    upstream_blocks_.clear();
    current_pool_size_ = 0;
  }

  /**
   * @brief Get the largest available block size and total free size in the specified free list
   *
   * This is intended only for debugging
   *
   * @param blocks The free list from which to return the summary
   * @return std::pair<std::size_t, std::size_t> Pair of largest available block, total free size
   */
  std::pair<std::size_t, std::size_t> free_list_summary(free_list const& blocks) {
    std::size_t largest{};
    std::size_t total{};
    std::for_each(blocks.cbegin(), blocks.cend(), [&largest, &total](auto const& block) {
      total += block.size();
      largest = std::max(largest, block.size());
    });
    return {largest, total};
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws nothing
   *
   * @param stream to execute on
   * @return std::pair contaiing free_size and total_size of memory
   */
  [[nodiscard]] std::pair<std::size_t, std::size_t>
  do_get_mem_info(cuda_stream_view stream) const override {
    // TODO implement this
    return {0, 0};
  }

private:
  Upstream*                  upstream_mr_; // The "heap" to allocate the pool from
  std::size_t                current_pool_size_{};
  std::optional<std::size_t> maximum_pool_size_{};

  // blocks allocated from upstream
  std::set<block_type, rmm::mr::detail::compare_blocks<block_type>> upstream_blocks_;
}; // namespace mr

} // namespace rmm::mr
