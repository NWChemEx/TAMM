#pragma once

#include "aligned.hpp"
#include "coalescing_free_list.hpp"
#include "device_memory_resource.hpp"
#include "stream_ordered_memory_resource.hpp"

#include <optional>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tamm::rmm::mr {

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
      if(upstream_mr == nullptr) { std::logic_error("Unexpected null upstream pointer."); }
      return upstream_mr;
    }()} {
    if(!rmm::detail::is_aligned(maximum_pool_size, rmm::detail::RMM_ALLOCATION_ALIGNMENT)) {
      std::logic_error(
        "Error, Maximum pool size required to be a multiple of 256/std::max_align_t bytes");
    }

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

protected:
  using free_list  = detail::coalescing_free_list;
  using block_type = free_list::block_type;
  using typename detail::stream_ordered_memory_resource<pool_memory_resource<Upstream>,
                                                        detail::coalescing_free_list>::split_block;

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
   * @brief Allocate initial memory for the pool
   *
   * If initial_size is unset, then queries the upstream memory resource for available memory if
   * upstream supports `get_mem_info`, or queries the device (using GPU API) for available memory
   * if not. Then attempts to initialize to half the available memory.
   *
   * @param maximum_size The optional maximum size for the pool
   */
  void initialize_pool(std::size_t maximum_size) {
    auto const block = block_from_upstream(maximum_size);
    if(block.has_value()) { this->insert_block(block.value()); }
    else {
      std::ostringstream os;
      os << "[TAMM ERROR] RMM initialize_pool() failed, too many processes per node!\n"
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
    for(auto block: upstream_blocks_) { get_upstream()->deallocate(block.pointer(), block.size()); }
    upstream_blocks_.clear();
  }

private:
  Upstream*   upstream_mr_; // The "heap" to allocate the pool from
  std::size_t maximum_pool_size_{};

  // blocks allocated from upstream
  std::set<block_type, rmm::mr::detail::compare_blocks<block_type>> upstream_blocks_;
}; // namespace mr

} // namespace tamm::rmm::mr
