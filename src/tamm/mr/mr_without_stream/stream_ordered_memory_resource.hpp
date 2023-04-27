#pragma once

#include "aligned.hpp"
#include "device_memory_resource.hpp"

#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <thread>
#include <unordered_map>

namespace tamm::rmm::mr::detail {

/**
 * @brief A CRTP helper function
 *
 * https://www.fluentcpp.com/2017/05/19/crtp-helper/
 *
 * Does two things:
 * 1. Makes "crtp" explicit in the inheritance structure of a CRTP base class.
 * 2. Avoids having to `static_cast` in a lot of places
 *
 * @tparam T The derived class in a CRTP hierarchy
 */
template<typename T>
struct crtp {
  [[nodiscard]] T&       underlying() { return static_cast<T&>(*this); }
  [[nodiscard]] T const& underlying() const { return static_cast<T const&>(*this); }
};

/**
 * @brief Base class for a stream-ordered memory resource
 *
 * This base class uses CRTP (https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
 * to provide static polymorphism to enable defining suballocator resources that maintain separate
 * pools per stream. All of the stream-ordering logic is contained in this class, but the logic
 * to determine how memory pools are managed and the type of allocation is implented in a derived
 * class and in a free list class.
 *
 * For example, a coalescing pool memory resource uses a coalescing_free_list and maintains data
 * structures for allocated blocks and has functions to allocate and free blocks and to expand the
 * pool.
 *
 * Classes derived from stream_ordered_memory_resource must implement the following four methods,
 * documented separately:
 *
 * 1. `std::size_t get_maximum_allocation_size() const`
 * 2. `block_type expand_pool(std::size_t size, free_list& blocks, gpu_stream_view stream)`
 * 3. `split_block allocate_from_block(block_type const& b, std::size_t size)`
 * 4. `block_type free_block(void* p, std::size_t size) noexcept`
 */
template<typename PoolResource, typename FreeListType>
class stream_ordered_memory_resource: public crtp<PoolResource>, public device_memory_resource {
public:
  ~stream_ordered_memory_resource() override { release(); }

  stream_ordered_memory_resource()                                                 = default;
  stream_ordered_memory_resource(stream_ordered_memory_resource const&)            = delete;
  stream_ordered_memory_resource(stream_ordered_memory_resource&&)                 = delete;
  stream_ordered_memory_resource& operator=(stream_ordered_memory_resource const&) = delete;
  stream_ordered_memory_resource& operator=(stream_ordered_memory_resource&&)      = delete;

protected:
  using free_list  = FreeListType;
  using block_type = typename free_list::block_type;

  // Derived classes must implement these four methods

  /// Pair representing a block that has been split for allocation
  using split_block = std::pair<block_type, block_type>;

  /**
   * @brief Returns the block `b` (last used) to the pool.
   *
   * @param block The block to insert into the pool.
   */
  void insert_block(block_type const& block) {
      // ABB this is incorrect since we need to insert into free_list and not to free_blocks_.insert()
      //free_blocks_.emplace_back(block);      
  }

  /**
   * @brief Allocates memory of size at least `bytes`.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param size The size in bytes of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t size) override {
    if(size <= 0) { return nullptr; }

    size = rmm::detail::align_up(size, rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
    if(!(size <= this->underlying().get_maximum_allocation_size())) {
      std::cerr << "Maximum allocation size exceeded! \n";
    }
    auto const block = this->underlying().get_block(size);

    return block.pointer();
  }

  /**
   * @brief Deallocate memory pointed to by `p`.
   *
   * @throws nothing
   *
   * @param p Pointer to be deallocated
   * @param size The size in bytes of the allocation to deallocate
   */
  void do_deallocate(void* ptr, std::size_t size) override {
    if(size <= 0 || ptr == nullptr) { return; }

    size             = rmm::detail::align_up(size, rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
    auto const block = this->underlying().free_block(ptr, size);

    free_blocks_.insert(block);
  }

private:
  /**
   * @brief Splits a block into an allocated block of `size` bytes and a remainder block, and
   * inserts the remainder into a free list.
   *
   * @param block The block to split into allocated and remainder portions.
   * @param size The size of the block to allocate from `b`.
   * @param blocks The `free_list` in which to insert the remainder block.
   * @return The allocated block.
   */
  block_type allocate_and_insert_remainder(block_type block, std::size_t size, free_list& blocks) {
    auto const [allocated, remainder] = this->underlying().allocate_from_block(block, size);
    if(remainder.is_valid()) { blocks.insert(remainder); }
    return allocated;
  }

  /**
   * @brief Get an available memory block of at least `size` bytes
   *
   * @param size The number of bytes to allocate
   * @return block_type A block of memory of at least `size` bytes
   */
  block_type get_block(std::size_t size) {
    // Try to find a satisfactory block in free list for the same stream (no sync required)
    auto iter = free_blocks_.find(stream_event);
    if(iter != free_blocks_.end()) {
      block_type const block = iter->second.get_block(size);
      if(block.is_valid()) { return allocate_and_insert_remainder(block, size, iter->second); }
    }

     free_list& blocks = (iter != free_blocks_.end()) ? iter->second
	 : free_blocks_[stream_event];

    // Try to find an existing block in another stream
    {
      block_type const block = get_block_from_other_stream(size, stream_event, blocks, false);
      if(block.is_valid()) { return block; }
    }

    // no large enough blocks available on other streams, so sync and merge until we find one
    {
      block_type const block = get_block_from_other_stream(size, stream_event, blocks, true);
      if(block.is_valid()) { return block; }
    }

    // no large enough blocks available after merging, so grow the pool
    block_type const block =
      this->underlying().expand_pool(size, blocks, gpu_stream_view{stream_event.stream});

    return allocate_and_insert_remainder(block, size, blocks);

//////////////////////////
    // this is the implementation for with out use of the "Streams"
    // Try to find a satisfactory block in free list for the same stream (no sync required)
    for(auto iter; iter < free_blocks_.end(); iter++) {
	block_type const block = iter->get_block(size);
	if(block.is_valid()) { return allocate_and_insert_remainder(block, size, *iter); }	
    }

    // no large enough blocks available on other streams, so sync and merge until we find one
    {
      block_type const block = get_block_from_other_stream(size, blocks);
      if(block.is_valid()) { return block; }
    }

    // no large enough blocks available after merging, so grow the pool
    block_type const block = this->underlying().expand_pool(size, blocks);

    return allocate_and_insert_remainder(block, size, blocks);    
  }

  /**
   * @brief Find a free block of at least `size` bytes in a `free_list` with a different
   * stream/event than `stream_event`.
   *
   * If an appropriate block is found in a free list F associated with event E,
   * `stream_event.stream` will be made to wait on event E.
   *
   * @param size The requested size of the allocation.
   * @return A block with non-null pointer and size >= `size`, or a nullptr block if none is
   *         available in `blocks`.
   */
  block_type get_block_from_other_stream(std::size_t size, free_list& blocks) {
    auto find_block = [&](auto iter) {
      auto  other_event  = iter->first.event;
      auto& other_blocks = iter->second;

      merge_lists(stream_event, blocks, other_event, std::move(other_blocks));

      free_blocks_.erase(iter);

      block_type const block = blocks.get_block(size); // get the best fit block in merged lists
      if(block.is_valid()) { return allocate_and_insert_remainder(block, size, blocks); }

      return block_type{};
    };

    for(auto iter = free_blocks_.begin(), next_iter = iter;
        iter != free_blocks_.end(); iter = next_iter) {
      ++next_iter; // Points to element after `iter` to allow erasing `iter` in the loop body

      block_type const block = find_block(iter);
      if(block.is_valid()) { return block; }
    }
    return block_type{};
  }

  void merge_lists(stream_event_pair stream_event, free_list& blocks, gpuEvent_t other_event,
                   free_list&& other_blocks) {
    // Since we found a block associated with a different stream, we have to insert a wait
    // on the stream's associated event into the allocating stream.

    tamm::gpuStreamWaitEvent(stream_event.stream, other_event);

    // Merge the two free lists
    blocks.insert(std::move(other_blocks));
  }

  /**
   * @brief Clear free lists
   *
   * Note: only called by destructor.
   */
  void release() {
    free_blocks_.clear();
  }

  // Vector of free_list
  std::vector<free_list> free_blocks_;
}; // namespace detail

} // namespace tamm::rmm::mr::detail
