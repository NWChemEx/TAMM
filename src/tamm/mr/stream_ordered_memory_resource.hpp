#pragma once

#include "aligned.hpp"
#include "device_memory_resource.hpp"

#include "tamm/errors.hpp" // tamm_terminate

#include <cstddef>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
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
  using lock_guard = std::lock_guard<std::mutex>;

  // Derived classes must implement these four methods

  /// Pair representing a block that has been split for allocation
  using split_block = std::pair<block_type, block_type>;

  /**
   * @brief Returns the block `b` (last used) to the pool.
   *
   * @param block The block to insert into the pool.
   */
  void insert_block(block_type const& block) { this->free_blocks_.insert(block); }

  /**
   * @brief Get the mutex guarding the free list.
   *
   * Derived classes must hold this lock when touching pool state (e.g. in
   * `release()`), since allocation and deallocation may run concurrently.
   *
   * @return std::mutex& the free-list mutex
   */
  std::mutex& get_mutex() { return mtx_; }

  /**
   * @brief Summarize the free list for diagnostics.
   *
   * Caller must hold `get_mutex()`.
   *
   * @return Pair of {largest free block, total free bytes}.
   */
  [[nodiscard]] std::pair<std::size_t, std::size_t> free_list_summary() const {
    return free_blocks_.summary();
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

    // Diagnostics are built while holding the lock, but tamm_terminate() must be called
    // *after* releasing it: tamm_terminate() calls exit(), which runs static destructors,
    // which destroy the pool singleton and re-enter release() on this same non-recursive
    // mutex from this same thread. Holding the lock across it deadlocks at exit.
    std::string failure_msg;
    {
      lock_guard lock(mtx_);

      std::size_t const requested = size;
      // align_up saturates at SIZE_MAX rather than wrapping, so an absurd request stays
      // absurd and is caught by the ceiling check below instead of collapsing to a small
      // value that would succeed.
      size = rmm::detail::align_up(size, rmm::detail::RMM_ALLOCATION_ALIGNMENT);

      if(!(size <= this->underlying().get_maximum_allocation_size())) {
        std::ostringstream os;
        os << "[TAMM ERROR] Maximum pool allocation size exceeded!\n"
           << "  requested   : " << format_bytes(requested)
           << (requested != size ? " (aligned up to " + format_bytes(size) + ")" : "") << "\n"
           << "  pool maximum: " << format_bytes(this->underlying().get_maximum_allocation_size())
           << "\n"
           << pool_state_report()
           << "  A single allocation cannot exceed the pool size. Increase the pool via\n"
           << "  TAMM_GPU_POOL / TAMM_CPU_POOL (percent of available memory), or reduce the\n"
           << "  tilesize so that individual blocks are smaller.\n"
           << __FILE__ << ":L" << __LINE__;
        failure_msg = os.str();
      }
      else {
        auto const block = this->underlying().get_block(size);
        if(block.is_valid()) { return block.pointer(); }
        failure_msg = no_block_message(size);
      }
    } // lock released here

    tamm_terminate(failure_msg);
    __builtin_unreachable();
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

    // As in do_allocate: any diagnostic must be reported *after* the lock is dropped,
    // because tamm_terminate() -> exit() -> static destructors -> release() re-enters this
    // same non-recursive mutex on this same thread.
    std::string failure_msg;
    {
      lock_guard lock(mtx_);

      size             = rmm::detail::align_up(size, rmm::detail::RMM_ALLOCATION_ALIGNMENT);
      auto const block = this->underlying().free_block(ptr, size, failure_msg);
      if(failure_msg.empty()) { free_blocks_.insert(block); }
    } // lock released here

    if(!failure_msg.empty()) { tamm_terminate(failure_msg); }
  }

private:
  /**
   * @brief Render a byte count as both an exact figure and a human-readable one.
   */
  static std::string format_bytes(std::size_t bytes) {
    static constexpr double kKiB = 1024.0;
    char const*             unit = "B";
    double                  val  = static_cast<double>(bytes);
    if(val >= kKiB * kKiB * kKiB) {
      val /= kKiB * kKiB * kKiB;
      unit = "GiB";
    }
    else if(val >= kKiB * kKiB) {
      val /= kKiB * kKiB;
      unit = "MiB";
    }
    else if(val >= kKiB) {
      val /= kKiB;
      unit = "KiB";
    }
    std::ostringstream os;
    os << bytes << " B";
    if(std::string{unit} != "B") {
      os << " (" << std::fixed << std::setprecision(2) << val << " " << unit << ")";
    }
    return os.str();
  }

  /**
   * @brief Describe the current pool occupancy, and diagnose *why* a request failed.
   *
   * The largest-free-block vs total-free comparison is what distinguishes a
   * fragmented pool from a merely exhausted one; the two have entirely
   * different remedies, so the message says which it is.
   *
   * Must be called with `mtx_` held, and must only touch `free_blocks_` directly or call
   * lock-free accessors -- calling a locking accessor (e.g. the public `free_summary()`)
   * from here would self-deadlock on this non-recursive mutex.
   */
  std::string pool_state_report() const {
    auto const [largest, total_free] = free_blocks_.summary();
    std::size_t const pool_total     = this->underlying().pool_size();

    std::ostringstream os;
    os << "  pool total  : " << format_bytes(pool_total) << "\n"
       << "  pool in use : " << format_bytes(pool_total >= total_free ? pool_total - total_free : 0)
       << "\n"
       << "  total free  : " << format_bytes(total_free) << "\n"
       << "  largest free: " << format_bytes(largest) << "\n"
       << "  free blocks : " << free_blocks_.size() << "\n";
    return os.str();
  }

  /**
   * @brief Splits a block into an allocated block of `size` bytes and a remainder block, and
   * inserts the remainder into a free list.
   *
   * @param block The block to split into allocated and remainder portions.
   * @param size The size of the block to allocate from `b`.
   * @return The allocated block.
   */
  block_type allocate_and_insert_remainder(block_type block, std::size_t size) {
    auto const [allocated, remainder] = this->underlying().allocate_from_block(block, size);
    if(remainder.is_valid()) { free_blocks_.insert(remainder); }
    return allocated;
  }

  /**
   * @brief Get an available memory block of at least `size` bytes
   *
   * @param size The number of bytes to allocate
   * @return block_type A block of memory of at least `size` bytes
   */
  /**
   * @brief Get an available memory block of at least `size` bytes.
   *
   * Must be called with `mtx_` held. Returns an invalid block if the request cannot be
   * satisfied; the caller is responsible for reporting the failure *after* releasing the
   * lock (see the note in `do_allocate`).
   */
  block_type get_block(std::size_t size) {
    block_type const block = free_blocks_.get_block(size);
    if(block.is_valid()) { return allocate_and_insert_remainder(block, size); }
    return block_type{};
  }

  /**
   * @brief Build the diagnostic for a failed allocation.
   *
   * Must be called with `mtx_` held (it inspects the free list). The returned string is
   * passed to tamm_terminate() only after the lock has been released.
   *
   * The pool is fixed-size -- it is never grown from upstream after construction -- so the
   * failure modes have different remedies and are reported separately.
   */
  std::string no_block_message(std::size_t size) const {
    auto const [largest, total_free] = free_blocks_.summary();
    std::size_t const pool_total     = this->underlying().pool_size();

    std::ostringstream os;
    os << "[TAMM ERROR] Pool allocation failed: no free block large enough.\n"
       << "  requested   : " << format_bytes(size) << " (alignment-adjusted)\n"
       << pool_state_report();

    if(size > pool_total) {
      os << "  cause       : REQUEST EXCEEDS POOL - this single request is larger than the\n"
         << "                entire pool, so no occupancy pattern could satisfy it.\n"
         << "                Reduce the problem size or tilesize. Raising TAMM_GPU_POOL /\n"
         << "                TAMM_CPU_POOL only helps if the hardware has the memory.\n";
    }
    else if(total_free < size) {
      std::size_t const in_use = pool_total >= total_free ? pool_total - total_free : 0;
      os << "  cause       : POOL EXHAUSTED - total free memory is less than the request.\n"
         << "  peak demand : " << format_bytes(in_use + size) << " (in use + requested)\n"
         << "                If peak demand exceeds the physical memory of this device, no\n"
         << "                TAMM_GPU_POOL / TAMM_CPU_POOL setting can satisfy it; reduce the\n"
         << "                problem size, tilesize, or ranks sharing this pool. Otherwise\n"
         << "                raise TAMM_GPU_POOL / TAMM_CPU_POOL.\n";
    }
    else {
      os << "  cause       : FRAGMENTATION - enough total free memory (" << format_bytes(total_free)
         << ")\n"
         << "                exists, but the largest contiguous block is only "
         << format_bytes(largest) << ".\n"
         << "                The pool does not grow or defragment. This usually means blocks\n"
         << "                are being freed with a size that differs from their allocation,\n"
         << "                or are leaked, which permanently splits the coalescing chain.\n";
    }
    os << __FILE__ << ":L" << __LINE__;
    return os.str();
  }

  /**
   * @brief Clear free lists
   *
   * Note: only called by destructor.
   */
  void release() {
    lock_guard lock(mtx_);
    free_blocks_.clear();
  }

  free_list  free_blocks_;
  std::mutex mtx_; // guards free_blocks_
}; // namespace detail

} // namespace tamm::rmm::mr::detail
