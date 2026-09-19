#pragma once

#include "free_list.hpp"
#include <iterator>

#include <algorithm>
#include <cassert>
#include <compare>
#include <cstddef>
#include <functional>
#include <iostream>
#include <set>

namespace tamm::rmm::mr::detail {

/**
 * @brief A simple block structure specifying the size and location of a block
 *        of memory, with a flag indicating whether it is the head of a block
 *        of memory allocated from the heap (or upstream allocator).
 */
struct block: public block_base {
  block() = default;
  block(char* ptr, std::size_t size, bool is_head):
    block_base{ptr}, size_bytes{size}, head{is_head} {}

  /**
   * @brief Returns the pointer to the memory represented by this block.
   *
   * @return the pointer to the memory represented by this block.
   */
  [[nodiscard]] inline char* pointer() const { return static_cast<char*>(ptr); }

  /**
   * @brief Returns the size of the memory represented by this block.
   *
   * @return the size in bytes of the memory represented by this block.
   */
  [[nodiscard]] inline std::size_t size() const { return size_bytes; }

  /**
   * @brief Returns whether this block is the start of an allocation from an upstream allocator.
   *
   * A block `b` may not be coalesced with a preceding contiguous block `a` if `b.is_head == true`.
   *
   * @return true if this block is the start of an allocation from an upstream allocator.
   */
  [[nodiscard]] inline bool is_head() const { return head; }

  /**
   * @brief Comparison operator to enable comparing blocks and storing in ordered containers.
   *
   * Orders by ptr address.

   * @param rhs
   * @return true if this block's ptr is < than `rhs` block pointer.
   * @return false if this block's ptr is >= than `rhs` block pointer.
   */
  /// Blocks are ordered by address. Spaceship gives <, <=, >, >= for free (C++20).
  [[nodiscard]] std::strong_ordering operator<=>(block const& rhs) const noexcept {
    return std::compare_three_way{}(pointer(), rhs.pointer());
  }
  /// Two blocks are the same block iff they start at the same address.
  [[nodiscard]] bool operator==(block const& rhs) const noexcept {
    return pointer() == rhs.pointer();
  }

  /**
   * @brief Coalesce two contiguous blocks into one.
   *
   * `this` must immediately precede `b` and both `this` and `b` must be from the same upstream
   * allocation. That is, `this->is_contiguous_before(b)`. Otherwise behavior is undefined.
   *
   * @param blk block to merge
   * @return The merged block
   */
  [[nodiscard]] inline block merge(block const& blk) const noexcept {
    assert(is_contiguous_before(blk));
    return {pointer(), size() + blk.size(), is_head()};
  }

  /**
   * @brief Verifies whether this block can be merged to the beginning of block b.
   *
   * @param blk The block to check for contiguity.
   * @return Returns true if this blocks's `ptr` + `size` == `b.ptr`, and `not b.is_head`,
             false otherwise.
   */
  [[nodiscard]] inline bool is_contiguous_before(block const& blk) const noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return (pointer() + size() == blk.ptr) and not(blk.is_head());
  }

  /**
   * @brief Is this block large enough to fit `sz` bytes?
   *
   * @param bytes The size in bytes to check for fit.
   * @return true if this block is at least `bytes` bytes
   */
  [[nodiscard]] inline bool fits(std::size_t bytes) const noexcept { return size() >= bytes; }

  // NOTE: the former is_better_fit() helper is gone. Best-fit selection now lives solely in
  // coalescing_free_list::size_index_ (ordered by size, then address), so there is exactly
  // one definition of "best fit" and no risk of the two drifting apart.

private:
  std::size_t size_bytes{}; ///< Size in bytes
  bool        head{};       ///< Indicates whether ptr was allocated from the heap
};

/**
 * @brief Comparator ordering blocks by (size, address) for the secondary size index.
 *
 * Ordering by size alone would not be a strict weak ordering over distinct blocks of equal
 * size, and `std::set` would silently drop all but one of them. The pointer is the
 * tie-breaker, which keeps every block addressable while still allowing `lower_bound(size)`
 * to locate the smallest block that fits in O(log n).
 */
struct size_ptr_less {
  using is_transparent = void;

  bool operator()(block const& lhs, block const& rhs) const noexcept {
    if(lhs.size() != rhs.size()) { return lhs.size() < rhs.size(); }
    return lhs.pointer() < rhs.pointer();
  }
  // Heterogeneous lookup by size: finds the first block whose size >= the key.
  bool operator()(std::size_t size, block const& rhs) const noexcept { return size < rhs.size(); }
  bool operator()(block const& lhs, std::size_t size) const noexcept { return lhs.size() < size; }
};

/**
 * @brief An ordered list of free memory blocks that coalesces contiguous blocks on insertion.
 *
 * @tparam list_type the type of the internal list data structure.
 */
struct coalescing_free_list: free_list<block> {
  coalescing_free_list()           = default;
  ~coalescing_free_list() override = default;

  coalescing_free_list(coalescing_free_list const&)            = delete;
  coalescing_free_list& operator=(coalescing_free_list const&) = delete;
  coalescing_free_list(coalescing_free_list&&)                 = delete;
  coalescing_free_list& operator=(coalescing_free_list&&)      = delete;

  /**
   * @brief Inserts a block into the `free_list` in the correct order, coalescing it with the
   *        preceding and following blocks if either is contiguous.
   *
   * @param b The block to insert.
   */
  void insert(block_type const& block) {
    if(!block.is_valid() || block.size() == 0) { return; }

    auto& blocks = block_set();

    // First block at an address strictly greater than `block` -- the right neighbour
    // candidate. O(log n) instead of the previous linear find_if.
    auto next = blocks.lower_bound(block);

    bool const merge_next = (next != blocks.end()) && block.is_contiguous_before(*next);

    bool     merge_prev = false;
    iterator prev{};
    if(next != blocks.begin()) {
      prev       = std::prev(next);
      merge_prev = prev->is_contiguous_before(block);
    }

    // Build the coalesced block, dropping any merged neighbours from BOTH indices.
    // std::set elements are immutable, so a merge is always erase-then-insert.
    block_type merged = block;

    if(merge_prev && merge_next) {
      merged = prev->merge(block).merge(*next);
      drop_from_size_index(*prev);
      drop_from_size_index(*next);
      blocks.erase(prev);
      blocks.erase(next);
    }
    else if(merge_prev) {
      merged = prev->merge(block);
      drop_from_size_index(*prev);
      blocks.erase(prev);
    }
    else if(merge_next) {
      merged = block.merge(*next);
      drop_from_size_index(*next);
      blocks.erase(next);
    }

    // The two indices must gain and lose blocks together. std::set silently ignores a
    // duplicate key, and the address set is keyed on address alone while size_index_ is
    // keyed on (size, address) -- so a duplicate address would be dropped here but still
    // accepted there, leaving a block in the size index that the address index does not
    // know about. get_block() would then hand out an address that is already live.
    // A duplicate address means the caller returned overlapping blocks; refuse to record it.
    auto const [addr_it, inserted] = block_set().insert(merged);
    (void) addr_it;
    assert(inserted && "coalescing_free_list: duplicate/overlapping block inserted");
    if(!inserted) { return; } // never touch size_index_ if the address insert failed

    size_index_.insert(merged);
  }

  // /**
  //  * @brief Moves blocks from free_list `other` into this free_list in their correct order,
  //  *        coalescing them with their preceding and following blocks if they are contiguous.
  //  *
  //  * @tparam InputIt iterator type
  //  * @param other free_list of blocks to insert
  //  */
  // void insert(free_list&& other) {
  //   using std::make_move_iterator;
  //   auto inserter = [this](block_type&& block) { this->insert(block); };
  //   std::for_each(make_move_iterator(other.begin()), make_move_iterator(other.end()), inserter);
  // }

  /**
   * @brief Finds the smallest block in the `free_list` large enough to fit `size` bytes.
   *
   * This is a "best fit" search.
   *
   * @param size The size in bytes of the desired block.
   * @return A block large enough to store `size` bytes.
   */
  block_type get_block(std::size_t size) {
    // Smallest block whose size >= `size`. This is the same best-fit choice the previous
    // linear min_element made, but in O(log n).
    auto const iter = size_index_.lower_bound(size);
    if(iter == size_index_.end()) { return block_type{}; } // no block large enough

    block_type const found = *iter;
    size_index_.erase(iter);

    // Every block in size_index_ must also be in the address index. Silently tolerating a
    // miss here would leave the block live in the address set while it is handed to the
    // caller -- i.e. the same memory allocated twice.
    auto& blocks  = block_set();
    auto  addr_it = blocks.find(found);
    assert(addr_it != blocks.end() && "coalescing_free_list: index desync in get_block");
    erase(addr_it);

    return found;
  }

  /**
   * @brief Erase all blocks, keeping both indices consistent.
   */
  void clear() noexcept {
    clear_blocks();
    size_index_.clear();
  }

private:
  /// Remove `blk` from the size index. Erases by exact (size, ptr) key, never by size
  /// alone -- multiple blocks may share a size.
  void drop_from_size_index(block_type const& blk) {
    auto const iter = size_index_.find(blk);
    if(iter != size_index_.end()) { size_index_.erase(iter); }
  }

  // Secondary index over the same blocks, ordered by (size, address), providing O(log n)
  // best-fit lookup. Kept in lockstep with the address-ordered set in the base class:
  // every block present in one is present in the other.
  std::set<block, size_ptr_less> size_index_;
}; // coalescing_free_list

} // namespace tamm::rmm::mr::detail
