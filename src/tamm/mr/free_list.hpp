#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iostream>
#include <set>
#include <utility>

namespace tamm::rmm::mr::detail {

struct block_base {
  void* ptr{}; ///< Raw memory pointer

  block_base() = default;
  block_base(void* ptr): ptr{ptr} {};

  /// Returns the raw pointer for this block
  [[nodiscard]] inline void* pointer() const { return ptr; }
  /// Returns true if this block is valid (non-null), false otherwise
  [[nodiscard]] inline bool is_valid() const { return pointer() != nullptr; }
};

/**
 * @brief Requirements on a block type stored in a `free_list`.
 */
template<typename T>
concept BlockConcept = requires(T blk, std::size_t n, T other) {
  { blk.pointer() } -> std::convertible_to<char*>;
  { blk.size() } -> std::convertible_to<std::size_t>;
  { blk.is_head() } -> std::convertible_to<bool>;
  { blk.is_valid() } -> std::convertible_to<bool>;
  { blk.fits(n) } -> std::convertible_to<bool>;
  { blk.is_contiguous_before(other) } -> std::convertible_to<bool>;
};

/**
 * @brief Comparator for block types based on pointer address.
 *
 * This comparator allows searching associative containers of blocks by pointer rather than
 * having to search by the contained type. Saves potentially error-prone temporary construction of
 * a block when you just want to search by pointer.
 */
template<typename block_type>
struct compare_blocks {
  // is_transparent (C++14 feature) allows search key type for set<block_type>::find()
  using is_transparent = void;

  bool operator()(block_type const& lhs, block_type const& rhs) const {
    return lhs.pointer() < rhs.pointer();
  }
  bool operator()(char const* ptr, block_type const& rhs) const { return ptr < rhs.pointer(); }
  bool operator()(block_type const& lhs, char const* ptr) const { return lhs.pointer() < ptr; };
};

/**
 * @brief Base class defining an interface for a list of free memory blocks.
 *
 * Blocks are held in an address-ordered `std::set`, giving O(log n) insertion, erasure and
 * neighbour lookup. The previous implementation used a `std::list`, which forced an O(n)
 * linear scan on every insert (to find the ordered position) and on every best-fit search.
 * In a CCSD contraction the free list routinely holds thousands of blocks and both
 * operations sit directly in the allocation hot path.
 *
 * @note The mutating operations (`insert_at`, `erase`, `clear`) are deliberately
 * **protected**. `coalescing_free_list` maintains a second, size-ordered index alongside
 * this one, and the two must be updated together; exposing raw mutators here would let a
 * caller desynchronize them silently. Derived classes expose only the synchronized API.
 *
 * @tparam BlockType the type of block stored in the list.
 */
template<BlockConcept BlockType>
class free_list {
public:
  free_list()          = default;
  virtual ~free_list() = default;

  free_list(free_list const&)            = delete;
  free_list& operator=(free_list const&) = delete;
  free_list(free_list&&)                 = delete;
  free_list& operator=(free_list&&)      = delete;

  using block_type     = BlockType;
  using set_type       = std::set<BlockType, compare_blocks<BlockType>>;
  using size_type      = typename set_type::size_type;
  using iterator       = typename set_type::iterator;
  using const_iterator = typename set_type::const_iterator;

  /// beginning of the free list
  [[nodiscard]] iterator begin() noexcept { return blocks.begin(); }
  /// beginning of the free list
  [[nodiscard]] const_iterator begin() const noexcept { return blocks.begin(); }
  /// beginning of the free list
  [[nodiscard]] const_iterator cbegin() const noexcept { return blocks.cbegin(); }

  /// end of the free list
  [[nodiscard]] iterator end() noexcept { return blocks.end(); }
  /// beginning of the free list
  [[nodiscard]] const_iterator end() const noexcept { return blocks.end(); }
  /// beginning of the free list
  [[nodiscard]] const_iterator cend() const noexcept { return blocks.cend(); }

  /**
   * @brief The size of the free list in blocks.
   *
   * @return size_type The number of blocks in the free list.
   */
  [[nodiscard]] size_type size() const noexcept { return blocks.size(); }

  /**
   * @brief checks whether the free_list is empty.
   *
   * @return true If there are blocks in the free_list.
   * @return false If there are no blocks in the free_list.
   */
  [[nodiscard]] bool is_empty() const noexcept { return blocks.empty(); }

  /**
   * @brief Summarize the contents of the free list.
   *
   * Intended for diagnostics when an allocation cannot be satisfied: the pair
   * distinguishes "the pool is fragmented" (large total, small largest) from
   * "the pool is exhausted" (both small).
   *
   * @return Pair of {largest available block, total free bytes}.
   */
  [[nodiscard]] std::pair<std::size_t, std::size_t> summary() const noexcept {
    std::size_t largest{0};
    std::size_t total{0};
    for(auto const& blk: blocks) {
      total += blk.size();
      largest = std::max(largest, blk.size());
    }
    return {largest, total};
  }

protected:
  /**
   * @brief Insert a block into the address-ordered set.
   *
   * @param block The block to insert.
   * @return iterator to the inserted block.
   */
  iterator insert_at(block_type const& block) { return blocks.insert(block).first; }

  /**
   * @brief Removes the block indicated by `iter` from the free list.
   *
   * @param iter An iterator referring to the block to erase.
   */
  void erase(const_iterator iter) { blocks.erase(iter); }

  /**
   * @brief Erase all blocks from the free_list.
   */
  void clear_blocks() noexcept { blocks.clear(); }

  /// Direct access for derived classes that must query neighbours.
  [[nodiscard]] set_type&       block_set() noexcept { return blocks; }
  [[nodiscard]] set_type const& block_set() const noexcept { return blocks; }

private:
  set_type blocks; // The internal container of blocks, ordered by address
};

} // namespace tamm::rmm::mr::detail
