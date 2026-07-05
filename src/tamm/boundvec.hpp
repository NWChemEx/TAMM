#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include "tamm/errors.hpp"

namespace tamm {

/**
 * @brief Vector of bounded length.
 *
 * A vector-like container whose maximum number of elements is fixed at
 * compile time as @p maxsize.  Storage is entirely on the stack (a
 * @c std::array), so no heap allocation ever occurs.
 *
 * @tparam T       Element type.
 * @tparam maxsize Compile-time upper bound on the number of elements.
 *
 * @todo Consider replacing with a fully-standard `std::inplace_vector`
 *       once C++26 support is available.
 */
template<typename T, std::size_t maxsize>
class BoundVec {
public:
  // -----------------------------------------------------------------------
  // Type aliases (mirror std::vector interface)
  // -----------------------------------------------------------------------
  using value_type             = T;
  using size_type              = std::size_t;
  using reference              = T&;
  using const_reference        = const T&;
  using pointer                = T*;
  using const_pointer          = const T*;
  using iterator               = typename std::array<T, maxsize>::iterator;
  using const_iterator         = typename std::array<T, maxsize>::const_iterator;
  using reverse_iterator       = typename std::array<T, maxsize>::reverse_iterator;
  using const_reverse_iterator = typename std::array<T, maxsize>::const_reverse_iterator;

  // -----------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------

  /** @brief Default constructor — produces an empty vector. */
  BoundVec() = default;

  /**
   * @brief Construct @p count copies of @p value.
   * @param count Number of elements to create.
   * @param value Value to copy into each element (defaults to T{}).
   * @pre   count <= maxsize
   */
  explicit BoundVec(size_type count, const T& value = T{}) {
    EXPECTS(count <= maxsize);
    for(size_type i = 0; i < count; ++i) { data_[size_++] = value; }
  }

  /**
   * @brief Construct from an initializer list.
   * @param list Elements to copy into the vector.
   * @pre   list.size() <= maxsize
   */
  BoundVec(std::initializer_list<T> list) {
    EXPECTS(list.size() <= maxsize);
    for(const auto& v: list) { data_[size_++] = v; }
  }

  /**
   * @brief Construct from a pair of iterators.
   * @tparam Iter  Input iterator type.
   * @param  first Beginning of the source range.
   * @param  last  One-past-end of the source range.
   * @pre    std::distance(first, last) <= maxsize
   *
   * Constrained to std::input_iterator so it does not compete with the
   * (count, value) constructor for calls like BoundVec(n, val).
   */
  template<std::input_iterator Iter>
  BoundVec(Iter first, Iter last) {
    for(; first != last; ++first) { push_back(*first); }
  }

  // -----------------------------------------------------------------------
  // Capacity
  // -----------------------------------------------------------------------

  /** @brief Return the number of live elements. */
  [[nodiscard]] size_type size() const noexcept { return size_; }

  /** @brief Return the compile-time maximum number of elements. */
  [[nodiscard]] static constexpr size_type max_size() noexcept { return maxsize; }

  /** @brief Return true iff the vector holds no elements. */
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  // -----------------------------------------------------------------------
  // Modifiers
  // -----------------------------------------------------------------------

  /** @brief Remove all elements (does not release storage). */
  void clear() noexcept { size_ = 0; }

  /**
   * @brief Append a copy of @p val.
   * @param val Value to append.
   * @pre   size() < maxsize
   */
  void push_back(const T& val) {
    EXPECTS(size_ < maxsize);
    data_[size_++] = val;
  }

  /**
   * @brief Append by moving @p val.
   * @param val Value to move-append.
   * @pre   size() < maxsize
   */
  void push_back(T&& val) {
    EXPECTS(size_ < maxsize);
    data_[size_++] = std::move(val);
  }

  /**
   * @brief Remove the last element.
   * @pre   !empty()
   */
  void pop_back() {
    EXPECTS(size_ > 0);
    --size_;
  }

  /**
   * @brief Resize the vector to @p sz elements.
   *
   * If @p sz > size(), new elements are value-initialised.  If @p sz <
   * size(), excess elements are discarded.
   *
   * @param sz New size.
   * @pre   sz <= maxsize
   */
  void resize(size_type sz) {
    // Fix: sz is unsigned so "sz >= 0" is always true and never fires.
    // The meaningful guard is an upper-bound check.
    EXPECTS(sz <= maxsize);
    if(sz > size_) {
      for(size_type i = size_; i < sz; ++i) { data_[i] = T{}; }
    }
    size_ = sz;
  }

  /**
   * @brief Insert a copy of @p val at the back (alias for push_back).
   * @param val  Value to insert.
   * @pre   size() < maxsize
   * @return Iterator to the inserted element.
   */
  iterator insert_back(const T& val) {
    push_back(val);
    return end() - 1;
  }

  /**
   * @brief Insert by moving @p val at the back (alias for push_back).
   * @param val  Value to move-insert.
   * @pre   size() < maxsize
   * @return Iterator to the inserted element.
   */
  iterator insert_back(T&& val) {
    push_back(std::move(val));
    return end() - 1;
  }

  /**
   * @brief Append the range [first, last) at the back.
   * @tparam InputIt Input iterator type.
   * @param  first   Beginning of the source range.
   * @param  last    One-past-end of the source range.
   * @pre    size() + distance(first,last) <= maxsize
   */
  template<std::input_iterator InputIt>
  void insert_back(InputIt first, InputIt last) {
    for(; first != last; ++first) { push_back(*first); }
  }

  /**
   * @brief Append @p count copies of @p value at the back.
   * @param count Number of copies to append.
   * @param value Value to copy.
   * @pre   size() + count <= maxsize
   */
  void insert_back(size_type count, const T& value) {
    for(size_type i = 0; i < count; ++i) { push_back(value); }
  }

  // -----------------------------------------------------------------------
  // Iterators
  // -----------------------------------------------------------------------

  /** @brief Return an iterator to the first live element. */
  iterator begin() noexcept { return data_.begin(); }
  /** @brief Return a const iterator to the first live element. */
  const_iterator begin() const noexcept { return data_.begin(); }

  /** @brief Return an iterator one past the last live element. */
  iterator end() noexcept { return data_.begin() + size_; }
  /** @brief Return a const iterator one past the last live element. */
  const_iterator end() const noexcept { return data_.begin() + size_; }

  /** @brief Const iterator to the first live element. */
  const_iterator cbegin() const noexcept { return data_.begin(); }
  /** @brief Const iterator one past the last live element. */
  const_iterator cend() const noexcept { return data_.begin() + size_; }

  /** @brief Reverse iterator to the last live element. */
  reverse_iterator       rbegin() noexcept { return reverse_iterator{end()}; }
  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator{end()}; }
  /** @brief Reverse iterator one before the first live element. */
  reverse_iterator       rend() noexcept { return reverse_iterator{begin()}; }
  const_reverse_iterator rend() const noexcept { return const_reverse_iterator{begin()}; }

  // -----------------------------------------------------------------------
  // Raw storage access
  // -----------------------------------------------------------------------

  /** @brief Pointer to the underlying contiguous storage (mutable). */
  [[nodiscard]] pointer data() noexcept { return data_.data(); }
  /** @brief Pointer to the underlying contiguous storage (const). */
  [[nodiscard]] const_pointer data() const noexcept { return data_.data(); }

  // -----------------------------------------------------------------------
  // Element access
  // -----------------------------------------------------------------------

  /**
   * @brief Unchecked element access.
   * @param i Index (0-based).
   * @pre   i < size()
   */
  reference operator[](size_type i) {
    EXPECTS(i < size_);
    return data_[i];
  }
  /** @brief Unchecked const element access. */
  const_reference operator[](size_type i) const {
    EXPECTS(i < size_);
    return data_[i];
  }

  /**
   * @brief Return a reference to the first element.
   * @pre   !empty()
   */
  reference front() {
    EXPECTS(size_ > 0);
    return data_.front();
  }
  /** @brief Return a const reference to the first element. */
  const_reference front() const {
    EXPECTS(size_ > 0);
    return data_.front();
  }

  /**
   * @brief Return a reference to the last live element.
   * @pre   !empty()
   */
  reference back() {
    EXPECTS(size_ > 0);
    return data_[size_ - 1];
  }
  /** @brief Return a const reference to the last live element. */
  const_reference back() const {
    EXPECTS(size_ > 0);
    return data_[size_ - 1];
  }

  // -----------------------------------------------------------------------
  // Comparison
  // -----------------------------------------------------------------------

  /** @brief Equality operator — two BoundVecs are equal iff their live
   *         elements are pairwise equal. */
  bool operator==(const BoundVec& rhs) const noexcept {
    if(size_ != rhs.size_) { return false; }
    for(size_type i = 0; i < size_; i++) {
      if(data_[i] != rhs.data_[i]) { return false; }
    }
    return true;
  }

private:
  /** @brief Size of the vector */
  size_type              size_{0};
  std::array<T, maxsize> data_{};
};

/**
 * @brief Dump a vector to an output stream (diagnostic helper).
 * @param os  Target output stream.
 * @param bv  Vector to print.
 * @return    Reference to @p os.
 */
template<typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const BoundVec<T, N>& bv) {
  os << '[';
  for(std::size_t i = 0; i < bv.size(); ++i) {
    if(i) os << ", ";
    os << bv[i];
  }
  return os << ']';
}

} // namespace tamm
