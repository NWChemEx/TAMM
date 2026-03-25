// Copyright 2016 Pacific Northwest National Laboratory
// C++20 modernization: concepts, std::destroy_at/construct_at,
// operator<=>, [[nodiscard]].

#pragma once

#include <array>
#include <concepts>    // std::copyable
#include <iosfwd>
#include <memory>      // std::construct_at, std::destroy_at
#include <string>

#include "tamm/errors.hpp"

namespace tamm {

/**
 * @brief Vector of bounded length.
 *
 * This class provides a vector-like interface while having bounded size like an
 * array. Bounds checks can be added in debug mode.
 *
 * C++20 changes vs. original:
 *  - Concept constraint: T must satisfy std::copyable.
 *  - Manual destructor calls (v.~value_type()) replaced by std::destroy_at.
 *  - Manual placement-new replaced by std::construct_at.
 *  - operator!= removed; falls out of defaulted operator== + <=> synthesis.
 *  - [[nodiscard]] applied to pure-query methods.
 *
 * @tparam T        Element type (must be std::copyable)
 * @tparam maxsize  Maximum size of the vector
 *
 * @todo Add bounds checking when BOUNDVEC_DEBUG option is enabled
 */
template<std::copyable T, int maxsize>
class BoundVec : public std::array<T, maxsize> {
public:
  using size_type              = typename std::array<T, maxsize>::size_type;
  using value_type             = T;
  using iterator               = typename std::array<T, maxsize>::iterator;
  using const_iterator         = typename std::array<T, maxsize>::const_iterator;
  using reverse_iterator       = typename std::array<T, maxsize>::reverse_iterator;
  using const_reverse_iterator = typename std::array<T, maxsize>::const_reverse_iterator;
  using reference              = typename std::array<T, maxsize>::reference;
  using const_reference        = typename std::array<T, maxsize>::const_reference;

  using std::array<T, maxsize>::begin;
  using std::array<T, maxsize>::rend;

  // ------------------------------------------------------------------
  // Constructors
  // ------------------------------------------------------------------

  /**
   * @brief Construct a zero-sized vector
   */
  BoundVec() noexcept : size_{0} {}

  /**
   * @brief Construct a vector of specified size, with an optional initial value
   *
   * @param[in] count size of constructed vector
   * @param[in] value initial value of all elements in the constructed vector
   */
  explicit BoundVec(size_type count, const T& value = T()) : size_{0} {
    for (size_type i = 0; i < count; ++i) push_back(value);
  }

  BoundVec(const BoundVec&)            = default;
  BoundVec(BoundVec&&)                 = default;
  BoundVec& operator=(const BoundVec&) = default;
  BoundVec& operator=(BoundVec&&)      = default;

  ~BoundVec() {
    // Destroy only the live elements [0, size_).
    for (size_type i = 0; i < size_; ++i)
      std::destroy_at(std::addressof(this->at(i)));
  }

  template<typename Itr>
  BoundVec(Itr first, Itr last) : size_{0} {
    for (auto it = first; it != last; ++it) push_back(*it);
  }

  BoundVec(std::initializer_list<T> init) : size_{0} {
    for (auto& v : init) push_back(v);
  }

  // ------------------------------------------------------------------
  // Capacity
  // ------------------------------------------------------------------

  /**
   * @brief Size of this vector
   *
   * @return Size of the vector
   */
  [[nodiscard]] constexpr size_type size() const noexcept { return size_; }

  /**
   * @brief Maximum number of elements this vector can hold
   *
   * @return Maximum size of this vector
   */
  [[nodiscard]] constexpr size_type max_size() const noexcept { return maxsize; }

  /**
   * @brief Is this vector empty
   *
   * @return True if this vector is empty, false otherwise.
   */
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

  // ------------------------------------------------------------------
  // Modifiers
  // ------------------------------------------------------------------

  /**
   * @brief Clear the contents of this vector
   */
  void clear() noexcept {
    for (size_type i = 0; i < size_; ++i)
      std::destroy_at(std::addressof(this->at(i)));
    size_ = 0;
  }

  /**
   * @brief Push one element to the back of the vector
   *
   * @param[in] value The value to be pushed
   *
   * @pre size() < max_size
   */
  void push_back(const T& value) {
    EXPECTS(size() < static_cast<size_type>(maxsize));
    std::construct_at(std::addressof(this->at(size_)), value);
    ++size_;
  }

  /**
   * @brief Push one element to the back of the vector (move overload)
   *
   * @param[in,out] value The value to be pushed
   *
   * @pre size() < max_size
   */
  void push_back(T&& value) {
    EXPECTS(size() < static_cast<size_type>(maxsize));
    std::construct_at(std::addressof(this->at(size_)), std::move(value));
    ++size_;
  }

  /**
   * @brief Remove one element from the back of the vector
   *
   * @pre size() > 0
   */
  void pop_back() noexcept {
    EXPECTS(size() > 0);
    std::destroy_at(std::addressof(back()));
    --size_;
  }

  /**
   * @brief Resize vector to desired size
   *
   * @pre size >= 0
   *
   * @param[in] sz Size desired for the vector
   */
  void resize(size_type sz) {
    EXPECTS(sz >= 0);
    EXPECTS(sz < static_cast<size_type>(maxsize));
    // Destroy elements that go away.
    for (size_type i = sz; i < size_; ++i)
      std::destroy_at(std::addressof(this->at(i)));
    // Default-construct new elements.
    for (size_type i = size_; i < sz; ++i)
      std::construct_at(std::addressof(this->at(i)));
    size_ = sz;
  }

  /**
   * @brief Insert a sequence of elements, specified using iterators, at the back of the vector
   *
   * @param[in] first Starting iterator position for elements to be inserted
   * @param[in] last  Ending iterator position for the elements to be inserted
   *
   * @pre size() + std::distance(first, last) <= maxsize
   */
  template<typename InputIt>
  void insert_back(InputIt first, InputIt last) {
    EXPECTS(size_ + std::distance(first, last) <= static_cast<std::ptrdiff_t>(maxsize));
    for (auto it = first; it != last; ++it) push_back(*it);
  }

  /**
   * @brief Insert a given value multiple times at the back of the vector
   *
   * @param[in] count Number of times the given value is to be inserted
   * @param[in] value Value to be inserted
   *
   * @pre size() + count <= maxsize
   */
  void insert_back(size_type count, const T& value) {
    EXPECTS(size_ + count <= static_cast<size_type>(maxsize));
    for (size_type i = 0; i < count; ++i) push_back(value);
  }

  // ------------------------------------------------------------------
  // Iterators (active range only)
  // ------------------------------------------------------------------

  /**
   * @brief Obtain end iterator past the last element in the vector
   *
   * @return The end iterator
   */
  iterator       end()       noexcept { return std::array<T,maxsize>::begin() + size_; }

  /**
   * @brief Obtain end iterator past the last element in the vector
   *
   * @return Const end iterator
   */
  const_iterator end() const noexcept { return std::array<T,maxsize>::begin() + size_; }

  // ------------------------------------------------------------------
  // Element access
  // ------------------------------------------------------------------

  /**
   * @brief Obtain reference to first element in the vector
   *
   * @pre size() > 0
   *
   * @return Reference to first element
   */
  [[nodiscard]] reference front() noexcept {
    EXPECTS(size() > 0);
    return this->at(0);
  }

  /**
   * @brief Obtain a const reference to first element in the vector
   *
   * @pre size() > 0
   *
   * @return Const reference to first element
   */
  [[nodiscard]] const_reference front() const noexcept {
    EXPECTS(size() > 0);
    return this->at(0);
  }

  /**
   * @brief Obtain reference to the last element in the vector
   *
   * @pre size() > 0
   *
   * @return Reference to last element
   */
  [[nodiscard]] reference back() noexcept {
    EXPECTS(size() > 0);
    return this->at(size_ - 1);
  }

  /**
   * @brief Obtain a const reference to the last element in the vector
   *
   * @pre size() > 0
   *
   * @return Const reference to last element
   */
  [[nodiscard]] const_reference back() const noexcept {
    EXPECTS(size() > 0);
    return this->at(size_ - 1);
  }

  // ------------------------------------------------------------------
  // Comparison (C++20: single operator== synthesises operator!=)
  // ------------------------------------------------------------------

  /**
   * @brief Equality operator to compare two vectors
   *
   * @param[in] lhs One vector to be compared
   * @param[in] rhs The other vector to be compared
   *
   * @return True if vectors are equal in size and elements
   */
  [[nodiscard]] friend bool
  operator==(const BoundVec& lhs, const BoundVec& rhs) noexcept {
    return lhs.size() == rhs.size() &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
  }
  // operator!= is synthesised automatically in C++20.

private:
  /**
   * @brief Size of the vector
   */
  size_type size_;
}; // class BoundVec

/**
 * @brief Dump a vector to an output stream
 *
 * @param[in,out] os Output stream to write to
 * @param[in] bvec   The vector to be output
 *
 * @return The modified output stream
 */
template<std::copyable T, int maxsize>
inline std::ostream& operator<<(std::ostream& os, const BoundVec<T, maxsize>& bvec) {
  os << "[ ";
  for (const auto& el : bvec) os << el << ' ';
  return os << ']';
}

} // namespace tamm
