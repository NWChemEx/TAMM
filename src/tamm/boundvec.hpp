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
 * Provides a std::vector-like interface backed by a fixed-size std::array.
 * The active size is tracked separately; elements beyond size() are
 * default-constructed but not considered live.
 *
 * C++20 changes vs. original:
 *  - Concept constraint: T must satisfy std::copyable.
 *  - Manual destructor calls (v.~value_type()) replaced by std::destroy_at.
 *  - Manual placement-new replaced by std::construct_at.
 *  - operator!= removed; falls out of defaulted operator== + <=> synthesis.
 *  - [[nodiscard]] applied to pure-query methods.
 *
 * @tparam T        Element type (must be std::copyable)
 * @tparam maxsize  Maximum capacity
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

  /// Default: empty vector.
  BoundVec() noexcept : size_{0} {}

  /// Construct with `count` copies of `value`.
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

  [[nodiscard]] constexpr size_type size()     const noexcept { return size_; }
  [[nodiscard]] constexpr size_type max_size() const noexcept { return maxsize; }
  [[nodiscard]] constexpr bool      empty()    const noexcept { return size_ == 0; }

  // ------------------------------------------------------------------
  // Modifiers
  // ------------------------------------------------------------------

  void clear() noexcept {
    for (size_type i = 0; i < size_; ++i)
      std::destroy_at(std::addressof(this->at(i)));
    size_ = 0;
  }

  void push_back(const T& value) {
    EXPECTS(size() < static_cast<size_type>(maxsize));
    std::construct_at(std::addressof(this->at(size_)), value);
    ++size_;
  }

  void push_back(T&& value) {
    EXPECTS(size() < static_cast<size_type>(maxsize));
    std::construct_at(std::addressof(this->at(size_)), std::move(value));
    ++size_;
  }

  void pop_back() noexcept {
    EXPECTS(size() > 0);
    std::destroy_at(std::addressof(back()));
    --size_;
  }

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

  template<typename InputIt>
  void insert_back(InputIt first, InputIt last) {
    EXPECTS(size_ + std::distance(first, last) <= static_cast<std::ptrdiff_t>(maxsize));
    for (auto it = first; it != last; ++it) push_back(*it);
  }

  void insert_back(size_type count, const T& value) {
    EXPECTS(size_ + count <= static_cast<size_type>(maxsize));
    for (size_type i = 0; i < count; ++i) push_back(value);
  }

  // ------------------------------------------------------------------
  // Iterators (active range only)
  // ------------------------------------------------------------------

  iterator       end()       noexcept { return std::array<T,maxsize>::begin() + size_; }
  const_iterator end() const noexcept { return std::array<T,maxsize>::begin() + size_; }

  // ------------------------------------------------------------------
  // Element access
  // ------------------------------------------------------------------

  [[nodiscard]] reference front() noexcept {
    EXPECTS(size() > 0);
    return this->at(0);
  }
  [[nodiscard]] const_reference front() const noexcept {
    EXPECTS(size() > 0);
    return this->at(0);
  }
  [[nodiscard]] reference back() noexcept {
    EXPECTS(size() > 0);
    return this->at(size_ - 1);
  }
  [[nodiscard]] const_reference back() const noexcept {
    EXPECTS(size() > 0);
    return this->at(size_ - 1);
  }

  // ------------------------------------------------------------------
  // Comparison (C++20: single operator== synthesises operator!=)
  // ------------------------------------------------------------------

  [[nodiscard]] friend bool
  operator==(const BoundVec& lhs, const BoundVec& rhs) noexcept {
    return lhs.size() == rhs.size() &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
  }
  // operator!= is synthesised automatically in C++20.

private:
  size_type size_;
};

// ---------------------------------------------------------------------------
// Stream output
// ---------------------------------------------------------------------------
template<std::copyable T, int maxsize>
inline std::ostream& operator<<(std::ostream& os, const BoundVec<T, maxsize>& bvec) {
  os << "[ ";
  for (const auto& el : bvec) os << el << ' ';
  return os << ']';
}

} // namespace tamm
