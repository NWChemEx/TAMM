// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMM_BOUNDVEC_HPP_
#define TAMM_BOUNDVEC_HPP_

#include <array>
#include <iosfwd>
#include <string>

#include "tamm/errors.hpp"

namespace tamm {

/**
 * @brief Vector of bounded length.
 *
 * This class provides a vector-like interface while having bounded size like an
 * array. Bounds checks can be added in debug mode.
 *
 * @tparam T Type of each element contained in the vector
 * @tparam maxsize Maximum size of the vector
 *
 * @todo Add bounds checking when BOUNDVEC_DEBUG option is enabled
 */
template<typename T, int maxsize>
class BoundVec : public std::array<T, maxsize> {
public:
    using size_type        = typename std::array<T, maxsize>::size_type;
    using value_type       = typename std::array<T, maxsize>::value_type;
    using iterator         = typename std::array<T, maxsize>::iterator;
    using const_iterator   = typename std::array<T, maxsize>::const_iterator;
    using reverse_iterator = typename std::array<T, maxsize>::reverse_iterator;
    using const_reverse_iterator =
      typename std::array<T, maxsize>::const_reverse_iterator;
    using reference       = typename std::array<T, maxsize>::reference;
    using const_reference = typename std::array<T, maxsize>::const_reference;

    using std::array<T, maxsize>::begin;
    using std::array<T, maxsize>::rend;

    /**
     * @brief Constructor a zero-sized vector
     */
    BoundVec() : size_{0} {}

    /**
     * @brief Construct a vector of specified size, with an optional initial value
     * 
     * @param[in] count size of constructor vector
     * @param[in] value initial value of all elements in the constructed vector
     */
    explicit BoundVec(size_type count, const T& value = T()) : size_{0} {
        for(size_type i = 0; i < count; i++) { push_back(value); }
    }

    BoundVec(const BoundVec&) = default;
    BoundVec(BoundVec&&) = default;
    BoundVec& operator = (const BoundVec&) = default;
    BoundVec& operator = (BoundVec&&) = default;
    ~BoundVec() {
        for(auto& v : *this) {
            v.~value_type();
        }
    }

    template<typename Itr>
    BoundVec(Itr first, Itr last) : size_{0} {
        for(auto itr = first; itr != last; ++itr) { push_back(*itr); }
    }

    BoundVec(std::initializer_list<T> init) : size_{0} {
        for(auto v : init) { push_back(v); }
    }

    /**
     * @brief Size of this vector
     * 
     * @return Size of the vector
     */
    constexpr size_type size() const { return size_; }

    /**
     * @brief Maximum number of elements this vector can hold
     * 
     * @return Maximum size of this vector
     */
    constexpr size_type max_size() const { return maxsize; }

    /**
     * @brief Is this vector empty
     * 
     * @return True if this vector is empty, false otherwise.
     */
    constexpr bool empty() const { return size_ == 0; }

    /**
     * @brief Clear the contents of this vector
     */
    void clear() {
        for(auto& v : *this) {
            v.~value_type();
        }
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
        EXPECTS(size() < maxsize);
        this->at(size_++) = value;
    }

    /**
     * @brief Push one element to the back of the vector
     * 
     * @param[in,out] value The value to be pushed
     * 
     * @pre size() < max_size
     */
    void push_back(T&& value) {
        EXPECTS(size() < maxsize);
        this->at(size_++) = std::move(value);
    }

    /**
     * @brief Remove one element from the back of the vector
     * 
     * @pre size() > 0
     */
    void pop_back() {
        EXPECTS(size() > 0);
        back().~value_type();
        size_ -= 1;
    }

    /**
     * @brief Resize vector to desired size
     * 
     * @pre size >=0 
     * 
     * @param[in] size Size desired for the vector
     */
    void resize(size_type size) {
        EXPECTS(size >= 0);
        for(size_t i = size; size < size_; i++) {
            this->at(i).~value_type();
        }
        size_ = size;
    }

    /**
     * @brief Insert a sequence of elements, specified using iterators, at the back of the vector
     * 
     * @param[in] first Starting iterator position for tasks to be inserted
     * @param[in] last Ending iterator position for the tasks to be inserted
     * 
     * @pre size() + std::distance(@param first, @param last) <= maxsize
     */ 
    template<typename InputIt>
    void insert_back(InputIt first, InputIt last) {
        EXPECTS(size_ + (last - first) <= maxsize);
        for(auto itr = first; itr != last; ++itr) { push_back(*itr); }
    }

    /**
     * @brief Insert a given value multiple times at the back of the vector
     * 
     * @param[in] count Number of times the given value is to be inserted
     * @param[in] value Value to be inserted
     * 
     * @pre size() + @param count <= maxsize
     */
    void insert_back(size_type count, const T& value) {
        EXPECTS(size_ + count <= maxsize);
        for(size_type i = 0; i < count; i++) { push_back(value); }
    }

    // BoundVec<T, maxsize>& operator = (BoundVec<T, maxsize>& bvec) {
    //   size_ = bvec.size_;
    //   std::copy(bvec.begin(), bvec.end(), begin());
    //   return *this;
    // }

    // BoundVec<T, maxsize>& operator=(const BoundVec<T, maxsize>& bvec) {
    //     size_ = bvec.size_;
    //     std::copy(bvec.begin(), bvec.end(), begin());
    //     return *this;
    // }

    // BoundVec<T, maxsize>& operator=(BoundVec<T, maxsize>&& bvec) {
    //     clear();
    //     for(auto&& bv : bvec) { push_back(std::move(bv)); }
    //     return *this;
    // }

    /**
     * @brief Obtain end iterator past the last element in the vector
     * 
     * @return The end iterator
     */
    iterator end() { return std::array<T, maxsize>::begin() + size_; }

    /**
     * @brief Obtain end iterator past the last element in the vector
     * 
     * @return Const end iterator
     */
    const_iterator end() const {
        return std::array<T, maxsize>::begin() + size_;
    }

    // reverse_iterator rbegin() const {
    //     return std::array<T, maxsize>::begin() + size_;
    // }

    /**
     * Obtain reference to first element in the vector
     * 
     * @pre size() > 0
     * 
     * @return Reference to first element
     */
    reference front() {
        EXPECTS(size() > 0);
        return this->at(0);
    }

    /**
     * Obtain a const reference to first element in the vector
     * 
     * @pre size() > 0
     * 
     * @return Reference to first element
     */
    const_reference front() const {
        EXPECTS(size() > 0);
        return this->at(0);
    }

    /**
     * Obtain reference to the last element in the vector
     * 
     * @pre size() > 0
     * 
     * @return Reference to last element
     */
    reference back() {
        EXPECTS(size() > 0);
        return this->at(size_ - 1);
    }

    /**
     * Obtain a const reference to the last element in the vector
     * 
     * @pre size() > 0
     * 
     * @return Const reference to last element
     */
    const_reference back() const {
        EXPECTS(size() > 0);
        return this->at(size_ - 1);
    }

private:
    /**
     * @brief Size of the vector
     */ 
    size_type size_;

    /**
     * @brief Equality operator to compare two vectors
     * 
     * @param[in] lhs One vector to be compared
     * 
     * @param[in] rhs The other vector to be compared
     */
    friend bool operator==(const BoundVec<T, maxsize>& lhs,
                           const BoundVec<T, maxsize>& rhs) {
        return lhs.size() == rhs.size() &&
               std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }

    /**
     * @brief Inequality operator to compare two vectors
     * 
     * @param[in] lhs One vector to be compared
     * 
     * @param[in] rhs The other vector to be compared
     */
    friend bool operator!=(const BoundVec<T, maxsize>& lhs,
                           const BoundVec<T, maxsize>& rhs) {
        return !(lhs == rhs);
    }
}; // class BoundVec

/**
 * @brief Dump a vector to an output stream
 * 
 * @param[in,out] os Output stream to write to
 * 
 * @param[in] bvec The vector to be output
 * 
 * @return The modified output stream @param os
 */
template<typename T, int maxsize>
inline std::ostream& operator<<(std::ostream& os, const BoundVec<T, maxsize>& bvec) {
    os << std::string{"[ "};
    for(auto el : bvec) { os << el << " "; }
    os << std::string{"]"};
    return os;
}

} // namespace tamm

#endif // TAMM_BOUNDVEC_HPP_
