// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMM_BOUNDVEC_H_
#define TAMM_BOUNDVEC_H_

#include <array>
#include <iosfwd>
#include <string>

#include "errors.h"

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

    BoundVec() : size_{0} {}

    explicit BoundVec(size_type count, const T& value = T()) : size_{0} {
        for(size_type i = 0; i < count; i++) { push_back(value); }
    }

    BoundVec(const BoundVec& bv) : size_{0} {
        for(auto& value : bv) { push_back(value); }
    }

    BoundVec(BoundVec&& bv) : size_{0} {
        for(auto&& value : bv) { push_back(std::move(value)); }
    }

    template<typename Itr>
    BoundVec(Itr first, Itr last) : size_{0} {
        for(auto itr = first; itr != last; ++itr) { push_back(*itr); }
    }

    BoundVec(std::initializer_list<T> init) : size_{0} {
        for(auto v : init) { push_back(v); }
    }

    size_type size() const { return size_; }

    size_type max_size() const { return maxsize; }

    bool empty() const { return size_ == 0; }

    void clear() { size_ = 0; }

    void push_back(const T& value) {
        EXPECTS(size_ < maxsize);
        this->at(size_++) = value;
    }

    void push_back(T&& value) {
        EXPECTS(size_ < maxsize);
        this->at(size_++) = std::move(value);
    }

    void pop_back() {
        EXPECTS(size_ > 0);
        size_ -= 1;
    }

    void resize(size_type size) {
        EXPECTS(size >= 0);
        size_ = size;
    }

    template<typename InputIt>
    void insert_back(InputIt first, InputIt last) {
        EXPECTS(size_ + (last - first) <= maxsize);
        for(auto itr = first; itr != last; ++itr) { push_back(*itr); }
    }

    void insert_back(size_type count, const T& value) {
        EXPECTS(size_ + count <= maxsize);
        for(size_type i = 0; i < count; i++) { push_back(value); }
    }

    // BoundVec<T, maxsize>& operator = (BoundVec<T, maxsize>& bvec) {
    //   size_ = bvec.size_;
    //   std::copy(bvec.begin(), bvec.end(), begin());
    //   return *this;
    // }

    BoundVec<T, maxsize>& operator=(const BoundVec<T, maxsize>& bvec) {
        size_ = bvec.size_;
        std::copy(bvec.begin(), bvec.end(), begin());
        return *this;
    }

    BoundVec<T, maxsize>& operator=(BoundVec<T, maxsize>&& bvec) {
        clear();
        for(auto&& bv : bvec) { push_back(std::move(bv)); }
        return *this;
    }

    iterator end() { return std::array<T, maxsize>::begin() + size_; }

    const_iterator end() const {
        return std::array<T, maxsize>::begin() + size_;
    }

    reverse_iterator rbegin() const {
        return std::array<T, maxsize>::begin() + size_;
    }

    reference front() {
        EXPECTS(size_ > 0);
        return this->at(0);
    }

    const_reference front() const {
        EXPECTS(size_ > 0);
        return this->at(0);
    }

    reference back() {
        EXPECTS(size_ > 0);
        return this->at(size_ - 1);
    }

    const_reference back() const {
        EXPECTS(size_ > 0);
        return this->at(size_ - 1);
    }

    private:
    size_type size_;
    friend bool operator==(const BoundVec<T, maxsize>& lhs,
                           const BoundVec<T, maxsize>& rhs) {
        return lhs.size() == rhs.size() &&
               std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }

    friend bool operator!=(const BoundVec<T, maxsize>& lhs,
                           const BoundVec<T, maxsize>& rhs) {
        return !(lhs == rhs);
    }
};

template<typename T, int maxsize>
std::ostream& operator<<(std::ostream& os, const BoundVec<T, maxsize>& bvec) {
    os << std::string{"[ "};
    for(auto el : bvec) { os << el << " "; }
    os << std::string{"]"};
    return os;
}

} // namespace tamm

#endif // TAMM_BOUNDVEC_H_
