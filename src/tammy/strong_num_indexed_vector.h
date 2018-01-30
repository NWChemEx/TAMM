// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMY_STRONGNUM_INDEXED_VECTOR_H__
#define TAMMY_STRONGNUM_INDEXED_VECTOR_H__

#include "strong_num.h"

namespace tammy {

/**
 * @brief Vector wrapper with strongly typed indeces
 *
 * This class describes a vector wrapper with strongly typed indeces, mainly used in index space construction for avoiding implicit casts
 *
 * @code
 * class StrongIntSpace;
 * using StrongInt = StrongNum<StrongIntSpace, int>;
 * using StrongIntIndexedVector = StrongNumIndexedVector <int, StrongInt>;
 * @endcode
 *
 * @tparam T Data type for vector contents
 * @tparam Index Strongly typed index type
 */
template<typename T, typename Index>
class StrongNumIndexedVector : public std::vector<T> {
 public:
  using std::vector<T>::vector;
  using size_type = typename std::vector<T>::size_type;
  
  StrongNumIndexedVector() = default;

  StrongNumIndexedVector(const std::vector<T>& vec)
      : std::vector<T>{vec} {}

  StrongNumIndexedVector(const StrongNumIndexedVector<T,Index>& svec)
      : std::vector<T>{svec} {}

  const T& operator [] (Index sint) const {
    return std::vector<T>::operator[](sint.template value<size_type>());
  }

  T& operator [] (Index sint) {
    return std::vector<T>::operator[](sint.template value<size_type>());
  }
};

} // namespace tammy


#endif // TAMMY_STRONGNUM_INDEXED_VECTOR_H__