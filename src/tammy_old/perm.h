#ifndef TAMMY_PERM_H_
#define TAMMY_PERM_H_

#include <algorithm>

#include "tammy/errors.h"
#include "tammy/types.h"

/**
 * @file Operations on permutations
 * @defgroup perm
 */

namespace tammy {

/**
 * @ingroup perm
 * @brief Compute permutation to be performed to permute vector @p from to vector @p to.
 * @param from Source vector for the permutation
 * @param to Target vector for the permutation
 * @pre @p from and @p to are permutations of each other
 * @pre from.size() == to.size()
 * @return Vector to permute @p from to @p to.
 * @post Return ret such that:
 * ensures 0<=i<from.size(): to[i] = from[ret[i]]
 */
inline PermVec
perm_compute(const IndexLabelVec& from, const IndexLabelVec& to) {
  PermVec layout;

  EXPECTS(from.size() == to.size());
  for(auto p : to) {
    auto itr = std::find(from.begin(), from.end(), p);
    EXPECTS(itr != from.end());
    layout.push_back(itr - from.begin());
  }
  return layout;
}

/**
 * @ingroup perm
 * @brief Returns number of inversions (https://en.wikipedia.org/wiki/Inversion_(discrete_mathematics)) involved in sorting this permutation.
 * The input vector should be a permutation vector.
 *
 * @param perm Permutation vector
 * @pre @p perm is a permutation of integers [0,perm.size()-1].
 * @return Number of inversions
 */
inline int
perm_count_inversions(const PermVec& perm) {
  int num_inversions = 0;
  PermVec perm_sort{perm};
#if 0
  std::sort(perm_sort.begin(), perm_sort.end());
  EXPECTS(std::adjacent_find(perm_sort.begin(), perm_sort.end()) == perm_sort.end());
  using size_type = PermVec::size_type;
  for(size_type i=0; i<perm.size(); i++) {
    auto itr = std::find(perm_sort.begin(), perm_sort.end(), perm[i]);
    EXPECTS(itr != perm.end());
    num_inversions += std::abs((itr - perm.begin()) - i);
  }
#else
  std::sort(perm_sort.begin(), perm_sort.end());
  EXPECTS(std::adjacent_find(perm_sort.begin(), perm_sort.end()) == perm_sort.end());
  //using size_type = PermVec::size_type;
  for(size_t i=0; i<perm.size(); i++) {
    auto itr = std::find(perm.begin(), perm.end(), i);
    EXPECTS(itr != perm.end());
    num_inversions += std::abs((itr - perm.begin()) - i);
  }
#endif
  return num_inversions / 2;
}

/**
 * @ingroup perm
 * @brief Apply a permutation to a given vector
 *
 * @tparam T Type of tensor vector element
 * @param label Vector to be permuted
 * @param perm Permutation to be applied on @p label
 * @return Permuted vector
 * @pre label.size() == perm.size, say, n = label.size()
 * @post Returns ret such that
 * 0<=i<n: ret[i] = label[perm[i]].
 *
 * @note
 * perm_apply(from, perm_compute(from,to)) == to.
 * perm_compute(from, perm_apply(from, perm)) == perm.
 */
template<typename T>
inline TensorVec<T>
perm_apply(const TensorVec<T>& label, const PermVec& perm) {
  TensorVec<T> ret;
  // std::cerr<<__FUNCTION__<<":"<<__LINE__<<": label="<<label<<std::endl;
  // std::cerr<<__FUNCTION__<<":"<<__LINE__<<": perm="<<perm<<std::endl;
  EXPECTS(label.size() == perm.size());
  using size_type = PermVec::size_type;
  for(size_type i=0; i<label.size(); i++) {
    ret.push_back(label[perm[i]]);
  }
  return ret;
}

/**
 * @ingroup perm
 * @brief Compose two permutations.
 *
 * @param p1 First permutation
 * @param p2 Second permutation
 * @return Composed permutation ret = p2 . p1
 * @pre p1.size() == p2.size(). say p1.size() == n.
 * @pre p1 and p2 are permutations of [0..n-1].
 * @post Returns ret such that ret = p2.p1, as in,
 * 0<=i<n: ret[i] = p1[p2[i]]
 */
inline PermVec
perm_compose(const PermVec& p1, const PermVec& p2) {
  PermVec ret(p1.size());
  EXPECTS(p1.size() == p2.size());
  for(unsigned i=0; i<p1.size(); i++) {
    ret[i] = p1[p2[i]];
  }
  return ret;
}

/**
 * @ingroup perm
 * @brief Checks whether the argument is a permutation vector
 *
 * @param perm Vector to be checked
 * @return true if @p perm is a permutation of integers in [0,perm.size()-1]. false otherwise.
 */
inline bool
is_permutation(PermVec perm) {
  std::sort(perm.begin(), perm.end());
  // return std::adjacent_find(perm.begin(), perm.end()) == perm.end();
  for(size_t i=0; i<perm.size(); i++) {
    if(perm[i] != i)
      return false;
  }
  return true;
}

/**
 * @ingroup perm
 * @brief Invert a permutation
 *
 * @param perm Input permutation
 * @return Return ret = perm^{-1}
 * @pre is_permutation(perm).
 * @post Returns ret such that.
 * 0<=i<perm.size(): ret[perm[i]] = i
 *
 * @note
 * @code
 * Identity(n) = [0, 1, ..., n-1].
 * perm_compose(perm, perm_invert(perm)) = Identity(n).
 * @endcode
 */
inline PermVec
perm_invert(const PermVec& perm) {
  PermVec ret(perm.size());
  EXPECTS(is_permutation(perm));
  for(unsigned i=0; i<perm.size(); i++) {
    auto itr = std::find(perm.begin(), perm.end(), i);
    EXPECTS(itr != perm.end());
    ret[i] = itr - perm.begin();
  }
  return ret;
}

} // namespace tammy

#endif  // TAMMY_PERM_H_
