#ifndef TAMMX_PERM_H_
#define TAMMX_PERM_H_

#include <algorithm>

#include "tammx/errors.h"
#include "tammx/types.h"

namespace tammx {

/**
 * requires from.size() == to.size()
 * Return ret such that.
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
   Returns number of inversions involved in sorting this permutation
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
  for(int i=0; i<perm.size(); i++) {
    auto itr = std::find(perm.begin(), perm.end(), i);
    EXPECTS(itr != perm.end());
    num_inversions += std::abs((itr - perm.begin()) - i);
  }
#endif
  return num_inversions / 2;
}

/**
 * requires label.size() == perm.size. say n = label.size().
 * Returns ret such that
 * 0<=i<n: ret[i] = label[perm[i]].
 *
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
 * requires p1.size() == p2.size(). say p1.size() ==n.
 * requires p1 and p2 are permutations of [0..n-1].
 * Returns ret such that.
 * 0<=i<n: ret[i] = p1[p2[i]]
 *
 * ret = p2 . p1
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

inline bool
is_permutation(PermVec perm) {
  std::sort(perm.begin(), perm.end());
  // return std::adjacent_find(perm.begin(), perm.end()) == perm.end();
  for(int i=0 ;i<perm.size(); i++) {
    if(perm[i] != i)
      return false;
  }
  return true;
}

/**
 * requires is_permutation(perm).
 * say n = perm.size().
 * Returns ret such that.
 * 0<=i<n: ret[perm[i]] = i
 *
 * ret = perm^{-1}
 *
 * Identity(n) = [0, 1, ..., n-1].
 * perm_compose(perm, perm_invert(perm)) = Identity(n).
 
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

} // namespace tammx

#endif  // TAMMX_PERM_H_
