//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------
#ifndef TAMM_TENSOR_ANTISYMM_H_
#define TAMM_TENSOR_ANTISYMM_H_

#include <algorithm>
#include <vector>
#include "tensor/variables.h"

namespace tamm {

/**
 * Triangular loop class
 */
class tloop {
 public:
  /**
   *Constructor
   */
  tloop();

  /**
   * Destructor
   */
  ~tloop();

  /**
   * Constructor
   *
   */
  tloop(const std::vector<size_t>& _indices, const int& _nloops);

  const std::vector<int>& vec() const;

  bool nextIter();

 private:
  int lb, ub;
  std::vector<size_t> indices;
  int nloops;
  bool first_time;
  std::vector<int> loops;
  void complete(); /* loops.push_back() */
  /**
   * int nextpos
   * @param[in] pos
   * @param[out] ret
   */
  int nextpos(int pos);
};

/**
 * Antisymm iterator. Iterate input (3 2)(4) as (2 3 4),(2 4 3),(3,4,2) where
 * (a<b)(c)
 */
class antisymm {
 public:
  /**
   * Constructor
   */
  antisymm();

  /**
   * Destructor
   */
  ~antisymm();

  /**
   * Constructor
   * @param[in] name IndexName of the iterator
   * @param[in] s1 number of anti-symmetry group from A tensor
   * @param[in] s2 number of anti-symmetry group from B tensor
   */
  antisymm(const std::vector<size_t>& vtab, const std::vector<IndexName>& name,
           int s1, int s2);

  /**
   * Check if this iterator is empty
   */
  bool empty() const;

  /**
   * Reset this anti-symmetry iterator
   */
  void reset();

  /**
   * Get current value of iterator.
   * @param[in] vec get current value and store it in vec
   * @return return false if end of iteration
   */
  inline bool next(
      std::vector<size_t>*
          vec); /*< enumerate next permuation for computation, store in &vec */

  /* following 3 no use, just for passing compilation for iterGroup */
  int sign() const;
  const std::vector<size_t>& v_range();
  const std::vector<size_t>& v_offset();

 private:
  tloop tl0;                 /*< initial setting of tloop */
  tloop tl1;                 /*< relicate for iterating */
  std::vector<size_t> slist; /*< current value */
  std::vector<size_t> s1;    /*< current value from A tensor */
  std::vector<size_t> s2;    /*< current value from B tensor */
  bool empty_;               /*< check if antisymm is empty */
};

inline tloop::tloop() {}

inline tloop::~tloop() {}

inline tloop::tloop(const std::vector<size_t>& _indices, const int& _nloops)
    : indices(_indices), nloops(_nloops), first_time(true), lb(0) {
  ub = indices.size();
  complete();
}

inline const std::vector<int>& tloop::vec() const { return loops; }

inline bool tloop::nextIter() {
  if (first_time == true) {
    first_time = false;
    return true;
  }
  while (loops.size() > 0 &&
         (indices.size() - nextpos(loops.back()) <= nloops - loops.size()))
    loops.pop_back();
  if (loops.size() == 0) return false;
  loops.back() = nextpos(loops.back());
  complete();
  return true;
}

inline void tloop::complete() {
  if (loops.size() == 0) loops.push_back(lb);
  while (loops.size() < nloops) loops.push_back(loops.back() + 1);
}

inline int tloop::nextpos(int pos) {
  assert(pos < indices.size());
  int ret;
  for (ret = pos + 1; ret < indices.size() && indices[ret] == indices[pos];
       ret++) {
  }
  return ret;
}

inline antisymm::antisymm() {}

inline antisymm::~antisymm() {}

inline antisymm::antisymm(const std::vector<size_t>& vtab,
                          const std::vector<IndexName>& name, int n1, int n2) {
  int n = name.size();
  slist.resize(n);
  if (n == 0) {
    empty_ = true;
  } else {
    empty_ = false;
    // const std::vector<size_t>& vtab = Table::value(); /* get value from table
    // */
    for (int i = 0; i < n; i++) slist[i] = vtab[name[i]];
    sort(slist.begin(), slist.end());
  }
  tl0 = tloop(slist, n1);
  tl1 = tl0;
}

inline bool antisymm::empty() const { return empty_; }

inline void antisymm::reset() { tl1 = tl0; }

inline int antisymm::sign() const { return 1; }

inline const std::vector<size_t>& antisymm::v_range() {}

inline const std::vector<size_t>& antisymm::v_offset() {}

inline bool antisymm::next(std::vector<size_t>* vec) {
  if (!tl1.nextIter()) return false;
  const std::vector<int>& itr = tl1.vec();
  s1.clear();
  s2.clear();
  int p = 0;
  for (int i = 0; i < itr.size(); i++) {
    while (p < itr[i]) {
      s2.push_back(slist[p]);
      p++;
    }
    s1.push_back(slist[itr[i]]);
    p = itr[i] + 1;
  }
  while (p < slist.size()) {
    s2.push_back(slist[p]);
    p++;
  }
  *vec = s1;
  vec->insert(vec->end(), s2.begin(), s2.end());
  return true;
}

} /* namespace tamm */

#endif  // TAMM_TENSOR_ANTISYMM_H_
