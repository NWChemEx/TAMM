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
#ifndef TAMM_INDEX_SORT_H_
#define TAMM_INDEX_SORT_H_

#include <cstddef>
#include <cassert>
#include <type_traits>

#include "tammx/expects.h"

namespace tammx {

void copy(const double *sbuf, double *dbuf, size_t sz, double alpha);

void index_sort_2(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                  int p1, int p2, double alpha);
void index_sort_3(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                  size_t sz3, int p1, int p2, int p3, double alpha);
void index_sort_4(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                  size_t sz3, size_t sz4, int p1, int p2, int p3, int p4,
                  double alpha);

/**
   pi(pvec)(ivec) = (ivec[pvec[0]-1], ivec[pvec[1]-1], ...)
   0<=i<sz1, 0<=j<sz2: dbuf[pi(pvec)(ivec)] = alpha * sbuf[ivec]
 */
template<typename T>
void
index_sort(const void* svbuf, void* dvbuf, int ndim, const size_t *sizes,
                const int *perm, T alpha) {
  bool val = std::is_same<double,T>::value; Expects(val);
  const double *sbuf = reinterpret_cast<const double*>(svbuf);
  double *dbuf = reinterpret_cast<double*>(dvbuf);
  assert(sbuf);
  assert(dbuf);
  assert(ndim >= 0);
  assert(ndim == 0 || sizes);
  assert(ndim == 0 || perm);

  if (ndim == 0) {
    *dbuf = *sbuf * alpha;
  } else if (ndim == 1) {
    copy(sbuf, dbuf, sizes[0], alpha);
  } else if (ndim == 2) {
    index_sort_2(sbuf, dbuf, sizes[0], sizes[1], perm[0], perm[1], alpha);
  } else if (ndim == 3) {
    index_sort_3(sbuf, dbuf, sizes[0], sizes[1], sizes[2], perm[0], perm[1],
                 perm[2], alpha);
  } else if (ndim == 4) {
    index_sort_4(sbuf, dbuf, sizes[0], sizes[1], sizes[2], sizes[3], perm[0],
                 perm[1], perm[2], perm[3], alpha);
  } else {
    assert(0);
  }
}

/**
   dbuf[0..sz-1] = alpha * sbuf[0..sz-1]
 */
void copy_add(const double *sbuf, double *dbuf, size_t sz,
              double alpha);

void index_sortacc_2(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                     int p1, int p2, double alpha);
void index_sortacc_3(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                     size_t sz3, int p1, int p2, int p3, double alpha);
void index_sortacc_4(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                     size_t sz3, size_t sz4, int p1, int p2, int p3, int p4,
                     double alpha);

/**
   pi(pvec)(ivec) = (ivec[pvec[0]-1], ivec[pvec[1]-1], ...)
   0<=i<sz1, 0<=j<sz2: dbuf[pi(pvec)(ivec)] += ialpha * sbuf[ivec]
 */
template<typename T>
void
index_sortacc(const void* svbuf, void* dvbuf, int ndim,
                   const size_t *sizes, const int *perm, T ialpha) {
  //bool val = std::is_same<double,T>::value; Expects(val);
  const double *sbuf = reinterpret_cast<const double*>(svbuf);
  double alpha = static_cast<double>(ialpha);
  double *dbuf = reinterpret_cast<double*>(dvbuf);
  assert(sbuf);
  assert(dbuf);
  assert(ndim >= 0);
  assert(ndim == 0 || sizes);
  assert(ndim == 0 || perm);

  if (ndim == 0) {
    *dbuf += *sbuf * alpha;
  } else if (ndim == 1) {
    copy_add(sbuf, dbuf, sizes[0], alpha);
  } else if (ndim == 2) {
    index_sortacc_2(sbuf, dbuf, sizes[0], sizes[1], perm[0], perm[1], alpha);
  } else if (ndim == 3) {
    index_sortacc_3(sbuf, dbuf, sizes[0], sizes[1], sizes[2], perm[0], perm[1],
                    perm[2], alpha);
  } else if (ndim == 4) {
    index_sortacc_4(sbuf, dbuf, sizes[0], sizes[1], sizes[2], sizes[3], perm[0],
                    perm[1], perm[2], perm[3], alpha);
  } else {
    assert(0);
  }
}

} // namespace tammx

#endif  // TAMM_TENSOR_INDEX_SORT_H_
