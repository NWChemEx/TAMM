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

#include "tammy/errors.h"

/**
 * @defgroup index_sort
 *
 * @file index_sort
 * @brief Wrapper for index permutation routines. The implementation of these methods can be performed using optimized libraries.
 *
 */

namespace tammy {

/**
 * @brief Scale elements in source array and assign to target array
 * @param[in] sbuf Pointer to source buffer
 * @param[in] dbuf Pointer to target buffer
 * @param[in] sz Number of elements in the buffer
 * @param[in] alpha Scaling factor
 *
 * @post
 * @code
 * 0<=i<=sz: dbuf[i] = alpha * sbuf[i];
 * @endcode
 */
void copy(const double *sbuf, double *dbuf, size_t sz, double alpha);

/**
 * @brief Scale elements in source 2-d array and assign to target 2-d array, with permutation (p1,p2).
 *
 * @param[in] sbuf Pointer to source buffer
 * @param[in] dbuf Pointer to target buffer
 * @param[in] sz1, sz2, .. Size of dimension 1 of source array
 * @param[in] p1, p2, .. Index permutation order
 * @param[in] alpha Scaling factor
 * @pre p1, p2, .. in {1, 2, ..} and pi != pj if i != j
 */
void index_sort_2(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                  int p1, int p2, double alpha);

/**
 * @copydoc index_sort_2
 */
void index_sort_3(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                  size_t sz3, int p1, int p2, int p3, double alpha);
void index_sort_4(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                  size_t sz3, size_t sz4, int p1, int p2, int p3, int p4,
                  double alpha);

/**
 * @brief Generic index permutation interface used by rest of the code.
 * @tparam T type of each element
 * @param sbuf Pointer to source buffer
 * @param dbuf Pointer to target buffer
 * @param ndim Number of dimension (for both buffers)
 * @param sizes Size of dimensions for source buffer (array of ndim elements)
 * @param perm Permutation to be applied (array of ndim elements)
 * @param ialpha Scaling factor
 *
 * @pre ndim >= 0
 * @pre sbuf != nullptr
 * @pre dbuf != nullptr
 * @pre ndim>0 => (sizes != nullptr and sizes[0..ndim] is valid)
 * @pre ndim>0 => (perm != nullptr and perm[0..ndim] is valid)
 *
 * @post
 * @code
 * pi(pvec)(ivec) = (ivec[pvec[0]-1], ivec[pvec[1]-1], ...);
 * 0**ndim <= ivec <= sizes : dbuf[pi(pvec)(ivec)] = alpha * sbuf[ivec];
 * @endcode
 */
template<typename T>
void
index_sort(const double* sbuf, double* dbuf, int ndim, const size_t *sizes,
                const int *perm, T ialpha) {
  bool val = std::is_convertible<T,double>::value; EXPECTS(val);
  EXPECTS(sbuf);
  EXPECTS(dbuf);
  EXPECTS(ndim >= 0);
  EXPECTS(ndim == 0 || sizes);
  EXPECTS(ndim == 0 || perm);

  double alpha = static_cast<double>(ialpha);
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
 * @brief Scale elements in source array and assign to target array
 * @param[in] sbuf Pointer to source buffer
 * @param[in] dbuf Pointer to target buffer
 * @param[in] sz Number of elements in the buffer
 * @param[in] alpha Scaling factor
 *
 * @post
 * @code
 * 0<=i<=sz: dbuf[i] += alpha * sbuf[i];
 * @endcode
 */void copy_add(const double *sbuf, double *dbuf, size_t sz,
              double alpha);

/**
 * @brief Same as index_sort_2, but add to the target buffer
 * @copydetails index_sort_2
 */
void index_sortacc_2(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                     int p1, int p2, double alpha);

/**
 * @brief Same as index_sort_3, but add to the target buffer
 * @copydetails index_sort_3
 */
void index_sortacc_3(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                     size_t sz3, int p1, int p2, int p3, double alpha);

/**
 * @brief Same as index_sort_4, but add to the target buffer
 * @copydetails index_sort_4
 */
void index_sortacc_4(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                     size_t sz3, size_t sz4, int p1, int p2, int p3, int p4,
                     double alpha);

/**
   pi(pvec)(ivec) = (ivec[pvec[0]-1], ivec[pvec[1]-1], ...)
   0<=i<sz1, 0<=j<sz2: dbuf[pi(pvec)(ivec)] += ialpha * sbuf[ivec]
 */
/**
 * Same as index_sort() but add to the target buffer
 * @copydetails index_sort
 */
template<typename T>
void
index_sortacc(const double* sbuf, double* dbuf, int ndim,
                   const size_t *sizes, const int *perm, T ialpha) {
  bool val = std::is_convertible<T,double>::value; EXPECTS(val);
  double alpha = static_cast<double>(ialpha);
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

} // namespace tammy

#endif  // TAMM_TENSOR_INDEX_SORT_H_
