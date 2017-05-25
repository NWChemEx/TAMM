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
#ifndef TAMM_TENSOR_INDEX_SORT_H_
#define TAMM_TENSOR_INDEX_SORT_H_

#include <cstddef>

namespace tamm {

void index_sort(const double *sbuf, double *dbuf, int ndim, const size_t *sizes,
                const int *perm, double alpha);

void index_sortacc(const double *sbuf, double *dbuf, int ndim,
                   const size_t *sizes, const int *perm, double alpha);

} /*namespace tamm*/

#endif  // TAMM_TENSOR_INDEX_SORT_H_
