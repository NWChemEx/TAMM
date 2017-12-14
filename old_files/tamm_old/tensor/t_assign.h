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
#ifndef TAMM_TENSOR_T_ASSIGN_H_
#define TAMM_TENSOR_T_ASSIGN_H_

#include <vector>
#include "tensor/capi.h"
#include "tensor/expression.h"
#include "tensor/func.h"
#include "tensor/gmem.h"
#include "tensor/preprocess.h"
#include "tensor/variables.h"

namespace tamm {
/**
 * CCSD assignment computation, ADD_HASH_BLOCK
 * @param[in] d_a, k_a_offset, d_c, k_c_offset from FORTRAN
 * @param[in] tC += tA * coef
 * @param[in] out_itr outer iterator group
 */
void t_assign2(const Tensor& tC, const std::vector<IndexName>& c_ids,
               const Tensor& tA, const std::vector<IndexName>& a_ids,
               IterGroup<triangular>* out_itr, double coef,
               gmem::Handle sync_ga = gmem::NULL_HANDLE, int spos = 0);

/**
 * Simply wrap t_assign2
 * @param[in] d_a, k_a_offset, d_c, k_c_offset from FORTRAN
 * @param[in] a Assignment store all the input data needed for t_assign2
 */
void t_assign3(Assignment* a, gmem::Handle sync_ga = gmem::NULL_HANDLE,
               int spos = 0);

}  // namespace tamm

#endif  // TAMM_TENSOR_T_ASSIGN_H_
