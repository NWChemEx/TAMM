#ifndef __tamm_assign_h__
#define __tamm_assign_h__

#include "func.h"
#include "preprocess.h"
#include "variables.h"
#include "capi.h"
#include "expression.h"
#include "ga_abstract.h"

namespace tamm {
/**
 * CCSD assignment computation, ADD_HASH_BLOCK
 * @param[in] d_a, k_a_offset, d_c, k_c_offset from FORTRAN
 * @param[in] tC += tA * coef
 * @param[in] out_itr outer iterator group
 */
void t_assign2(Tensor& tC, 
               const vector<IndexName> &c_ids, 
               Tensor& tA, const vector<IndexName> &a_ids, 
               IterGroup<triangular>& out_itr, double coef, 
               gmem::Handle sync_ga=gmem::NULL_HANDLE, int spos=0);

/**
 * Simply wrap t_assign2
 * @param[in] d_a, k_a_offset, d_c, k_c_offset from FORTRAN
 * @param[in] a Assignment store all the input data needed for t_assign2
 */
void t_assign3(Assignment& a, gmem::Handle sync_ga=gmem::NULL_HANDLE, int spos=0);

} // namespace tamm

#endif
