#ifndef __tamm_mult_h__
#define __tamm_mult_h__

//#include "global.h" // ga_sync
#include "capi.h"
#include "expression.h"
#include "func.h"
#include "gmem.h"
#include "preprocess.h"
#include "variables.h"

namespace tamm {

/**
 * CCSD(T) multiplication computation (antisymm part), DGEMM + SORTACC
 * @param[in] d_a, k_a_offset, d_b, k_b_offset
 * @param[in] a_c the result of the computation
 * @param[in] tC += tA * tB * coef
 * @param[in] sum_ids summation indices names
 * @param[in] sum_itr summation triangular iterator group
 * @param[in] cp_itr copy iterator group
 * @param[in] tid current value of the tC indices, to compute antisymm iterator
 * inside
 */
void t_mult(double* a_c, Tensor& tC, Tensor& tA, Tensor& tB, const double coef,
            const std::vector<IndexName>& sum_ids,
            IterGroup<triangular>& sum_itr, IterGroup<CopyIter>& cp_itr,
            const std::vector<size_t>& tid, Multiplication& m);

/**
 * Outer full loops, wrap t_mult, currently not in use
 * @param[in] d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset
 * @param[in] tC += tA * tB * coef
 * @param[in] sum_ids summation indices names
 * @param[in] sum_itr summation triangular iterator group
 * @param[in] cp_itr copy iterator group
 * @param[in] out_itr outer iterator each value will be pass to t_mult as tid
 */
void t_mult2(Tensor& tC, Tensor& tA, Tensor& tB, const double coef,
             const std::vector<IndexName>& sum_ids,
             IterGroup<triangular>& sum_itr, IterGroup<CopyIter>& cp_itr,
             IterGroup<triangular>& out_itr, Multiplication& m);

/**
 * CCSD multiplication computation, DGEMM + ADD_HASH_BLOCK
 * @param[in] d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset from FORTRAN
 * @param[in] tC += tA * tB * coef
 * @param[in] sum_ids summation indices names
 * @param[in] sum_itr summation triangular iterator group
 * @param[in] cp_itr copy iterator group
 * @param[in] out_itr outer iterator group
 */
void t_mult3(Tensor& tC, Tensor& tA, Tensor& tB, const double coef,
             const std::vector<IndexName>& sum_ids,
             IterGroup<triangular>& sum_itr, IterGroup<CopyIter>& cp_itr,
             IterGroup<triangular>& out_itr, Multiplication& m,
             gmem::Handle sync_ga = gmem::NULL_HANDLE, int spos = 0);

/**
 * Simply wrap t_mult3
 * @param[in] d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset from FORTRAN
 * @param[in] m Multiplication store all the input data needed for t_mult3
 */
void t_mult4(Multiplication& m, gmem::Handle sync_ga = gmem::NULL_HANDLE,
             int spos = 0);

}  // namespace tamm

#endif
