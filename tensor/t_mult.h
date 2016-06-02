#ifndef __ctce_mult_h__
#define __ctce_mult_h__

//#include "global.h" // ga_sync
#include "func.h"
#include "preprocess.h"
#include "variables.h"
#include "capi.h"
#include "expression.h"
#include "ga.h"

namespace ctce {

  extern "C" {

    /**
     * CCSD(T) multiplication computation (antisymm part), DGEMM + SORTACC
     * @param[in] d_a, k_a_offset, d_b, k_b_offset
     * @param[in] a_c the result of the computation
     * @param[in] tC += tA * tB * coef
     * @param[in] sum_ids summation indices names
     * @param[in] sum_itr summation triangular iterator group
     * @param[in] cp_itr copy iterator group
     * @param[in] tid current value of the tC indices, to compute antisymm iterator inside
     */
    void t_mult(Integer *d_a, Integer *k_a_offset, Integer *d_b, Integer *k_b_offset, double *a_c,
        Tensor &tC, Tensor &tA, Tensor &tB, const double coef,
        const std::vector<IndexName>& sum_ids, 
        IterGroup<triangular>& sum_itr,
        IterGroup<CopyIter>& cp_itr, 
        const std::vector<Integer>& tid);

    /**
     * Outer full loops, wrap t_mult, currently not in use
     * @param[in] d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset
     * @param[in] tC += tA * tB * coef
     * @param[in] sum_ids summation indices names
     * @param[in] sum_itr summation triangular iterator group
     * @param[in] cp_itr copy iterator group
     * @param[in] out_itr outer iterator each value will be pass to t_mult as tid
     */
    void t_mult2(Integer *d_a, Integer *k_a_offset, Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset,
        Tensor& tC, Tensor& tA, Tensor& tB, const double coef,
        const std::vector<IndexName>& sum_ids,
        IterGroup<triangular>& sum_itr,
        IterGroup<CopyIter>& cp_itr,
        IterGroup<triangular>& out_itr);

    /**
     * CCSD multiplication computation, DGEMM + ADD_HASH_BLOCK
     * @param[in] d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset from FORTRAN
     * @param[in] tC += tA * tB * coef
     * @param[in] sum_ids summation indices names
     * @param[in] sum_itr summation triangular iterator group
     * @param[in] cp_itr copy iterator group
     * @param[in] out_itr outer iterator group
     */
    void t_mult3(Integer *d_a, Integer *k_a_offset, Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset,
        Tensor& tC, Tensor& tA, Tensor& tB, const double coef,
        const std::vector<IndexName>& sum_ids,
        IterGroup<triangular>& sum_itr,
        IterGroup<CopyIter>& cp_itr,
								 IterGroup<triangular>& out_itr,
								 Multiplication& m);

    /**
     * Simply wrap t_mult3
     * @param[in] d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset from FORTRAN
     * @param[in] m Multiplication store all the input data needed for t_mult3
     */ 
    void t_mult4(Integer *d_a, Integer *k_a_offset, Integer *d_b, Integer *k_b_offset,
        Integer *d_c, Integer *k_c_offset, Multiplication& m);

  }

}; // namespace ctce

#endif
