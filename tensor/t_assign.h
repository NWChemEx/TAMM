#ifndef _ASSIGN_H_
#define _ASSIGN_H_

#include "func.h"
#include "preprocess.h"
#include "variables.h"
#include "capi.h"
#include "expression.h"
#include "ga.h"

namespace ctce {

  extern "C" {

    /**
     * CCSD assignment computation, ADD_HASH_BLOCK
     * @param[in] d_a, k_a_offset, d_c, k_c_offset from FORTRAN
     * @param[in] tC += tA * coef
     * @param[in] out_itr outer iterator group
     */
    void t_assign2(
        Integer* d_a, Integer* k_a_offset,
        Integer* d_c, Integer* k_c_offset,
        Tensor& tC, const vector<IndexName> &c_ids,
				Tensor& tA, const vector<IndexName> &a_ids,
				IterGroup<triangular>& out_itr, double coef,
                   int sync_ga=0);

    /**
     * Simply wrap t_assign2
     * @param[in] d_a, k_a_offset, d_c, k_c_offset from FORTRAN
     * @param[in] a Assignment store all the input data needed for t_assign2
     */
    void t_assign3(Integer* d_a, Integer* k_a_offset,
                   Integer* d_c, Integer* k_c_offset, Assignment& a, int sync_ga=0);

    void t_assign4(Integer* d_a, Integer* k_a_offset,
        Integer* d_c, Integer* k_c_offset, Assignment& a);

  }

}; // namespace ctce

#endif
