
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
#include <iostream>
#include "tansor/fapi.h"
#include "tensor/gmem.h"
#include "tensor/iterGroup.h"
#include "tenosr/t_mult.h"
#include "tensor/triangular.h"
#include "tensor/variables.h"

namespace tamm {

extern "C" {

static bool set_ccsd_t = true;
static Multiplication m0, m1, m2;

void gen_ccsd_t_cxx_() {
  if (set_ccsd_t) {
    // comment NAG std::cout << "tamm: generate ccsd_t singles and doubles 1
    // 2.\n";
    Tensor tC, tA, tB;
    std::vector<int> cpt;

    DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;

    tC = Tensor6(P4B, P5B, P6B, H1B, H2B, H3B, 0, 0, 0, 1, 1, 1, iVT_tensor,
                 dist_nw, dim_ov);
    tA = Tensor2(P4B, H1B, 0, 1, T_tensor, dist_nwma, dim_ov);
    tB = Tensor4(P5B, P6B, H2B, H3B, 0, 0, 1, 1, V_tensor, idist, dim_n);
    m0 = Multiplication(tC, tA, tB, 1.0);
    cpt = newVec<int>(2, 2, 2);
    m0.setCopyItr(cpt);

    tC = Tensor6(P4B, P5B, P6B, H1B, H2B, H3B, 0, 0, 0, 1, 1, 1, iVT_tensor,
                 dist_nw, dim_ov);
    tA = Tensor4(P4B, P5B, H1B, H7B, 0, 0, 1, 1, T_tensor, dist_nw, dim_ov);
    tB = Tensor4(H7B, P6B, H2B, H3B, 0, 1, 2, 2, V_tensor, idist, dim_n);
    m1 = Multiplication(tC, tA, tB, -1.0);
    cpt = newVec<int>(2, 1, 2);
    m1.setCopyItr(cpt);

    tC = Tensor6(P4B, P5B, P6B, H1B, H2B, H3B, 0, 0, 0, 1, 1, 1, iVT_tensor,
                 dist_nw, dim_ov);
    tA = Tensor4(P4B, P7B, H1B, H2B, 0, 0, 1, 1, T_tensor, dist_nw, dim_ov);
    tB = Tensor4(P5B, P6B, H3B, P7B, 0, 0, 1, 2, V_tensor, idist, dim_n);
    m2 = Multiplication(tC, tA, tB, -1.0);
    cpt = newVec<int>(2, 2, 1);
    m2.setCopyItr(cpt);

    set_ccsd_t = false;
  }
}

/* i0 ( p4 p5 p6 h1 h2 h3 )_vt + = 1 * P( 9 ) * t ( p4 h1 )_t * v ( p5 p6 h2 h3
 * )_v */
void ccsd_t_singles_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_b,
                           Integer *k_b_offset, double *a_c,
                           const std::vector<Integer> &tid) {
  t_mult(d_a, k_a_offset, d_b, k_b_offset, a_c, m0.tC(), m0.tA(), m0.tB(),
         m0.coef(), m0.sum_ids(), &m0.sum_itr(), &m0.cp_itr(), tid, &m0);
}

/* i0 ( p4 p5 p6 h1 h2 h3 )_vt + = -1 * P( 9 ) * Sum ( h7 ) * t ( p4 p5 h1 h7
 * )_t * v ( h7 p6 h2 h3 )_v */
void ccsd_t_doubles_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_b,
                           Integer *k_b_offset, double *a_c,
                           const std::vector<Integer> &tid) {
  t_mult(d_a, k_a_offset, d_b, k_b_offset, a_c, m1.tC(), m1.tA(), m1.tB(),
         m1.coef(), m1.sum_ids(), &m1.sum_itr(), &m1.cp_itr(), tid, &m1);
}

/* i0 ( p4 p5 p6 h1 h2 h3 )_vt + = -1 * P( 9 ) * Sum ( p7 ) * t ( p4 p7 h1 h2
 * )_t * v ( p5 p6 h3 p7 )_v */
void ccsd_t_doubles_2_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_b,
                           Integer *k_b_offset, double *a_c,
                           const std::vector<Integer> &tid) {
  t_mult(d_a, k_a_offset, d_b, k_b_offset, a_c, m2.tC(), m2.tA(), m2.tB(),
         m2.coef(), m2.sum_ids(), &m2.sum_itr(), &m2.cp_itr(), tid, &m2);
}

void ccsd_t_cxx_(Integer *k_t1_local, Integer *d_t1, Integer *k_t1_offset,
                 Integer *d_t2, Integer *k_t2_offset, Integer *d_v2,
                 Integer *k_v2_offset, double *energy1, double *energy2,
                 Integer *size_t1) {
#if 0
      double *p_k_t1_local = Variables::dbl_mb() + *k_t1_local;
      p_k_t1_local = static_cast<double *>(malloc((*size_t1)*sizeof(double)));
      memset(p_k_t1_local, 0 , (*size_t1)*sizeof(double));
      // ma_zero_(&dbl_mb[k_t1_local],size_t1);
      get_block_(d_t1, p_k_t1_local, size_t1, &Variables::izero());
#endif  // If 0

  // GA initialization
  int nprocs = gmem::ranks();
  int count = 0;
  int taskDim = 1;
  char taskStr[10] = "NXTASK2";
  gmem::Handle taskHandle =
      gmem::create(gmem::Int, taskDim, taskStr);  // global array for next task
  gmem::zero(taskHandle);                         // initialize to zero

  gmem::sync();

  // get next task
  int sub = 0;
  int next;
  next = static_cast<int>(gmem::atomic_fetch_add(taskHandle, sub, 1));

  //      printf("ccsdt#%d = %d\n",GA_Nodeid(),next);

  *energy1 = 0.0;
  *energy2 = 0.0;
  double energy[2];
  energy[0] = 0.0;
  energy[1] = 0.0;

  Tensor tC =
      Tensor6(P4B, P5B, P6B, H1B, H2B, H3B, 0, 0, 0, 1, 1, 1, iVT_tensor);
  IterGroup<triangular> out_itr;
  // genTrigIter(&out_itr,tC.name(),tC.ext_sym_group());
  genTrigIter(&out_itr, id2name(tC.ids()), ext_sym_group(tC.ids()));
  out_itr.setType(TRIG2);

  gen_ccsd_t_cxx_();  // generate singles and doubles expr

  std::vector<Integer> vec, rvec, ovec;
  out_itr.reset();

  // comment NAG std::cout << "singles sloop, doubles cxx.\n";

  // std::cout << "NAG at LINE --- "<< __LINE__<<"\n";

  while (out_itr.next(&vec)) {
    rvec = out_itr.v_range();
    ovec = out_itr.v_offset();

    if (next == count) {
      // printf("ccsdt#%d: %d==%d\n",GA_Nodeid(),next,count);

      if ((is_spatial_nonzero(vec, 0)) && (is_spin_nonzero(vec)) &&
          (is_spin_restricted_le(vec, 8))) {
        Integer rsize = compute_size(vec);
        double *buf_double = static_cast<double*>(malloc(rsize*sizeof(double)));
        double *buf_single = static_cast<double*>(malloc(rsize*sizeof(double)));
        memset(buf_single, 0, rsize * sizeof(double));
        memset(buf_double, 0, rsize * sizeof(double));

        Integer toggle = 2;
#if 0
        ccsd_t_singles_l_(buf_single, k_t1_local, d_v2, k_t1_offset,
                          k_v2_offset, &vec[3], &vec[4], &vec[5], &vec[0],
                          &vec[1], &vec[2], &toggle);
#else
        ccsd_t_singles_1_cxx_(k_t1_local, k_t1_offset, d_v2, k_v2_offset,
                              buf_single, vec);
#endif  // Fortran Functions

#if 0
        ccsd_t_doubles_(buf_double, d_t2, d_v2, k_t2_offset,
                        k_v2_offset, &vec[3], &vec[4], &vec[5], &vec[0],
                        &vec[1], &vec[2], &toggle);
#else
        ccsd_t_doubles_1_cxx_(d_t2, k_t2_offset, d_v2, k_v2_offset, buf_double,
                              vec);
        ccsd_t_doubles_2_cxx_(d_t2, k_t2_offset, d_v2, k_v2_offset, buf_double,
                              vec);
#endif  // Fortran functions
        double factor = computeFactor(vec);
        computeEnergy(rvec, ovec, energy1, energy2, buf_single, buf_double,
                      factor);
        free(buf_single);
        free(buf_double);
      }  // if spatial

      int sub = 0;
      next = static_cast<int>(gmem::atomic_fetch_add(taskHandle, sub, 1));
    }  // if next == count

    // std::cout << "NAG at LINE --- "<< __LINE__<<"\n";

    count = count + 1;
  }  // out_itr

  // std::cout << "NAG at LINE --- "<< __LINE__<<"\n";

  gmem::sync();
  gmem::destroy(taskHandle);  // free

  energy[0] = *energy1;
  energy[1] = *energy2;
  gmem::op(energy, 2, gmem::Plus);  // collect data
  *energy1 = energy[0];
  *energy2 = energy[1];

  //      free(p_k_t1_local);
}  // ccsd_t.F
}  // extern C
};  // namespace tamm
