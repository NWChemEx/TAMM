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
#include "tensor/corf.h"
#include "tensor/equations.h"
#include "tensor/input.h"
#include "tensor/schedulers.h"
#include "tensor/t_assign.h"
#include "tensor/t_mult.h"
#include "tensor/tensor.h"
#include "tensor/tensors_and_ops.h"
#include "tensor/variables.h"

/*
 *  ph {
 *
 *  index h1,h2,h3,h4,h5,h6,h7 = O;
 *  index p1,p2,p3,p4,p5,p6 = V;
 *
 *  array i0[V][O];
 *  array t_vo[V][O]: irrep_t;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array y_ov[O][V]: irrep_y;
 *  array y_oovv[O,O][V,V]: irrep_y;
 *  array ph_3_1[O][O];
 *  array ph_4_1[O,O][O,V];
 *
 *  ph_1:       i0[p2,h1] += 1 * t_vo[p2,h1];
 *  ph_2:       i0[p2,h1] += 1 * t_vvoo[p2,p3,h1,h4] * y_ov[h4,p3];
 *  ph_3_1:     ph_3_1[h7,h1] += 1 * t_vo[p3,h1] * y_ov[h7,p3];
 *  ph_3_2:     ph_3_1[h7,h1] += -1/2 * t_vvoo[p3,p4,h1,h5] *
 * y_oovv[h5,h7,p3,p4]; ph_3:       i0[p2,h1] += -1 * t_vo[p2,h7] *
 * ph_3_1[h7,h1]; ph_4_1:     ph_4_1[h4,h5,h1,p3] += -1 * t_vo[p6,h1] *
 * y_oovv[h4,h5,p3,p6]; ph_4:       i0[p2,h1] += -1/2 * t_vvoo[p2,p3,h4,h5] *
 * ph_4_1[h4,h5,h1,p3];
 *
 *  }
 */

extern "C" {
void ccsd_1prdm_ph_1_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_i0,
                      F77Integer *k_i0_offset);
void ccsd_1prdm_ph_2_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset,
                      F77Integer *d_y_ov, F77Integer *k_y_ov_offset, F77Integer *d_i0,
                      F77Integer *k_i0_offset);
void ccsd_1prdm_ph_3_1_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset,
                        F77Integer *d_y_ov, F77Integer *k_y_ov_offset,
                        F77Integer *d_ph_3_1, F77Integer *k_ph_3_1_offset);
void ccsd_1prdm_ph_3_2_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset,
                        F77Integer *d_y_oovv, F77Integer *k_y_oovv_offset,
                        F77Integer *d_ph_3_1, F77Integer *k_ph_3_1_offset);
void ccsd_1prdm_ph_3_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset,
                      F77Integer *d_ph_3_1, F77Integer *k_ph_3_1_offset,
                      F77Integer *d_i0, F77Integer *k_i0_offset);
void ccsd_1prdm_ph_4_1_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset,
                        F77Integer *d_y_oovv, F77Integer *k_y_oovv_offset,
                        F77Integer *d_ph_4_1, F77Integer *k_ph_4_1_offset);
void ccsd_1prdm_ph_4_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset,
                      F77Integer *d_ph_4_1, F77Integer *k_ph_4_1_offset,
                      F77Integer *d_i0, F77Integer *k_i0_offset);

void offset_ccsd_1prdm_ph_3_1_(F77Integer *l_ph_3_1_offset,
                               F77Integer *k_ph_3_1_offset, F77Integer *size_ph_3_1);
void offset_ccsd_1prdm_ph_4_1_(F77Integer *l_ph_4_1_offset,
                               F77Integer *k_ph_4_1_offset, F77Integer *size_ph_4_1);
}

namespace tamm {

void schedule_linear(std::map<std::string, tamm::Tensor> &tensors,
                     std::vector<Operation> &ops);
void schedule_linear_lazy(std::map<std::string, tamm::Tensor> &tensors,
                          std::vector<Operation> &ops);
void schedule_levels(std::map<std::string, tamm::Tensor> &tensors,
                     std::vector<Operation> &ops);

extern "C" {
void ccsd_1prdm_ph_cxx_(F77Integer *d_i0, F77Integer *d_t_vo, F77Integer *d_t_vvoo,
                        F77Integer *d_y_ov, F77Integer *d_y_oovv,
                        F77Integer *k_i0_offset, F77Integer *k_t_vo_offset,
                        F77Integer *k_t_vvoo_offset, F77Integer *k_y_ov_offset,
                        F77Integer *k_y_oovv_offset) {
  static bool set_ph = true;

  Assignment op_ph_1;
  Multiplication op_ph_2;
  Multiplication op_ph_3_1;
  Multiplication op_ph_3_2;
  Multiplication op_ph_3;
  Multiplication op_ph_4_1;
  Multiplication op_ph_4;

  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_ph) {
    ccsd_1prdm_ph_equations(&eqs);
    set_ph = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector<Operation> ops;
  tensors_and_ops(&eqs, &tensors, &ops);

  Tensor *i0 = &tensors["i0"];
  Tensor *t_vo = &tensors["t_vo"];
  Tensor *t_vvoo = &tensors["t_vvoo"];
  Tensor *y_ov = &tensors["y_ov"];
  Tensor *y_oovv = &tensors["y_oovv"];
  Tensor *ph_3_1 = &tensors["ph_3_1"];
  Tensor *ph_4_1 = &tensors["ph_4_1"];

  /* ----- Insert attach code ------ */
  i0->attach(*k_i0_offset, 0, *d_i0);
  t_vo->attach(*k_t_vo_offset, 0, *d_t_vo);
  t_vvoo->attach(*k_t_vvoo_offset, 0, *d_t_vvoo);
  y_ov->attach(*k_y_ov_offset, 0, *d_y_ov);
  y_oovv->attach(*k_y_oovv_offset, 0, *d_y_oovv);

#if 1
  schedule_levels(&tensors, &ops);
#else
  op_ph_1 = ops[0].add;
  op_ph_2 = ops[1].mult;
  op_ph_3_1 = ops[2].mult;
  op_ph_3_2 = ops[3].mult;
  op_ph_3 = ops[4].mult;
  op_ph_4_1 = ops[5].mult;
  op_ph_4 = ops[6].mult;

  CorFortran(1, op_ph_1, ccsd_1prdm_ph_1_);
  CorFortran(1, op_ph_2, ccsd_1prdm_ph_2_);
  CorFortran(1, ph_3_1, offset_ccsd_1prdm_ph_3_1_);
  CorFortran(1, op_ph_3_1, ccsd_1prdm_ph_3_1_);
  CorFortran(1, op_ph_3_2, ccsd_1prdm_ph_3_2_);
  CorFortran(1, op_ph_3, ccsd_1prdm_ph_3_);
  destroy(ph_3_1);
  CorFortran(1, ph_4_1, offset_ccsd_1prdm_ph_4_1_);
  CorFortran(1, op_ph_4_1, ccsd_1prdm_ph_4_1_);
  CorFortran(1, op_ph_4, ccsd_1prdm_ph_4_);
  destroy(ph_4_1);
#endif

  /* ----- Insert detach code ------ */
  i0->detach();
  t_vo->detach();
  t_vvoo->detach();
  y_ov->detach();
  y_oovv->detach();
}
}  // extern C
};  // namespace tamm
