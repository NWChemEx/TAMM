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
 *  hh {
 *
 *  index h1,h2,h3,h4,h5 = O;
 *  index p1,p2,p3,p4 = V;
 *
 *  array i0[O][O];
 *  array t_vo[V][O]: irrep_t;
 *  array y_ov[O][V]: irrep_y;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array y_oovv[O,O][V,V]: irrep_y;
 *
 *  hh_1:       i0[h2,h1] += -1 * t_vo[p3,h1] * y_ov[h2,p3];
 *  hh_2:       i0[h2,h1] += -1/2 * t_vvoo[p3,p4,h1,h5] * y_oovv[h2,h5,p3,p4];
 *
 *  }
 */

extern "C" {
void ccsd_1prdm_hh_1_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_y_ov,
                      F77Integer *k_y_ov_offset, F77Integer *d_i0,
                      F77Integer *k_i0_offset);
void ccsd_1prdm_hh_2_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset,
                      F77Integer *d_y_oovv, F77Integer *k_y_oovv_offset,
                      F77Integer *d_i0, F77Integer *k_i0_offset);
}

namespace tamm {

void schedule_linear(std::map<std::string, tamm::Tensor> &tensors,
                     std::vector<Operation> &ops);
void schedule_linear_lazy(std::map<std::string, tamm::Tensor> &tensors,
                          std::vector<Operation> &ops);
void schedule_levels(std::map<std::string, tamm::Tensor> &tensors,
                     std::vector<Operation> &ops);

extern "C" {
void ccsd_1prdm_hh_cxx_(F77Integer *d_i0, F77Integer *d_t_vo, F77Integer *d_t_vvoo,
                        F77Integer *d_y_ov, F77Integer *d_y_oovv,
                        F77Integer *k_i0_offset, F77Integer *k_t_vo_offset,
                        F77Integer *k_t_vvoo_offset, F77Integer *k_y_ov_offset,
                        F77Integer *k_y_oovv_offset) {
  static bool set_hh = true;

  Multiplication op_hh_1;
  Multiplication op_hh_2;

  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_hh) {
    ccsd_1prdm_hh_equations(&eqs);
    set_hh = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector<Operation> ops;
  tensors_and_ops(&eqs, &tensors, &ops);

  Tensor *i0 = &tensors["i0"];
  Tensor *t_vo = &tensors["t_vo"];
  Tensor *y_ov = &tensors["y_ov"];
  Tensor *t_vvoo = &tensors["t_vvoo"];
  Tensor *y_oovv = &tensors["y_oovv"];

  /* ----- Insert attach code ------ */
  i0->attach(*k_i0_offset, 0, *d_i0);
  t_vo->attach(*k_t_vo_offset, 0, *d_t_vo);
  t_vvoo->attach(*k_t_vvoo_offset, 0, *d_t_vvoo);
  y_ov->attach(*k_y_ov_offset, 0, *d_y_ov);
  y_oovv->attach(*k_y_oovv_offset, 0, *d_y_oovv);

#if 1
  schedule_levels(&tensors, &ops);
#else
  op_hh_1 = ops[0].mult;
  op_hh_2 = ops[1].mult;

  CorFortran(1, op_hh_1, ccsd_1prdm_hh_1_);
  CorFortran(1, op_hh_2, ccsd_1prdm_hh_2_);
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
