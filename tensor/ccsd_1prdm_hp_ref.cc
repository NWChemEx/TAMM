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
 *  hp {
 *
 *  index h1,h2 = O;
 *  index p1 = V;
 *
 *  array i0[O][V];
 *  array y_ov[O][V]: irrep_y;
 *
 *  hp_1:       i0[h2,p1] += 1 * y_ov[h2,p1];
 *
 *  }
 */

extern "C" {
void ccsd_1prdm_hp_1_(Integer *d_i0, Integer *d_y_ov, Integer *k_y_ov_offset,
                      Integer *k_i0_offset);
}

namespace tamm {

void schedule_linear(std::map<std::string, tamm::Tensor> &tensors,
                     std::vector<Operation> &ops);
void schedule_linear_lazy(std::map<std::string, tamm::Tensor> &tensors,
                          std::vector<Operation> &ops);
void schedule_levels(std::map<std::string, tamm::Tensor> &tensors,
                     std::vector<Operation> &ops);

extern "C" {
void ccsd_1prdm_hp_cxx_(Integer *d_i0, Integer *d_y_ov, Integer *k_i0_offset,
                        Integer *k_y_ov_offset) {
  static bool set_hp = true;

  Assignment op_hp_1;

  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_hp) {
    ccsd_1prdm_hp_equations(&eqs);
    set_hp = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector<Operation> ops;
  tensors_and_ops(&eqs, &tensors, &ops);

  Tensor *i0 = &tensors["i0"];
  Tensor *y_ov = &tensors["y_ov"];

  /* ----- Insert attach code ------ */
  i0->attach(*k_i0_offset, 0, *d_i0);
  y_ov->attach(*k_y_ov_offset, 0, *d_y_ov);

#if 1
  schedule_levels(&tensors, &ops);
#else
  op_hp_1 = ops[0].add;

  CorFortran(1, op_hp_1, ccsd_1prdm_hp_1_);
#endif

  /* ----- Insert detach code ------ */
  i0->detach();
  y_ov->detach();
}
}  // extern C
};  // namespace tamm
