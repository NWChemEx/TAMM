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
 *  pp {
 *  
 *  index h1,h2,h3,h4,h5 = O;
 *  index p1,p2,p3 = V;
 *  
 *  array i0[V][V];
 *  array t_vo[V][O]: irrep_t;
 *  array y_ov[O][V]: irrep_y;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array y_oovv[O,O][V,V]: irrep_y;
 *  
 *  pp_1:       i0[p1,p2] += 1 * t_vo[p1,h3] * y_ov[h3,p2];
 *  pp_2:       i0[p1,p2] += 1/2 * t_vvoo[p1,p3,h4,h5] * y_oovv[h4,h5,p2,p3];
 *  
 *  }
*/


extern "C" {
void ccsd_1prdm_pp_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                      Integer *d_y_ov, Integer *k_y_ov_offset, 
                      Integer *d_i0, Integer *k_i0_offset);
void ccsd_1prdm_pp_2_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset, 
                      Integer *d_y_oovv, Integer *k_y_oovv_offset, 
                      Integer *d_i0, Integer *k_i0_offset);
}

namespace tamm {

void schedule_linear(std::map<std::string, tamm::Tensor> &tensors, 
                     std::vector<Operation> &ops);
void schedule_linear_lazy(std::map<std::string, tamm::Tensor> &tensors, 
                          std::vector<Operation> &ops);
void schedule_levels(std::map<std::string, tamm::Tensor> &tensors, 
                     std::vector<Operation> &ops);

extern "C" {
  // void ccsd_1prdm_pp_cxx_(Integer *d_t_vvoo, Integer *d_i0, Integer *d_y_ov, Integer *d_y_oovv, Integer *d_t_vo, 
  //Integer *k_t_vvoo_offset, Integer *k_i0_offset, Integer *k_y_ov_offset, Integer *k_y_oovv_offset, Integer *k_t_vo_offset) {
void ccsd_1prdm_pp_cxx_(Integer *d_i0, Integer *d_t_vo, Integer *d_t_vvoo, 
                        Integer *d_y_ov, Integer *d_y_oovv, Integer *k_i0_offset, 
                        Integer *k_t_vo_offset, Integer *k_t_vvoo_offset, 
                        Integer *k_y_ov_offset, Integer *k_y_oovv_offset) {

  static bool set_pp = true;
  
  Multiplication op_pp_1;
  Multiplication op_pp_2;
  
  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_pp) {
    ccsd_1prdm_pp_equations(&eqs);
    set_pp = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector <Operation> ops;
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
    op_pp_1 = ops[0].mult;
    op_pp_2 = ops[1].mult;

    CorFortran(1, op_pp_1, ccsd_1prdm_pp_1_);
    CorFortran(1, op_pp_2, ccsd_1prdm_pp_2_);
  #endif

  /* ----- Insert detach code ------ */
  i0->detach();
  t_vo->detach(); 
  t_vvoo->detach(); 
  y_ov->detach(); 
  y_oovv->detach(); 
  }
} // extern C
}; // namespace tamm
