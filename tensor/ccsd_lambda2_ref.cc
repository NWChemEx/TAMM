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
 *  lambda2 {
 *
 *  index h1,h2,h3,h4,h5,h6,h7,h8,h9,h10 = O;
 *  index p1,p2,p3,p4,p5,p6,p7,p8,p9,p10 = V;
 *
 *  array i0[O,O][V,V];
 *  array v[N,N][N,N]: irrep_v;
 *  array y_ov[O][V]: irrep_y;
 *  array f[N][N]: irrep_f;
 *  array t_vo[V][O]: irrep_t;
 *  array y_oovv[O,O][V,V]: irrep_y;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array lambda2_7_1[O,O][O,O];
 *  array lambda2_8_1[O,V][O,V];
 *  array lambda2_2_1[O][V];
 *  array lambda2_15_1[O,O][O,O];
 *  array lambda2_11_1[O,O][O,V];
 *  array lambda2_15_2_1[O,O][O,V];
 *  array lambda2_3_1[O,O][O,V];
 *  array lambda2_10_1[O][O];
 *  array lambda2_14_1[V][V];
 *  array lambda2_16_1[O,V][O,V];
 *  array lambda2_6_4_1[O][V];
 *  array lambda2_5_2_1[O][V];
 *  array lambda2_13_1[O,O][O,V];
 *  array lambda2_5_1[O][O];
 *  array lambda2_7_2_1[O,O][O,V];
 *  array lambda2_6_1[V][V];
 *  array lambda2_12_1[O,O][O,V];
 *  array lambda2_16_1_1[O,O][O,V];
 *
 *  lambda2_1:       i0[h3,h4,p1,p2] += 1 * v[h3,h4,p1,p2];
 *  lambda2_2_1:     lambda2_2_1[h3,p1] += 1 * f[h3,p1];
 *  lambda2_2_2:     lambda2_2_1[h3,p1] += 1 * t_vo[p5,h6] * v[h3,h6,p1,p5];
 *  lambda2_2:       i0[h3,h4,p1,p2] += 1 * y_ov[h3,p1] * lambda2_2_1[h4,p2];
 *  lambda2_3_1:     lambda2_3_1[h3,h4,h7,p1] += 1 * v[h3,h4,h7,p1];
 *  lambda2_3_2:     lambda2_3_1[h3,h4,h7,p1] += -1 * t_vo[p5,h7] *
 * v[h3,h4,p1,p5];
 *  lambda2_3:       i0[h3,h4,p1,p2] += -1 * y_ov[h7,p1] *
 * lambda2_3_1[h3,h4,h7,p2];
 *  lambda2_4:       i0[h3,h4,p1,p2] += -1 * y_ov[h3,p5] * v[h4,p5,p1,p2];
 *  lambda2_5_1:     lambda2_5_1[h3,h9] += 1 * f[h3,h9];
 *  lambda2_5_2_1:   lambda2_5_2_1[h3,p5] += 1 * f[h3,p5];
 *  lambda2_5_2_2:   lambda2_5_2_1[h3,p5] += 1 * t_vo[p7,h8] * v[h3,h8,p5,p7];
 *  lambda2_5_2:     lambda2_5_1[h3,h9] += 1 * t_vo[p5,h9] *
 * lambda2_5_2_1[h3,p5];
 *  lambda2_5_3:     lambda2_5_1[h3,h9] += 1 * t_vo[p5,h6] * v[h3,h6,h9,p5];
 *  lambda2_5_4:     lambda2_5_1[h3,h9] += -1/2 * t_vvoo[p5,p6,h8,h9] *
 * v[h3,h8,p5,p6];
 *  lambda2_5:       i0[h3,h4,p1,p2] += -1 * y_oovv[h3,h9,p1,p2] *
 * lambda2_5_1[h4,h9];
 *  lambda2_6_1:     lambda2_6_1[p10,p1] += 1 * f[p10,p1];
 *  lambda2_6_2:     lambda2_6_1[p10,p1] += -1 * t_vo[p5,h6] * v[h6,p10,p1,p5];
 *  lambda2_6_3:     lambda2_6_1[p10,p1] += 1/2 * t_vvoo[p6,p10,h7,h8] *
 * v[h7,h8,p1,p6];
 *  lambda2_6_4_1:   lambda2_6_4_1[h6,p1] += 1 * t_vo[p7,h8] * v[h6,h8,p1,p7];
 *  lambda2_6_4:     lambda2_6_1[p10,p1] += -1 * t_vo[p10,h6] *
 * lambda2_6_4_1[h6,p1];
 *  lambda2_6:       i0[h3,h4,p1,p2] += 1 * y_oovv[h3,h4,p1,p10] *
 * lambda2_6_1[p10,p2];
 *  lambda2_7_1:     lambda2_7_1[h3,h4,h9,h10] += 1 * v[h3,h4,h9,h10];
 *  lambda2_7_2_1:   lambda2_7_2_1[h3,h4,h10,p5] += 1 * v[h3,h4,h10,p5];
 *  lambda2_7_2_2:   lambda2_7_2_1[h3,h4,h10,p5] += -1/2 * t_vo[p7,h10] *
 * v[h3,h4,p5,p7];
 *  lambda2_7_2:     lambda2_7_1[h3,h4,h9,h10] += -2 * t_vo[p5,h9] *
 * lambda2_7_2_1[h3,h4,h10,p5];
 *  lambda2_7_3:     lambda2_7_1[h3,h4,h9,h10] += 1/2 * t_vvoo[p5,p6,h9,h10] *
 * v[h3,h4,p5,p6];
 *  lambda2_7:       i0[h3,h4,p1,p2] += 1/2 * y_oovv[h9,h10,p1,p2] *
 * lambda2_7_1[h3,h4,h9,h10];
 *  lambda2_8_1:     lambda2_8_1[h3,p7,h9,p1] += 1 * v[h3,p7,h9,p1];
 *  lambda2_8_2:     lambda2_8_1[h3,p7,h9,p1] += -1 * t_vo[p5,h9] *
 * v[h3,p7,p1,p5];
 *  lambda2_8_3:     lambda2_8_1[h3,p7,h9,p1] += -1 * t_vvoo[p6,p7,h8,h9] *
 * v[h3,h8,p1,p6];
 *  lambda2_8:       i0[h3,h4,p1,p2] += -1 * y_oovv[h3,h9,p1,p7] *
 * lambda2_8_1[h4,p7,h9,p2];
 *  lambda2_9:       i0[h3,h4,p1,p2] += 1/2 * y_oovv[h3,h4,p5,p6] *
 * v[p5,p6,p1,p2];
 *  lambda2_10_1:    lambda2_10_1[h3,h9] += 1 * t_vo[p5,h9] * y_ov[h3,p5];
 *  lambda2_10_2:    lambda2_10_1[h3,h9] += -1/2 * t_vvoo[p5,p6,h7,h9] *
 * y_oovv[h3,h7,p5,p6];
 *  lambda2_10:      i0[h3,h4,p1,p2] += 1 * lambda2_10_1[h3,h9] *
 * v[h4,h9,p1,p2];
 *  lambda2_11_1:    lambda2_11_1[h3,h4,h5,p1] += -1 * t_vo[p6,h5] *
 * y_oovv[h3,h4,p1,p6];
 *  lambda2_11:      i0[h3,h4,p1,p2] += 1 * lambda2_11_1[h3,h4,h5,p1] *
 * f[h5,p2];
 *  lambda2_12_1:    lambda2_12_1[h3,h7,h6,p1] += 1 * t_vo[p5,h6] *
 * y_oovv[h3,h7,p1,p5];
 *  lambda2_12:      i0[h3,h4,p1,p2] += 1 * lambda2_12_1[h3,h7,h6,p1] *
 * v[h4,h6,h7,p2];
 *  lambda2_13_1:    lambda2_13_1[h3,h4,h6,p7] += -1 * t_vo[p5,h6] *
 * y_oovv[h3,h4,p5,p7];
 *  lambda2_13:      i0[h3,h4,p1,p2] += 1 * lambda2_13_1[h3,h4,h6,p7] *
 * v[h6,p7,p1,p2];
 *  lambda2_14_1:    lambda2_14_1[p6,p1] += 1 * t_vvoo[p5,p6,h7,h8] *
 * y_oovv[h7,h8,p1,p5];
 *  lambda2_14:      i0[h3,h4,p1,p2] += -1/2 * lambda2_14_1[p6,p1] *
 * v[h3,h4,p2,p6];
 *  lambda2_15_1:    lambda2_15_1[h3,h4,h8,h9] += 1 * t_vvoo[p5,p6,h8,h9] *
 * y_oovv[h3,h4,p5,p6];
 *  lambda2_15_2_1:  lambda2_15_2_1[h3,h4,h8,p5] += -1 * t_vo[p7,h8] *
 * y_oovv[h3,h4,p5,p7];
 *  lambda2_15_2:    lambda2_15_1[h3,h4,h8,h9] += 2 * t_vo[p5,h9] *
 * lambda2_15_2_1[h3,h4,h8,p5];
 *  lambda2_15:      i0[h3,h4,p1,p2] += 1/4 * lambda2_15_1[h3,h4,h8,h9] *
 * v[h8,h9,p1,p2];
 *  lambda2_16_1_1:  lambda2_16_1_1[h3,h6,h8,p1] += 1 * t_vo[p7,h8] *
 * y_oovv[h3,h6,p1,p7];
 *  lambda2_16_1:    lambda2_16_1[h3,p5,h8,p1] += 1 * t_vo[p5,h6] *
 * lambda2_16_1_1[h3,h6,h8,p1];
 *  lambda2_16:      i0[h3,h4,p1,p2] += -1 * lambda2_16_1[h3,p5,h8,p1] *
 * v[h4,h8,p2,p5];
 *
 *  }
*/

extern "C" {
void ccsd_lambda2_1_(Integer *d_v, Integer *k_v_offset, Integer *d_i0,
                     Integer *k_i0_offset);
void ccsd_lambda2_2_1_(Integer *d_f, Integer *k_f_offset,
                       Integer *d_lambda2_2_1, Integer *k_lambda2_2_1_offset);
void ccsd_lambda2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                       Integer *k_v_offset, Integer *d_lambda2_2_1,
                       Integer *k_lambda2_2_1_offset);
void ccsd_lambda2_2_(Integer *d_y_ov, Integer *k_y_ov_offset,
                     Integer *d_lambda2_2_1, Integer *k_lambda2_2_1_offset,
                     Integer *d_i0, Integer *k_i0_offset);
void ccsd_lambda2_3_1_(Integer *d_v, Integer *k_v_offset,
                       Integer *d_lambda2_3_1, Integer *k_lambda2_3_1_offset);
void ccsd_lambda2_3_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                       Integer *k_v_offset, Integer *d_lambda2_3_1,
                       Integer *k_lambda2_3_1_offset);
void ccsd_lambda2_3_(Integer *d_y_ov, Integer *k_y_ov_offset,
                     Integer *d_lambda2_3_1, Integer *k_lambda2_3_1_offset,
                     Integer *d_i0, Integer *k_i0_offset);
void ccsd_lambda2_4_(Integer *d_y_ov, Integer *k_y_ov_offset, Integer *d_v,
                     Integer *k_v_offset, Integer *d_i0, Integer *k_i0_offset);
void ccsd_lambda2_5_1_(Integer *d_f, Integer *k_f_offset,
                       Integer *d_lambda2_5_1, Integer *k_lambda2_5_1_offset);
void ccsd_lambda2_5_2_1_(Integer *d_f, Integer *k_f_offset,
                         Integer *d_lambda2_5_2_1,
                         Integer *k_lambda2_5_2_1_offset);
void ccsd_lambda2_5_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_lambda2_5_2_1,
                         Integer *k_lambda2_5_2_1_offset);
void ccsd_lambda2_5_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                       Integer *d_lambda2_5_2_1,
                       Integer *k_lambda2_5_2_1_offset, Integer *d_lambda2_5_1,
                       Integer *k_lambda2_5_1_offset);
void ccsd_lambda2_5_3_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                       Integer *k_v_offset, Integer *d_lambda2_5_1,
                       Integer *k_lambda2_5_1_offset);
void ccsd_lambda2_5_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                       Integer *d_v, Integer *k_v_offset,
                       Integer *d_lambda2_5_1, Integer *k_lambda2_5_1_offset);
void ccsd_lambda2_5_(Integer *d_y_oovv, Integer *k_y_oovv_offset,
                     Integer *d_lambda2_5_1, Integer *k_lambda2_5_1_offset,
                     Integer *d_i0, Integer *k_i0_offset);
void ccsd_lambda2_6_1_(Integer *d_f, Integer *k_f_offset,
                       Integer *d_lambda2_6_1, Integer *k_lambda2_6_1_offset);
void ccsd_lambda2_6_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                       Integer *k_v_offset, Integer *d_lambda2_6_1,
                       Integer *k_lambda2_6_1_offset);
void ccsd_lambda2_6_3_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                       Integer *d_v, Integer *k_v_offset,
                       Integer *d_lambda2_6_1, Integer *k_lambda2_6_1_offset);
void ccsd_lambda2_6_4_1_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_lambda2_6_4_1,
                         Integer *k_lambda2_6_4_1_offset);
void ccsd_lambda2_6_4_(Integer *d_t_vo, Integer *k_t_vo_offset,
                       Integer *d_lambda2_6_4_1,
                       Integer *k_lambda2_6_4_1_offset, Integer *d_lambda2_6_1,
                       Integer *k_lambda2_6_1_offset);
void ccsd_lambda2_6_(Integer *d_y_oovv, Integer *k_y_oovv_offset,
                     Integer *d_lambda2_6_1, Integer *k_lambda2_6_1_offset,
                     Integer *d_i0, Integer *k_i0_offset);
void ccsd_lambda2_7_1_(Integer *d_v, Integer *k_v_offset,
                       Integer *d_lambda2_7_1, Integer *k_lambda2_7_1_offset);
void ccsd_lambda2_7_2_1_(Integer *d_v, Integer *k_v_offset,
                         Integer *d_lambda2_7_2_1,
                         Integer *k_lambda2_7_2_1_offset);
void ccsd_lambda2_7_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_lambda2_7_2_1,
                         Integer *k_lambda2_7_2_1_offset);
void ccsd_lambda2_7_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                       Integer *d_lambda2_7_2_1,
                       Integer *k_lambda2_7_2_1_offset, Integer *d_lambda2_7_1,
                       Integer *k_lambda2_7_1_offset);
void ccsd_lambda2_7_3_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                       Integer *d_v, Integer *k_v_offset,
                       Integer *d_lambda2_7_1, Integer *k_lambda2_7_1_offset);
void ccsd_lambda2_7_(Integer *d_y_oovv, Integer *k_y_oovv_offset,
                     Integer *d_lambda2_7_1, Integer *k_lambda2_7_1_offset,
                     Integer *d_i0, Integer *k_i0_offset);
void ccsd_lambda2_8_1_(Integer *d_v, Integer *k_v_offset,
                       Integer *d_lambda2_8_1, Integer *k_lambda2_8_1_offset);
void ccsd_lambda2_8_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                       Integer *k_v_offset, Integer *d_lambda2_8_1,
                       Integer *k_lambda2_8_1_offset);
void ccsd_lambda2_8_3_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                       Integer *d_v, Integer *k_v_offset,
                       Integer *d_lambda2_8_1, Integer *k_lambda2_8_1_offset);
void ccsd_lambda2_8_(Integer *d_y_oovv, Integer *k_y_oovv_offset,
                     Integer *d_lambda2_8_1, Integer *k_lambda2_8_1_offset,
                     Integer *d_i0, Integer *k_i0_offset);
void ccsd_lambda2_9_(Integer *d_y_oovv, Integer *k_y_oovv_offset, Integer *d_v,
                     Integer *k_v_offset, Integer *d_i0, Integer *k_i0_offset);
void ccsd_lambda2_10_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                        Integer *d_y_ov, Integer *k_y_ov_offset,
                        Integer *d_lambda2_10_1,
                        Integer *k_lambda2_10_1_offset);
void ccsd_lambda2_10_2_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                        Integer *d_y_oovv, Integer *k_y_oovv_offset,
                        Integer *d_lambda2_10_1,
                        Integer *k_lambda2_10_1_offset);
void ccsd_lambda2_10_(Integer *d_lambda2_10_1, Integer *k_lambda2_10_1_offset,
                      Integer *d_v, Integer *k_v_offset, Integer *d_i0,
                      Integer *k_i0_offset);
void ccsd_lambda2_11_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                        Integer *d_y_oovv, Integer *k_y_oovv_offset,
                        Integer *d_lambda2_11_1,
                        Integer *k_lambda2_11_1_offset);
void ccsd_lambda2_11_(Integer *d_lambda2_11_1, Integer *k_lambda2_11_1_offset,
                      Integer *d_f, Integer *k_f_offset, Integer *d_i0,
                      Integer *k_i0_offset);
void ccsd_lambda2_12_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                        Integer *d_y_oovv, Integer *k_y_oovv_offset,
                        Integer *d_lambda2_12_1,
                        Integer *k_lambda2_12_1_offset);
void ccsd_lambda2_12_(Integer *d_lambda2_12_1, Integer *k_lambda2_12_1_offset,
                      Integer *d_v, Integer *k_v_offset, Integer *d_i0,
                      Integer *k_i0_offset);
void ccsd_lambda2_13_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                        Integer *d_y_oovv, Integer *k_y_oovv_offset,
                        Integer *d_lambda2_13_1,
                        Integer *k_lambda2_13_1_offset);
void ccsd_lambda2_13_(Integer *d_lambda2_13_1, Integer *k_lambda2_13_1_offset,
                      Integer *d_v, Integer *k_v_offset, Integer *d_i0,
                      Integer *k_i0_offset);
void ccsd_lambda2_14_1_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                        Integer *d_y_oovv, Integer *k_y_oovv_offset,
                        Integer *d_lambda2_14_1,
                        Integer *k_lambda2_14_1_offset);
void ccsd_lambda2_14_(Integer *d_lambda2_14_1, Integer *k_lambda2_14_1_offset,
                      Integer *d_v, Integer *k_v_offset, Integer *d_i0,
                      Integer *k_i0_offset);
void ccsd_lambda2_15_1_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                        Integer *d_y_oovv, Integer *k_y_oovv_offset,
                        Integer *d_lambda2_15_1,
                        Integer *k_lambda2_15_1_offset);
void ccsd_lambda2_15_2_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                          Integer *d_y_oovv, Integer *k_y_oovv_offset,
                          Integer *d_lambda2_15_2_1,
                          Integer *k_lambda2_15_2_1_offset);
void ccsd_lambda2_15_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                        Integer *d_lambda2_15_2_1,
                        Integer *k_lambda2_15_2_1_offset,
                        Integer *d_lambda2_15_1,
                        Integer *k_lambda2_15_1_offset);
void ccsd_lambda2_15_(Integer *d_lambda2_15_1, Integer *k_lambda2_15_1_offset,
                      Integer *d_v, Integer *k_v_offset, Integer *d_i0,
                      Integer *k_i0_offset);
void ccsd_lambda2_16_1_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                          Integer *d_y_oovv, Integer *k_y_oovv_offset,
                          Integer *d_lambda2_16_1_1,
                          Integer *k_lambda2_16_1_1_offset);
void ccsd_lambda2_16_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                        Integer *d_lambda2_16_1_1,
                        Integer *k_lambda2_16_1_1_offset,
                        Integer *d_lambda2_16_1,
                        Integer *k_lambda2_16_1_offset);
void ccsd_lambda2_16_(Integer *d_lambda2_16_1, Integer *k_lambda2_16_1_offset,
                      Integer *d_v, Integer *k_v_offset, Integer *d_i0,
                      Integer *k_i0_offset);

void offset_ccsd_lambda2_2_1_(Integer *l_lambda2_2_1_offset,
                              Integer *k_lambda2_2_1_offset,
                              Integer *size_lambda2_2_1);
void offset_ccsd_lambda2_3_1_(Integer *l_lambda2_3_1_offset,
                              Integer *k_lambda2_3_1_offset,
                              Integer *size_lambda2_3_1);
void offset_ccsd_lambda2_5_1_(Integer *l_lambda2_5_1_offset,
                              Integer *k_lambda2_5_1_offset,
                              Integer *size_lambda2_5_1);
void offset_ccsd_lambda2_5_2_1_(Integer *l_lambda2_5_2_1_offset,
                                Integer *k_lambda2_5_2_1_offset,
                                Integer *size_lambda2_5_2_1);
void offset_ccsd_lambda2_6_1_(Integer *l_lambda2_6_1_offset,
                              Integer *k_lambda2_6_1_offset,
                              Integer *size_lambda2_6_1);
void offset_ccsd_lambda2_6_4_1_(Integer *l_lambda2_6_4_1_offset,
                                Integer *k_lambda2_6_4_1_offset,
                                Integer *size_lambda2_6_4_1);
void offset_ccsd_lambda2_7_1_(Integer *l_lambda2_7_1_offset,
                              Integer *k_lambda2_7_1_offset,
                              Integer *size_lambda2_7_1);
void offset_ccsd_lambda2_7_2_1_(Integer *l_lambda2_7_2_1_offset,
                                Integer *k_lambda2_7_2_1_offset,
                                Integer *size_lambda2_7_2_1);
void offset_ccsd_lambda2_8_1_(Integer *l_lambda2_8_1_offset,
                              Integer *k_lambda2_8_1_offset,
                              Integer *size_lambda2_8_1);
void offset_ccsd_lambda2_10_1_(Integer *l_lambda2_10_1_offset,
                               Integer *k_lambda2_10_1_offset,
                               Integer *size_lambda2_10_1);
void offset_ccsd_lambda2_11_1_(Integer *l_lambda2_11_1_offset,
                               Integer *k_lambda2_11_1_offset,
                               Integer *size_lambda2_11_1);
void offset_ccsd_lambda2_12_1_(Integer *l_lambda2_12_1_offset,
                               Integer *k_lambda2_12_1_offset,
                               Integer *size_lambda2_12_1);
void offset_ccsd_lambda2_13_1_(Integer *l_lambda2_13_1_offset,
                               Integer *k_lambda2_13_1_offset,
                               Integer *size_lambda2_13_1);
void offset_ccsd_lambda2_14_1_(Integer *l_lambda2_14_1_offset,
                               Integer *k_lambda2_14_1_offset,
                               Integer *size_lambda2_14_1);
void offset_ccsd_lambda2_15_1_(Integer *l_lambda2_15_1_offset,
                               Integer *k_lambda2_15_1_offset,
                               Integer *size_lambda2_15_1);
void offset_ccsd_lambda2_15_2_1_(Integer *l_lambda2_15_2_1_offset,
                                 Integer *k_lambda2_15_2_1_offset,
                                 Integer *size_lambda2_15_2_1);
void offset_ccsd_lambda2_16_1_1_(Integer *l_lambda2_16_1_1_offset,
                                 Integer *k_lambda2_16_1_1_offset,
                                 Integer *size_lambda2_16_1_1);
void offset_ccsd_lambda2_16_1_(Integer *l_lambda2_16_1_offset,
                               Integer *k_lambda2_16_1_offset,
                               Integer *size_lambda2_16_1);
}

namespace tamm {

extern "C" {
void ccsd_lambda2_cxx_(Integer *d_f, Integer *d_i0, Integer *d_t_vo,
                       Integer *d_t_vvoo, Integer *d_v, Integer *d_y_ov,
                       Integer *d_y_oovv, Integer *k_f_offset,
                       Integer *k_i0_offset, Integer *k_t_vo_offset,
                       Integer *k_t_vvoo_offset, Integer *k_v_offset,
                       Integer *k_y_ov_offset, Integer *k_y_oovv_offset) {
  static bool set_lambda2 = true;

  Assignment op_lambda2_1;
  Assignment op_lambda2_2_1;
  Assignment op_lambda2_3_1;
  Assignment op_lambda2_5_1;
  Assignment op_lambda2_5_2_1;
  Assignment op_lambda2_6_1;
  Assignment op_lambda2_7_1;
  Assignment op_lambda2_7_2_1;
  Assignment op_lambda2_8_1;
  Multiplication op_lambda2_2_2;
  Multiplication op_lambda2_2;
  Multiplication op_lambda2_3_2;
  Multiplication op_lambda2_3;
  Multiplication op_lambda2_4;
  Multiplication op_lambda2_5_2_2;
  Multiplication op_lambda2_5_2;
  Multiplication op_lambda2_5_3;
  Multiplication op_lambda2_5_4;
  Multiplication op_lambda2_5;
  Multiplication op_lambda2_6_2;
  Multiplication op_lambda2_6_3;
  Multiplication op_lambda2_6_4_1;
  Multiplication op_lambda2_6_4;
  Multiplication op_lambda2_6;
  Multiplication op_lambda2_7_2_2;
  Multiplication op_lambda2_7_2;
  Multiplication op_lambda2_7_3;
  Multiplication op_lambda2_7;
  Multiplication op_lambda2_8_2;
  Multiplication op_lambda2_8_3;
  Multiplication op_lambda2_8;
  Multiplication op_lambda2_9;
  Multiplication op_lambda2_10_1;
  Multiplication op_lambda2_10_2;
  Multiplication op_lambda2_10;
  Multiplication op_lambda2_11_1;
  Multiplication op_lambda2_11;
  Multiplication op_lambda2_12_1;
  Multiplication op_lambda2_12;
  Multiplication op_lambda2_13_1;
  Multiplication op_lambda2_13;
  Multiplication op_lambda2_14_1;
  Multiplication op_lambda2_14;
  Multiplication op_lambda2_15_1;
  Multiplication op_lambda2_15_2_1;
  Multiplication op_lambda2_15_2;
  Multiplication op_lambda2_15;
  Multiplication op_lambda2_16_1_1;
  Multiplication op_lambda2_16_1;
  Multiplication op_lambda2_16;

  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_lambda2) {
    ccsd_lambda2_equations(&eqs);
    set_lambda2 = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector<Operation> ops;
  tensors_and_ops(&eqs, &tensors, &ops);

  Tensor *i0 = &tensors["i0"];
  Tensor *v = &tensors["v"];
  Tensor *y_ov = &tensors["y_ov"];
  Tensor *f = &tensors["f"];
  Tensor *t_vo = &tensors["t_vo"];
  Tensor *y_oovv = &tensors["y_oovv"];
  Tensor *t_vvoo = &tensors["t_vvoo"];
  Tensor *lambda2_7_1 = &tensors["lambda2_7_1"];
  Tensor *lambda2_8_1 = &tensors["lambda2_8_1"];
  Tensor *lambda2_2_1 = &tensors["lambda2_2_1"];
  Tensor *lambda2_15_1 = &tensors["lambda2_15_1"];
  Tensor *lambda2_11_1 = &tensors["lambda2_11_1"];
  Tensor *lambda2_15_2_1 = &tensors["lambda2_15_2_1"];
  Tensor *lambda2_3_1 = &tensors["lambda2_3_1"];
  Tensor *lambda2_10_1 = &tensors["lambda2_10_1"];
  Tensor *lambda2_14_1 = &tensors["lambda2_14_1"];
  Tensor *lambda2_16_1 = &tensors["lambda2_16_1"];
  Tensor *lambda2_6_4_1 = &tensors["lambda2_6_4_1"];
  Tensor *lambda2_5_2_1 = &tensors["lambda2_5_2_1"];
  Tensor *lambda2_13_1 = &tensors["lambda2_13_1"];
  Tensor *lambda2_5_1 = &tensors["lambda2_5_1"];
  Tensor *lambda2_7_2_1 = &tensors["lambda2_7_2_1"];
  Tensor *lambda2_6_1 = &tensors["lambda2_6_1"];
  Tensor *lambda2_12_1 = &tensors["lambda2_12_1"];
  Tensor *lambda2_16_1_1 = &tensors["lambda2_16_1_1"];

  /* ----- Insert attach code ------ */
  v->set_dist(idist);
  i0->attach(*k_i0_offset, 0, *d_i0);
  f->attach(*k_f_offset, 0, *d_f);
  v->attach(*k_v_offset, 0, *d_v);

  t_vo->attach(*k_t_vo_offset, 0, *d_t_vo);
  t_vvoo->attach(*k_t_vvoo_offset, 0, *d_t_vvoo);
  y_ov->attach(*k_y_ov_offset, 0, *d_y_ov);
  y_oovv->attach(*k_y_oovv_offset, 0, *d_y_oovv);
  y_ov->set_irrep(Variables::irrep_y());
  y_oovv->set_irrep(Variables::irrep_y());

#if 1
  schedule_linear(&tensors, &ops);
  // schedule_linear_lazy(&tensors, &ops);
  //  schedule_levels(&tensors, &ops);
#else
  op_lambda2_1 = ops[0].add;
  op_lambda2_2_1 = ops[1].add;
  op_lambda2_2_2 = ops[2].mult;
  op_lambda2_2 = ops[3].mult;
  op_lambda2_3_1 = ops[4].add;
  op_lambda2_3_2 = ops[5].mult;
  op_lambda2_3 = ops[6].mult;
  op_lambda2_4 = ops[7].mult;
  op_lambda2_5_1 = ops[8].add;
  op_lambda2_5_2_1 = ops[9].add;
  op_lambda2_5_2_2 = ops[10].mult;
  op_lambda2_5_2 = ops[11].mult;
  op_lambda2_5_3 = ops[12].mult;
  op_lambda2_5_4 = ops[13].mult;
  op_lambda2_5 = ops[14].mult;
  op_lambda2_6_1 = ops[15].add;
  op_lambda2_6_2 = ops[16].mult;
  op_lambda2_6_3 = ops[17].mult;
  op_lambda2_6_4_1 = ops[18].mult;
  op_lambda2_6_4 = ops[19].mult;
  op_lambda2_6 = ops[20].mult;
  op_lambda2_7_1 = ops[21].add;
  op_lambda2_7_2_1 = ops[22].add;
  op_lambda2_7_2_2 = ops[23].mult;
  op_lambda2_7_2 = ops[24].mult;
  op_lambda2_7_3 = ops[25].mult;
  op_lambda2_7 = ops[26].mult;
  op_lambda2_8_1 = ops[27].add;
  op_lambda2_8_2 = ops[28].mult;
  op_lambda2_8_3 = ops[29].mult;
  op_lambda2_8 = ops[30].mult;
  op_lambda2_9 = ops[31].mult;
  op_lambda2_10_1 = ops[32].mult;
  op_lambda2_10_2 = ops[33].mult;
  op_lambda2_10 = ops[34].mult;
  op_lambda2_11_1 = ops[35].mult;
  op_lambda2_11 = ops[36].mult;
  op_lambda2_12_1 = ops[37].mult;
  op_lambda2_12 = ops[38].mult;
  op_lambda2_13_1 = ops[39].mult;
  op_lambda2_13 = ops[40].mult;
  op_lambda2_14_1 = ops[41].mult;
  op_lambda2_14 = ops[42].mult;
  op_lambda2_15_1 = ops[43].mult;
  op_lambda2_15_2_1 = ops[44].mult;
  op_lambda2_15_2 = ops[45].mult;
  op_lambda2_15 = ops[46].mult;
  op_lambda2_16_1_1 = ops[47].mult;
  op_lambda2_16_1 = ops[48].mult;
  op_lambda2_16 = ops[49].mult;

  CorFortran(1, &op_lambda2_1, ccsd_lambda2_1_);
  CorFortran(1, lambda2_2_1, offset_ccsd_lambda2_2_1_);
  CorFortran(1, &op_lambda2_2_1, ccsd_lambda2_2_1_);
  CorFortran(1, &op_lambda2_2_2, ccsd_lambda2_2_2_);
  CorFortran(1, &op_lambda2_2, ccsd_lambda2_2_);
  destroy(lambda2_2_1);
  CorFortran(1, lambda2_3_1, offset_ccsd_lambda2_3_1_);
  CorFortran(1, &op_lambda2_3_1, ccsd_lambda2_3_1_);
  CorFortran(1, &op_lambda2_3_2, ccsd_lambda2_3_2_);
  CorFortran(1, &op_lambda2_3, ccsd_lambda2_3_);
  destroy(lambda2_3_1);
  CorFortran(1, &op_lambda2_4, ccsd_lambda2_4_);
  CorFortran(1, lambda2_5_1, offset_ccsd_lambda2_5_1_);
  CorFortran(1, &op_lambda2_5_1, ccsd_lambda2_5_1_);
  CorFortran(1, lambda2_5_2_1, offset_ccsd_lambda2_5_2_1_);
  CorFortran(1, &op_lambda2_5_2_1, ccsd_lambda2_5_2_1_);
  CorFortran(1, &op_lambda2_5_2_2, ccsd_lambda2_5_2_2_);
  CorFortran(1, &op_lambda2_5_2, ccsd_lambda2_5_2_);
  destroy(lambda2_5_2_1);
  CorFortran(1, &op_lambda2_5_3, ccsd_lambda2_5_3_);
  CorFortran(1, &op_lambda2_5_4, ccsd_lambda2_5_4_);
  CorFortran(1, &op_lambda2_5, ccsd_lambda2_5_);
  destroy(lambda2_5_1);
  CorFortran(1, lambda2_6_1, offset_ccsd_lambda2_6_1_);
  CorFortran(1, &op_lambda2_6_1, ccsd_lambda2_6_1_);
  CorFortran(1, &op_lambda2_6_2, ccsd_lambda2_6_2_);
  CorFortran(1, &op_lambda2_6_3, ccsd_lambda2_6_3_);
  CorFortran(1, lambda2_6_4_1, offset_ccsd_lambda2_6_4_1_);
  CorFortran(1, &op_lambda2_6_4_1, ccsd_lambda2_6_4_1_);
  CorFortran(1, &op_lambda2_6_4, ccsd_lambda2_6_4_);
  destroy(lambda2_6_4_1);
  CorFortran(1, &op_lambda2_6, ccsd_lambda2_6_);
  destroy(lambda2_6_1);
  CorFortran(1, lambda2_7_1, offset_ccsd_lambda2_7_1_);
  CorFortran(1, &op_lambda2_7_1, ccsd_lambda2_7_1_);
  CorFortran(1, lambda2_7_2_1, offset_ccsd_lambda2_7_2_1_);
  CorFortran(1, &op_lambda2_7_2_1, ccsd_lambda2_7_2_1_);
  CorFortran(1, &op_lambda2_7_2_2, ccsd_lambda2_7_2_2_);
  CorFortran(1, &op_lambda2_7_2, ccsd_lambda2_7_2_); // solved  by replacing the multiplier from -2 to -1
  destroy(lambda2_7_2_1);
  CorFortran(1, &op_lambda2_7_3, ccsd_lambda2_7_3_);
  CorFortran(1, &op_lambda2_7, ccsd_lambda2_7_);
  destroy(lambda2_7_1);
  CorFortran(1, lambda2_8_1, offset_ccsd_lambda2_8_1_);
  CorFortran(1, &op_lambda2_8_1, ccsd_lambda2_8_1_);
  CorFortran(1, &op_lambda2_8_2, ccsd_lambda2_8_2_);
  CorFortran(1, &op_lambda2_8_3, ccsd_lambda2_8_3_);
  CorFortran(1, &op_lambda2_8, ccsd_lambda2_8_);
  destroy(lambda2_8_1);
  CorFortran(1, &op_lambda2_9, ccsd_lambda2_9_);
  CorFortran(1, lambda2_10_1, offset_ccsd_lambda2_10_1_);
  CorFortran(1, &op_lambda2_10_1, ccsd_lambda2_10_1_);
  CorFortran(1, &op_lambda2_10_2, ccsd_lambda2_10_2_);
  CorFortran(1, &op_lambda2_10, ccsd_lambda2_10_);
  destroy(lambda2_10_1);
#if 1  // following block work entirely in fortran or c++
  CorFortran(1, lambda2_11_1, offset_ccsd_lambda2_11_1_);
  CorFortran(1, &op_lambda2_11_1, ccsd_lambda2_11_1_);
  CorFortran(1, &op_lambda2_11, ccsd_lambda2_11_);
#else
  CorFortran(0, lambda2_11_1, offset_ccsd_lambda2_11_1_);
  CorFortran(0, &op_lambda2_11_1, ccsd_lambda2_11_1_);
  CorFortran(0, &op_lambda2_11, ccsd_lambda2_11_);
#endif  // code selection if 1 or 0 etc
  destroy(lambda2_11_1);
#if 1  // following block work entirely in fortran or c++, ok after bug fix
  CorFortran(1, lambda2_12_1, offset_ccsd_lambda2_12_1_);
  CorFortran(1, &op_lambda2_12_1, ccsd_lambda2_12_1_);
  CorFortran(1, &op_lambda2_12, ccsd_lambda2_12_);
#else
  CorFortran(0, lambda2_12_1, offset_ccsd_lambda2_12_1_);
  CorFortran(0, &op_lambda2_12_1, ccsd_lambda2_12_1_);
  CorFortran(0, &op_lambda2_12, ccsd_lambda2_12_);
#endif  // code selection if 1 or 0 etc
  destroy(lambda2_12_1);
  CorFortran(1, lambda2_13_1, offset_ccsd_lambda2_13_1_);
  CorFortran(1, &op_lambda2_13_1, ccsd_lambda2_13_1_);
  CorFortran(1, &op_lambda2_13, ccsd_lambda2_13_);
  destroy(lambda2_13_1);
  CorFortran(1, lambda2_14_1, offset_ccsd_lambda2_14_1_);
  CorFortran(1, &op_lambda2_14_1, ccsd_lambda2_14_1_);
  CorFortran(1, &op_lambda2_14, ccsd_lambda2_14_);
  destroy(lambda2_14_1);
  CorFortran(1, lambda2_15_1, offset_ccsd_lambda2_15_1_);
  CorFortran(1, &op_lambda2_15_1, ccsd_lambda2_15_1_);
  CorFortran(1, lambda2_15_2_1, offset_ccsd_lambda2_15_2_1_);
  CorFortran(1, &op_lambda2_15_2_1, ccsd_lambda2_15_2_1_);
  CorFortran(1, &op_lambda2_15_2, ccsd_lambda2_15_2_); // solved  by replacing the multiplier from 2 to 1
  destroy(lambda2_15_2_1);
  CorFortran(1, &op_lambda2_15, ccsd_lambda2_15_); 
  destroy(lambda2_15_1);
#if 1
#if 0  // following block work entirely in fortran or c++
  CorFortran(0, lambda2_16_1, offset_ccsd_lambda2_16_1_);
  CorFortran(0, lambda2_16_1_1, offset_ccsd_lambda2_16_1_1_);
  CorFortran(0, &op_lambda2_16_1_1, ccsd_lambda2_16_1_1_);
  CorFortran(0, &op_lambda2_16_1, ccsd_lambda2_16_1_);
  destroy(lambda2_16_1_1);
  CorFortran(0, &op_lambda2_16, ccsd_lambda2_16_);
#else
  CorFortran(1, lambda2_16_1, offset_ccsd_lambda2_16_1_);
  CorFortran(1, lambda2_16_1_1, offset_ccsd_lambda2_16_1_1_);
  CorFortran(1, &op_lambda2_16_1_1, ccsd_lambda2_16_1_1_);
  CorFortran(1, &op_lambda2_16_1, ccsd_lambda2_16_1_);
  destroy(lambda2_16_1_1);
  CorFortran(1, &op_lambda2_16, ccsd_lambda2_16_);
#endif  // code selection if 1 or 0 etc
  destroy(lambda2_16_1);
#else
  // order of code below not ok
  CorFortran(0, lambda2_16_1_1, offset_ccsd_lambda2_16_1_1_);
  CorFortran(0, &op_lambda2_16_1_1, ccsd_lambda2_16_1_1_);
  CorFortran(0, lambda2_16_1, offset_ccsd_lambda2_16_1_);
  CorFortran(0, &op_lambda2_16_1, ccsd_lambda2_16_1_);
  destroy(lambda2_16_1_1);
  CorFortran(0, &op_lambda2_16, ccsd_lambda2_16_);
  destroy(lambda2_16_1);
#endif  // code selection if 1 or 0 etc
#endif  // code selection if 1 or 0 etc
  /* ----- Insert detach code ------ */
  f->detach();
  i0->detach();
  v->detach();
  t_vo->detach();
  t_vvoo->detach();
  y_ov->detach();
  y_oovv->detach();
  }
}  // extern C
};  // namespace tamm
