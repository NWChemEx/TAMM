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
 *  x2 {
 *
 *  index h1,h2,h3,h4,h5,h6,h7,h8,h9,h10 = O;
 *  index p1,p2,p3,p4,p5,p6,p7,p8,p9 = V;
 *
 *  array i0[V][O,O];
 *  array x_o[][O]: irrep_x;
 *  array v[N,N][N,N]: irrep_v;
 *  array t_vo[V][O]: irrep_t;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array f[N][N]: irrep_f;
 *  array x_voo[V][O,O]: irrep_x;
 *  array x2_2_1[O][O];
 *  array x2_1_2_1[O,V][O,V];
 *  array x2_6_1[O][O,O];
 *  array x2_4_2_1[O,O][O,V];
 *  array x2_3_1[V][V];
 *  array x2_6_1_1[O,O][O,O];
 *  array x2_7_1[][V];
 *  array x2_8_1[O][O,V];
 *  array x2_5_1[O,V][O,V];
 *  array x2_6_1_2_1[O,O][O,V];
 *  array x2_6_2_1[O][V];
 *  array x2_1_1[O,V][O,O];
 *  array x2_4_1[O,O][O,O];
 *  array x2_6_3_1[O,O][O,V];
 *  array x2_1_3_1[O][V];
 *  array x2_2_2_1[O][V];
 *  array x2_1_4_1[O,O][O,V];
 *
 *  x2_1_1:     x2_1_1[h9,p3,h1,h2] += 1 * v[h9,p3,h1,h2];
 *  x2_1_2_1:   x2_1_2_1[h9,p3,h1,p5] += 1 * v[h9,p3,h1,p5];
 *  x2_1_2_2:   x2_1_2_1[h9,p3,h1,p5] += -1/2 * t_vo[p6,h1] * v[h9,p3,p5,p6];
 *  x2_1_2:     x2_1_1[h9,p3,h1,h2] += -1 * t_vo[p5,h1] * x2_1_2_1[h9,p3,h2,p5];
 *  x2_1_3_1:   x2_1_3_1[h9,p8] += 1 * f[h9,p8];
 *  x2_1_3_2:   x2_1_3_1[h9,p8] += 1 * t_vo[p6,h7] * v[h7,h9,p6,p8];
 *  x2_1_3:     x2_1_1[h9,p3,h1,h2] += -1 * t_vvoo[p3,p8,h1,h2] *
 * x2_1_3_1[h9,p8];
 *  x2_1_4_1:   x2_1_4_1[h6,h9,h1,p5] += 1 * v[h6,h9,h1,p5];
 *  x2_1_4_2:   x2_1_4_1[h6,h9,h1,p5] += -1 * t_vo[p7,h1] * v[h6,h9,p5,p7];
 *  x2_1_4:     x2_1_1[h9,p3,h1,h2] += 1 * t_vvoo[p3,p5,h1,h6] *
 * x2_1_4_1[h6,h9,h2,p5];
 *  x2_1_5:     x2_1_1[h9,p3,h1,h2] += 1/2 * t_vvoo[p5,p6,h1,h2] *
 * v[h9,p3,p5,p6];
 *  x2_1:       i0[p4,h1,h2] += -1 * x_o[h9] * x2_1_1[h9,p4,h1,h2];
 *  x2_2_1:     x2_2_1[h8,h1] += 1 * f[h8,h1];
 *  x2_2_2_1:   x2_2_2_1[h8,p9] += 1 * f[h8,p9];
 *  x2_2_2_2:   x2_2_2_1[h8,p9] += 1 * t_vo[p6,h7] * v[h7,h8,p6,p9];
 *  x2_2_2:     x2_2_1[h8,h1] += 1 * t_vo[p9,h1] * x2_2_2_1[h8,p9];
 *  x2_2_3:     x2_2_1[h8,h1] += -1 * t_vo[p5,h6] * v[h6,h8,h1,p5];
 *  x2_2_4:     x2_2_1[h8,h1] += -1/2 * t_vvoo[p5,p6,h1,h7] * v[h7,h8,p5,p6];
 *  x2_2:       i0[p3,h1,h2] += -1 * x_voo[p3,h1,h8] * x2_2_1[h8,h2];
 *  x2_3_1:     x2_3_1[p3,p8] += 1 * f[p3,p8];
 *  x2_3_2:     x2_3_1[p3,p8] += 1 * t_vo[p5,h6] * v[h6,p3,p5,p8];
 *  x2_3_3:     x2_3_1[p3,p8] += 1/2 * t_vvoo[p3,p5,h6,h7] * v[h6,h7,p5,p8];
 *  x2_3:       i0[p4,h1,h2] += 1 * x_voo[p8,h1,h2] * x2_3_1[p4,p8];
 *  x2_4_1:     x2_4_1[h9,h10,h1,h2] += 1 * v[h9,h10,h1,h2];
 *  x2_4_2_1:   x2_4_2_1[h9,h10,h1,p5] += 1 * v[h9,h10,h1,p5];
 *  x2_4_2_2:   x2_4_2_1[h9,h10,h1,p5] += -1/2 * t_vo[p6,h1] * v[h9,h10,p5,p6];
 *  x2_4_2:     x2_4_1[h9,h10,h1,h2] += -1 * t_vo[p5,h1] *
 * x2_4_2_1[h9,h10,h2,p5];
 *  x2_4_3:     x2_4_1[h9,h10,h1,h2] += 1/2 * t_vvoo[p5,p6,h1,h2] *
 * v[h9,h10,p5,p6];
 *  x2_4:       i0[p3,h1,h2] += 1/2 * x_voo[p3,h9,h10] * x2_4_1[h9,h10,h1,h2];
 *  x2_5_1:     x2_5_1[h7,p3,h1,p8] += 1 * v[h7,p3,h1,p8];
 *  x2_5_2:     x2_5_1[h7,p3,h1,p8] += 1 * t_vo[p5,h1] * v[h7,p3,p5,p8];
 *  x2_5:       i0[p4,h1,h2] += -1 * x_voo[p8,h1,h7] * x2_5_1[h7,p4,h2,p8];
 *  x2_6_1_1:   x2_6_1_1[h8,h10,h1,h2] += 1 * v[h8,h10,h1,h2];
 *  x2_6_1_2_1: x2_6_1_2_1[h8,h10,h1,p5] += 1 * v[h8,h10,h1,p5];
 *  x2_6_1_2_2: x2_6_1_2_1[h8,h10,h1,p5] += -1/2 * t_vo[p6,h1] *
 * v[h8,h10,p5,p6];
 *  x2_6_1_2:   x2_6_1_1[h8,h10,h1,h2] += -1 * t_vo[p5,h1] *
 * x2_6_1_2_1[h8,h10,h2,p5];
 *  x2_6_1_3:   x2_6_1_1[h8,h10,h1,h2] += 1/2 * t_vvoo[p5,p6,h1,h2] *
 * v[h8,h10,p5,p6];
 *  x2_6_1:     x2_6_1[h10,h1,h2] += -1 * x_o[h8] * x2_6_1_1[h8,h10,h1,h2];
 *  x2_6_2_1:   x2_6_2_1[h10,p5] += 1 * f[h10,p5];
 *  x2_6_2_2:   x2_6_2_1[h10,p5] += -1 * t_vo[p6,h7] * v[h7,h10,p5,p6];
 *  x2_6_2:     x2_6_1[h10,h1,h2] += 1 * x_voo[p5,h1,h2] * x2_6_2_1[h10,p5];
 *  x2_6_3_1:   x2_6_3_1[h8,h10,h1,p9] += 1 * v[h8,h10,h1,p9];
 *  x2_6_3_2:   x2_6_3_1[h8,h10,h1,p9] += 1 * t_vo[p5,h1] * v[h8,h10,p5,p9];
 *  x2_6_3:     x2_6_1[h10,h1,h2] += -1 * x_voo[p9,h1,h8] *
 * x2_6_3_1[h8,h10,h2,p9];
 *  x2_6:       i0[p3,h1,h2] += 1 * t_vo[p3,h10] * x2_6_1[h10,h1,h2];
 *  x2_7_1:     x2_7_1[p5] += -1 * x_voo[p8,h6,h7] * v[h6,h7,p5,p8];
 *  x2_7:       i0[p3,h1,h2] += 1/2 * t_vvoo[p3,p5,h1,h2] * x2_7_1[p5];
 *  x2_8_1:     x2_8_1[h6,h1,p5] += 1 * x_voo[p8,h1,h7] * v[h6,h7,p5,p8];
 *  x2_8:       i0[p3,h1,h2] += 1 * t_vvoo[p3,p5,h1,h6] * x2_8_1[h6,h2,p5];
 *
 *  }
 */

extern "C" {
void ipccsd_x2_1_1_(F77Integer *d_v, F77Integer *k_v_offset, F77Integer *d_x2_1_1,
                    F77Integer *k_x2_1_1_offset);
void ipccsd_x2_1_2_1_(F77Integer *d_v, F77Integer *k_v_offset, F77Integer *d_x2_1_2_1,
                      F77Integer *k_x2_1_2_1_offset);
void ipccsd_x2_1_2_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                      F77Integer *k_v_offset, F77Integer *d_x2_1_2_1,
                      F77Integer *k_x2_1_2_1_offset);
void ipccsd_x2_1_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset,
                    F77Integer *d_x2_1_2_1, F77Integer *k_x2_1_2_1_offset,
                    F77Integer *d_x2_1_1, F77Integer *k_x2_1_1_offset);
void ipccsd_x2_1_3_1_(F77Integer *d_f, F77Integer *k_f_offset, F77Integer *d_x2_1_3_1,
                      F77Integer *k_x2_1_3_1_offset);
void ipccsd_x2_1_3_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                      F77Integer *k_v_offset, F77Integer *d_x2_1_3_1,
                      F77Integer *k_x2_1_3_1_offset);
void ipccsd_x2_1_3_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset,
                    F77Integer *d_x2_1_3_1, F77Integer *k_x2_1_3_1_offset,
                    F77Integer *d_x2_1_1, F77Integer *k_x2_1_1_offset);
void ipccsd_x2_1_4_1_(F77Integer *d_v, F77Integer *k_v_offset, F77Integer *d_x2_1_4_1,
                      F77Integer *k_x2_1_4_1_offset);
void ipccsd_x2_1_4_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                      F77Integer *k_v_offset, F77Integer *d_x2_1_4_1,
                      F77Integer *k_x2_1_4_1_offset);
void ipccsd_x2_1_4_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset,
                    F77Integer *d_x2_1_4_1, F77Integer *k_x2_1_4_1_offset,
                    F77Integer *d_x2_1_1, F77Integer *k_x2_1_1_offset);
void ipccsd_x2_1_5_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset, F77Integer *d_v,
                    F77Integer *k_v_offset, F77Integer *d_x2_1_1,
                    F77Integer *k_x2_1_1_offset);
void ipccsd_x2_1_(F77Integer *d_x_o, F77Integer *k_x_o_offset, F77Integer *d_x2_1_1,
                  F77Integer *k_x2_1_1_offset, F77Integer *d_i0,
                  F77Integer *k_i0_offset);
void ipccsd_x2_2_1_(F77Integer *d_f, F77Integer *k_f_offset, F77Integer *d_x2_2_1,
                    F77Integer *k_x2_2_1_offset);
void ipccsd_x2_2_2_1_(F77Integer *d_f, F77Integer *k_f_offset, F77Integer *d_x2_2_2_1,
                      F77Integer *k_x2_2_2_1_offset);
void ipccsd_x2_2_2_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                      F77Integer *k_v_offset, F77Integer *d_x2_2_2_1,
                      F77Integer *k_x2_2_2_1_offset);
void ipccsd_x2_2_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset,
                    F77Integer *d_x2_2_2_1, F77Integer *k_x2_2_2_1_offset,
                    F77Integer *d_x2_2_1, F77Integer *k_x2_2_1_offset);
void ipccsd_x2_2_3_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                    F77Integer *k_v_offset, F77Integer *d_x2_2_1,
                    F77Integer *k_x2_2_1_offset);
void ipccsd_x2_2_4_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset, F77Integer *d_v,
                    F77Integer *k_v_offset, F77Integer *d_x2_2_1,
                    F77Integer *k_x2_2_1_offset);
void ipccsd_x2_2_(F77Integer *d_x_voo, F77Integer *k_x_voo_offset, F77Integer *d_x2_2_1,
                  F77Integer *k_x2_2_1_offset, F77Integer *d_i0,
                  F77Integer *k_i0_offset);
void ipccsd_x2_3_1_(F77Integer *d_f, F77Integer *k_f_offset, F77Integer *d_x2_3_1,
                    F77Integer *k_x2_3_1_offset);
void ipccsd_x2_3_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                    F77Integer *k_v_offset, F77Integer *d_x2_3_1,
                    F77Integer *k_x2_3_1_offset);
void ipccsd_x2_3_3_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset, F77Integer *d_v,
                    F77Integer *k_v_offset, F77Integer *d_x2_3_1,
                    F77Integer *k_x2_3_1_offset);
void ipccsd_x2_3_(F77Integer *d_x_voo, F77Integer *k_x_voo_offset, F77Integer *d_x2_3_1,
                  F77Integer *k_x2_3_1_offset, F77Integer *d_i0,
                  F77Integer *k_i0_offset);
void ipccsd_x2_4_1_(F77Integer *d_v, F77Integer *k_v_offset, F77Integer *d_x2_4_1,
                    F77Integer *k_x2_4_1_offset);
void ipccsd_x2_4_2_1_(F77Integer *d_v, F77Integer *k_v_offset, F77Integer *d_x2_4_2_1,
                      F77Integer *k_x2_4_2_1_offset);
void ipccsd_x2_4_2_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                      F77Integer *k_v_offset, F77Integer *d_x2_4_2_1,
                      F77Integer *k_x2_4_2_1_offset);
void ipccsd_x2_4_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset,
                    F77Integer *d_x2_4_2_1, F77Integer *k_x2_4_2_1_offset,
                    F77Integer *d_x2_4_1, F77Integer *k_x2_4_1_offset);
void ipccsd_x2_4_3_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset, F77Integer *d_v,
                    F77Integer *k_v_offset, F77Integer *d_x2_4_1,
                    F77Integer *k_x2_4_1_offset);
void ipccsd_x2_4_(F77Integer *d_x_voo, F77Integer *k_x_voo_offset, F77Integer *d_x2_4_1,
                  F77Integer *k_x2_4_1_offset, F77Integer *d_i0,
                  F77Integer *k_i0_offset);
void ipccsd_x2_5_1_(F77Integer *d_v, F77Integer *k_v_offset, F77Integer *d_x2_5_1,
                    F77Integer *k_x2_5_1_offset);
void ipccsd_x2_5_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                    F77Integer *k_v_offset, F77Integer *d_x2_5_1,
                    F77Integer *k_x2_5_1_offset);
void ipccsd_x2_5_(F77Integer *d_x_voo, F77Integer *k_x_voo_offset, F77Integer *d_x2_5_1,
                  F77Integer *k_x2_5_1_offset, F77Integer *d_i0,
                  F77Integer *k_i0_offset);
void ipccsd_x2_6_1_1_(F77Integer *d_v, F77Integer *k_v_offset, F77Integer *d_x2_6_1_1,
                      F77Integer *k_x2_6_1_1_offset);
void ipccsd_x2_6_1_2_1_(F77Integer *d_v, F77Integer *k_v_offset,
                        F77Integer *d_x2_6_1_2_1, F77Integer *k_x2_6_1_2_1_offset);
void ipccsd_x2_6_1_2_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                        F77Integer *k_v_offset, F77Integer *d_x2_6_1_2_1,
                        F77Integer *k_x2_6_1_2_1_offset);
void ipccsd_x2_6_1_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset,
                      F77Integer *d_x2_6_1_2_1, F77Integer *k_x2_6_1_2_1_offset,
                      F77Integer *d_x2_6_1_1, F77Integer *k_x2_6_1_1_offset);
void ipccsd_x2_6_1_3_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset, F77Integer *d_v,
                      F77Integer *k_v_offset, F77Integer *d_x2_6_1_1,
                      F77Integer *k_x2_6_1_1_offset);
void ipccsd_x2_6_1_(F77Integer *d_x_o, F77Integer *k_x_o_offset, F77Integer *d_x2_6_1_1,
                    F77Integer *k_x2_6_1_1_offset, F77Integer *d_x2_6_1,
                    F77Integer *k_x2_6_1_offset);
void ipccsd_x2_6_2_1_(F77Integer *d_f, F77Integer *k_f_offset, F77Integer *d_x2_6_2_1,
                      F77Integer *k_x2_6_2_1_offset);
void ipccsd_x2_6_2_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                      F77Integer *k_v_offset, F77Integer *d_x2_6_2_1,
                      F77Integer *k_x2_6_2_1_offset);
void ipccsd_x2_6_2_(F77Integer *d_x_voo, F77Integer *k_x_voo_offset,
                    F77Integer *d_x2_6_2_1, F77Integer *k_x2_6_2_1_offset,
                    F77Integer *d_x2_6_1, F77Integer *k_x2_6_1_offset);
void ipccsd_x2_6_3_1_(F77Integer *d_v, F77Integer *k_v_offset, F77Integer *d_x2_6_3_1,
                      F77Integer *k_x2_6_3_1_offset);
void ipccsd_x2_6_3_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                      F77Integer *k_v_offset, F77Integer *d_x2_6_3_1,
                      F77Integer *k_x2_6_3_1_offset);
void ipccsd_x2_6_3_(F77Integer *d_x_voo, F77Integer *k_x_voo_offset,
                    F77Integer *d_x2_6_3_1, F77Integer *k_x2_6_3_1_offset,
                    F77Integer *d_x2_6_1, F77Integer *k_x2_6_1_offset);
void ipccsd_x2_6_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_x2_6_1,
                  F77Integer *k_x2_6_1_offset, F77Integer *d_i0,
                  F77Integer *k_i0_offset);
void ipccsd_x2_7_1_(F77Integer *d_x_voo, F77Integer *k_x_voo_offset, F77Integer *d_v,
                    F77Integer *k_v_offset, F77Integer *d_x2_7_1,
                    F77Integer *k_x2_7_1_offset);
void ipccsd_x2_7_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset,
                  F77Integer *d_x2_7_1, F77Integer *k_x2_7_1_offset, F77Integer *d_i0,
                  F77Integer *k_i0_offset);
void ipccsd_x2_8_1_(F77Integer *d_x_voo, F77Integer *k_x_voo_offset, F77Integer *d_v,
                    F77Integer *k_v_offset, F77Integer *d_x2_8_1,
                    F77Integer *k_x2_8_1_offset);
void ipccsd_x2_8_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset,
                  F77Integer *d_x2_8_1, F77Integer *k_x2_8_1_offset, F77Integer *d_i0,
                  F77Integer *k_i0_offset);

void offset_ipccsd_x2_1_1_(F77Integer *l_x2_1_1_offset, F77Integer *k_x2_1_1_offset,
                           F77Integer *size_x2_1_1);
void offset_ipccsd_x2_1_2_1_(F77Integer *l_x2_1_2_1_offset,
                             F77Integer *k_x2_1_2_1_offset,
                             F77Integer *size_x2_1_2_1);
void offset_ipccsd_x2_1_3_1_(F77Integer *l_x2_1_3_1_offset,
                             F77Integer *k_x2_1_3_1_offset,
                             F77Integer *size_x2_1_3_1);
void offset_ipccsd_x2_1_4_1_(F77Integer *l_x2_1_4_1_offset,
                             F77Integer *k_x2_1_4_1_offset,
                             F77Integer *size_x2_1_4_1);
void offset_ipccsd_x2_2_1_(F77Integer *l_x2_2_1_offset, F77Integer *k_x2_2_1_offset,
                           F77Integer *size_x2_2_1);
void offset_ipccsd_x2_2_2_1_(F77Integer *l_x2_2_2_1_offset,
                             F77Integer *k_x2_2_2_1_offset,
                             F77Integer *size_x2_2_2_1);
void offset_ipccsd_x2_3_1_(F77Integer *l_x2_3_1_offset, F77Integer *k_x2_3_1_offset,
                           F77Integer *size_x2_3_1);
void offset_ipccsd_x2_4_1_(F77Integer *l_x2_4_1_offset, F77Integer *k_x2_4_1_offset,
                           F77Integer *size_x2_4_1);
void offset_ipccsd_x2_4_2_1_(F77Integer *l_x2_4_2_1_offset,
                             F77Integer *k_x2_4_2_1_offset,
                             F77Integer *size_x2_4_2_1);
void offset_ipccsd_x2_5_1_(F77Integer *l_x2_5_1_offset, F77Integer *k_x2_5_1_offset,
                           F77Integer *size_x2_5_1);
void offset_ipccsd_x2_6_1_1_(F77Integer *l_x2_6_1_1_offset,
                             F77Integer *k_x2_6_1_1_offset,
                             F77Integer *size_x2_6_1_1);
void offset_ipccsd_x2_6_1_2_1_(F77Integer *l_x2_6_1_2_1_offset,
                               F77Integer *k_x2_6_1_2_1_offset,
                               F77Integer *size_x2_6_1_2_1);
void offset_ipccsd_x2_6_1_(F77Integer *l_x2_6_1_offset, F77Integer *k_x2_6_1_offset,
                           F77Integer *size_x2_6_1);
void offset_ipccsd_x2_6_2_1_(F77Integer *l_x2_6_2_1_offset,
                             F77Integer *k_x2_6_2_1_offset,
                             F77Integer *size_x2_6_2_1);
void offset_ipccsd_x2_6_3_1_(F77Integer *l_x2_6_3_1_offset,
                             F77Integer *k_x2_6_3_1_offset,
                             F77Integer *size_x2_6_3_1);
void offset_ipccsd_x2_7_1_(F77Integer *l_x2_7_1_offset, F77Integer *k_x2_7_1_offset,
                           F77Integer *size_x2_7_1);
void offset_ipccsd_x2_8_1_(F77Integer *l_x2_8_1_offset, F77Integer *k_x2_8_1_offset,
                           F77Integer *size_x2_8_1);
}

namespace tamm {

extern "C" {
void ipccsd_x2_cxx_(F77Integer *d_f, F77Integer *d_i0, F77Integer *d_t_vo,
                    F77Integer *d_t_vvoo, F77Integer *d_v, F77Integer *d_x_o,
                    F77Integer *d_x_voo, F77Integer *k_f_offset, F77Integer *k_i0_offset,
                    F77Integer *k_t_vo_offset, F77Integer *k_t_vvoo_offset,
                    F77Integer *k_v_offset, F77Integer *k_x_o_offset,
                    F77Integer *k_x_voo_offset) {
  static bool set_x2 = true;

  Assignment op_x2_1_1;
  Assignment op_x2_1_2_1;
  Assignment op_x2_1_3_1;
  Assignment op_x2_1_4_1;
  Assignment op_x2_2_1;
  Assignment op_x2_2_2_1;
  Assignment op_x2_3_1;
  Assignment op_x2_4_1;
  Assignment op_x2_4_2_1;
  Assignment op_x2_5_1;
  Assignment op_x2_6_1_1;
  Assignment op_x2_6_1_2_1;
  Assignment op_x2_6_2_1;
  Assignment op_x2_6_3_1;
  Multiplication op_x2_1_2_2;
  Multiplication op_x2_1_2;
  Multiplication op_x2_1_3_2;
  Multiplication op_x2_1_3;
  Multiplication op_x2_1_4_2;
  Multiplication op_x2_1_4;
  Multiplication op_x2_1_5;
  Multiplication op_x2_1;
  Multiplication op_x2_2_2_2;
  Multiplication op_x2_2_2;
  Multiplication op_x2_2_3;
  Multiplication op_x2_2_4;
  Multiplication op_x2_2;
  Multiplication op_x2_3_2;
  Multiplication op_x2_3_3;
  Multiplication op_x2_3;
  Multiplication op_x2_4_2_2;
  Multiplication op_x2_4_2;
  Multiplication op_x2_4_3;
  Multiplication op_x2_4;
  Multiplication op_x2_5_2;
  Multiplication op_x2_5;
  Multiplication op_x2_6_1_2_2;
  Multiplication op_x2_6_1_2;
  Multiplication op_x2_6_1_3;
  Multiplication op_x2_6_1;
  Multiplication op_x2_6_2_2;
  Multiplication op_x2_6_2;
  Multiplication op_x2_6_3_2;
  Multiplication op_x2_6_3;
  Multiplication op_x2_6;
  Multiplication op_x2_7_1;
  Multiplication op_x2_7;
  Multiplication op_x2_8_1;
  Multiplication op_x2_8;

  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_x2) {
    ipccsd_x2_equations(&eqs);
    set_x2 = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector<Operation> ops;
  tensors_and_ops(&eqs, &tensors, &ops);

  Tensor *i0 = &tensors["i0"];
  Tensor *x_o = &tensors["x_o"];
  Tensor *v = &tensors["v"];
  Tensor *t_vo = &tensors["t_vo"];
  Tensor *t_vvoo = &tensors["t_vvoo"];
  Tensor *f = &tensors["f"];
  Tensor *x_voo = &tensors["x_voo"];
  Tensor *x2_2_1 = &tensors["x2_2_1"];
  Tensor *x2_1_2_1 = &tensors["x2_1_2_1"];
  Tensor *x2_6_1 = &tensors["x2_6_1"];
  Tensor *x2_4_2_1 = &tensors["x2_4_2_1"];
  Tensor *x2_3_1 = &tensors["x2_3_1"];
  Tensor *x2_6_1_1 = &tensors["x2_6_1_1"];
  Tensor *x2_7_1 = &tensors["x2_7_1"];
  Tensor *x2_8_1 = &tensors["x2_8_1"];
  Tensor *x2_5_1 = &tensors["x2_5_1"];
  Tensor *x2_6_1_2_1 = &tensors["x2_6_1_2_1"];
  Tensor *x2_6_2_1 = &tensors["x2_6_2_1"];
  Tensor *x2_1_1 = &tensors["x2_1_1"];
  Tensor *x2_4_1 = &tensors["x2_4_1"];
  Tensor *x2_6_3_1 = &tensors["x2_6_3_1"];
  Tensor *x2_1_3_1 = &tensors["x2_1_3_1"];
  Tensor *x2_2_2_1 = &tensors["x2_2_2_1"];
  Tensor *x2_1_4_1 = &tensors["x2_1_4_1"];

  v->set_dist(idist);
  t_vo->set_dist(dist_nw);
  f->attach(*k_f_offset, 0, *d_f);
  i0->attach(*k_i0_offset, 0, *d_i0);
  t_vo->attach(*k_t_vo_offset, 0, *d_t_vo);
  t_vvoo->attach(*k_t_vvoo_offset, 0, *d_t_vvoo);
  v->attach(*k_v_offset, 0, *d_v);
  x_o->attach(*k_x_o_offset, 0, *d_x_o);
  x_voo->attach(*k_x_voo_offset, 0, *d_x_voo);
  x_o->set_irrep(Variables::irrep_x());
  x_voo->set_irrep(Variables::irrep_x());
  i0->set_irrep(Variables::irrep_x());
  x2_6_1->set_irrep(Variables::irrep_x());
  x2_7_1->set_irrep(Variables::irrep_x());
  x2_8_1->set_irrep(Variables::irrep_x());

  for (int i = 0; i < eqs.op_entries.size(); i++) {
    switch (eqs.op_entries[i].optype) {
      case OpTypeAdd: {
        Tensor *t_alhs =
            &tensors[eqs.tensor_entries.at(eqs.op_entries[i].add.tc).name];
        t_alhs->set_irrep(
            (&tensors[eqs.tensor_entries.at(eqs.op_entries[i].add.ta).name])
                ->irrep());
        break;
      }
      case OpTypeMult: {
        Tensor *t_mlhs =
            &tensors[eqs.tensor_entries.at(eqs.op_entries[i].mult.tc).name];
        t_mlhs->set_irrep(
            (&tensors[eqs.tensor_entries.at(eqs.op_entries[i].mult.ta).name])
                ->irrep() ^
            (&tensors[eqs.tensor_entries.at(eqs.op_entries[i].mult.tb).name])
                ->irrep());
        break;
      }
    }
  }

#if 1
  // schedule_linear(&tensors, &ops);
  // schedule_linear_lazy(&tensors, &ops);
  schedule_levels(&tensors, &ops);
#else
  op_x2_1_1 = ops[0].add;
  op_x2_1_2_1 = ops[1].add;
  op_x2_1_2_2 = ops[2].mult;
  op_x2_1_2 = ops[3].mult;
  op_x2_1_3_1 = ops[4].add;
  op_x2_1_3_2 = ops[5].mult;
  op_x2_1_3 = ops[6].mult;
  op_x2_1_4_1 = ops[7].add;
  op_x2_1_4_2 = ops[8].mult;
  op_x2_1_4 = ops[9].mult;
  op_x2_1_5 = ops[10].mult;
  op_x2_1 = ops[11].mult;
  op_x2_2_1 = ops[12].add;
  op_x2_2_2_1 = ops[13].add;
  op_x2_2_2_2 = ops[14].mult;
  op_x2_2_2 = ops[15].mult;
  op_x2_2_3 = ops[16].mult;
  op_x2_2_4 = ops[17].mult;
  op_x2_2 = ops[18].mult;
  op_x2_3_1 = ops[19].add;
  op_x2_3_2 = ops[20].mult;
  op_x2_3_3 = ops[21].mult;
  op_x2_3 = ops[22].mult;
  op_x2_4_1 = ops[23].add;
  op_x2_4_2_1 = ops[24].add;
  op_x2_4_2_2 = ops[25].mult;
  op_x2_4_2 = ops[26].mult;
  op_x2_4_3 = ops[27].mult;
  op_x2_4 = ops[28].mult;
  op_x2_5_1 = ops[29].add;
  op_x2_5_2 = ops[30].mult;
  op_x2_5 = ops[31].mult;
  op_x2_6_1_1 = ops[32].add;
  op_x2_6_1_2_1 = ops[33].add;
  op_x2_6_1_2_2 = ops[34].mult;
  op_x2_6_1_2 = ops[35].mult;
  op_x2_6_1_3 = ops[36].mult;
  op_x2_6_1 = ops[37].mult;
  op_x2_6_2_1 = ops[38].add;
  op_x2_6_2_2 = ops[39].mult;
  op_x2_6_2 = ops[40].mult;
  op_x2_6_3_1 = ops[41].add;
  op_x2_6_3_2 = ops[42].mult;
  op_x2_6_3 = ops[43].mult;
  op_x2_6 = ops[44].mult;
  op_x2_7_1 = ops[45].mult;
  op_x2_7 = ops[46].mult;
  op_x2_8_1 = ops[47].mult;
  op_x2_8 = ops[48].mult;

  CorFortran(1, x2_1_1, offset_ipccsd_x2_1_1_);
  CorFortran(1, &op_x2_1_1, ipccsd_x2_1_1_);
  CorFortran(1, x2_1_2_1, offset_ipccsd_x2_1_2_1_);
  CorFortran(1, &op_x2_1_2_1, ipccsd_x2_1_2_1_);
  CorFortran(1, &op_x2_1_2_2, ipccsd_x2_1_2_2_);
  CorFortran(1, &op_x2_1_2, ipccsd_x2_1_2_);
  destroy(x2_1_2_1);
  CorFortran(1, x2_1_3_1, offset_ipccsd_x2_1_3_1_);
  CorFortran(1, &op_x2_1_3_1, ipccsd_x2_1_3_1_);
  CorFortran(1, &op_x2_1_3_2, ipccsd_x2_1_3_2_);
  CorFortran(1, &op_x2_1_3, ipccsd_x2_1_3_);
  destroy(x2_1_3_1);
  CorFortran(1, x2_1_4_1, offset_ipccsd_x2_1_4_1_);
  CorFortran(1, &op_x2_1_4_1, ipccsd_x2_1_4_1_);
  CorFortran(1, &op_x2_1_4_2, ipccsd_x2_1_4_2_);
  CorFortran(1, &op_x2_1_4, ipccsd_x2_1_4_);
  destroy(x2_1_4_1);
  CorFortran(1, &op_x2_1_5, ipccsd_x2_1_5_);
  CorFortran(1, &op_x2_1, ipccsd_x2_1_); /** @bug */
  destroy(x2_1_1);
  CorFortran(1, x2_2_1, offset_ipccsd_x2_2_1_);
  CorFortran(1, &op_x2_2_1, ipccsd_x2_2_1_);
  CorFortran(1, x2_2_2_1, offset_ipccsd_x2_2_2_1_);
  CorFortran(1, &op_x2_2_2_1, ipccsd_x2_2_2_1_);
  CorFortran(1, &op_x2_2_2_2, ipccsd_x2_2_2_2_);
  CorFortran(1, &op_x2_2_2, ipccsd_x2_2_2_);
  destroy(x2_2_2_1);
  CorFortran(1, &op_x2_2_3, ipccsd_x2_2_3_);
  CorFortran(1, &op_x2_2_4, ipccsd_x2_2_4_);
  CorFortran(1, &op_x2_2, ipccsd_x2_2_); /** @bug */
  destroy(x2_2_1);
  CorFortran(1, x2_3_1, offset_ipccsd_x2_3_1_);
  CorFortran(1, &op_x2_3_1, ipccsd_x2_3_1_);
  CorFortran(1, &op_x2_3_2, ipccsd_x2_3_2_);
  CorFortran(1, &op_x2_3_3, ipccsd_x2_3_3_);
  CorFortran(1, &op_x2_3, ipccsd_x2_3_); /** @bug */
  destroy(x2_3_1);
  CorFortran(1, x2_4_1, offset_ipccsd_x2_4_1_);
  CorFortran(1, &op_x2_4_1, ipccsd_x2_4_1_);
  CorFortran(1, x2_4_2_1, offset_ipccsd_x2_4_2_1_);
  CorFortran(1, &op_x2_4_2_1, ipccsd_x2_4_2_1_);
  CorFortran(1, &op_x2_4_2_2, ipccsd_x2_4_2_2_);
  CorFortran(1, &op_x2_4_2, ipccsd_x2_4_2_);
  destroy(x2_4_2_1);
  CorFortran(1, &op_x2_4_3, ipccsd_x2_4_3_);
  CorFortran(1, &op_x2_4, ipccsd_x2_4_); /** @bug */
  destroy(x2_4_1);
  CorFortran(1, x2_5_1, offset_ipccsd_x2_5_1_);
  CorFortran(1, &op_x2_5_1, ipccsd_x2_5_1_);
  CorFortran(1, &op_x2_5_2, ipccsd_x2_5_2_);
  CorFortran(1, &op_x2_5, ipccsd_x2_5_); /** @bug */
  destroy(x2_5_1);
  CorFortran(1, x2_6_1, offset_ipccsd_x2_6_1_); /** @bug -- */
  CorFortran(1, x2_6_1_1, offset_ipccsd_x2_6_1_1_);
  CorFortran(1, &op_x2_6_1_1, ipccsd_x2_6_1_1_);
  CorFortran(1, x2_6_1_2_1, offset_ipccsd_x2_6_1_2_1_);
  CorFortran(1, &op_x2_6_1_2_1, ipccsd_x2_6_1_2_1_);
  CorFortran(1, &op_x2_6_1_2_2, ipccsd_x2_6_1_2_2_);
  CorFortran(1, &op_x2_6_1_2, ipccsd_x2_6_1_2_);
  destroy(x2_6_1_2_1);
  CorFortran(1, &op_x2_6_1_3, ipccsd_x2_6_1_3_);
  CorFortran(1, &op_x2_6_1, ipccsd_x2_6_1_); /** @bug -- */
  destroy(x2_6_1_1);
  CorFortran(1, x2_6_2_1, offset_ipccsd_x2_6_2_1_);
  CorFortran(1, &op_x2_6_2_1, ipccsd_x2_6_2_1_);
  CorFortran(1, &op_x2_6_2_2, ipccsd_x2_6_2_2_);
  CorFortran(1, &op_x2_6_2, ipccsd_x2_6_2_); /** @bug */
  destroy(x2_6_2_1);
  CorFortran(1, x2_6_3_1, offset_ipccsd_x2_6_3_1_);
  CorFortran(1, &op_x2_6_3_1, ipccsd_x2_6_3_1_);
  CorFortran(1, &op_x2_6_3_2, ipccsd_x2_6_3_2_);
  CorFortran(1, &op_x2_6_3, ipccsd_x2_6_3_); /** @bug */
  destroy(x2_6_3_1);
  CorFortran(1, &op_x2_6, ipccsd_x2_6_); /** @bug */
  destroy(x2_6_1);
  CorFortran(1, x2_7_1, offset_ipccsd_x2_7_1_); /** @bug -- */
  CorFortran(1, &op_x2_7_1, ipccsd_x2_7_1_);    /** @bug */
  CorFortran(1, &op_x2_7, ipccsd_x2_7_);        /** @bug */
  destroy(x2_7_1);
  CorFortran(1, x2_8_1, offset_ipccsd_x2_8_1_); /** @bug -- */
  CorFortran(1, &op_x2_8_1, ipccsd_x2_8_1_);    /** @bug */
  CorFortran(1, &op_x2_8, ipccsd_x2_8_);        /** @bug */
  destroy(x2_8_1);
#endif  // Use fortrain functions

  /* ----- Insert detach code ------ */

  f->detach();
  i0->detach();
  t_vo->detach();
  t_vvoo->detach();
  v->detach();
  x_o->detach();
  x_voo->detach();
}
}  // extern C
};  // namespace tamm
