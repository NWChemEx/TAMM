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
 *  index p1,p2,p3,p4,p5,p6,p7,p8,p9,p10 = V;
 *
 *  array i0[V,V][O];
 *  array x_v[V][]: irrep_x;
 *  array v[N,N][N,N]: irrep_v;
 *  array x_vvo[V,V][O]: irrep_x;
 *  array f[N][N]: irrep_f;
 *  array t_vo[V][O]: irrep_t;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array x2_10_1[V,O][V];
 *  array x2_2_1[O][O];
 *  array x2_6_1[O,V][O];
 *  array x2_6_5_1[O,O][O];
 *  array x2_3_1[V][V];
 *  array x2_6_7_1[O,O][V];
 *  array x2_8_1[O][];
 *  array x2_7_1[V,V][V];
 *  array x2_6_2_1[O][V];
 *  array x2_8_1_1[O][V];
 *  array x2_9_1[O,O][O];
 *  array x2_6_6_1[V,O][V];
 *  array x2_9_3_1[O,O][V];
 *  array x2_4_1[O,V][O,V];
 *  array x2_6_5_3_1[O,O][V];
 *  array x2_6_3_1[O,O][O,V];
 *  array x2_2_2_1[O][V];
 *
 *  x2_1:       i0[p3,p4,h2] += 1 * x_v[p5] * v[p3,p4,h2,p5];
 *  x2_2_1:     x2_2_1[h8,h1] += 1 * f[h8,h1];
 *  x2_2_2_1:   x2_2_2_1[h8,p9] += 1 * f[h8,p9];
 *  x2_2_2_2:   x2_2_2_1[h8,p9] += 1 * t_vo[p6,h7] * v[h7,h8,p6,p9];
 *  x2_2_2:     x2_2_1[h8,h1] += 1 * t_vo[p9,h1] * x2_2_2_1[h8,p9];
 *  x2_2_3:     x2_2_1[h8,h1] += -1 * t_vo[p5,h6] * v[h6,h8,h1,p5];
 *  x2_2_4:     x2_2_1[h8,h1] += -1/2 * t_vvoo[p5,p6,h1,h7] * v[h7,h8,p5,p6];
 *  x2_2:       i0[p3,p4,h2] += -1 * x_vvo[p3,p4,h8] * x2_2_1[h8,h2];
 *  x2_3_1:     x2_3_1[p3,p8] += 1 * f[p3,p8];
 *  x2_3_2:     x2_3_1[p3,p8] += 1 * t_vo[p5,h6] * v[h6,p3,p5,p8];
 *  x2_3_3:     x2_3_1[p3,p8] += 1/2 * t_vvoo[p3,p5,h6,h7] * v[h6,h7,p5,p8];
 *  x2_3:       i0[p3,p4,h1] += 1 * x_vvo[p3,p8,h1] * x2_3_1[p4,p8];
 *  x2_4_1:     x2_4_1[h7,p3,h1,p8] += 1 * v[h7,p3,h1,p8];
 *  x2_4_2:     x2_4_1[h7,p3,h1,p8] += 1 * t_vo[p5,h1] * v[h7,p3,p5,p8];
 *  x2_4:       i0[p3,p4,h2] += -1 * x_vvo[p3,p8,h7] * x2_4_1[h7,p4,h2,p8];
 *  x2_5:       i0[p3,p4,h1] += 1/2 * x_vvo[p5,p6,h1] * v[p3,p4,p5,p6];
 *  x2_6_1:     x2_6_1[h9,p3,h2] += 1/2 * x_v[p6] * v[h9,p3,h2,p6];
 *  x2_6_2_1:   x2_6_2_1[h9,p5] += 1 * f[h9,p5];
 *  x2_6_2_2:   x2_6_2_1[h9,p5] += -1 * t_vo[p6,h7] * v[h7,h9,p5,p6];
 *  x2_6_2:     x2_6_1[h9,p3,h1] += -1/2 * x_vvo[p3,p5,h1] * x2_6_2_1[h9,p5];
 *  x2_6_3_1:   x2_6_3_1[h8,h9,h1,p10] += 1 * v[h8,h9,h1,p10];
 *  x2_6_3_2:   x2_6_3_1[h8,h9,h1,p10] += 1 * t_vo[p5,h1] * v[h8,h9,p5,p10];
 *  x2_6_3:     x2_6_1[h9,p3,h2] += 1/2 * x_vvo[p3,p10,h8] *
 * x2_6_3_1[h8,h9,h2,p10];
 *  x2_6_4:     x2_6_1[h9,p3,h1] += 1/4 * x_vvo[p6,p7,h1] * v[h9,p3,p6,p7];
 *  x2_6_5_1:   x2_6_5_1[h9,h10,h2] += 1/2 * x_v[p7] * v[h9,h10,h2,p7];
 *  x2_6_5_2:   x2_6_5_1[h9,h10,h1] += 1/4 * x_vvo[p7,p8,h1] * v[h9,h10,p7,p8];
 *  x2_6_5_3_1: x2_6_5_3_1[h9,h10,p5] += 1 * x_v[p8] * v[h9,h10,p5,p8];
 *  x2_6_5_3:   x2_6_5_1[h9,h10,h1] += 1/2 * t_vo[p5,h1] *
 * x2_6_5_3_1[h9,h10,p5];
 *  x2_6_5:     x2_6_1[h9,p3,h1] += -1/2 * t_vo[p3,h10] * x2_6_5_1[h9,h10,h1];
 *  x2_6_6_1:   x2_6_6_1[p3,h9,p5] += 1 * x_v[p7] * v[h9,p3,p5,p7];
 *  x2_6_6:     x2_6_1[h9,p3,h1] += 1/2 * t_vo[p5,h1] * x2_6_6_1[p3,h9,p5];
 *  x2_6_7_1:   x2_6_7_1[h6,h9,p5] += 1 * x_v[p8] * v[h6,h9,p5,p8];
 *  x2_6_7:     x2_6_1[h9,p3,h1] += -1/2 * t_vvoo[p3,p5,h1,h6] *
 * x2_6_7_1[h6,h9,p5];
 *  x2_6:       i0[p3,p4,h1] += -2 * t_vo[p3,h9] * x2_6_1[h9,p4,h1];
 *  x2_7_1:     x2_7_1[p3,p4,p5] += 1 * x_v[p6] * v[p3,p4,p5,p6];
 *  x2_7:       i0[p3,p4,h1] += 1 * t_vo[p5,h1] * x2_7_1[p3,p4,p5];
 *  x2_8_1_1:   x2_8_1_1[h5,p9] += 1 * f[h5,p9];
 *  x2_8_1_2:   x2_8_1_1[h5,p9] += -1 * t_vo[p6,h7] * v[h5,h7,p6,p9];
 *  x2_8_1:     x2_8_1[h5] += 1 * x_v[p9] * x2_8_1_1[h5,p9];
 *  x2_8_2:     x2_8_1[h5] += -1/2 * x_vvo[p7,p8,h6] * v[h5,h6,p7,p8];
 *  x2_8:       i0[p3,p4,h1] += -1 * t_vvoo[p3,p4,h1,h5] * x2_8_1[h5];
 *  x2_9_1:     x2_9_1[h5,h6,h2] += 1/2 * x_v[p7] * v[h5,h6,h2,p7];
 *  x2_9_2:     x2_9_1[h5,h6,h1] += 1/4 * x_vvo[p7,p8,h1] * v[h5,h6,p7,p8];
 *  x2_9_3_1:   x2_9_3_1[h5,h6,p7] += 1 * x_v[p8] * v[h5,h6,p7,p8];
 *  x2_9_3:     x2_9_1[h5,h6,h1] += 1/2 * t_vo[p7,h1] * x2_9_3_1[h5,h6,p7];
 *  x2_9:       i0[p3,p4,h1] += 1 * t_vvoo[p3,p4,h5,h6] * x2_9_1[h5,h6,h1];
 *  x2_10_1:    x2_10_1[p3,h6,p5] += 1 * x_v[p7] * v[h6,p3,p5,p7];
 *  x2_10_2:    x2_10_1[p3,h6,p5] += -1 * x_vvo[p3,p8,h7] * v[h6,h7,p5,p8];
 *  x2_10:      i0[p3,p4,h1] += 1 * t_vvoo[p3,p5,h1,h6] * x2_10_1[p4,h6,p5];
 *
 *  }
 */

extern "C" {
void eaccsd_x2_1_(Integer *d_x_v, Integer *k_x_v_offset, Integer *d_v,
                  Integer *k_v_offset, Integer *d_i0, Integer *k_i0_offset);
void eaccsd_x2_2_1_(Integer *d_f, Integer *k_f_offset, Integer *d_x2_2_1,
                    Integer *k_x2_2_1_offset);
void eaccsd_x2_2_2_1_(Integer *d_f, Integer *k_f_offset, Integer *d_x2_2_2_1,
                      Integer *k_x2_2_2_1_offset);
void eaccsd_x2_2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                      Integer *k_v_offset, Integer *d_x2_2_2_1,
                      Integer *k_x2_2_2_1_offset);
void eaccsd_x2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                    Integer *d_x2_2_2_1, Integer *k_x2_2_2_1_offset,
                    Integer *d_x2_2_1, Integer *k_x2_2_1_offset);
void eaccsd_x2_2_3_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                    Integer *k_v_offset, Integer *d_x2_2_1,
                    Integer *k_x2_2_1_offset);
void eaccsd_x2_2_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset, Integer *d_v,
                    Integer *k_v_offset, Integer *d_x2_2_1,
                    Integer *k_x2_2_1_offset);
void eaccsd_x2_2_(Integer *d_x_vvo, Integer *k_x_vvo_offset, Integer *d_x2_2_1,
                  Integer *k_x2_2_1_offset, Integer *d_i0,
                  Integer *k_i0_offset);
void eaccsd_x2_3_1_(Integer *d_f, Integer *k_f_offset, Integer *d_x2_3_1,
                    Integer *k_x2_3_1_offset);
void eaccsd_x2_3_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                    Integer *k_v_offset, Integer *d_x2_3_1,
                    Integer *k_x2_3_1_offset);
void eaccsd_x2_3_3_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset, Integer *d_v,
                    Integer *k_v_offset, Integer *d_x2_3_1,
                    Integer *k_x2_3_1_offset);
void eaccsd_x2_3_(Integer *d_x_vvo, Integer *k_x_vvo_offset, Integer *d_x2_3_1,
                  Integer *k_x2_3_1_offset, Integer *d_i0,
                  Integer *k_i0_offset);
void eaccsd_x2_4_1_(Integer *d_v, Integer *k_v_offset, Integer *d_x2_4_1,
                    Integer *k_x2_4_1_offset);
void eaccsd_x2_4_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                    Integer *k_v_offset, Integer *d_x2_4_1,
                    Integer *k_x2_4_1_offset);
void eaccsd_x2_4_(Integer *d_x_vvo, Integer *k_x_vvo_offset, Integer *d_x2_4_1,
                  Integer *k_x2_4_1_offset, Integer *d_i0,
                  Integer *k_i0_offset);
void eaccsd_x2_5_(Integer *d_x_vvo, Integer *k_x_vvo_offset, Integer *d_v,
                  Integer *k_v_offset, Integer *d_i0, Integer *k_i0_offset);
void eaccsd_x2_6_1_(Integer *d_x_v, Integer *k_x_v_offset, Integer *d_v,
                    Integer *k_v_offset, Integer *d_x2_6_1,
                    Integer *k_x2_6_1_offset);
void eaccsd_x2_6_2_1_(Integer *d_f, Integer *k_f_offset, Integer *d_x2_6_2_1,
                      Integer *k_x2_6_2_1_offset);
void eaccsd_x2_6_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                      Integer *k_v_offset, Integer *d_x2_6_2_1,
                      Integer *k_x2_6_2_1_offset);
void eaccsd_x2_6_2_(Integer *d_x_vvo, Integer *k_x_vvo_offset,
                    Integer *d_x2_6_2_1, Integer *k_x2_6_2_1_offset,
                    Integer *d_x2_6_1, Integer *k_x2_6_1_offset);
void eaccsd_x2_6_3_1_(Integer *d_v, Integer *k_v_offset, Integer *d_x2_6_3_1,
                      Integer *k_x2_6_3_1_offset);
void eaccsd_x2_6_3_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                      Integer *k_v_offset, Integer *d_x2_6_3_1,
                      Integer *k_x2_6_3_1_offset);
void eaccsd_x2_6_3_(Integer *d_x_vvo, Integer *k_x_vvo_offset,
                    Integer *d_x2_6_3_1, Integer *k_x2_6_3_1_offset,
                    Integer *d_x2_6_1, Integer *k_x2_6_1_offset);
void eaccsd_x2_6_4_(Integer *d_x_vvo, Integer *k_x_vvo_offset, Integer *d_v,
                    Integer *k_v_offset, Integer *d_x2_6_1,
                    Integer *k_x2_6_1_offset);
void eaccsd_x2_6_5_1_(Integer *d_x_v, Integer *k_x_v_offset, Integer *d_v,
                      Integer *k_v_offset, Integer *d_x2_6_5_1,
                      Integer *k_x2_6_5_1_offset);
void eaccsd_x2_6_5_2_(Integer *d_x_vvo, Integer *k_x_vvo_offset, Integer *d_v,
                      Integer *k_v_offset, Integer *d_x2_6_5_1,
                      Integer *k_x2_6_5_1_offset);
void eaccsd_x2_6_5_3_1_(Integer *d_x_v, Integer *k_x_v_offset, Integer *d_v,
                        Integer *k_v_offset, Integer *d_x2_6_5_3_1,
                        Integer *k_x2_6_5_3_1_offset);
void eaccsd_x2_6_5_3_(Integer *d_t_vo, Integer *k_t_vo_offset,
                      Integer *d_x2_6_5_3_1, Integer *k_x2_6_5_3_1_offset,
                      Integer *d_x2_6_5_1, Integer *k_x2_6_5_1_offset);
void eaccsd_x2_6_5_(Integer *d_t_vo, Integer *k_t_vo_offset,
                    Integer *d_x2_6_5_1, Integer *k_x2_6_5_1_offset,
                    Integer *d_x2_6_1, Integer *k_x2_6_1_offset);
void eaccsd_x2_6_6_1_(Integer *d_x_v, Integer *k_x_v_offset, Integer *d_v,
                      Integer *k_v_offset, Integer *d_x2_6_6_1,
                      Integer *k_x2_6_6_1_offset);
void eaccsd_x2_6_6_(Integer *d_t_vo, Integer *k_t_vo_offset,
                    Integer *d_x2_6_6_1, Integer *k_x2_6_6_1_offset,
                    Integer *d_x2_6_1, Integer *k_x2_6_1_offset);
void eaccsd_x2_6_7_1_(Integer *d_x_v, Integer *k_x_v_offset, Integer *d_v,
                      Integer *k_v_offset, Integer *d_x2_6_7_1,
                      Integer *k_x2_6_7_1_offset);
void eaccsd_x2_6_7_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                    Integer *d_x2_6_7_1, Integer *k_x2_6_7_1_offset,
                    Integer *d_x2_6_1, Integer *k_x2_6_1_offset);
void eaccsd_x2_6_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_x2_6_1,
                  Integer *k_x2_6_1_offset, Integer *d_i0,
                  Integer *k_i0_offset);
void eaccsd_x2_7_1_(Integer *d_x_v, Integer *k_x_v_offset, Integer *d_v,
                    Integer *k_v_offset, Integer *d_x2_7_1,
                    Integer *k_x2_7_1_offset);
void eaccsd_x2_7_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_x2_7_1,
                  Integer *k_x2_7_1_offset, Integer *d_i0,
                  Integer *k_i0_offset);
void eaccsd_x2_8_1_1_(Integer *d_f, Integer *k_f_offset, Integer *d_x2_8_1_1,
                      Integer *k_x2_8_1_1_offset);
void eaccsd_x2_8_1_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                      Integer *k_v_offset, Integer *d_x2_8_1_1,
                      Integer *k_x2_8_1_1_offset);
void eaccsd_x2_8_1_(Integer *d_x_v, Integer *k_x_v_offset, Integer *d_x2_8_1_1,
                    Integer *k_x2_8_1_1_offset, Integer *d_x2_8_1,
                    Integer *k_x2_8_1_offset);
void eaccsd_x2_8_2_(Integer *d_x_vvo, Integer *k_x_vvo_offset, Integer *d_v,
                    Integer *k_v_offset, Integer *d_x2_8_1,
                    Integer *k_x2_8_1_offset);
void eaccsd_x2_8_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                  Integer *d_x2_8_1, Integer *k_x2_8_1_offset, Integer *d_i0,
                  Integer *k_i0_offset);
void eaccsd_x2_9_1_(Integer *d_x_v, Integer *k_x_v_offset, Integer *d_v,
                    Integer *k_v_offset, Integer *d_x2_9_1,
                    Integer *k_x2_9_1_offset);
void eaccsd_x2_9_2_(Integer *d_x_vvo, Integer *k_x_vvo_offset, Integer *d_v,
                    Integer *k_v_offset, Integer *d_x2_9_1,
                    Integer *k_x2_9_1_offset);
void eaccsd_x2_9_3_1_(Integer *d_x_v, Integer *k_x_v_offset, Integer *d_v,
                      Integer *k_v_offset, Integer *d_x2_9_3_1,
                      Integer *k_x2_9_3_1_offset);
void eaccsd_x2_9_3_(Integer *d_t_vo, Integer *k_t_vo_offset,
                    Integer *d_x2_9_3_1, Integer *k_x2_9_3_1_offset,
                    Integer *d_x2_9_1, Integer *k_x2_9_1_offset);
void eaccsd_x2_9_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                  Integer *d_x2_9_1, Integer *k_x2_9_1_offset, Integer *d_i0,
                  Integer *k_i0_offset);
void eaccsd_x2_10_1_(Integer *d_x_v, Integer *k_x_v_offset, Integer *d_v,
                     Integer *k_v_offset, Integer *d_x2_10_1,
                     Integer *k_x2_10_1_offset);
void eaccsd_x2_10_2_(Integer *d_x_vvo, Integer *k_x_vvo_offset, Integer *d_v,
                     Integer *k_v_offset, Integer *d_x2_10_1,
                     Integer *k_x2_10_1_offset);
void eaccsd_x2_10_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                   Integer *d_x2_10_1, Integer *k_x2_10_1_offset, Integer *d_i0,
                   Integer *k_i0_offset);

void offset_eaccsd_x2_2_1_(Integer *l_x2_2_1_offset, Integer *k_x2_2_1_offset,
                           Integer *size_x2_2_1);
void offset_eaccsd_x2_2_2_1_(Integer *l_x2_2_2_1_offset,
                             Integer *k_x2_2_2_1_offset,
                             Integer *size_x2_2_2_1);
void offset_eaccsd_x2_3_1_(Integer *l_x2_3_1_offset, Integer *k_x2_3_1_offset,
                           Integer *size_x2_3_1);
void offset_eaccsd_x2_4_1_(Integer *l_x2_4_1_offset, Integer *k_x2_4_1_offset,
                           Integer *size_x2_4_1);
void offset_eaccsd_x2_6_1_(Integer *l_x2_6_1_offset, Integer *k_x2_6_1_offset,
                           Integer *size_x2_6_1);
void offset_eaccsd_x2_6_2_1_(Integer *l_x2_6_2_1_offset,
                             Integer *k_x2_6_2_1_offset,
                             Integer *size_x2_6_2_1);
void offset_eaccsd_x2_6_3_1_(Integer *l_x2_6_3_1_offset,
                             Integer *k_x2_6_3_1_offset,
                             Integer *size_x2_6_3_1);
void offset_eaccsd_x2_6_5_1_(Integer *l_x2_6_5_1_offset,
                             Integer *k_x2_6_5_1_offset,
                             Integer *size_x2_6_5_1);
void offset_eaccsd_x2_6_5_3_1_(Integer *l_x2_6_5_3_1_offset,
                               Integer *k_x2_6_5_3_1_offset,
                               Integer *size_x2_6_5_3_1);
void offset_eaccsd_x2_6_6_1_(Integer *l_x2_6_6_1_offset,
                             Integer *k_x2_6_6_1_offset,
                             Integer *size_x2_6_6_1);
void offset_eaccsd_x2_6_7_1_(Integer *l_x2_6_7_1_offset,
                             Integer *k_x2_6_7_1_offset,
                             Integer *size_x2_6_7_1);
void offset_eaccsd_x2_7_1_(Integer *l_x2_7_1_offset, Integer *k_x2_7_1_offset,
                           Integer *size_x2_7_1);
void offset_eaccsd_x2_8_1_1_(Integer *l_x2_8_1_1_offset,
                             Integer *k_x2_8_1_1_offset,
                             Integer *size_x2_8_1_1);
void offset_eaccsd_x2_8_1_(Integer *l_x2_8_1_offset, Integer *k_x2_8_1_offset,
                           Integer *size_x2_8_1);
void offset_eaccsd_x2_9_1_(Integer *l_x2_9_1_offset, Integer *k_x2_9_1_offset,
                           Integer *size_x2_9_1);
void offset_eaccsd_x2_9_3_1_(Integer *l_x2_9_3_1_offset,
                             Integer *k_x2_9_3_1_offset,
                             Integer *size_x2_9_3_1);
void offset_eaccsd_x2_10_1_(Integer *l_x2_10_1_offset,
                            Integer *k_x2_10_1_offset, Integer *size_x2_10_1);
}

namespace tamm {

extern "C" {
void eaccsd_x2_cxx_(Fint *d_f1, Fint *d_i0, Fint *d_t1, Fint *d_t2, Fint *d_v2,
                    Fint *d_x1, Fint *d_x2, Fint *k_f1_offset,
                    Fint *k_i0_offset, Fint *k_t1_offset, Fint *k_t2_offset,
                    Fint *k_v2_offset, Fint *k_x1_offset, Fint *k_x2_offset) {
  static bool set_x2 = true;

  Assignment op_x2_2_1;
  Assignment op_x2_2_2_1;
  Assignment op_x2_3_1;
  Assignment op_x2_4_1;
  Assignment op_x2_6_2_1;
  Assignment op_x2_6_3_1;
  Assignment op_x2_8_1_1;
  Multiplication op_x2_1;
  Multiplication op_x2_2_2_2;
  Multiplication op_x2_2_2;
  Multiplication op_x2_2_3;
  Multiplication op_x2_2_4;
  Multiplication op_x2_2;
  Multiplication op_x2_3_2;
  Multiplication op_x2_3_3;
  Multiplication op_x2_3;
  Multiplication op_x2_4_2;
  Multiplication op_x2_4;
  Multiplication op_x2_5;
  Multiplication op_x2_6_1;
  Multiplication op_x2_6_2_2;
  Multiplication op_x2_6_2;
  Multiplication op_x2_6_3_2;
  Multiplication op_x2_6_3;
  Multiplication op_x2_6_4;
  Multiplication op_x2_6_5_1;
  Multiplication op_x2_6_5_2;
  Multiplication op_x2_6_5_3_1;
  Multiplication op_x2_6_5_3;
  Multiplication op_x2_6_5;
  Multiplication op_x2_6_6_1;
  Multiplication op_x2_6_6;
  Multiplication op_x2_6_7_1;
  Multiplication op_x2_6_7;
  Multiplication op_x2_6;
  Multiplication op_x2_7_1;
  Multiplication op_x2_7;
  Multiplication op_x2_8_1_2;
  Multiplication op_x2_8_1;
  Multiplication op_x2_8_2;
  Multiplication op_x2_8;
  Multiplication op_x2_9_1;
  Multiplication op_x2_9_2;
  Multiplication op_x2_9_3_1;
  Multiplication op_x2_9_3;
  Multiplication op_x2_9;
  Multiplication op_x2_10_1;
  Multiplication op_x2_10_2;
  Multiplication op_x2_10;

  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_x2) {
    eaccsd_x2_equations(&eqs);
    set_x2 = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector<Operation> ops;
  tensors_and_ops(&eqs, &tensors, &ops);

  Tensor *i0 = &tensors["i0"];
  Tensor *x_v = &tensors["x_v"];
  Tensor *v = &tensors["v"];
  Tensor *x_vvo = &tensors["x_vvo"];
  Tensor *f = &tensors["f"];
  Tensor *t_vo = &tensors["t_vo"];
  Tensor *t_vvoo = &tensors["t_vvoo"];
  Tensor *x2_10_1 = &tensors["x2_10_1"];
  Tensor *x2_2_1 = &tensors["x2_2_1"];
  Tensor *x2_6_1 = &tensors["x2_6_1"];
  Tensor *x2_6_5_1 = &tensors["x2_6_5_1"];
  Tensor *x2_3_1 = &tensors["x2_3_1"];
  Tensor *x2_6_7_1 = &tensors["x2_6_7_1"];
  Tensor *x2_8_1 = &tensors["x2_8_1"];
  Tensor *x2_7_1 = &tensors["x2_7_1"];
  Tensor *x2_6_2_1 = &tensors["x2_6_2_1"];
  Tensor *x2_8_1_1 = &tensors["x2_8_1_1"];
  Tensor *x2_9_1 = &tensors["x2_9_1"];
  Tensor *x2_6_6_1 = &tensors["x2_6_6_1"];
  Tensor *x2_9_3_1 = &tensors["x2_9_3_1"];
  Tensor *x2_4_1 = &tensors["x2_4_1"];
  Tensor *x2_6_5_3_1 = &tensors["x2_6_5_3_1"];
  Tensor *x2_6_3_1 = &tensors["x2_6_3_1"];
  Tensor *x2_2_2_1 = &tensors["x2_2_2_1"];

  /* ----- Insert attach code ------ */
  v->set_dist(idist);

  i0->attach(*k_i0_offset, 0, *d_i0);
  x_v->attach(*k_x1_offset, 0, *d_x1);
  f->attach(*k_f1_offset, 0, *d_f1);
  t_vo->attach(*k_t1_offset, 0, *d_t1);
  v->attach(*k_v2_offset, 0, *d_v2);
  x_vvo->attach(*k_x2_offset, 0, *d_x2);
  t_vvoo->attach(*k_t2_offset, 0, *d_t2);

  i0->set_irrep(Variables::irrep_x());
  x_v->set_irrep(Variables::irrep_x());
  x_vvo->set_irrep(Variables::irrep_x());
  x2_10_1->set_irrep(Variables::irrep_x());
  x2_6_1->set_irrep(Variables::irrep_x());
  x2_6_5_1->set_irrep(Variables::irrep_x());
  x2_6_7_1->set_irrep(Variables::irrep_x());
  x2_8_1->set_irrep(Variables::irrep_x());
  x2_7_1->set_irrep(Variables::irrep_x());
  x2_9_1->set_irrep(Variables::irrep_x());
  x2_6_6_1->set_irrep(Variables::irrep_x());
  x2_9_3_1->set_irrep(Variables::irrep_x());
  x2_6_5_3_1->set_irrep(Variables::irrep_x());

  for(int i=0; i<eqs.op_entries.size(); i++) {
      switch(eqs.op_entries[i].optype) {
      case OpTypeAdd:
      {
          Tensor *t_alhs = &tensors[eqs.tensor_entries.at(eqs.op_entries[i].add.tc).name];
          t_alhs->set_irrep((&tensors[eqs.tensor_entries.at(eqs.op_entries[i].add.ta).name])->irrep());
          break;
      }
      case OpTypeMult:
       {   Tensor *t_mlhs = &tensors[eqs.tensor_entries.at(eqs.op_entries[i].mult.tc).name];
          t_mlhs->set_irrep((&tensors[eqs.tensor_entries.at(eqs.op_entries[i].mult.ta).name])->irrep() ^ 
                          (&tensors[eqs.tensor_entries.at(eqs.op_entries[i].mult.tb).name])->irrep());
          break;
       }
      }
  }

#if 1
  schedule_levels(&tensors, &ops);
#else
  op_x2_1 = ops[0].mult;
  op_x2_2_1 = ops[1].add;
  op_x2_2_2_1 = ops[2].add;
  op_x2_2_2_2 = ops[3].mult;
  op_x2_2_2 = ops[4].mult;
  op_x2_2_3 = ops[5].mult;
  op_x2_2_4 = ops[6].mult;
  op_x2_2 = ops[7].mult;
  op_x2_3_1 = ops[8].add;
  op_x2_3_2 = ops[9].mult;
  op_x2_3_3 = ops[10].mult;
  op_x2_3 = ops[11].mult;
  op_x2_4_1 = ops[12].add;
  op_x2_4_2 = ops[13].mult;
  op_x2_4 = ops[14].mult;
  op_x2_5 = ops[15].mult;
  op_x2_6_1 = ops[16].mult;
  op_x2_6_2_1 = ops[17].add;
  op_x2_6_2_2 = ops[18].mult;
  op_x2_6_2 = ops[19].mult;
  op_x2_6_3_1 = ops[20].add;
  op_x2_6_3_2 = ops[21].mult;
  op_x2_6_3 = ops[22].mult;
  op_x2_6_4 = ops[23].mult;
  op_x2_6_5_1 = ops[24].mult;
  op_x2_6_5_2 = ops[25].mult;
  op_x2_6_5_3_1 = ops[26].mult;
  op_x2_6_5_3 = ops[27].mult;
  op_x2_6_5 = ops[28].mult;
  op_x2_6_6_1 = ops[29].mult;
  op_x2_6_6 = ops[30].mult;
  op_x2_6_7_1 = ops[31].mult;
  op_x2_6_7 = ops[32].mult;
  op_x2_6 = ops[33].mult;
  op_x2_7_1 = ops[34].mult;
  op_x2_7 = ops[35].mult;
  op_x2_8_1_1 = ops[36].add;
  op_x2_8_1_2 = ops[37].mult;
  op_x2_8_1 = ops[38].mult;
  op_x2_8_2 = ops[39].mult;
  op_x2_8 = ops[40].mult;
  op_x2_9_1 = ops[41].mult;
  op_x2_9_2 = ops[42].mult;
  op_x2_9_3_1 = ops[43].mult;
  op_x2_9_3 = ops[44].mult;
  op_x2_9 = ops[45].mult;
  op_x2_10_1 = ops[46].mult;
  op_x2_10_2 = ops[47].mult;
  op_x2_10 = ops[48].mult;

  CorFortran(1, &op_x2_1, eaccsd_x2_1_);
  CorFortran(1, &x2_2_1, offset_eaccsd_x2_2_1_);
  CorFortran(1, &op_x2_2_1, eaccsd_x2_2_1_);
  CorFortran(1, &x2_2_2_1, offset_eaccsd_x2_2_2_1_);
  CorFortran(1, &op_x2_2_2_1, eaccsd_x2_2_2_1_);
  CorFortran(1, &op_x2_2_2_2, eaccsd_x2_2_2_2_);
  CorFortran(1, &op_x2_2_2, eaccsd_x2_2_2_);
  destroy(x2_2_2_1);
  CorFortran(1, &op_x2_2_3, eaccsd_x2_2_3_);
  CorFortran(1, &op_x2_2_4, eaccsd_x2_2_4_);
  CorFortran(1, &op_x2_2, eaccsd_x2_2_);
  destroy(x2_2_1);
  CorFortran(1, &x2_3_1, offset_eaccsd_x2_3_1_);
  CorFortran(1, &op_x2_3_1, eaccsd_x2_3_1_);
  CorFortran(1, &op_x2_3_2, eaccsd_x2_3_2_);
  CorFortran(1, &op_x2_3_3, eaccsd_x2_3_3_);
  CorFortran(1, &op_x2_3, eaccsd_x2_3_);
  destroy(x2_3_1);
  CorFortran(1, &x2_4_1, offset_eaccsd_x2_4_1_);
  CorFortran(1, &op_x2_4_1, eaccsd_x2_4_1_);
  CorFortran(1, &op_x2_4_2, eaccsd_x2_4_2_);
  CorFortran(1, &op_x2_4, eaccsd_x2_4_);
  destroy(x2_4_1);
  CorFortran(1, &op_x2_5, eaccsd_x2_5_);
  CorFortran(1, &x2_6_1, offset_eaccsd_x2_6_1_);
  CorFortran(1, &op_x2_6_1, eaccsd_x2_6_1_);
  CorFortran(1, &x2_6_2_1, offset_eaccsd_x2_6_2_1_);
  CorFortran(1, &op_x2_6_2_1, eaccsd_x2_6_2_1_);
  CorFortran(1, &op_x2_6_2_2, eaccsd_x2_6_2_2_);
  CorFortran(1, &op_x2_6_2, eaccsd_x2_6_2_);
  destroy(x2_6_2_1);
  CorFortran(1, &x2_6_3_1, offset_eaccsd_x2_6_3_1_);
  CorFortran(1, &op_x2_6_3_1, eaccsd_x2_6_3_1_);
  CorFortran(1, &op_x2_6_3_2, eaccsd_x2_6_3_2_);
  CorFortran(1, &op_x2_6_3, eaccsd_x2_6_3_);
  destroy(x2_6_3_1);
  CorFortran(1, &op_x2_6_4, eaccsd_x2_6_4_);
  CorFortran(1, &x2_6_5_1, offset_eaccsd_x2_6_5_1_);
  CorFortran(1, &op_x2_6_5_1, eaccsd_x2_6_5_1_);
  CorFortran(1, &op_x2_6_5_2, eaccsd_x2_6_5_2_);
  CorFortran(1, &x2_6_5_3_1, offset_eaccsd_x2_6_5_3_1_);
  CorFortran(1, &op_x2_6_5_3_1, eaccsd_x2_6_5_3_1_);
  CorFortran(1, &op_x2_6_5_3, eaccsd_x2_6_5_3_);
  destroy(x2_6_5_3_1);
  CorFortran(1, &op_x2_6_5, eaccsd_x2_6_5_);
  destroy(x2_6_5_1);
  CorFortran(1, &x2_6_6_1, offset_eaccsd_x2_6_6_1_);
  CorFortran(1, &op_x2_6_6_1, eaccsd_x2_6_6_1_);
  CorFortran(1, &op_x2_6_6, eaccsd_x2_6_6_);
  destroy(x2_6_6_1);
  CorFortran(1, &x2_6_7_1, offset_eaccsd_x2_6_7_1_);
  CorFortran(1, &op_x2_6_7_1, eaccsd_x2_6_7_1_);
  CorFortran(1, &op_x2_6_7, eaccsd_x2_6_7_);  // @bug Some problem in op_x2_6_7
                                             //   when executed in Fortran when
                                             //   rest is executed with C
  destroy(x2_6_7_1);
  CorFortran(1, &op_x2_6, eaccsd_x2_6_);
  destroy(x2_6_1);
  CorFortran(1, &x2_7_1, offset_eaccsd_x2_7_1_);
  CorFortran(1, &op_x2_7_1, eaccsd_x2_7_1_);
  CorFortran(1, &op_x2_7, eaccsd_x2_7_);
  destroy(x2_7_1);
  CorFortran(1, &x2_8_1_1, offset_eaccsd_x2_8_1_1_);
  CorFortran(1, &op_x2_8_1_1, eaccsd_x2_8_1_1_);
  CorFortran(1, &op_x2_8_1_2, eaccsd_x2_8_1_2_);
  CorFortran(1, &x2_8_1, offset_eaccsd_x2_8_1_);
  CorFortran(1, &op_x2_8_1, eaccsd_x2_8_1_);
  CorFortran(1, &op_x2_8_2, eaccsd_x2_8_2_);
  CorFortran(1, &op_x2_8, eaccsd_x2_8_);
  destroy(x2_8_1);
  destroy(x2_8_1_1);
  CorFortran(1, &x2_9_1, offset_eaccsd_x2_9_1_);
  CorFortran(1, &op_x2_9_1, eaccsd_x2_9_1_);
  CorFortran(1, &op_x2_9_2, eaccsd_x2_9_2_);
  CorFortran(1, &x2_9_3_1, offset_eaccsd_x2_9_3_1_);
  CorFortran(1, &op_x2_9_3_1, eaccsd_x2_9_3_1_);
  CorFortran(1, &op_x2_9_3, eaccsd_x2_9_3_);
  destroy(x2_9_3_1);
  CorFortran(1, &op_x2_9, eaccsd_x2_9_);
  destroy(x2_9_1);
  CorFortran(1, &x2_10_1, offset_eaccsd_x2_10_1_);
  CorFortran(1, &op_x2_10_1, eaccsd_x2_10_1_);
  CorFortran(1, &op_x2_10_2, eaccsd_x2_10_2_);
  CorFortran(1, &op_x2_10, eaccsd_x2_10_);
  destroy(x2_10_1);
#endif  // Use c scheduler

  /* ----- Insert detach code ------ */
  i0->detach();
  x_v->detach();
  f->detach();
  t_vo->detach();
  v->detach();
  x_vvo->detach();
  t_vvoo->detach();
}
}  // extern C
};  // namespace tamm
