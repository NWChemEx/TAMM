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
 *  t2 {
 *
 *  index h1,h2,h3,h4,h5,h6,h7,h8,h9,h10 = O;
 *  index p1,p2,p3,p4,p5,p6 = V;
 *
 *  array i0[V,V][O,O];
 *  array v[N,N][N,N]: irrep_v;
 *  array t_vo[V][O]: irrep_t;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array f[N][N]: irrep_f;
 *  array t2_3_1[V,V][O,V];
 *  array t2_2_2_3[O,V][O,V];
 *  array t2_2_2_1[O,O][O,O];
 *  array t2_2_2_2_1[O,O][O,V];
 *  array t2_2_1[O,V][O,O];
 *
 *  t2_1:       i0[p3,p4,h1,h2] += 1 * v[p3,p4,h1,h2];
 *  t2_2_1:     t2_2_1[h10,p3,h1,h2] += 1 * v[h10,p3,h1,h2];
 *  t2_2_2_1:   t2_2_2_1[h8,h10,h1,h2] += 1 * v[h8,h10,h1,h2];
 *  t2_2_2_2_1: t2_2_2_2_1[h8,h10,h1,p5] += 1 * v[h8,h10,h1,p5];
 *  t2_2_2_2_2: t2_2_2_2_1[h8,h10,h1,p5] += -1/2 * t_vo[p6,h1] *
 * v[h8,h10,p5,p6];
 *  t2_2_2_2:   t2_2_2_1[h8,h10,h1,h2] += -1 * t_vo[p5,h1] *
 * t2_2_2_2_1[h8,h10,h2,p5];
 *  t2_2_2:     t2_2_1[h10,p3,h1,h2] += 1/2 * t_vo[p3,h8] *
 * t2_2_2_1[h8,h10,h1,h2];
 *  t2_2_2_3:   t2_2_2_3[h10,p3,h1,p5] += 1 * v[h10,p3,h1,p5];
 *  t2_2_2_4:   t2_2_2_3[h10,p3,h1,p5] += -1/2 * t_vo[p6,h1] * v[h10,p3,p5,p6];
 *  t2_2_3:     t2_2_1[h10,p3,h1,h2] += -1 * t_vo[p5,h1] *
 * t2_2_2_3[h10,p3,h2,p5];
 *  t2_2:       i0[p3,p4,h1,h2] += -1 * t_vo[p3,h10] * t2_2_1[h10,p4,h1,h2];
 *  t2_3_1:     t2_3_1[p3,p4,h1,p5] += 1 * v[p3,p4,h1,p5];
 *  t2_3_2:     t2_3_1[p3,p4,h1,p5] += -1/2 * t_vo[p6,h1] * v[p3,p4,p5,p6];
 *  t2_3:       i0[p3,p4,h1,h2] += -1 * t_vo[p5,h1] * t2_3_1[p3,p4,h2,p5];
 *  t2_4:       i0[p3,p4,h1,h2] += -1 * t_vvoo[p3,p4,h1,h5] * f[h5,h2];
 *  t2_5:       i0[p3,p4,h1,h2] += 1 * t_vvoo[p3,p5,h1,h2] * f[p4,p5];
 *
 *  }
 */

extern "C" {
void cc2_t2_1_(F77Integer *d_v, F77Integer *k_v_offset, F77Integer *d_i0,
               F77Integer *k_i0_offset);
void cc2_t2_2_1_(F77Integer *d_v, F77Integer *k_v_offset, F77Integer *d_t2_2_1,
                 F77Integer *k_t2_2_1_offset);
void cc2_t2_2_2_1_(F77Integer *d_v, F77Integer *k_v_offset, F77Integer *d_t2_2_2_1,
                   F77Integer *k_t2_2_2_1_offset);
void cc2_t2_2_2_2_1_(F77Integer *d_v, F77Integer *k_v_offset, F77Integer *d_t2_2_2_2_1,
                     F77Integer *k_t2_2_2_2_1_offset);
void cc2_t2_2_2_2_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                     F77Integer *k_v_offset, F77Integer *d_t2_2_2_2_1,
                     F77Integer *k_t2_2_2_2_1_offset);
void cc2_t2_2_2_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset,
                   F77Integer *d_t2_2_2_2_1, F77Integer *k_t2_2_2_2_1_offset,
                   F77Integer *d_t2_2_2_1, F77Integer *k_t2_2_2_1_offset);
void cc2_t2_2_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_t2_2_2_1,
                 F77Integer *k_t2_2_2_1_offset, F77Integer *d_t2_2_1,
                 F77Integer *k_t2_2_1_offset);
void cc2_t2_2_2_3_(F77Integer *d_v, F77Integer *k_v_offset, F77Integer *d_t2_2_2_3,
                   F77Integer *k_t2_2_2_3_offset);
void cc2_t2_2_2_4_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                   F77Integer *k_v_offset, F77Integer *d_t2_2_2_3,
                   F77Integer *k_t2_2_2_3_offset);
void cc2_t2_2_3_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_t2_2_2_3,
                 F77Integer *k_t2_2_2_3_offset, F77Integer *d_t2_2_1,
                 F77Integer *k_t2_2_1_offset);
void cc2_t2_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_t2_2_1,
               F77Integer *k_t2_2_1_offset, F77Integer *d_i0, F77Integer *k_i0_offset);
void cc2_t2_3_1_(F77Integer *d_v, F77Integer *k_v_offset, F77Integer *d_t2_3_1,
                 F77Integer *k_t2_3_1_offset);
void cc2_t2_3_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                 F77Integer *k_v_offset, F77Integer *d_t2_3_1,
                 F77Integer *k_t2_3_1_offset);
void cc2_t2_3_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_t2_3_1,
               F77Integer *k_t2_3_1_offset, F77Integer *d_i0, F77Integer *k_i0_offset);
void cc2_t2_4_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset, F77Integer *d_f,
               F77Integer *k_f_offset, F77Integer *d_i0, F77Integer *k_i0_offset);
void cc2_t2_5_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset, F77Integer *d_f,
               F77Integer *k_f_offset, F77Integer *d_i0, F77Integer *k_i0_offset);

void offset_cc2_t2_2_1_(F77Integer *l_t2_2_1_offset, F77Integer *k_t2_2_1_offset,
                        F77Integer *size_t2_2_1);
void offset_cc2_t2_2_2_1_(F77Integer *l_t2_2_2_1_offset,
                          F77Integer *k_t2_2_2_1_offset, F77Integer *size_t2_2_2_1);
void offset_cc2_t2_2_2_2_1_(F77Integer *l_t2_2_2_2_1_offset,
                            F77Integer *k_t2_2_2_2_1_offset,
                            F77Integer *size_t2_2_2_2_1);
void offset_cc2_t2_2_2_3_(F77Integer *l_t2_2_2_3_offset,
                          F77Integer *k_t2_2_2_3_offset, F77Integer *size_t2_2_2_3);
void offset_cc2_t2_3_1_(F77Integer *l_t2_3_1_offset, F77Integer *k_t2_3_1_offset,
                        F77Integer *size_t2_3_1);
}

namespace tamm {

extern "C" {
// void cc2_t2_cxx(Integer *d_i0,Integer *d_t_vvoo,Integer *d_f,Integer
// *d_t_vo,Integer *d_v,Integer *k_i0_offset,Integer *k_t_vvoo_offset,Integer
// *k_f_offset,Integer *k_t_vo_offset,Integer *k_v_offset){
void cc2_t2_cxx_(F77Integer *d_f, F77Integer *d_i0, F77Integer *d_t_vo,
                 F77Integer *d_t_vvoo, F77Integer *d_v, F77Integer *k_f_offset,
                 F77Integer *k_i0_offset, F77Integer *k_t_vo_offset,
                 F77Integer *k_t_vvoo_offset, F77Integer *k_v_offset) {
  static bool set_t2 = true;

  Assignment op_t2_1;
  Assignment op_t2_2_1;
  Assignment op_t2_2_2_1;
  Assignment op_t2_2_2_2_1;
  Assignment op_t2_2_2_3;
  Assignment op_t2_3_1;
  Multiplication op_t2_2_2_2_2;
  Multiplication op_t2_2_2_2;
  Multiplication op_t2_2_2;
  Multiplication op_t2_2_2_4;
  Multiplication op_t2_2_3;
  Multiplication op_t2_2;
  Multiplication op_t2_3_2;
  Multiplication op_t2_3;
  Multiplication op_t2_4;
  Multiplication op_t2_5;

  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_t2) {
    cc2_t2_equations(&eqs);
    set_t2 = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector<Operation> ops;
  tensors_and_ops(&eqs, &tensors, &ops);

  Tensor *i0 = &tensors["i0"];
  Tensor *v = &tensors["v"];
  Tensor *t_vo = &tensors["t_vo"];
  Tensor *t_vvoo = &tensors["t_vvoo"];
  Tensor *f = &tensors["f"];
  Tensor *t2_3_1 = &tensors["t2_3_1"];
  Tensor *t2_2_2_1 = &tensors["t2_2_2_1"];
  Tensor *t2_2_2_3 = &tensors["t2_2_2_3"];
  Tensor *t2_2_2_2_1 = &tensors["t2_2_2_2_1"];
  Tensor *t2_2_1 = &tensors["t2_2_1"];

  op_t2_1 = ops[0].add;
  op_t2_2_1 = ops[1].add;
  op_t2_2_2_1 = ops[2].add;
  op_t2_2_2_2_1 = ops[3].add;
  op_t2_2_2_2_2 = ops[4].mult;
  op_t2_2_2_2 = ops[5].mult;
  op_t2_2_2 = ops[6].mult;
  op_t2_2_2_3 = ops[7].add;
  op_t2_2_2_4 = ops[8].mult;
  op_t2_2_3 = ops[9].mult;
  op_t2_2 = ops[10].mult;
  op_t2_3_1 = ops[11].add;
  op_t2_3_2 = ops[12].mult;
  op_t2_3 = ops[13].mult;
  op_t2_4 = ops[14].mult;
  op_t2_5 = ops[15].mult;

  /* ----- Insert attach code ------ */
  v->set_dist(idist);
  t_vo->set_dist(dist_nwma);
  f->attach(*k_f_offset, 0, *d_f);
  i0->attach(*k_i0_offset, 0, *d_i0);
  t_vo->attach(*k_t_vo_offset, 0, *d_t_vo);
  t_vvoo->attach(*k_t_vvoo_offset, 0, *d_t_vvoo);
  v->attach(*k_v_offset, 0, *d_v);

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
  CorFortran(1, &op_t2_1, cc2_t2_1_);
  CorFortran(1, &op_t2_2_1, ofsset_cc2_t2_2_1_);
  CorFortran(1, &op_t2_2_1, cc2_t2_2_1_);
  CorFortran(1, &op_t2_2_2_1, ofsset_cc2_t2_2_2_1_);
  CorFortran(1, &op_t2_2_2_1, cc2_t2_2_2_1_);
  CorFortran(1, &op_t2_2_2_2_1, ofsset_cc2_t2_2_2_2_1_);
  CorFortran(1, &op_t2_2_2_2_1, cc2_t2_2_2_2_1_);
  CorFortran(1, &op_t2_2_2_2_2, cc2_t2_2_2_2_2_);
  CorFortran(1, &op_t2_2_2_2, cc2_t2_2_2_2_);
  destroy(t2_2_2_2_1);
  CorFortran(1, &op_t2_2_2, cc2_t2_2_2_);
  destroy(t2_2_2_1);
  CorFortran(1, &op_t2_2_2_3, ofsset_cc2_t2_2_2_3_);
  CorFortran(1, &op_t2_2_2_3, cc2_t2_2_2_3_);
  CorFortran(1, &op_t2_2_2_4, cc2_t2_2_2_4_);
  CorFortran(1, &op_t2_2_3, cc2_t2_2_3_);
  destroy(t2_2_2_3);
  CorFortran(1, &op_t2_2, cc2_t2_2_);
  destroy(t2_2_1);
  CorFortran(1, &op_t2_3_1, ofsset_cc2_t2_3_1_);
  CorFortran(1, &op_t2_3_1, cc2_t2_3_1_);
  CorFortran(1, &op_t2_3_2, cc2_t2_3_2_);
  CorFortran(1, &op_t2_3, cc2_t2_3_);
  destroy(t2_3_1);
  CorFortran(1, &op_t2_4, cc2_t2_4_);
  CorFortran(1, &op_t2_5, cc2_t2_5_);
#endif  // 1 -> do not use fortran

  /* ----- Insert detach code ------ */
  f->detach();
  i0->detach();
  t_vo->detach();
  t_vvoo->detach();
  v->detach();
}
}  // extern C
};  // namespace tamm
