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
 * i0 ( p2 h1 )_f + = 1 * f ( p2 h1 )_f DONE
 * i0 ( p2 h1 )_tf + = -1 * Sum ( h7 ) * t ( p2 h7 )_t * i1 ( h7 h1 )_f DONE
 *     i1 ( h7 h1 )_f + = 1 * f ( h7 h1 )_f DONE
 *     i1 ( h7 h1 )_ft + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i2 ( h7 p3 )_f DONE
 *         i2 ( h7 p3 )_f + = 1 * f ( h7 p3 )_f DONE
 *         i2 ( h7 p3 )_vt + = -1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h7 p3
 * p5 )_v         DONE
 *     i1 ( h7 h1 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 h7 h1 p4
 * )_v             NOPE
 *     i1 ( h7 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v (
 * h5 h7 p3 p4 )_v  NOPE
 * i0 ( p2 h1 )_tf + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i1 ( p2 p3 )_f DONE
 *     i1 ( p2 p3 )_f + = 1 * f ( p2 p3 )_f DONE
 *     i1 ( p2 p3 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 p2 p3 p4
 * )_v             NOPE
 * i0 ( p2 h1 )_vt + = -1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 p2 h1 p3 )_v
 * NOPE
 * i0 ( p2 h1 )_tf + = 1 * Sum ( p7 h8 ) * t ( p2 p7 h1 h8 )_t * i1 ( h8 p7 )_f
 * DONE
 *     i1 ( h8 p7 )_f + = 1 * f ( h8 p7 )_f DONE
 *     i1 ( h8 p7 )_vt + = 1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h8 p5 p7
 * )_v              NOPE
 * i0 ( p2 h1 )_vt + = -1/2 * Sum ( h4 h5 p3 ) * t ( p2 p3 h4 h5 )_t * i1 ( h4
 * h5 h1 p3 )_v     NOPE
 *     i1 ( h4 h5 h1 p3 )_v + = 1 * v ( h4 h5 h1 p3 )_v DONE
 *     i1 ( h4 h5 h1 p3 )_vt + = -1 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h4 h5 p3
 * p6 )_v          NOPE
 * i0 ( p2 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 p2
 * p3 p4 )_v      DONE
 */

/*
 *  t1 {
 *
 *  index h1,h2,h3,h4,h5,h6,h7,h8 = O;
 *  index p1,p2,p3,p4,p5,p6,p7 = V;
 *
 *  array i0[V][O];
 *  array f[N][N]: irrep_f;
 *  array v[N,N][N,N]: irrep_v;
 *  array t_vo[V][O]: irrep_t;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array t1_2_1[O][O];
 *  array t1_2_2_1[O][V];
 *  array t1_3_1[V][V];
 *  array t1_5_1[O][V];
 *  array t1_6_1[O,O][O,V];
 *
 *  t1_1:       i0[p2,h1] += 1 * f[p2,h1];
 *  t1_2_1:     t1_2_1[h7,h1] += 1 * f[h7,h1];
 *  t1_2_2_1:   t1_2_2_1[h7,p3] += 1 * f[h7,p3];
 *  t1_2_2_2:   t1_2_2_1[h7,p3] += -1 * t_vo[p5,h6] * v[h6,h7,p3,p5];
 *  t1_2_2:     t1_2_1[h7,h1] += 1 * t_vo[p3,h1] * t1_2_2_1[h7,p3];
 *  t1_2_3:     t1_2_1[h7,h1] += -1 * t_vo[p4,h5] * v[h5,h7,h1,p4];
 *  t1_2_4:     t1_2_1[h7,h1] += -1/2 * t_vvoo[p3,p4,h1,h5] * v[h5,h7,p3,p4];
 *  t1_2:       i0[p2,h1] += -1 * t_vo[p2,h7] * t1_2_1[h7,h1];
 *  t1_3_1:     t1_3_1[p2,p3] += 1 * f[p2,p3];
 *  t1_3_2:     t1_3_1[p2,p3] += -1 * t_vo[p4,h5] * v[h5,p2,p3,p4];
 *  t1_3:       i0[p2,h1] += 1 * t_vo[p3,h1] * t1_3_1[p2,p3];
 *  t1_4:       i0[p2,h1] += -1 * t_vo[p3,h4] * v[h4,p2,h1,p3];
 *  t1_5_1:     t1_5_1[h8,p7] += 1 * f[h8,p7];
 *  t1_5_2:     t1_5_1[h8,p7] += 1 * t_vo[p5,h6] * v[h6,h8,p5,p7];
 *  t1_5:       i0[p2,h1] += 1 * t_vvoo[p2,p7,h1,h8] * t1_5_1[h8,p7];
 *  t1_6_1:     t1_6_1[h4,h5,h1,p3] += 1 * v[h4,h5,h1,p3];
 *  t1_6_2:     t1_6_1[h4,h5,h1,p3] += -1 * t_vo[p6,h1] * v[h4,h5,p3,p6];
 *  t1_6:       i0[p2,h1] += -1/2 * t_vvoo[p2,p3,h4,h5] * t1_6_1[h4,h5,h1,p3];
 *  t1_7:       i0[p2,h1] += -1/2 * t_vvoo[p3,p4,h1,h5] * v[h5,p2,p3,p4];
 *
 *  }
 */

extern "C" {
void ccsd_t1_1_(F77Integer *d_f, F77Integer *k_f_offset, F77Integer *d_i0,
                F77Integer *k_i0_offset);
void ccsd_t1_2_1_(F77Integer *d_f, F77Integer *k_f_offset, F77Integer *d_t1_2_1,
                  F77Integer *k_t1_2_1_offset);
void ccsd_t1_2_2_1_(F77Integer *d_f, F77Integer *k_f_offset, F77Integer *d_t1_2_2_1,
                    F77Integer *k_t1_2_2_1_offset);
void ccsd_t1_2_2_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                    F77Integer *k_v_offset, F77Integer *d_t1_2_2_1,
                    F77Integer *k_t1_2_2_1_offset);
void ccsd_t1_2_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_t1_2_2_1,
                  F77Integer *k_t1_2_2_1_offset, F77Integer *d_t1_2_1,
                  F77Integer *k_t1_2_1_offset);
void ccsd_t1_2_3_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                  F77Integer *k_v_offset, F77Integer *d_t1_2_1,
                  F77Integer *k_t1_2_1_offset);
void ccsd_t1_2_4_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset, F77Integer *d_v,
                  F77Integer *k_v_offset, F77Integer *d_t1_2_1,
                  F77Integer *k_t1_2_1_offset);
void ccsd_t1_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_t1_2_1,
                F77Integer *k_t1_2_1_offset, F77Integer *d_i0, F77Integer *k_i0_offset);
void ccsd_t1_3_1_(F77Integer *d_f, F77Integer *k_f_offset, F77Integer *d_t1_3_1,
                  F77Integer *k_t1_3_1_offset);
void ccsd_t1_3_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                  F77Integer *k_v_offset, F77Integer *d_t1_3_1,
                  F77Integer *k_t1_3_1_offset);
void ccsd_t1_3_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_t1_3_1,
                F77Integer *k_t1_3_1_offset, F77Integer *d_i0, F77Integer *k_i0_offset);
void ccsd_t1_4_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                F77Integer *k_v_offset, F77Integer *d_i0, F77Integer *k_i0_offset);
void ccsd_t1_5_1_(F77Integer *d_f, F77Integer *k_f_offset, F77Integer *d_t1_5_1,
                  F77Integer *k_t1_5_1_offset);
void ccsd_t1_5_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                  F77Integer *k_v_offset, F77Integer *d_t1_5_1,
                  F77Integer *k_t1_5_1_offset);
void ccsd_t1_5_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset, F77Integer *d_t1_5_1,
                F77Integer *k_t1_5_1_offset, F77Integer *d_i0, F77Integer *k_i0_offset);
void ccsd_t1_6_1_(F77Integer *d_v, F77Integer *k_v_offset, F77Integer *d_t1_6_1,
                  F77Integer *k_t1_6_1_offset);
void ccsd_t1_6_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_v,
                  F77Integer *k_v_offset, F77Integer *d_t1_6_1,
                  F77Integer *k_t1_6_1_offset);
void ccsd_t1_6_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset, F77Integer *d_t1_6_1,
                F77Integer *k_t1_6_1_offset, F77Integer *d_i0, F77Integer *k_i0_offset);
void ccsd_t1_7_(F77Integer *d_t_vvoo, F77Integer *k_t_vvoo_offset, F77Integer *d_v,
                F77Integer *k_v_offset, F77Integer *d_i0, F77Integer *k_i0_offset);

void offset_ccsd_t1_2_1_(F77Integer *l_t1_2_1_offset, F77Integer *k_t1_2_1_offset,
                         F77Integer *size_t1_2_1);
void offset_ccsd_t1_2_2_1_(F77Integer *l_t1_2_2_1_offset,
                           F77Integer *k_t1_2_2_1_offset, F77Integer *size_t1_2_2_1);
void offset_ccsd_t1_3_1_(F77Integer *l_t1_3_1_offset, F77Integer *k_t1_3_1_offset,
                         F77Integer *size_t1_3_1);
void offset_ccsd_t1_5_1_(F77Integer *l_t1_5_1_offset, F77Integer *k_t1_5_1_offset,
                         F77Integer *size_t1_5_1);
void offset_ccsd_t1_6_1_(F77Integer *l_t1_6_1_offset, F77Integer *k_t1_6_1_offset,
                         F77Integer *size_t1_6_1);
}

namespace tamm {

extern "C" {
//  void ccsd_t1_cxx_(Integer *d_t_vvoo, Integer *d_i0, Integer *d_v, Integer
//  *d_t_vo, Integer *d_f, Integer *k_t_vvoo_offset, Integer *k_i0_offset,
//  Integer *k_v_offset, Integer *k_t_vo_offset, Integer *k_f_offset) {

void ccsd_t1_cxx_(F77Integer *d_f, F77Integer *d_i0, F77Integer *d_t_vo,
                  F77Integer *d_t_vvoo, F77Integer *d_v, F77Integer *k_f_offset,
                  F77Integer *k_i0_offset, F77Integer *k_t_vo_offset,
                  F77Integer *k_t_vvoo_offset, F77Integer *k_v_offset) {
  static bool set_t1 = true;

  Assignment op_t1_1;
  Assignment op_t1_2_1;
  Assignment op_t1_2_2_1;
  Assignment op_t1_3_1;
  Assignment op_t1_5_1;
  Assignment op_t1_6_1;
  Multiplication op_t1_2_2_2;
  Multiplication op_t1_2_2;
  Multiplication op_t1_2_3;
  Multiplication op_t1_2_4;
  Multiplication op_t1_2;
  Multiplication op_t1_3_2;
  Multiplication op_t1_3;
  Multiplication op_t1_4;
  Multiplication op_t1_5_2;
  Multiplication op_t1_5;
  Multiplication op_t1_6_2;
  Multiplication op_t1_6;
  Multiplication op_t1_7;

  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_t1) {
    ccsd_t1_equations(&eqs);
    set_t1 = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector<Operation> ops;
  tensors_and_ops(&eqs, &tensors, &ops);

  Tensor *i0 = &tensors["i0"];
  Tensor *f = &tensors["f"];
  Tensor *v = &tensors["v"];
  Tensor *t_vo = &tensors["t_vo"];
  Tensor *t_vvoo = &tensors["t_vvoo"];
  Tensor *t1_2_1 = &tensors["t1_2_1"];
  Tensor *t1_2_2_1 = &tensors["t1_2_2_1"];
  Tensor *t1_3_1 = &tensors["t1_3_1"];
  Tensor *t1_5_1 = &tensors["t1_5_1"];
  Tensor *t1_6_1 = &tensors["t1_6_1"];

  /* ----- Insert attach code ------ */
  v->set_dist(idist);
  i0->attach(*k_i0_offset, 0, *d_i0);
  f->attach(*k_f_offset, 0, *d_f);
  v->attach(*k_v_offset, 0, *d_v);
  t_vo->set_dist(dist_nwma);
  t_vo->attach(*k_t_vo_offset, 0, *d_t_vo);
  t_vvoo->attach(*k_t_vvoo_offset, 0, *d_t_vvoo);

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
  op_t1_1 = ops[0].add;
  op_t1_2_1 = ops[1].add;
  op_t1_2_2_1 = ops[2].add;
  op_t1_2_2_2 = ops[3].mult;
  op_t1_2_2 = ops[4].mult;
  op_t1_2_3 = ops[5].mult;
  op_t1_2_4 = ops[6].mult;
  op_t1_2 = ops[7].mult;
  op_t1_3_1 = ops[8].add;
  op_t1_3_2 = ops[9].mult;
  op_t1_3 = ops[10].mult;
  op_t1_4 = ops[11].mult;
  op_t1_5_1 = ops[12].add;
  op_t1_5_2 = ops[13].mult;
  op_t1_5 = ops[14].mult;
  op_t1_6_1 = ops[15].add;
  op_t1_6_2 = ops[16].mult;
  op_t1_6 = ops[17].mult;
  op_t1_7 = ops[18].mult;

  CorFortran(0, &op_t1_1, ccsd_t1_1_);
  CorFortran(0, t1_2_1, offset_ccsd_t1_2_1_);
  CorFortran(0, &op_t1_2_1, ccsd_t1_2_1_);
  CorFortran(0, t1_2_2_1, offset_ccsd_t1_2_2_1_);
  CorFortran(0, &op_t1_2_2_1, ccsd_t1_2_2_1_);
  CorFortran(0, &op_t1_2_2_2, ccsd_t1_2_2_2_);
  CorFortran(0, &op_t1_2_2, ccsd_t1_2_2_);
  destroy(t1_2_2_1);
  CorFortran(0, &op_t1_2_3, ccsd_t1_2_3_);
  CorFortran(0, &op_t1_2_4, ccsd_t1_2_4_);
  CorFortran(0, &op_t1_2, ccsd_t1_2_);
  destroy(t1_2_1);
  CorFortran(1, t1_3_1, offset_ccsd_t1_3_1_);
  CorFortran(1, &op_t1_3_1, ccsd_t1_3_1_);
  CorFortran(1, &op_t1_3_2, ccsd_t1_3_2_);
  CorFortran(1, &op_t1_3, ccsd_t1_3_);
  destroy(t1_3_1);
/*    CorFortran(1, op_t1_4, ccsd_t1_4_);
    CorFortran(1, t1_5_1, offset_ccsd_t1_5_1_);
    CorFortran(1, op_t1_5_1, ccsd_t1_5_1_);
    CorFortran(1, op_t1_5_2, ccsd_t1_5_2_);
    CorFortran(1, op_t1_5, ccsd_t1_5_);
    destroy(t1_5_1);
    CorFortran(1, t1_6_1, offset_ccsd_t1_6_1_);
    CorFortran(1, op_t1_6_1, ccsd_t1_6_1_);
    CorFortran(1, op_t1_6_2, ccsd_t1_6_2_);
    CorFortran(1, op_t1_6, ccsd_t1_6_);
    destroy(t1_6_1);
    CorFortran(1, op_t1_7, ccsd_t1_7_); */
#endif

  /* ----- Insert detach code ------ */
  f->detach();
  i0->detach();
  v->detach();
  t_vo->detach();
  t_vvoo->detach();
}
}  // extern C
};  // namespace tamm
