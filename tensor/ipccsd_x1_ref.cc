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
 *  x1 {
 *
 *  index h1,h2,h3,h4,h5,h6,h7,h8 = O;
 *  index p1,p2,p3,p4,p5,p6,p7 = V;
 *
 *  array x1_1_1[O][O];
 *  array f[N][N]: irrep_f;
 *  array x1_1_2_1[O][V];
 *  array t_vo[V][O]: irrep_t;
 *  array v[N,N][N,N]: irrep_v;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array i0[][O] : irrep_x;
 *  array x_o[][O]: irrep_x;
 *  array x1_2_1[O][V];
 *  array x_voo[V][O,O]: irrep_x;
 *  array x1_3_1[O,O][O,V];
 *
 *  x1_1_1:     x1_1_1[h6,h1] += 1 * f[h6,h1];
 *  x1_1_1_2_1:   x1_1_2_1[h6,p7] += 1 * f[h6,p7];
 *  x1_1_1_2_2:   x1_1_2_1[h6,p7] += 1 * t_vo[p4,h5] * v[h5,h6,p4,p7];
 *  x1_1_1_2:     x1_1_1[h6,h1] += 1 * t_vo[p7,h1] * x1_1_2_1[h6,p7];
 *  x1_1_1_3:     x1_1_1[h6,h1] += -1 * t_vo[p3,h4] * v[h4,h6,h1,p3];
 *  x1_1_1_4:     x1_1_1[h6,h1] += -1/2 * t_vvoo[p3,p4,h1,h5] * v[h5,h6,p3,p4];
 *  x1_1:       i0[h1] += -1 * x_o[h6] * x1_1_1[h6,h1];
 *  x1_2_1:     x1_2_1[h6,p7] += 1 * f[h6,p7];
 *  x1_2_2:     x1_2_1[h6,p7] += 1 * t_vo[p3,h4] * v[h4,h6,p3,p7];
 *  x1_2:       i0[h1] += 1 * x_voo[p7,h1,h6] * x1_2_1[h6,p7];
 *  x1_3_1:     x1_3_1[h6,h8,h1,p7] += 1 * v[h6,h8,h1,p7];
 *  x1_3_2:     x1_3_1[h6,h8,h1,p7] += 1 * t_vo[p3,h1] * v[h6,h8,p3,p7];
 *  x1_3:       i0[h1] += -1/2 * x_voo[p7,h6,h8] * x1_3_1[h6,h8,h1,p7];
 *
 *  }
 */

extern "C" {
void ipccsd_x1_1_1_(Fint *d_f, Fint *k_f_offset, Fint *d_x1_1_1,
                    Fint *k_x1_1_1_offset);
void ipccsd_x1_1_2_1_(Fint *d_f, Fint *k_f_offset, Fint *d_x1_1_1_1,
                      Fint *k_x1_1_1_1_offset);
void ipccsd_x1_1_2_2_(Fint *d_t_vo, Fint *k_t_vo_offset, Fint *d_v,
                      Fint *k_v_offset, Fint *d_x1_1_1_1,
                      Fint *k_x1_1_1_1_offset);
void ipccsd_x1_1_2_(Fint *d_t_vo, Fint *k_t_vo_offset, Fint *d_x1_1_1_1,
                    Fint *k_x1_1_1_1_offset, Fint *d_x1_1_1,
                    Fint *k_x1_1_1_offset);
void ipccsd_x1_1_3_(Fint *d_t_vo, Fint *k_t_vo_offset, Fint *d_v,
                    Fint *k_v_offset, Fint *d_x1_1_1, Fint *k_x1_1_1_offset);
void ipccsd_x1_1_4_(Fint *d_t_vvoo, Fint *k_t_vvoo_offset, Fint *d_v,
                    Fint *k_v_offset, Fint *d_x1_1_1, Fint *k_x1_1_1_offset);
void ipccsd_x1_1_(Fint *d_x_o, Fint *k_x_o_offset, Fint *d_x1_1_1,
                  Fint *k_x1_1_1_offset, Fint *d_i0, Fint *k_i0_offset);
void ipccsd_x1_2_1_(Fint *d_f, Fint *k_f_offset, Fint *d_x1_2_1,
                    Fint *k_x1_2_1_offset);
void ipccsd_x1_2_2_(Fint *d_t_vo, Fint *k_t_vo_offset, Fint *d_v,
                    Fint *k_v_offset, Fint *d_x1_2_1, Fint *k_x1_2_1_offset);
void ipccsd_x1_2_(Fint *d_x_voo, Fint *k_x_voo_offset, Fint *d_x1_2_1,
                  Fint *k_x1_2_1_offset, Fint *d_i0, Fint *k_i0_offset);
void ipccsd_x1_3_1_(Fint *d_v, Fint *k_v_offset, Fint *d_x1_3_1,
                    Fint *k_x1_3_1_offset);
void ipccsd_x1_3_2_(Fint *d_t_vo, Fint *k_t_vo_offset, Fint *d_v,
                    Fint *k_v_offset, Fint *d_x1_3_1, Fint *k_x1_3_1_offset);
void ipccsd_x1_3_(Fint *d_t_vo, Fint *k_t_vo_offset, Fint *d_v,
                  Fint *k_v_offset, Fint *d_x1_3_1, Fint *k_x1_3_1_offset);

void offset_ipccsd_x1_1_1_(Fint *l_x1_1_1_offset, Fint *k_x1_1_1_offset,
                           Fint *size_x1_1_1);
void offset_ipccsd_x1_1_2_1_(Fint *l_x1_1_1_1_offset, Fint *k_x1_1_1_1_offset,
                             Fint *size_x1_1_1_1);
void offset_ipccsd_x1_2_1_(Fint *l_x1_2_1_offset, Fint *k_x1_2_1_offset,
                           Fint *size_x1_2_1);
void offset_ipccsd_x1_3_1_(Fint *l_x1_3_1_offset, Fint *k_x1_3_1_offset,
                           Fint *size_x1_3_1);
}

namespace tamm {

extern "C" {
void ipccsd_x1_cxx_(Fint *d_f1, Fint *d_i0, Fint *d_t_vo, Fint *d_t_vvoo,
                    Fint *d_v2, Fint *d_x1, Fint *d_x2, Fint *k_f1_offset,
                    Fint *k_i0_offset, Fint *k_t_vo_offset,
                    Fint *k_t_vvoo_offset, Fint *k_v2_offset, Fint *k_x1_offset,
                    Fint *k_x2_offset) {
  static bool set_x1 = true;

  Assignment op_x1_1_1;
  Assignment op_x1_1_2_1;
  Assignment op_x1_2_1;
  Assignment op_x1_3_1;
  Multiplication op_x1_1_2_2;
  Multiplication op_x1_1_2;
  Multiplication op_x1_1_3;
  Multiplication op_x1_1_4;
  Multiplication op_x1_1;
  Multiplication op_x1_2_2;
  Multiplication op_x1_2;
  Multiplication op_x1_3_2;
  Multiplication op_x1_3;

  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_x1) {
    ipccsd_x1_equations(&eqs);
    set_x1 = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector<Operation> ops;
  tensors_and_ops(&eqs, &tensors, &ops);

  Tensor *x1_1_1 = &tensors["x1_1_1"];
  Tensor *f = &tensors["f"];
  Tensor *x1_1_2_1 = &tensors["x1_1_2_1"];
  Tensor *t_vo = &tensors["t_vo"];
  Tensor *v = &tensors["v"];
  Tensor *t_vvoo = &tensors["t_vvoo"];
  Tensor *i0 = &tensors["i0"];
  Tensor *x_o = &tensors["x_o"];
  Tensor *x1_2_1 = &tensors["x1_2_1"];
  Tensor *x_voo = &tensors["x_voo"];
  Tensor *x1_3_1 = &tensors["x1_3_1"];

  v->set_dist(idist);
  t_vo->set_dist(dist_nw);
  f->attach(*k_f1_offset, 0, *d_f1);
  i0->attach(*k_i0_offset, 0, *d_i0);
  t_vo->attach(*k_t_vo_offset, 0, *d_t_vo);
  t_vvoo->attach(*k_t_vvoo_offset, 0, *d_t_vvoo);
  v->attach(*k_v2_offset, 0, *d_v2);
  x_o->attach(*k_x1_offset, 0, *d_x1);
  x_voo->attach(*k_x2_offset, 0, *d_x2);
  x_o->set_irrep(Variables::irrep_x());
  x_voo->set_irrep(Variables::irrep_x());
  i0->set_irrep(Variables::irrep_x());

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
  op_x1_1_1 = ops[0].add;
  op_x1_1_2_1 = ops[1].add;
  op_x1_1_2_2 = ops[2].mult;
  op_x1_1_2 = ops[3].mult;
  op_x1_1_3 = ops[4].mult;
  op_x1_1_4 = ops[5].mult;
  op_x1_1 = ops[6].mult;
  op_x1_2_1 = ops[7].add;
  op_x1_2_2 = ops[8].mult;
  op_x1_2 = ops[9].mult;
  op_x1_3_1 = ops[10].add;
  op_x1_3_2 = ops[11].mult;
  op_x1_3 = ops[12].mult;

  CorFortran(1, x1_1_1, offset_ipccsd_x1_1_1_);
  CorFortran(1, &op_x1_1_1, ipccsd_x1_1_1_);
  CorFortran(1, x1_1_2_1, offset_ipccsd_x1_1_2_1_);
  CorFortran(1, &op_x1_1_2_1, ipccsd_x1_1_2_1_);
  CorFortran(1, &op_x1_1_2_2, ipccsd_x1_1_2_2_);
  CorFortran(1, &op_x1_1_2, ipccsd_x1_1_2_);
  destroy(x1_1_2_1);
  CorFortran(1, &op_x1_1_3, ipccsd_x1_1_3_);
  CorFortran(1, &op_x1_1_4, ipccsd_x1_1_4_);
  CorFortran(1, &op_x1_1, ipccsd_x1_1_);
  destroy(x1_1_1);
  CorFortran(1, x1_2_1, offset_ipccsd_x1_2_1_);
  CorFortran(1, &op_x1_2_1, ipccsd_x1_2_1_);
  CorFortran(1, &op_x1_2_2, ipccsd_x1_2_2_);
  CorFortran(1, &op_x1_2,
             ipccsd_x1_2_); /*@bug @fixme Does not work in C mode: works now*/
  destroy(x1_2_1);
  CorFortran(1, x1_3_1, offset_ipccsd_x1_3_1_);
  CorFortran(1, &op_x1_3_1, ipccsd_x1_3_1_);
  CorFortran(1, &op_x1_3_2, ipccsd_x1_3_2_);
  CorFortran(1, &op_x1_3,
             ipccsd_x1_3_); /*@bug @fixme Does not work in C mode: works now*/
  destroy(x1_3_1);
#endif  // Use fortran functions

  f->detach();
  i0->detach();
  t_vo->detach();
  t_vvoo->detach();
  v->detach();
  x_o->detach();
  x_voo->detach();
}
}  // extern C
}  // namespace tamm
