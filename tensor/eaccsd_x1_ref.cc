#include "variables.h"
#include <iostream>
#include "tensor.h"
#include "t_mult.h"
#include "t_assign.h"
#include "input.h"
#include "corf.h"
#include "equations.h"
#include "tensors_and_ops.h"

/*
 *  x1 {
 *  
 *  index h1,h2,h3,h4,h5,h6 = O;
 *  index p1,p2,p3,p4,p5,p6,p7 = V;
 *  
 *  array i0[V][O];
 *  array x_vo[V][O]: irrep_x;
 *  array f[N][N]: irrep_f;
 *  array t_vo[V][O]: irrep_t;
 *  array v[N,N][N,N]: irrep_v;
 *  array x_vvoo[V,V][O,O]: irrep_x;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array x1_4_1_1[O][V];
 *  array x1_4_1[O][O];
 *  array x1_2_1[O][V];
 *  array x1_5_1[O,O][O,V];
 *  array x1_1_1[V][V];
 *  
 *  x1_1_1:     x1_1_1[p2,p6] += 1 * f[p2,p6];
 *  x1_1_2:     x1_1_1[p2,p6] += 1 * t_vo[p3,h4] * v[h4,p2,p3,p6];
 *  x1_1:       i0[p2,h1] += 1 * x_vo[p6,h1] * x1_1_1[p2,p6];
 *  x1_2_1:     x1_2_1[h6,p7] += 1 * f[h6,p7];
 *  x1_2_2:     x1_2_1[h6,p7] += 1 * t_vo[p3,h4] * v[h4,h6,p3,p7];
 *  x1_2:       i0[p2,h1] += 1 * x_vvoo[p2,p7,h1,h6] * x1_2_1[h6,p7];
 *  x1_3:       i0[p2,h1] += -1/2 * x_vvoo[p4,p5,h1,h3] * v[h3,p2,p4,p5];
 *  x1_4_1_1:   x1_4_1_1[h3,p7] += 1 * f[h3,p7];
 *  x1_4_1_2:   x1_4_1_1[h3,p7] += -1 * t_vo[p4,h5] * v[h3,h5,p4,p7];
 *  x1_4_1:     x1_4_1[h3,h1] += 1 * x_vo[p7,h1] * x1_4_1_1[h3,p7];
 *  x1_4_2:     x1_4_1[h3,h1] += 1/2 * x_vvoo[p5,p6,h1,h4] * v[h3,h4,p5,p6];
 *  x1_4:       i0[p2,h1] += -1 * t_vo[p2,h3] * x1_4_1[h3,h1];
 *  x1_5_1:     x1_5_1[h4,h5,h1,p3] += 1 * x_vo[p6,h1] * v[h4,h5,p3,p6];
 *  x1_5:       i0[p2,h1] += 1/2 * t_vvoo[p2,p3,h4,h5] * x1_5_1[h4,h5,h1,p3];
 *  
 *  }
*/


extern "C" {
  void eaccsd_x1_1_1_(Integer *d_f, Integer *k_f_offset,Integer *d_x1_1_1, Integer *k_x1_1_1_offset);
  void eaccsd_x1_1_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_x1_1_1, Integer *k_x1_1_1_offset);
  void eaccsd_x1_1_(Integer *d_x_vo, Integer *k_x_vo_offset,Integer *d_x1_1_1, Integer *k_x1_1_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void eaccsd_x1_2_1_(Integer *d_f, Integer *k_f_offset,Integer *d_x1_2_1, Integer *k_x1_2_1_offset);
  void eaccsd_x1_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_x1_2_1, Integer *k_x1_2_1_offset);
  void eaccsd_x1_2_(Integer *d_x_vvoo, Integer *k_x_vvoo_offset,Integer *d_x1_2_1, Integer *k_x1_2_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void eaccsd_x1_3_(Integer *d_x_vvoo, Integer *k_x_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);
  void eaccsd_x1_4_1_1_(Integer *d_f, Integer *k_f_offset,Integer *d_x1_4_1_1, Integer *k_x1_4_1_1_offset);
  void eaccsd_x1_4_1_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_x1_4_1_1, Integer *k_x1_4_1_1_offset);
  void eaccsd_x1_4_1_(Integer *d_x_vo, Integer *k_x_vo_offset,Integer *d_x1_4_1_1, Integer *k_x1_4_1_1_offset,Integer *d_x1_4_1, Integer *k_x1_4_1_offset);
  void eaccsd_x1_4_2_(Integer *d_x_vvoo, Integer *k_x_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_x1_4_1, Integer *k_x1_4_1_offset);
  void eaccsd_x1_4_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_x1_4_1, Integer *k_x1_4_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void eaccsd_x1_5_1_(Integer *d_x_vo, Integer *k_x_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_x1_5_1, Integer *k_x1_5_1_offset);
  void eaccsd_x1_5_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_x1_5_1, Integer *k_x1_5_1_offset,Integer *d_i0, Integer *k_i0_offset);

  void offset_eaccsd_x1_1_1_(Integer *l_x1_1_1_offset, Integer *k_x1_1_1_offset, Integer *size_x1_1_1);
  void offset_eaccsd_x1_2_1_(Integer *l_x1_2_1_offset, Integer *k_x1_2_1_offset, Integer *size_x1_2_1);
  void offset_eaccsd_x1_4_1_1_(Integer *l_x1_4_1_1_offset, Integer *k_x1_4_1_1_offset, Integer *size_x1_4_1_1);
  void offset_eaccsd_x1_4_1_(Integer *l_x1_4_1_offset, Integer *k_x1_4_1_offset, Integer *size_x1_4_1);
  void offset_eaccsd_x1_5_1_(Integer *l_x1_5_1_offset, Integer *k_x1_5_1_offset, Integer *size_x1_5_1);
}

namespace ctce {

void schedule_levels(std::map<std::string, ctce::Tensor> &tensors,
                     std::vector<Operation> &ops);

extern "C" {
  void eaccsd_x1_cxx_(Integer *d_t_vvoo, Integer *d_f, Integer *d_i0, Integer *d_x_vvoo, Integer *d_x_vo, Integer *d_t_vo, Integer *d_v, 
  Integer *k_t_vvoo_offset, Integer *k_f_offset, Integer *k_i0_offset, Integer *k_x_vvoo_offset, Integer *k_x_vo_offset, Integer *k_t_vo_offset, Integer *k_v_offset) {

  static bool set_x1 = true;
  
  Assignment op_x1_1_1;
  Assignment op_x1_2_1;
  Assignment op_x1_4_1_1;
  Multiplication op_x1_1_2;
  Multiplication op_x1_1;
  Multiplication op_x1_2_2;
  Multiplication op_x1_2;
  Multiplication op_x1_3;
  Multiplication op_x1_4_1_2;
  Multiplication op_x1_4_1;
  Multiplication op_x1_4_2;
  Multiplication op_x1_4;
  Multiplication op_x1_5_1;
  Multiplication op_x1_5;
  
  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_x1) {
    eaccsd_x1_equations(eqs);
    set_x1 = false;
  }

  std::map<std::string, ctce::Tensor> tensors;
  std::vector <Operation> ops;
  tensors_and_ops(eqs, tensors, ops);

  Tensor *i0 = &tensors["i0"];
  Tensor *x_vo = &tensors["x_vo"];
  Tensor *f = &tensors["f"];
  Tensor *t_vo = &tensors["t_vo"];
  Tensor *v = &tensors["v"];
  Tensor *x_vvoo = &tensors["x_vvoo"];
  Tensor *t_vvoo = &tensors["t_vvoo"];
  Tensor *x1_4_1_1 = &tensors["x1_4_1_1"];
  Tensor *x1_4_1 = &tensors["x1_4_1"];
  Tensor *x1_2_1 = &tensors["x1_2_1"];
  Tensor *x1_5_1 = &tensors["x1_5_1"];
  Tensor *x1_1_1 = &tensors["x1_1_1"];

  /* ----- Insert attach code ------ */
  v->set_dist(idist);
  i0->attach(*k_i0_offset, 0, *d_i0);
  f->attach(*k_f_offset, 0, *d_f);
  v->attach(*k_v_offset, 0, *d_v);

  #if 1
    schedule_levels(tensors, ops);
  #else
    op_x1_1_1 = ops[0].add;
    op_x1_1_2 = ops[1].mult;
    op_x1_1 = ops[2].mult;
    op_x1_2_1 = ops[3].add;
    op_x1_2_2 = ops[4].mult;
    op_x1_2 = ops[5].mult;
    op_x1_3 = ops[6].mult;
    op_x1_4_1_1 = ops[7].add;
    op_x1_4_1_2 = ops[8].mult;
    op_x1_4_1 = ops[9].mult;
    op_x1_4_2 = ops[10].mult;
    op_x1_4 = ops[11].mult;
    op_x1_5_1 = ops[12].mult;
    op_x1_5 = ops[13].mult;

    CorFortran(1, x1_1_1, offset_eaccsd_x1_1_1_);
    CorFortran(1, op_x1_1_1, eaccsd_x1_1_1_);
    CorFortran(1, op_x1_1_2, eaccsd_x1_1_2_);
    CorFortran(1, op_x1_1, eaccsd_x1_1_);
    destroy(x1_1_1);
    CorFortran(1, x1_2_1, offset_eaccsd_x1_2_1_);
    CorFortran(1, op_x1_2_1, eaccsd_x1_2_1_);
    CorFortran(1, op_x1_2_2, eaccsd_x1_2_2_);
    CorFortran(1, op_x1_2, eaccsd_x1_2_);
    destroy(x1_2_1);
    CorFortran(1, op_x1_3, eaccsd_x1_3_);
    CorFortran(1, x1_4_1_1, offset_eaccsd_x1_4_1_1_);
    CorFortran(1, op_x1_4_1_1, eaccsd_x1_4_1_1_);
    CorFortran(1, op_x1_4_1_2, eaccsd_x1_4_1_2_);
    CorFortran(1, x1_4_1, offset_eaccsd_x1_4_1_);
    CorFortran(1, op_x1_4_1, eaccsd_x1_4_1_);
    destroy(x1_4_1_1);
    CorFortran(1, op_x1_4_2, eaccsd_x1_4_2_);
    CorFortran(1, op_x1_4, eaccsd_x1_4_);
    destroy(x1_4_1);
    CorFortran(1, x1_5_1, offset_eaccsd_x1_5_1_);
    CorFortran(1, op_x1_5_1, eaccsd_x1_5_1_);
    CorFortran(1, op_x1_5, eaccsd_x1_5_);
    destroy(x1_5_1);
  #endif

  /* ----- Insert detach code ------ */
  f->detach();
  i0->detach();
  v->detach();
  }
} // extern C
}; // namespace ctce
