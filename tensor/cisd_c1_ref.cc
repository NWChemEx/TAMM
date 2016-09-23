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
 *  c1 {
 *  
 *  index h1,h2,h3,h4,h5 = O;
 *  index p1,p2,p3,p4 = V;
 *  
 *  array i0[V][O];
 *  array f[N][N]: irrep_f;
 *  array t_vo[V][O]: irrep_t;
 *  array v[N,N][N,N]: irrep_v;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array e[][]: irrep_e;
 *  
 *  c1_1:       i0[p2,h1] += 1 * f[p2,h1];
 *  c1_2:       i0[p2,h1] += -1 * t_vo[p2,h3] * f[h3,h1];
 *  c1_3:       i0[p2,h1] += 1 * t_vo[p3,h1] * f[p2,p3];
 *  c1_4:       i0[p2,h1] += -1 * t_vo[p3,h4] * v[h4,p2,h1,p3];
 *  c1_5:       i0[p2,h1] += 1 * t_vvoo[p2,p4,h1,h3] * f[h3,p4];
 *  c1_6:       i0[p2,h1] += -1/2 * t_vvoo[p2,p3,h4,h5] * v[h4,h5,h1,p3];
 *  c1_7:       i0[p2,h1] += -1/2 * t_vvoo[p3,p4,h1,h5] * v[h5,p2,p3,p4];
 *  c1_8:       i0[p2,h1] += -1 * e * t_vo[p2,h1];
 *  
 *  }
*/


extern "C" {
  void cisd_c1_1_(Integer *d_f, Integer *k_f_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c1_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_f, Integer *k_f_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c1_3_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_f, Integer *k_f_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c1_4_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c1_5_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_f, Integer *k_f_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c1_6_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c1_7_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c1_8_(Integer *d_e, Integer *k_e_offset,Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_i0, Integer *k_i0_offset);

}

namespace ctce {

void schedule_linear(std::map<std::string, ctce::Tensor> &tensors, std::vector<Operation> &ops);
void schedule_linear_lazy(std::map<std::string, ctce::Tensor> &tensors, std::vector<Operation> &ops);
void schedule_levels(std::map<std::string, ctce::Tensor> &tensors, std::vector<Operation> &ops);

extern "C" {
  //void cisd_c1_cxx_(Integer *d_t_vvoo, Integer *d_e, Integer *d_f, Integer *d_i0, Integer *d_t_vo, Integer *d_v,
  //Integer *k_t_vvoo_offset, Integer *k_e_offset, Integer *k_f_offset, Integer *k_i0_offset, Integer *k_t_vo_offset, Integer *k_v_offset) {

  void cisd_c1_cxx_(Integer *d_e, Integer *d_f, Integer *d_i0, Integer *d_t_vo, Integer *d_t_vvoo, Integer *d_v,
                    Integer *k_e_offset, Integer *k_f_offset, Integer *k_i0_offset,
                    Integer *k_t_vo_offset, Integer *k_t_vvoo_offset, Integer *k_v_offset){

  static bool set_c1 = true;
  
  Assignment op_c1_1;
  Multiplication op_c1_2;
  Multiplication op_c1_3;
  Multiplication op_c1_4;
  Multiplication op_c1_5;
  Multiplication op_c1_6;
  Multiplication op_c1_7;
  Multiplication op_c1_8;
  
  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_c1) {
    cisd_c1_equations(eqs);
    set_c1 = false;
  }

  std::map<std::string, ctce::Tensor> tensors;
  std::vector <Operation> ops;
  tensors_and_ops(eqs, tensors, ops);

  Tensor *i0 = &tensors["i0"];
  Tensor *f = &tensors["f"];
  Tensor *t_vo = &tensors["t_vo"];
  Tensor *v = &tensors["v"];
  Tensor *t_vvoo = &tensors["t_vvoo"];
  Tensor *e = &tensors["e"];

  /* ----- Insert attach code ------ */
  v->set_dist(dist_nw);
  t_vo->set_dist(dist_nw);
  e->attach(*k_e_offset, 0, *d_e);
  f->attach(*k_f_offset, 0, *d_f);
  i0->attach(*k_i0_offset, 0, *d_i0);
  t_vo->attach(*k_t_vo_offset, 0, *d_t_vo);
  t_vvoo->attach(*k_t_vvoo_offset, 0, *d_t_vvoo);
  v->attach(*k_v_offset, 0, *d_v);

  #if 1
    schedule_levels(tensors, ops);
  #else
    op_c1_1 = ops[0].add;
    op_c1_2 = ops[1].mult;
    op_c1_3 = ops[2].mult;
    op_c1_4 = ops[3].mult;
    op_c1_5 = ops[4].mult;
    op_c1_6 = ops[5].mult;
    op_c1_7 = ops[6].mult;
    op_c1_8 = ops[7].mult;

    CorFortran(1, op_c1_1, cisd_c1_1_);
    CorFortran(1, op_c1_2, cisd_c1_2_);
    CorFortran(1, op_c1_3, cisd_c1_3_);
    CorFortran(1, op_c1_4, cisd_c1_4_);
    CorFortran(1, op_c1_5, cisd_c1_5_);
    CorFortran(1, op_c1_6, cisd_c1_6_);
    CorFortran(1, op_c1_7, cisd_c1_7_);
    CorFortran(1, op_c1_8, cisd_c1_8_);
  #endif

  /* ----- Insert detach code ------ */
  e->detach();
  f->detach();
  i0->detach();
  v->detach();
  t_vo->detach();
  t_vvoo->detach();
  }
} // extern C
}; // namespace ctce
