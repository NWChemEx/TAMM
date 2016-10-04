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
 *  e {
 *  
 *  index h1,h2,h3,h4 = O;
 *  index p1,p2 = V;
 *  
 *  array i0[][];
 *  array t_vo[V][O]: irrep_t;
 *  array f[N][N]: irrep_f;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array v[N,N][N,N]: irrep_v;
 *  
 *  e_1:       i0 += 1 * t_vo[p2,h1] * f[h1,p2];
 *  e_2:       i0 += 1/4 * t_vvoo[p1,p2,h3,h4] * v[h3,h4,p1,p2];
 *  
 *  }
*/


extern "C" {
  void cisd_e_1_(Fint *d_t_vo, Fint *k_t_vo_offset,Fint *d_f, Fint *k_f_offset,Fint *d_i0, Fint *k_i0_offset);
  void cisd_e_2_(Fint *d_t_vvoo, Fint *k_t_vvoo_offset,Fint *d_v, Fint *k_v_offset,Fint *d_i0, Fint *k_i0_offset);

}

namespace tamm {

void schedule_linear(std::map<std::string, tamm::Tensor> &tensors, std::vector<Operation> &ops);
void schedule_linear_lazy(std::map<std::string, tamm::Tensor> &tensors, std::vector<Operation> &ops);
void schedule_levels(std::map<std::string, tamm::Tensor> &tensors, std::vector<Operation> &ops);

extern "C" {
  //void cisd_e_cxx_(Fint *d_i0, Fint *d_t_vvoo, Fint *d_v, Fint *d_t_vo, Fint *d_f,
  //Fint *k_i0_offset, Fint *k_t_vvoo_offset, Fint *k_v_offset, Fint *k_t_vo_offset, Fint *k_f_offset) {
  void cisd_e_cxx_(Fint *d_f, Fint *d_i0, Fint *d_t_vo, Fint *d_t_vvoo, Fint *d_v,
                     Fint *k_f_offset, Fint *k_i0_offset,
                     Fint *k_t_vo_offset, Fint *k_t_vvoo_offset, Fint *k_v_offset){

  static bool set_e = true;
  
  Multiplication op_e_1;
  Multiplication op_e_2;
  
  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_e) {
    cisd_e_equations(eqs);
    set_e = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector <Operation> ops;
  tensors_and_ops(eqs, tensors, ops);

  Tensor *i0 = &tensors["i0"];
  Tensor *t_vo = &tensors["t_vo"];
  Tensor *f = &tensors["f"];
  Tensor *t_vvoo = &tensors["t_vvoo"];
  Tensor *v = &tensors["v"];

  /* ----- Insert attach code ------ */
    v->set_dist(dist_nw);
    t_vo->set_dist(dist_nw);
    f->attach(*k_f_offset, 0, *d_f);
    i0->attach(*k_i0_offset, 0, *d_i0);
    t_vo->attach(*k_t_vo_offset, 0, *d_t_vo);
    t_vvoo->attach(*k_t_vvoo_offset, 0, *d_t_vvoo);
    v->attach(*k_v_offset, 0, *d_v);

  #if 0
    schedule_levels(tensors, ops);
  #else
    op_e_1 = ops[0].mult;
    op_e_2 = ops[1].mult;

    CorFortran(1, op_e_1, cisd_e_1_);
    CorFortran(1, op_e_2, cisd_e_2_);
  #endif

  /* ----- Insert detach code ------ */
    f->detach();
    i0->detach();
    v->detach();
    t_vo->detach();
    t_vvoo->detach();
  }
} // extern C
}; // namespace tamm
