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
 *  t1 {
 *  
 *  index h1,h2,h3,h4,h5,h6,h7,h8 = O;
 *  index p1,p2,p3,p4,p5,p6,p7 = V;
 *  
 *  array i0[V][O];
 *  array f[N][N]: irrep_f;
 *  array t_vo[V][O]: irrep_t;
 *  array v[N,N][N,N]: irrep_v;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array t1_2_1[O][O];
 *  array t1_2_2_1[O][V];
 *  array t1_5_1[O][V];
 *  array t1_6_1[O,O][O,V];
 *  array t1_3_1[V][V];
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
  void cc2_t1_1_(Integer *d_f, Integer *k_f_offset,Integer *d_i0, Integer *k_i0_offset);
  void cc2_t1_2_1_(Integer *d_f, Integer *k_f_offset,Integer *d_t1_2_1, Integer *k_t1_2_1_offset);
  void cc2_t1_2_2_1_(Integer *d_f, Integer *k_f_offset,Integer *d_t1_2_2_1, Integer *k_t1_2_2_1_offset);
  void cc2_t1_2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t1_2_2_1, Integer *k_t1_2_2_1_offset);
  void cc2_t1_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t1_2_2_1, Integer *k_t1_2_2_1_offset,Integer *d_t1_2_1, Integer *k_t1_2_1_offset);
  void cc2_t1_2_3_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t1_2_1, Integer *k_t1_2_1_offset);
  void cc2_t1_2_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t1_2_1, Integer *k_t1_2_1_offset);
  void cc2_t1_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t1_2_1, Integer *k_t1_2_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void cc2_t1_3_1_(Integer *d_f, Integer *k_f_offset,Integer *d_t1_3_1, Integer *k_t1_3_1_offset);
  void cc2_t1_3_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t1_3_1, Integer *k_t1_3_1_offset);
  void cc2_t1_3_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t1_3_1, Integer *k_t1_3_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void cc2_t1_4_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);
  void cc2_t1_5_1_(Integer *d_f, Integer *k_f_offset,Integer *d_t1_5_1, Integer *k_t1_5_1_offset);
  void cc2_t1_5_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t1_5_1, Integer *k_t1_5_1_offset);
  void cc2_t1_5_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_t1_5_1, Integer *k_t1_5_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void cc2_t1_6_1_(Integer *d_v, Integer *k_v_offset,Integer *d_t1_6_1, Integer *k_t1_6_1_offset);
  void cc2_t1_6_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t1_6_1, Integer *k_t1_6_1_offset);
  void cc2_t1_6_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_t1_6_1, Integer *k_t1_6_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void cc2_t1_7_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);

  void offset_cc2_t1_2_1_(Integer *l_t1_2_1_offset, Integer *k_t1_2_1_offset, Integer *size_t1_2_1);
  void offset_cc2_t1_2_2_1_(Integer *l_t1_2_2_1_offset, Integer *k_t1_2_2_1_offset, Integer *size_t1_2_2_1);
  void offset_cc2_t1_3_1_(Integer *l_t1_3_1_offset, Integer *k_t1_3_1_offset, Integer *size_t1_3_1);
  void offset_cc2_t1_5_1_(Integer *l_t1_5_1_offset, Integer *k_t1_5_1_offset, Integer *size_t1_5_1);
  void offset_cc2_t1_6_1_(Integer *l_t1_6_1_offset, Integer *k_t1_6_1_offset, Integer *size_t1_6_1);
}

namespace ctce {
  void schedule_linear(std::vector<Tensor> &tensors,
                       std::vector<Operation> &ops);
  void schedule_linear_lazy(std::vector<Tensor> &tensors,
                            std::vector<Operation> &ops);
  void schedule_levels(std::vector<Tensor> &tensors,
                            std::vector<Operation> &ops);

  extern "C" {
    void cc2_t1_cxx_(Integer *d_t_vvoo,Integer *d_i0,Integer *d_v,Integer *d_t_vo,Integer *d_f,Integer *k_t_vvoo_offset,Integer *k_i0_offset,Integer *k_v_offset,Integer *k_t_vo_offset,Integer *k_f_offset){
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
        cc2_t1_equations(eqs);
        set_t1 = false;
      }

      std::vector <Tensor> tensors;
      std::vector <Operation> ops;
      tensors_and_ops(eqs, tensors, ops);

      Tensor *i0 = &tensors[0];
      Tensor *f = &tensors[1];
      Tensor *t1_2_1 = &tensors[2];
      Tensor *t1_2_2_1 = &tensors[3];
      Tensor *t_vo = &tensors[4];
      Tensor *v = &tensors[5];
      Tensor *t_vvoo = &tensors[6];
      Tensor *t1_3_1 = &tensors[7];
      Tensor *t1_5_1 = &tensors[8];
      Tensor *t1_6_1 = &tensors[9];

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
      
/* ----- Insert attach code ------ */
      v->set_dist(idist);
      t_vo->set_dist(dist_nwma);
      f->attach(*k_f_offset, 0, *d_f);
      i0->attach(*k_i0_offset, 0, *d_i0);
      t_vo->attach(*k_t_vo_offset, 0, *d_t_vo);
      t_vvoo->attach(*k_t_vvoo_offset, 0, *d_t_vvoo);
      v->attach(*k_v_offset, 0, *d_v);

#if 1
      // schedule_linear(tensors, ops);
      // schedule_linear_lazy(tensors, ops);
       schedule_levels(tensors, ops);
#else
      CorFortran(1, op_t1_1, cc2_t1_1_);
      CorFortran(1, op_t1_2_1, ofsset_cc2_t1_2_1_);
      CorFortran(1, op_t1_2_1, cc2_t1_2_1_);
      CorFortran(1, op_t1_2_2_1, ofsset_cc2_t1_2_2_1_);
      CorFortran(1, op_t1_2_2_1, cc2_t1_2_2_1_);
      CorFortran(1, op_t1_2_2_2, cc2_t1_2_2_2_);
      CorFortran(1, op_t1_2_2, cc2_t1_2_2_);
      destroy(t1_2_2_1);
      CorFortran(1, op_t1_2_3, cc2_t1_2_3_);
      CorFortran(1, op_t1_2_4, cc2_t1_2_4_);
      CorFortran(1, op_t1_2, cc2_t1_2_);
      destroy(t1_2_1);
      CorFortran(1, op_t1_3_1, ofsset_cc2_t1_3_1_);
      CorFortran(1, op_t1_3_1, cc2_t1_3_1_);
      CorFortran(1, op_t1_3_2, cc2_t1_3_2_);
      CorFortran(1, op_t1_3, cc2_t1_3_);
      destroy(t1_3_1);
      CorFortran(1, op_t1_4, cc2_t1_4_);
      CorFortran(1, op_t1_5_1, ofsset_cc2_t1_5_1_);
      CorFortran(1, op_t1_5_1, cc2_t1_5_1_);
      CorFortran(1, op_t1_5_2, cc2_t1_5_2_);
      CorFortran(1, op_t1_5, cc2_t1_5_);
      destroy(t1_5_1);
      CorFortran(1, op_t1_6_1, ofsset_cc2_t1_6_1_);
      CorFortran(1, op_t1_6_1, cc2_t1_6_1_);
      CorFortran(1, op_t1_6_2, cc2_t1_6_2_);
      CorFortran(1, op_t1_6, cc2_t1_6_);
      destroy(t1_6_1);
      CorFortran(1, op_t1_7, cc2_t1_7_);
#endif
      
/* ----- Insert detach code ------ */
      f->detach();
      i0->detach();
      t_vo->detach();
      t_vvoo->detach();
      v->detach();

    }
  } // extern C
}; // namespace ctce
