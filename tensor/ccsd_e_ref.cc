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
 * i0 ( )_tf + = 1 * Sum ( p5 h6 ) * t ( p5 h6 )_t * i1 ( h6 p5 )_f
 *    i1 ( h6 p5 )_f + = 1 * f ( h6 p5 )_f
 *    i1 ( h6 p5 )_vt + = 1/2 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 h6 p3 p5 )_v
 * i0 ( )_vt + = 1/4 * Sum ( h3 h4 p1 p2 ) * t ( p1 p2 h3 h4 )_t * v ( h3 h4 p1 p2 )_v
*/

/*
 * e_1: i0 ( )_tf + = 1 * Sum ( p5 h6 ) * t ( p5 h6 )_t * i1 ( h6 p5 )_f
 * e_1_1:   i1 ( h6 p5 )_f + = 1 * f ( h6 p5 )_f
 * e_1_2:   i1 ( h6 p5 )_vt + = 1/2 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 h6 p3 p5 )_v
 * e_2: i0 ( )_vt + = 1/4 * Sum ( h3 h4 p1 p2 ) * t ( p1 p2 h3 h4 )_t * v ( h3 h4 p1 p2 )_v
*/


extern "C" {
  void offset_ccsd_e_1_1_(Integer *l_i1_offset, Integer *k_i1_offset, Integer *size_i1);
  void ccsd_e_1_(Integer *d_t1, Integer *k_t1_offset, Integer *d_v2, Integer *k_v2_offset,
                      Integer *d_i2, Integer *k_i2_offset);
  void ccsd_e_copy_fock_to_t_(Integer *d_f1, Integer *k_f1_offset, Integer *d_i1, Integer *k_i1_offset);
  void ccsd_e_1_2_(Integer *d_t1, Integer *k_t1_offset, Integer *d_v2, Integer *k_v2_offset,
                      Integer *d_i2, Integer *k_i2_offset);
  void ccsd_e_2_(Integer *d_t1, Integer *k_t1_offset, Integer *d_v2, Integer *k_v2_offset,
                      Integer *d_i2, Integer *k_i2_offset);
}


namespace ctce {

  void schedule_linear(std::vector<Tensor> &tensors,
                       std::vector<Operation> &ops);
  void schedule_linear_lazy(std::vector<Tensor> &tensors,
                            std::vector<Operation> &ops);
  void schedule_levels(std::vector<Tensor> &tensors,
                            std::vector<Operation> &ops);

  extern "C" {

    void ccsd_e_cxx_(Integer *d_f1, Integer *d_i0, Integer *d_t1, Integer *d_t2, Integer *d_v2,
                      Integer *k_f1_offset, Integer *k_i0_offset,
                      Integer *k_t1_offset, Integer *k_t2_offset, Integer *k_v2_offset) {
      static bool set_e = true;
      Assignment e_1_1;
      Multiplication e_1, e_1_2, e_2;

      DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;

      static Equations eqs;

      if (set_e) {
          ccsd_e_equations(eqs);
          set_e = false;
      }
      std::vector<Tensor> tensors;
      std::vector<Operation> ops;

      tensors_and_ops(eqs,tensors, ops);

      Tensor *i0 = &tensors[0];
      Tensor *f = &tensors[1];
      Tensor *v = &tensors[2];
      Tensor *t1 = &tensors[3];
      Tensor *t2 = &tensors[4];
      Tensor *i1_1 = &tensors[5];

      v->set_dist(idist);
      t1->set_dist(dist_nwma);
      f->attach(*k_f1_offset, 0, *d_f1);
      i0->attach(*k_i0_offset, 0, *d_i0);
      t1->attach(*k_t1_offset, 0, *d_t1);
      t2->attach(*k_t2_offset, 0, *d_t2);
      v->attach(*k_v2_offset, 0, *d_v2);

#if 0
      // schedule_linear(tensors, ops);
      // schedule_linear_lazy(tensors, ops);
      schedule_levels(tensors, ops);
#else
      e_1_1 = ops[0].add;
      e_1_2 = ops[1].mult;
      e_1 = ops[2].mult;
      e_2 = ops[3].mult;

      CorFortran(1, i1_1, offset_ccsd_e_1_1_);
      CorFortran(1,e_1_1, ccsd_e_copy_fock_to_t_);
      CorFortran(1,e_1_2,ccsd_e_1_2_);
      CorFortran(0,e_1,ccsd_e_1_);
      CorFortran(1,e_2,ccsd_e_2_);
      destroy(i1_1);
#endif
      f->detach();
      i0->detach();
      t1->detach();
      t2->detach();
      v->detach();
    }
  } // extern C
}; // namespace ctce

