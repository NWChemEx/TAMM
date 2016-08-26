#include "variables.h"
#include <iostream>
#include "tensor.h"
#include "t_mult.h"
#include "t_assign.h"
#include "input.h"
#include "corf.h"
#include "equations.h"
#include "tensors_and_ops.h"
//#include "expression.h"
//kbn p2
//     i0 ( p2 h1 )_xf + = -1 * Sum ( h6 ) * x ( p2 h6 )_x * i1 ( h6 h1 )_f
//         i1 ( h6 h1 )_f + = 1 * f ( h6 h1 )_f
//         i1 ( h6 h1 )_ft + = 1 * Sum ( p7 ) * t ( p7 h1 )_t * i2 ( h6 p7 )_f
//             i2 ( h6 p7 )_f + = 1 * f ( h6 p7 )_f
//             i2 ( h6 p7 )_vt + = 1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 h6 p4 p7 )_v
//         i1 ( h6 h1 )_vt + = -1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 h6 h1 p3 )_v
//         i1 ( h6 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 h6 p3 p4 )_v
//ckbn p2
//     i0 ( p2 h1 )_xf + = 1 * Sum ( p7 h6 ) * x ( p2 p7 h1 h6 )_x * i1 ( h6 p7 )_f
//         i1 ( h6 p7 )_f + = 1 * f ( h6 p7 )_f
//         i1 ( h6 p7 )_vt + = 1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 h6 p3 p7 )_v
//ckbn p2
//     i0 ( p2 h1 )_xv + = -1/2 * Sum ( p7 h6 h8 ) * x ( p2 p7 h6 h8 )_x * i1 ( h6 h8 h1 p7 )_v
//         i1 ( h6 h8 h1 p7 )_v + = 1 * v ( h6 h8 h1 p7 )_v
//         i1 ( h6 h8 h1 p7 )_vt + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * v ( h6 h8 p3 p7 )_v

extern "C" {

  void ipccsd_x1_1_1_(Integer *d_f1, Integer *k_f1_offset, Integer *d_i1, Integer *k_i1_offset);


}


namespace ctce {

  void schedule_linear(std::vector<Tensor> &tensors,
                       std::vector<Operation> &ops);
  void schedule_linear_lazy(std::vector<Tensor> &tensors,
                            std::vector<Operation> &ops);
  void schedule_levels(std::vector<Tensor> &tensors,
                            std::vector<Operation> &ops);

  extern "C" {
    
    void ipccsd_x1_cxx_(Integer *d_f1, Integer *d_i0, Integer *d_t1, Integer *d_t2, 
Integer *d_v2, Integer *d_x1, Integer *d_x2, Integer *k_f1_offset, 
Integer *k_i0_offset, Integer *k_t1_offset, Integer *k_t2_offset, 
Integer *k_v2_offset, Integer *k_x1_offset, Integer *k_x2_offset) {
      static bool set_t1 = true;
      Assignment a_t1_1, a_t1_2_1, a_t1_2_2_1, a_t1_3_1, a_t1_5_1, a_t1_6_1;
      Multiplication m_t1_2_2_2, m_t1_2_2, m_t1_2_3, m_t1_2_4, m_t1_2, m_t1_3_2, m_t1_3, m_t1_4, m_t1_5_2, m_t1_5, m_t1_6_2, m_t1_6, m_t1_7;

      DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;

      static Equations eqs;

      if (set_t1) {
        ccsd_t1_equations(eqs);
        set_t1 = false;
      }
      std::vector<Tensor> tensors;
      std::vector<Operation> ops;

      tensors_and_ops(eqs,tensors, ops);

      Tensor *i0 = &tensors[0];
      Tensor *f = &tensors[1];
      Tensor *v = &tensors[2];
      Tensor *t1 = &tensors[3];
      Tensor *t2 = &tensors[4];
      Tensor *x1 = &tensors[5];
      Tensor *x2 = &tensors[6];

/*
      Tensor *i1_2 = &tensors[5];
      Tensor *i1_2_2 = &tensors[6];
      Tensor *i1_3 = &tensors[7];
      Tensor *i1_5 = &tensors[8];
      Tensor *i1_6 = &tensors[9];
*/

      v->set_dist(idist);
      t1->set_dist(dist_nwma);
      f->attach(*k_f1_offset, 0, *d_f1);
      i0->attach(*k_i0_offset, 0, *d_i0);
      t1->attach(*k_t1_offset, 0, *d_t1);
      t2->attach(*k_t2_offset, 0, *d_t2);
      v->attach(*k_v2_offset, 0, *d_v2);
      x1->attach(*k_x1_offset, 0, *d_x1);
      x2->attach(*k_x2_offset, 0, *d_x2);

#if 1
      // schedule_linear(tensors, ops);
      // schedule_linear_lazy(tensors, ops);
       schedule_levels(tensors, ops);
#else

      a_t1_1 = ops[0].add;
      a_t1_2_1 = ops[1].add;
      a_t1_2_2_1 = ops[2].add;
      m_t1_2_2_2 = ops[3].mult;
      m_t1_2_2 = ops[4].mult;
      m_t1_2_3 = ops[5].mult;
      m_t1_2_4 = ops[6].mult;
      m_t1_2 = ops[7].mult;
      a_t1_3_1 = ops[8].add;
      m_t1_3_2 = ops[9].mult;
      m_t1_3 = ops[10].mult;
      m_t1_4 = ops[11].mult;
      a_t1_5_1 = ops[12].add;
      m_t1_5_2 = ops[13].mult;
      m_t1_5 = ops[14].mult;
      a_t1_6_1 = ops[15].add;
      m_t1_6_2 = ops[16].mult;
      m_t1_6 = ops[17].mult;
      m_t1_7 = ops[18].mult;

      CorFortran(1,a_t1_1,ccsd_t1_1_);
      CorFortran(1,i1_2,offset_ccsd_t1_2_1_);
      CorFortran(1,a_t1_2_1,ccsd_t1_2_1_);
      CorFortran(1,i1_2_2,offset_ccsd_t1_2_2_1_);
      CorFortran(1,a_t1_2_2_1,ccsd_t1_2_2_1_);
      CorFortran(1,m_t1_2_2_2,ccsd_t1_2_2_2_);
      CorFortran(1,m_t1_2_2,ccsd_t1_2_2_);
      destroy(i1_2_2);
      CorFortran(1,m_t1_2_3,ccsd_t1_2_3_);
      CorFortran(1,m_t1_2_4,ccsd_t1_2_4_);
      CorFortran(1,m_t1_2,ccsd_t1_2_);
      destroy(i1_2);
      CorFortran(1,i1_3,offset_ccsd_t1_3_1_);
      CorFortran(1,a_t1_3_1,ccsd_t1_3_1_);
      CorFortran(1,m_t1_3_2,ccsd_t1_3_2_);
      CorFortran(1,m_t1_3,ccsd_t1_3_);
      destroy(i1_3);
      CorFortran(1,m_t1_4,ccsd_t1_4_);
      CorFortran(1,i1_5,offset_ccsd_t1_5_1_);
      CorFortran(1,a_t1_5_1,ccsd_t1_5_1_);
      CorFortran(1,m_t1_5_2,ccsd_t1_5_2_);
      CorFortran(1,m_t1_5,ccsd_t1_5_);
      destroy(i1_5);
      CorFortran(1,i1_6,offset_ccsd_t1_6_1_);
      CorFortran(1,a_t1_6_1,ccsd_t1_6_1_);
      CorFortran(1,m_t1_6_2,ccsd_t1_6_2_);
      CorFortran(1,m_t1_6,ccsd_t1_6_);
      destroy(i1_6);
      CorFortran(1,m_t1_7,ccsd_t1_7_);
#endif
      f->detach();
      i0->detach();
      t1->detach();
      t2->detach();
      v->detach();
    }
  } // extern C
}; // namespace ctce


