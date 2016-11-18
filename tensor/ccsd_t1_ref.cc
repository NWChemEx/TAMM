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
 * t1_1: i0 ( p2 h1 )_f + = 1 * f ( p2 h1 )_f
 * t1_2_1: i1 ( h7 h1 )_f + = 1 * f ( h7 h1 )_f
 * t1_2_2_1: i2 ( h7 p3 )_f + = 1 * f ( h7 p3 )_f
 * t1_2_2_2: i2 ( h7 p3 )_vt + = -1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h7
 * p3 p5 )_v
 * t1_2_2: i1 ( h7 h1 )_ft + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i2 ( h7 p3 )_f
 * t1_2_3: i1 ( h7 h1 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 h7 h1
 * p4 )_v
 * t1_2_4: i1 ( h7 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v
 * ( h5 h7 p3 p4 )_v
 * t1_2: i0 ( p2 h1 )_tf + = -1 * Sum ( h7 ) * t ( p2 h7 )_t * i1 ( h7 h1 )_f
 * t1_3_1: i1 ( p2 p3 )_f + = 1 * f ( p2 p3 )_f
 * t1_3_2: i1 ( p2 p3 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 p2 p3
 * p4 )_v
 * t1_3: i0 ( p2 h1 )_tf + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i1 ( p2 p3 )_f
 * t1_4: i0 ( p2 h1 )_vt + = -1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 p2 h1
 * p3 )_v
 * t1_5_1: i1 ( h8 p7 )_f + = 1 * f ( h8 p7 )_f
 * t1_5_2: i1 ( h8 p7 )_vt + = 1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h8 p5
 * p7 )_v
 * t1_5: i0 ( p2 h1 )_tf + = 1 * Sum ( p7 h8 ) * t ( p2 p7 h1 h8 )_t * i1 ( h8
 * p7 )_f
 * t1_6_1: i1 ( h4 h5 h1 p3 )_v + = 1 * v ( h4 h5 h1 p3 )_v
 * t1_6_2: i1 ( h4 h5 h1 p3 )_vt + = -1 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h4 h5
 * p3 p6 )_v
 * t1_6: i0 ( p2 h1 )_vt + = -1/2 * Sum ( h4 h5 p3 ) * t ( p2 p3 h4 h5 )_t * i1
 * ( h4 h5 h1 p3 )_v
 * t1_7: i0 ( p2 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v (
 * h5 p2 p3 p4 )_v
 */

extern "C" {
void ccsd_t_vo_1_(Integer *d_f1, Integer *k_f1_offset, Integer *d_i0,
                  Integer *k_i0_offset);
void ccsd_t_vo_2_1_(Integer *d_f1, Integer *k_f1_offset, Integer *d_i1,
                    Integer *k_i1_offset);
void ccsd_t_vo_2_2_1_(Integer *d_f1, Integer *k_f1_offset, Integer *d_i2,
                      Integer *k_i2_offset);
void ccsd_t_vo_2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v2,
                      Integer *k_v2_offset, Integer *d_i2,
                      Integer *k_i2_offset);
void ccsd_t_vo_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_i2,
                    Integer *k_i2_offset, Integer *d_i1, Integer *k_i1_offset);
void ccsd_t_vo_2_3_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v2,
                    Integer *k_v2_offset, Integer *d_i1, Integer *k_i1_offset);
void ccsd_t_vo_2_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset, Integer *d_v2,
                    Integer *k_v2_offset, Integer *d_i1, Integer *k_i1_offset);
void ccsd_t_vo_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_i1,
                  Integer *k_i1_offset, Integer *d_i0, Integer *k_i0_offset);
void ccsd_t_vo_3_1_(Integer *d_f1, Integer *k_f1_offset, Integer *d_i1,
                    Integer *k_i1_offset);
void ccsd_t_vo_3_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v2,
                    Integer *k_v2_offset, Integer *d_i1, Integer *k_i1_offset);
void ccsd_t_vo_3_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_i1,
                  Integer *k_i1_offset, Integer *d_i0, Integer *k_i0_offset);
void ccsd_t_vo_4_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v2,
                  Integer *k_v2_offset, Integer *d_i0, Integer *k_i0_offset);
void ccsd_t_vo_5_1_(Integer *d_f1, Integer *k_f1_offset, Integer *d_i1,
                    Integer *k_i1_offset);
void ccsd_t_vo_5_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v2,
                    Integer *k_v2_offset, Integer *d_i1, Integer *k_i1_offset);
void ccsd_t_vo_5_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset, Integer *d_i1,
                  Integer *k_i1_offset, Integer *d_i0, Integer *k_i0_offset);
void ccsd_t_vo_6_1_(Integer *d_v2, Integer *k_v2_offset, Integer *d_i1,
                    Integer *k_i1_offset);
void ccsd_t_vo_6_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v2,
                    Integer *k_v2_offset, Integer *d_i1, Integer *k_i1_offset);
void ccsd_t_vo_6_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset, Integer *d_i1,
                  Integer *k_i1_offset, Integer *d_i0, Integer *k_i0_offset);
void ccsd_t_vo_7_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset, Integer *d_v2,
                  Integer *k_v2_offset, Integer *d_i0, Integer *k_i0_offset);

void offset_ccsd_t_vo_2_1_(Integer *l_i1_offset, Integer *k_i1_offset,
                           Integer *size_i1);
void offset_ccsd_t_vo_2_2_1_(Integer *l_i2_offset, Integer *k_i2_offset,
                             Integer *size_i2);
void offset_ccsd_t_vo_3_1_(Integer *l_i1_offset, Integer *k_i1_offset,
                           Integer *size_i1);
void offset_ccsd_t_vo_5_1_(Integer *l_i1_offset, Integer *k_i1_offset,
                           Integer *size_i1);
void offset_ccsd_t_vo_6_1_(Integer *l_i1_offset, Integer *k_i1_offset,
                           Integer *size_i1);
}

namespace tamm {

extern "C" {

void ccsd_t1_cxx_(Integer *d_f1, Integer *d_i0, Integer *d_t_vo,
                  Integer *d_t_vvoo, Integer *d_v2, Integer *k_f1_offset,
                  Integer *k_i0_offset, Integer *k_t_vo_offset,
                  Integer *k_t_vvoo_offset, Integer *k_v2_offset) {
  static bool set_t1 = true;
  Assignment a_t1_1, a_t1_2_1, a_t1_2_2_1, a_t1_3_1, a_t1_5_1, a_t1_6_1;
  Multiplication m_t1_2_2_2, m_t1_2_2, m_t1_2_3, m_t1_2_4, m_t1_2, m_t1_3_2,
      m_t1_3, m_t1_4, m_t1_5_2, m_t1_5, m_t1_6_2, m_t1_6, m_t1_7;

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

  v->set_dist(idist);
  t_vo->set_dist(dist_nwma);
  f->attach(*k_f1_offset, 0, *d_f1);
  i0->attach(*k_i0_offset, 0, *d_i0);
  t_vo->attach(*k_t_vo_offset, 0, *d_t_vo);
  t_vvoo->attach(*k_t_vvoo_offset, 0, *d_t_vvoo);
  v->attach(*k_v2_offset, 0, *d_v2);

#if 1
  // schedule_linear(tensors, ops);
  // schedule_linear_lazy(tensors, ops);
  schedule_levels(&tensors, &ops);
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

  CorFortran(1, &a_t1_1, ccsd_t_vo_1_);
  CorFortran(1, &i1_2, offset_ccsd_t_vo_2_1_);
  CorFortran(1, &a_t1_2_1, ccsd_t_vo_2_1_);
  CorFortran(1, &i1_2_2, offset_ccsd_t_vo_2_2_1_);
  CorFortran(1, &a_t1_2_2_1, ccsd_t_vo_2_2_1_);
  CorFortran(1, &m_t1_2_2_2, ccsd_t_vo_2_2_2_);
  CorFortran(1, &m_t1_2_2, ccsd_t_vo_2_2_);
  destroy(i1_2_2);
  CorFortran(1, &m_t1_2_3, ccsd_t_vo_2_3_);
  CorFortran(1, &m_t1_2_4, ccsd_t_vo_2_4_);
  CorFortran(1, &m_t1_2, ccsd_t_vo_2_);
  destroy(i1_2);
  CorFortran(1, &i1_3, offset_ccsd_t_vo_3_1_);
  CorFortran(1, &a_t1_3_1, ccsd_t_vo_3_1_);
  CorFortran(1, &m_t1_3_2, ccsd_t_vo_3_2_);
  CorFortran(1, &m_t1_3, ccsd_t_vo_3_);
  destroy(i1_3);
  CorFortran(1, &m_t1_4, ccsd_t_vo_4_);
  CorFortran(1, &i1_5, offset_ccsd_t_vo_5_1_);
  CorFortran(1, &a_t1_5_1, ccsd_t_vo_5_1_);
  CorFortran(1, &m_t1_5_2, ccsd_t_vo_5_2_);
  CorFortran(1, &m_t1_5, ccsd_t_vo_5_);
  destroy(i1_5);
  CorFortran(1, &i1_6, offset_ccsd_t_vo_6_1_);
  CorFortran(1, &a_t1_6_1, ccsd_t_vo_6_1_);
  CorFortran(1, &m_t1_6_2, ccsd_t_vo_6_2_);
  CorFortran(1, &m_t1_6, ccsd_t_vo_6_);
  destroy(i1_6);
  CorFortran(1, &m_t1_7, ccsd_t_vo_7_);
#endif  // Use c scheduler
  f->detach();
  i0->detach();
  t_vo->detach();
  t_vvoo->detach();
  v->detach();
}
}  // extern C
};  // namespace tamm
