#include "variables.h"
#include <iostream>
#include "tensor.h"
#include "t_mult.h"
#include "t_assign.h"
#include "input.h"
#include "corf.h"

/*
 * i0 ( p2 h1 )_f + = 1 * f ( p2 h1 )_f                                                         DONE
 * i0 ( p2 h1 )_tf + = -1 * Sum ( h7 ) * t ( p2 h7 )_t * i1 ( h7 h1 )_f                         DONE
 *     i1 ( h7 h1 )_f + = 1 * f ( h7 h1 )_f                                                     DONE
 *     i1 ( h7 h1 )_ft + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i2 ( h7 p3 )_f                      DONE
 *         i2 ( h7 p3 )_f + = 1 * f ( h7 p3 )_f                                                 DONE
 *         i2 ( h7 p3 )_vt + = -1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h7 p3 p5 )_v         DONE
 *     i1 ( h7 h1 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 h7 h1 p4 )_v             NOPE
 *     i1 ( h7 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 h7 p3 p4 )_v  NOPE
 * i0 ( p2 h1 )_tf + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i1 ( p2 p3 )_f                          DONE
 *     i1 ( p2 p3 )_f + = 1 * f ( p2 p3 )_f                                                     DONE
 *     i1 ( p2 p3 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 p2 p3 p4 )_v             NOPE
 * i0 ( p2 h1 )_vt + = -1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 p2 h1 p3 )_v                 NOPE
 * i0 ( p2 h1 )_tf + = 1 * Sum ( p7 h8 ) * t ( p2 p7 h1 h8 )_t * i1 ( h8 p7 )_f                 DONE
 *     i1 ( h8 p7 )_f + = 1 * f ( h8 p7 )_f                                                     DONE
 *     i1 ( h8 p7 )_vt + = 1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h8 p5 p7 )_v              NOPE
 * i0 ( p2 h1 )_vt + = -1/2 * Sum ( h4 h5 p3 ) * t ( p2 p3 h4 h5 )_t * i1 ( h4 h5 h1 p3 )_v     NOPE
 *     i1 ( h4 h5 h1 p3 )_v + = 1 * v ( h4 h5 h1 p3 )_v                                         DONE
 *     i1 ( h4 h5 h1 p3 )_vt + = -1 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h4 h5 p3 p6 )_v          NOPE
 * i0 ( p2 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 p2 p3 p4 )_v      DONE
 */

/* 
 * t1_1: i0 ( p2 h1 )_f + = 1 * f ( p2 h1 )_f
 * t1_2_1: i1 ( h7 h1 )_f + = 1 * f ( h7 h1 )_f 
 * t1_2_2_1: i2 ( h7 p3 )_f + = 1 * f ( h7 p3 )_f 
 * t1_2_2_2: i2 ( h7 p3 )_vt + = -1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h7 p3 p5 )_v 
 * t1_2_2: i1 ( h7 h1 )_ft + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i2 ( h7 p3 )_f 
 * t1_2_3: i1 ( h7 h1 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 h7 h1 p4 )_v 
 * t1_2_4: i1 ( h7 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 h7 p3 p4 )_v 
 * t1_2: i0 ( p2 h1 )_tf + = -1 * Sum ( h7 ) * t ( p2 h7 )_t * i1 ( h7 h1 )_f 
 * t1_3_1: i1 ( p2 p3 )_f + = 1 * f ( p2 p3 )_f 
 * t1_3_2: i1 ( p2 p3 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 p2 p3 p4 )_v 
 * t1_3: i0 ( p2 h1 )_tf + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i1 ( p2 p3 )_f 
 * t1_4: i0 ( p2 h1 )_vt + = -1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 p2 h1 p3 )_v 
 * t1_5_1: i1 ( h8 p7 )_f + = 1 * f ( h8 p7 )_f 
 * t1_5_2: i1 ( h8 p7 )_vt + = 1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h8 p5 p7 )_v 
 * t1_5: i0 ( p2 h1 )_tf + = 1 * Sum ( p7 h8 ) * t ( p2 p7 h1 h8 )_t * i1 ( h8 p7 )_f 
 * t1_6_1: i1 ( h4 h5 h1 p3 )_v + = 1 * v ( h4 h5 h1 p3 )_v 
 * t1_6_2: i1 ( h4 h5 h1 p3 )_vt + = -1 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h4 h5 p3 p6 )_v 
 * t1_6: i0 ( p2 h1 )_vt + = -1/2 * Sum ( h4 h5 p3 ) * t ( p2 p3 h4 h5 )_t * i1 ( h4 h5 h1 p3 )_v 
 * t1_7: i0 ( p2 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 p2 p3 p4 )_v 
 */

extern "C" {
  void ccsd_t1_1_(Integer *d_f1, Integer *k_f1_offset, Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t1_2_1_(Integer *d_f1, Integer *k_f1_offset, Integer *d_i1, Integer *k_i1_offset);
  void ccsd_t1_2_2_1_(Integer *d_f1, Integer *k_f1_offset, Integer *d_i2, Integer *k_i2_offset);
  void ccsd_t1_2_2_2_(Integer *d_t1, Integer *k_t1_offset, Integer *d_v2, Integer *k_v2_offset,
                      Integer *d_i2, Integer *k_i2_offset);
  void ccsd_t1_2_2_(Integer *d_t1, Integer *k_t1_offset, Integer *d_i2, Integer *k_i2_offset,
                    Integer *d_i1, Integer *k_i1_offset);
  void ccsd_t1_2_3_(Integer *d_t1, Integer *k_t1_offset, Integer *d_v2, Integer *k_v2_offset,
                    Integer *d_i1, Integer *k_i1_offset);
  void ccsd_t1_2_4_(Integer *d_t2, Integer *k_t2_offset, Integer *d_v2, Integer *k_v2_offset,
                    Integer *d_i1, Integer *k_i1_offset);
  void ccsd_t1_2_(Integer *d_t1, Integer *k_t1_offset, Integer *d_i1, Integer *k_i1_offset, 
                  Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t1_3_1_(Integer *d_f1, Integer *k_f1_offset, Integer *d_i1, Integer *k_i1_offset);
  void ccsd_t1_3_2_(Integer *d_t1, Integer *k_t1_offset, Integer *d_v2, Integer *k_v2_offset,
                    Integer *d_i1, Integer *k_i1_offset);
  void ccsd_t1_3_(Integer *d_t1, Integer *k_t1_offset, Integer *d_i1, Integer *k_i1_offset, 
                  Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t1_4_(Integer *d_t1, Integer *k_t1_offset, Integer *d_v2, Integer *k_v2_offset, 
                  Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t1_5_1_(Integer *d_f1, Integer *k_f1_offset, Integer *d_i1, Integer *k_i1_offset);
  void ccsd_t1_5_2_(Integer *d_t1, Integer *k_t1_offset, Integer *d_v2, Integer *k_v2_offset,
                    Integer *d_i1, Integer *k_i1_offset);
  void ccsd_t1_5_(Integer *d_t2, Integer *k_t2_offset, Integer *d_i1, Integer *k_i1_offset, 
                  Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t1_6_1_(Integer *d_v2, Integer *k_v2_offset, Integer *d_i1, Integer *k_i1_offset);
  void ccsd_t1_6_2_(Integer *d_t1, Integer *k_t1_offset, Integer *d_v2, Integer *k_v2_offset,
                    Integer *d_i1, Integer *k_i1_offset);
  void ccsd_t1_6_(Integer *d_t2, Integer *k_t2_offset, Integer *d_i1, Integer *k_i1_offset, 
                  Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t1_7_(Integer *d_t2, Integer *k_t2_offset, Integer *d_v2, Integer *k_v2_offset,
                  Integer *d_i0, Integer *k_i0_offset);

  void offset_ccsd_t1_2_1_(Integer *l_i1_offset, Integer *k_i1_offset, Integer *size_i1);
  void offset_ccsd_t1_2_2_1_(Integer *l_i2_offset, Integer *k_i2_offset, Integer *size_i2);
  void offset_ccsd_t1_3_1_(Integer *l_i1_offset, Integer *k_i1_offset, Integer *size_i1);
  void offset_ccsd_t1_5_1_(Integer *l_i1_offset, Integer *k_i1_offset, Integer *size_i1);
  void offset_ccsd_t1_6_1_(Integer *l_i1_offset, Integer *k_i1_offset, Integer *size_i1);
}


namespace ctce {

  static RangeEntry ranges[] = {
    {"O", TO},
    {"V", TV},
    {"N", TN}
  };

  static RangeEntry *O = &ranges[0];
  static RangeEntry *V = &ranges[1];
  static RangeEntry *N = &ranges[2];


  static IndexEntry indices[] = {
    {"p1", &ranges[1], P1B},
    {"p2", &ranges[1], P2B},
    {"p3", &ranges[1], P3B},
    {"p4", &ranges[1], P4B},
    {"p5", &ranges[1], P5B},
    {"p6", &ranges[1], P6B},
    {"p7", &ranges[1], P7B},
    {"p8", &ranges[1], P8B},
    {"p9", &ranges[1], P9B},
    {"p10", &ranges[1], P10B},
    {"p11", &ranges[1], P11B},
    {"p12", &ranges[1], P12B},
    {"h1", &ranges[0], H1B},
    {"h2", &ranges[0], H2B},
    {"h3", &ranges[0], H3B},
    {"h4", &ranges[0], H4B},
    {"h5", &ranges[0], H5B},
    {"h6", &ranges[0], H6B},
    {"h7", &ranges[0], H7B},
    {"h8", &ranges[0], H8B},
    {"h9", &ranges[0], H9B},
    {"h10", &ranges[0], H10B},
    {"h11", &ranges[0], H11B},
    {"h12", &ranges[0], H12B},
  };

  static IndexEntry *p1 = &indices[P1B];
  static IndexEntry *p2 = &indices[P2B];
  static IndexEntry *p3 = &indices[P3B];
  static IndexEntry *p4 = &indices[P4B];
  static IndexEntry *p5 = &indices[P5B];
  static IndexEntry *p6 = &indices[P6B];
  static IndexEntry *p7 = &indices[P7B];
  static IndexEntry *p8 = &indices[P8B];
  static IndexEntry *p9 = &indices[P9B];
  static IndexEntry *p10 = &indices[P10B];
  static IndexEntry *p11 = &indices[P11B];
  static IndexEntry *p12 = &indices[P12B];
  static IndexEntry *h1 = &indices[H1B];
  static IndexEntry *h2 = &indices[H2B];
  static IndexEntry *h3 = &indices[H3B];
  static IndexEntry *h4 = &indices[H4B];
  static IndexEntry *h5 = &indices[H5B];
  static IndexEntry *h6 = &indices[H6B];
  static IndexEntry *h7 = &indices[H7B];
  static IndexEntry *h8 = &indices[H8B];
  static IndexEntry *h9 = &indices[H9B];
  static IndexEntry *h10 = &indices[H10B];
  static IndexEntry *h11 = &indices[H11B];
  static IndexEntry *h12 = &indices[H12B];

  static TensorEntry tensors[] = {
    {"i0", {V, O}, 2, 1},
    {"f", {N, N}, 2, 1},
    {"v", {N, N, N, N}, 4, 2},
    {"t1", {V, O}, 2, 1},
    {"t2", {V, V, O, O}, 4, 2},
    {"i1_2", {O, O}, 2, 1},
    {"i1_2_2", {O, V}, 2, 1},
    {"i1_3", {V, V}, 2, 1},
    {"i1_5", {O, V}, 2, 1},
    {"i1_6", {O, O, O, V}, 4, 2},
  };

  static IndexEntry *ie = indices;
  static TensorEntry *ti0 = &tensors[0];
  static TensorEntry *tf = &tensors[1];
  static TensorEntry *tv = &tensors[2];
  static TensorEntry *tt1 = &tensors[3];
  static TensorEntry *tt2 = &tensors[4];
  static TensorEntry *ti1_2 = &tensors[5];
  static TensorEntry *ti1_2_2 = &tensors[6];
  static TensorEntry *ti1_3 = &tensors[7];
  static TensorEntry *ti1_5 = &tensors[8];
  static TensorEntry *ti1_6 = &tensors[9];

  static AddOp addops[] = {
    {ti0, tf, 1.0, {p2, h1}, {p2, h1}},             //t1_1
    {ti1_2, tf, 1.0, {h7, h1}, {h7, h1}},           //t1_2_1
    {ti1_2_2, tf, 1.0, {h7,p3}, {h7,p3}},           //t1_2_2_1
    {ti1_3, tf, 1.0, {p2,p3}, {p2,p3}},             //t1_3_1
    {ti1_5, tf, 1.0, {h8,p7}, {h8,p7}},             //t1_5_1
    {ti1_6, tv, 1.0, {h4,h5,h1,p3}, {h4,h5,h1,p3}}, //t1_6_1    
  };

  static MultOp multops[] = {
    {ti1_2_2, tt1, tv, -1.0, {h7, p3}, {p5,h6}, {h6,h7,p3,p5}},     //t1_2_2_2
    {ti1_2, tt1, ti1_2_2, 1.0, {h7,h1}, {p3,h1}, {h7,p3}},          //t1_2_2
    {ti1_2, tt1, tv, -1.0, {h7, h1}, {p4,h5}, {h5,h7,h1,p4}},       //t1_2_3
    {ti1_2, tt2, tv, -0.5, {h7,h1}, {p3,p4,h1,h5}, {h5,h7,p3,p4}},  //t1_2_4
    {ti0, tt1, ti1_2, -1.0, {p2,h1}, {p2,h7}, {h7,h1}},             //t1_2
    {ti1_3, tt1, tv, -1.0, {p2,p3}, {p4,h5}, {h5,p2,p3,p4}},        //t1_3_2
    {ti0, tt1, ti1_3, 1.0, {p2,h1}, {p3,h1}, {p2,p3}},              //t1_3
    {ti0, tt1, tv, -1.0, {p2,h1}, {p3,h4}, {h4,p2,h1,p3}},          //t1_4
    {ti1_5, tt1, tv, 1.0, {h8,p7}, {p5,h6}, {h6,h8,p5,p7}},         //t1_5_2
    {ti0, tt2, ti1_5, 1.0, {p2,h1}, {p2,p7,h1,h8}, {h8,p7}},        //t1_5
    {ti1_6, tt1, tv, -1.0, {h4,h5,h1,p3}, {p6,h1}, {h4,h5,p3,p6}},  //t1_6_2
    {ti0, tt2, ti1_6, -0.5, {p2,h1}, {p2,p3,h4,h5}, {h4,h5,h1,p3}}, //t1_6
    {ti0, tt2, tv, -0.5, {p2,h1}, {p3,p4,h1,h5}, {h5,p2,p3,p4}},    //t1_7
  };

  static Operation ops[] = {
    { OpTypeAdd, &addops[0]},    // t1_1
    { OpTypeAdd, &addops[1]},    // t1_2_1
    { OpTypeAdd, &addops[2]},    // t1_2_2_1
    { OpTypeMult, &multops[0]},  // t1_2_2_2
    { OpTypeMult, &multops[1]},  // t1_2_2
    { OpTypeMult, &multops[2]},  // t1_2_3
    { OpTypeMult, &multops[3]},  // t1_2_4
    { OpTypeMult, &multops[4]},  // t1_2
    { OpTypeAdd, &addops[3]},    // t1_3_1
    { OpTypeMult, &multops[5]},  // t1_3_2
    { OpTypeMult, &multops[6]},  // t1_3
    { OpTypeMult, &multops[7]},  // t1_4
    { OpTypeAdd, &addops[4]},    // t1_5_1
    { OpTypeMult, &multops[8]},  // t1_5_2
    { OpTypeMult, &multops[9]},  // t1_5
    { OpTypeAdd, &addops[5]},    // t1_6_1
    { OpTypeMult, &multops[10]}, // t1_6_2
    { OpTypeMult, &multops[11]}, // t1_6
    { OpTypeMult, &multops[12]}, // t1_7
  };

  static int num_ranges = sizeof(ranges)/sizeof(ranges[0]);
  static int num_indices = sizeof(indices)/sizeof(indices[0]);
  static int num_tensors = sizeof(tensors)/sizeof(tensors[0]);
  static int num_operations = sizeof(ops) / sizeof(ops[0]);

  extern "C" {
    
    void ccsd_t1_cxx_(Integer *d_f1, Integer *d_i0, Integer *d_t1, Integer *d_t2, Integer *d_v2, 
                      Integer *k_f1_offset, Integer *k_i0_offset,
                      Integer *k_t1_offset, Integer *k_t2_offset, Integer *k_v2_offset) {
      static bool set_t1 = true;
      Assignment a_t1_1, a_t1_2_1, a_t1_2_2_1, a_t1_3_1, a_t1_5_1, a_t1_6_1;
      Multiplication m_t1_2_2_2, m_t1_2_2, m_t1_2_3, m_t1_2_4, m_t1_2, m_t1_3_2, m_t1_3, m_t1_4, m_t1_5_2, m_t1_5, m_t1_6_2, m_t1_6, m_t1_7;

      DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;

      if (set_t1) {
        input_initialize(num_ranges, ranges,
                         num_indices, indices,
                         num_tensors, tensors,
                         num_operations, ops);
        tensors[3].tensor.set_dist(dist_nwma);
        tensors[2].tensor.set_dist(idist);
        set_t1 = false;
      }

      Tensor *i0 = &tensors[0].tensor;
      Tensor *f = &tensors[1].tensor;
      Tensor *v = &tensors[2].tensor;
      Tensor *t1 = &tensors[3].tensor;
      Tensor *t2 = &tensors[4].tensor;
      Tensor *i1_2 = &tensors[5].tensor;
      Tensor *i1_2_2 = &tensors[6].tensor;
      Tensor *i1_3 = &tensors[7].tensor;
      Tensor *i1_5 = &tensors[8].tensor;
      Tensor *i1_6 = &tensors[9].tensor;

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

      f->attach(*k_f1_offset, 0, *d_f1);
      i0->attach(*k_i0_offset, 0, *d_i0);
      t1->attach(*k_t1_offset, 0, *d_t1);
      t2->attach(*k_t2_offset, 0, *d_t2);
      v->attach(*k_v2_offset, 0, *d_v2);

      CorFortran(0,a_t1_1,ccsd_t1_1_);
      CorFortran(0,i1_2,offset_ccsd_t1_2_1_);
      CorFortran(0,a_t1_2_1,ccsd_t1_2_1_);
      CorFortran(0,i1_2_2,offset_ccsd_t1_2_2_1_);
      CorFortran(0,a_t1_2_2_1,ccsd_t1_2_2_1_);
      CorFortran(0,m_t1_2_2_2,ccsd_t1_2_2_2_);
      CorFortran(0,m_t1_2_2,ccsd_t1_2_2_);
      destroy(i1_2_2);
      CorFortran(0,m_t1_2_3,ccsd_t1_2_3_);
      CorFortran(0,m_t1_2_4,ccsd_t1_2_4_);
      CorFortran(0,m_t1_2,ccsd_t1_2_);
      destroy(i1_2);
      CorFortran(0,i1_3,offset_ccsd_t1_3_1_);
      CorFortran(0,a_t1_3_1,ccsd_t1_3_1_);
      CorFortran(0,m_t1_3_2,ccsd_t1_3_2_);
      CorFortran(0,m_t1_3,ccsd_t1_3_);
      destroy(i1_3);
      CorFortran(0,m_t1_4,ccsd_t1_4_);
      CorFortran(0,i1_5,offset_ccsd_t1_5_1_);
      CorFortran(0,a_t1_5_1,ccsd_t1_5_1_);
      CorFortran(0,m_t1_5_2,ccsd_t1_5_2_);
      CorFortran(0,m_t1_5,ccsd_t1_5_);
      destroy(i1_5);
      CorFortran(0,i1_6,offset_ccsd_t1_6_1_);
      CorFortran(0,a_t1_6_1,ccsd_t1_6_1_);
      CorFortran(0,m_t1_6_2,ccsd_t1_6_2_);
      CorFortran(0,m_t1_6,ccsd_t1_6_);
      destroy(i1_6);
      CorFortran(0,m_t1_7,ccsd_t1_7_);
    }
  } // extern C
}; // namespace ctce

