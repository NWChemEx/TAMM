#include "variables.h"
#include <iostream>
#include "tensor.h"
#include "t_mult.h"
#include "t_assign.h"
#include "input.h"

namespace ctce {

  static Assignment a_t1_1, a_t1_2_1, a_t1_2_2_1, a_t1_3_1, a_t1_5_1, a_t1_6_1;
  static Multiplication m_t1_2_2_2, m_t1_2_2, m_t1_2_3, m_t1_2_4, m_t1_2, m_t1_3_2, m_t1_3, m_t1_4, m_t1_5_2, m_t1_5, m_t1_6_2, m_t1_6, m_t1_7;

  static Tensor i0, f, v, t1, t2;
  static Tensor i1_2, i1_2_2, i1_3, i1_5, i1_6;

  static RangeEntry ranges[] = {
    {"O", TO},
    {"V", TV},
    {"N", TN}
  };

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

  static TensorEntry tensors[] = {
    {"i0", {&ranges[1], &ranges[0]}, 2, 1, NULL},
    {"f", {&ranges[2], &ranges[2]}, 2, 1, NULL},
    {"v", {&ranges[2], &ranges[2], &ranges[2], &ranges[2]}, 4, 2, NULL},
    {"t1", {&ranges[1], &ranges[0]}, 2, 1, NULL},
    {"t2", {&ranges[1], &ranges[1], &ranges[0], &ranges[0]}, 4, 2, NULL},
    {"i1_2", {&ranges[0], &ranges[0]}, 2, 1, NULL},
    {"i1_2_2", {&ranges[0], &ranges[1]}, 2, 1, NULL},
    {"i1_3", {&ranges[1], &ranges[1]}, 2, 1, NULL},
    {"i1_5", {&ranges[0], &ranges[1]}, 2, 1, NULL},
    {"i1_6", {&ranges[0], &ranges[0], &ranges[0], &ranges[1]}, 4, 2, NULL},
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
/* t1_1: i0 ( p2 h1 )_f + = 1 * f ( p2 h1 )_f */
    {ti0, tf, 1.0, {ie+P2B, ie+H1B}, {ie+P2B, ie+H1B}}, //t1_1
/* t1_2_1: i1 ( h7 h1 )_f + = 1 * f ( h7 h1 )_f */
    {ti1_2, tf, 1.0, {ie+H7B, ie+H1B}, {ie+H7B, ie+H1B}}, //t1_2_1
/* t1_2_2_1: i2 ( h7 p3 )_f + = 1 * f ( h7 p3 )_f */
    {ti1_2_2, tf, 1.0, {ie+H7B,ie+P3B}, {ie+H7B,ie+P3B}}, //t1_2_2_1
/* t1_3_1: i1 ( p2 p3 )_f + = 1 * f ( p2 p3 )_f */
    {ti1_3, tf, 1.0, {ie+P2B,ie+P3B}, {ie+P2B,ie+P3B}}, //t1_3_1
/* t1_5_1: i1 ( h8 p7 )_f + = 1 * f ( h8 p7 )_f */
    {ti1_5, tf, 1.0, {ie+H8B,ie+P7B}, {ie+H8B,ie+P7B}}, //t1_5_1
/* t1_6_1: i1 ( h4 h5 h1 p3 )_v + = 1 * v ( h4 h5 h1 p3 )_v */
    {ti1_6, tv, 1.0, {ie+H4B,ie+H5B,ie+H1B,ie+P3B}, {ie+H4B,ie+H5B,ie+H1B,ie+P3B}}, //t1_6_1    
  };

  static MultOp multops[] = {
/* t1_2_2_2: i2 ( h7 p3 )_vt + = -1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h7 p3 p5 )_v */
   {ti1_2_2, tt1, tv, -1.0, {ie+H7B, ie+P3B}, {ie+P5B,ie+H6B}, {ie+H6B,ie+H7B,ie+P3B,ie+P5B}},
/* t1_2_2: i1 ( h7 h1 )_ft + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i2 ( h7 p3 )_f */
   {ti1_2, tt1, ti1_2_2, 1.0, {ie+H7B,ie+H1B}, {ie+P3B,ie+H1B}, {ie+H7B,ie+P3B}},
/* t1_2_3: i1 ( h7 h1 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 h7 h1 p4 )_v */
    {ti1_2, tt1, tv, -1.0, {ie+H7B, ie+H1B}, {ie+P4B,ie+H5B}, {ie+H5B,ie+H7B,ie+H1B,ie+P4B}},
/* t1_2_4: i1 ( h7 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 h7 p3 p4 )_v */
   {ti1_2, tt2, tv, -0.5, {ie+H7B,ie+H1B}, {ie+P3B,ie+P4B,ie+H1B,ie+H5B}, {ie+H5B,ie+H7B,ie+P3B,ie+P4B}},
/* t1_2: i0 ( p2 h1 )_tf + = -1 * Sum ( h7 ) * t ( p2 h7 )_t * i1 ( h7 h1 )_f */
   {ti0, tt1, ti1_2, -1.0, {ie+P2B,ie+H1B}, {ie+P2B,ie+H7B}, {ie+H7B,ie+H1B}},
/* t1_3_2: i1 ( p2 p3 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 p2 p3 p4 )_v */
   {ti1_3, tt1, tv, -1.0, {ie+P2B,ie+P3B}, {ie+P4B,ie+H5B}, {ie+H5B,ie+P2B,ie+P3B,ie+P4B}},
/* t1_3: i0 ( p2 h1 )_tf + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i1 ( p2 p3 )_f */
   {ti0, tt1, ti1_3, 1.0, {ie+P2B,ie+H1B}, {ie+P3B,ie+H1B}, {ie+P2B,ie+P3B}},
/* t1_4: i0 ( p2 h1 )_vt + = -1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 p2 h1 p3 )_v */
   {ti0, tt1, tv, -1.0, {ie+P2B,ie+H1B}, {ie+P3B,ie+H4B}, {ie+H4B,ie+P2B,ie+H1B,ie+P3B}},
/* t1_5_2: i1 ( h8 p7 )_vt + = 1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h8 p5 p7 )_v */
   {ti1_5, tt1, tv, 1.0, {ie+H8B,ie+P7B}, {ie+P5B,ie+H6B}, {ie+H6B,ie+H8B,ie+P5B,ie+P7B}},
/* t1_5: i0 ( p2 h1 )_tf + = 1 * Sum ( p7 h8 ) * t ( p2 p7 h1 h8 )_t * i1 ( h8 p7 )_f */
   {ti0, tt2, ti1_5, 1.0, {ie+P2B,ie+H1B}, {ie+P2B,ie+P7B,ie+H1B,ie+H8B}, {ie+H8B,ie+P7B}},
/* t1_6_2: i1 ( h4 h5 h1 p3 )_vt + = -1 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h4 h5 p3 p6 )_v */
   {ti1_6, tt1, tv, -1.0, {ie+H4B,ie+H5B,ie+H1B,ie+P3B}, {ie+P6B,ie+H1B}, {ie+H4B,ie+H5B,ie+P3B,ie+P6B}},
/* t1_6: i0 ( p2 h1 )_vt + = -1/2 * Sum ( h4 h5 p3 ) * t ( p2 p3 h4 h5 )_t * i1 ( h4 h5 h1 p3 )_v */
   {ti0, tt2, ti1_6, -0.5, {ie+P2B,ie+H1B}, {ie+P2B,ie+P3B,ie+H4B,ie+H5B}, {ie+H4B,ie+H5B,ie+H1B,ie+P3B}},
/* t1_7: i0 ( p2 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 p2 p3 p4 )_v */
   {ti0, tt2, tv, -0.5, {ie+P2B,ie+H1B}, {ie+P3B,ie+P4B,ie+H1B,ie+H5B}, {ie+H5B,ie+P2B,ie+P3B,ie+P4B}},
  };

/* t1_1: i0 ( p2 h1 )_f + = 1 * f ( p2 h1 )_f */
/* t1_2_1: i1 ( h7 h1 )_f + = 1 * f ( h7 h1 )_f */
/* t1_2_2_1: i2 ( h7 p3 )_f + = 1 * f ( h7 p3 )_f */
/* t1_2_2_2: i2 ( h7 p3 )_vt + = -1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h7 p3 p5 )_v */
/* t1_2_2: i1 ( h7 h1 )_ft + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i2 ( h7 p3 )_f */
/* t1_2_3: i1 ( h7 h1 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 h7 h1 p4 )_v */
/* t1_2_4: i1 ( h7 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 h7 p3 p4 )_v */
/* t1_2: i0 ( p2 h1 )_tf + = -1 * Sum ( h7 ) * t ( p2 h7 )_t * i1 ( h7 h1 )_f */
/* t1_3_1: i1 ( p2 p3 )_f + = 1 * f ( p2 p3 )_f */
/* t1_3_2: i1 ( p2 p3 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 p2 p3 p4 )_v */
/* t1_3: i0 ( p2 h1 )_tf + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i1 ( p2 p3 )_f */
/* t1_4: i0 ( p2 h1 )_vt + = -1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 p2 h1 p3 )_v */
/* t1_5_1: i1 ( h8 p7 )_f + = 1 * f ( h8 p7 )_f */
/* t1_5_2: i1 ( h8 p7 )_vt + = 1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h8 p5 p7 )_v */
/* t1_5: i0 ( p2 h1 )_tf + = 1 * Sum ( p7 h8 ) * t ( p2 p7 h1 h8 )_t * i1 ( h8 p7 )_f */
/* t1_6_1: i1 ( h4 h5 h1 p3 )_v + = 1 * v ( h4 h5 h1 p3 )_v */
/* t1_6_2: i1 ( h4 h5 h1 p3 )_vt + = -1 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h4 h5 p3 p6 )_v */
/* t1_6: i0 ( p2 h1 )_vt + = -1/2 * Sum ( h4 h5 p3 ) * t ( p2 p3 h4 h5 )_t * i1 ( h4 h5 h1 p3 )_v */
/* t1_7: i0 ( p2 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 p2 p3 p4 )_v */

  static Operation ops[] = {
    { OpTypeAdd, &addops[0]},    /* t1_1*/
    { OpTypeAdd, &addops[1]},    /* t1_2_1*/
    { OpTypeAdd, &addops[2]},    /* t1_2_2_1*/
    { OpTypeMult, &multops[0]},  /* t1_2_2_2*/
    { OpTypeMult, &multops[1]},  /* t1_2_2*/
    { OpTypeMult, &multops[2]},  /* t1_2_3*/
    { OpTypeMult, &multops[3]},  /* t1_2_4*/
    { OpTypeMult, &multops[4]},  /* t1_2*/
    { OpTypeAdd, &addops[3]},    /* t1_3_1*/
    { OpTypeMult, &multops[5]},  /* t1_3_2*/
    { OpTypeMult, &multops[6]},  /* t1_3*/
    { OpTypeMult, &multops[7]},  /* t1_4*/
    { OpTypeAdd, &addops[4]},    /* t1_5_1*/
    { OpTypeMult, &multops[8]},  /* t1_5_2*/
    { OpTypeMult, &multops[9]},  /* t1_5*/
    { OpTypeAdd, &addops[5]},    /* t1_6_1*/
    { OpTypeMult, &multops[10]},  /* t1_6_2*/
    { OpTypeMult, &multops[11]},  /* t1_6*/
    { OpTypeMult, &multops[12]},  /* t1_7*/
  };

  static int num_ranges = sizeof(ranges)/sizeof(ranges[0]);
  static int num_indices = sizeof(indices)/sizeof(indices[0]);
  static int num_tensors = sizeof(tensors)/sizeof(tensors[0]);
  static int num_operations = sizeof(ops) / sizeof(ops[0]);

  extern "C" {

    void gen_expr_t1_cxx_() {

      static bool set_t1 = true;
      //Tensor tC, tA, tB;

      DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;

      if (set_t1) {

      input_initialize(num_ranges, ranges,
                       num_indices, indices,
                       num_tensors, tensors,
                       num_operations, ops);

      // i0 = Tensor2(TV,TO,dist_nw);
      // f = Tensor2(TN,TN, dist_nw);
      // v = Tensor4(TN,TN,TN,TN,idist);
      // t1 = Tensor2(TV,TO,dist_nwma);
      // t2 = Tensor4(TV,TV,TO,TO,dist_nw);
      // i1_2 = Tensor2(TO,TO,dist_nw);
      // i1_2_2 = Tensor2(TO,TV,dist_nw);
      // i1_3 = Tensor2(TV,TV,dist_nw);
      // i1_5 = Tensor2(TO,TV,dist_nw);
      // i1_6 = Tensor4(TO,TO,TO,TV,dist_nw);

      tensors[3].tensor->set_dist(dist_nwma);
      tensors[2].tensor->set_dist(idist);

      input_ops_initialize(num_ranges, ranges,
                           num_indices, indices,
                           num_tensors, tensors,
                           num_operations, ops);
      /*
     i0 ( p2 h1 )_f + = 1 * f ( p2 h1 )_f                                                         DONE
     i0 ( p2 h1 )_tf + = -1 * Sum ( h7 ) * t ( p2 h7 )_t * i1 ( h7 h1 )_f                         DONE
         i1 ( h7 h1 )_f + = 1 * f ( h7 h1 )_f                                                     DONE
         i1 ( h7 h1 )_ft + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i2 ( h7 p3 )_f                      DONE
             i2 ( h7 p3 )_f + = 1 * f ( h7 p3 )_f                                                 DONE
             i2 ( h7 p3 )_vt + = -1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h7 p3 p5 )_v         DONE
         i1 ( h7 h1 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 h7 h1 p4 )_v             NOPE
         i1 ( h7 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 h7 p3 p4 )_v  NOPE
     i0 ( p2 h1 )_tf + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i1 ( p2 p3 )_f                          DONE
         i1 ( p2 p3 )_f + = 1 * f ( p2 p3 )_f                                                     DONE
         i1 ( p2 p3 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 p2 p3 p4 )_v             NOPE
     i0 ( p2 h1 )_vt + = -1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 p2 h1 p3 )_v                 NOPE
     i0 ( p2 h1 )_tf + = 1 * Sum ( p7 h8 ) * t ( p2 p7 h1 h8 )_t * i1 ( h8 p7 )_f                 DONE
         i1 ( h8 p7 )_f + = 1 * f ( h8 p7 )_f                                                     DONE
         i1 ( h8 p7 )_vt + = 1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h8 p5 p7 )_v              NOPE
     i0 ( p2 h1 )_vt + = -1/2 * Sum ( h4 h5 p3 ) * t ( p2 p3 h4 h5 )_t * i1 ( h4 h5 h1 p3 )_v     NOPE
         i1 ( h4 h5 h1 p3 )_v + = 1 * v ( h4 h5 h1 p3 )_v                                         DONE
         i1 ( h4 h5 h1 p3 )_vt + = -1 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h4 h5 p3 p6 )_v          NOPE
     i0 ( p2 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 p2 p3 p4 )_v      DONE
	*/

        /* i0 ( p2 h1 )_f + = 1 * f ( p2 h1 )_f */
      //a_t1_1 = Assignment(i0,f,1.0,ivec(P2B,H1B), ivec(P2B,H1B));
      a_t1_1 = ops[0].add;

        /* i1 ( h7 h1 )_f + = 1 * f ( h7 h1 )_f */
      //a_t1_2_1 = Assignment(i1_2,f,1.0, ivec(H7B,H1B), ivec(H7B,H1B));
         a_t1_2_1 = ops[1].add;

        /* i2 ( h7 p3 )_f + = 1 * f ( h7 p3 )_f */
         //a_t1_2_2_1 = Assignment(i1_2_2,f,1.0, ivec(H7B,P3B), ivec(H7B,P3B));
         a_t1_2_2_1 = ops[2].add;

        /* i2 ( h7 p3 )_vt + = -1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h7 p3 p5 )_v */
        // m_t1_2_2_2 = Multiplication(i1_2_2,ivec(H7B,P3B),
	// 			    t1,ivec(P5B,H6B),
	// 			    v,ivec(H6B,H7B,P3B,P5B),
	// 			    -1.0);
         m_t1_2_2_2 = ops[3].mult;

        /* i1 ( h7 h1 )_ft + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i2 ( h7 p3 )_f */
        // m_t1_2_2 = Multiplication(i1_2,ivec(H7B,H1B),
	// 			  t1,ivec(P3B,H1B),
	// 			  i1_2_2,ivec(H7B,P3B),
	// 			  1.0);
         m_t1_2_2 = ops[4].mult;

        /* i1 ( h7 h1 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 h7 h1 p4 )_v */
        // m_t1_2_3 = Multiplication(i1_2,ivec(H7B,H1B),
	// 			  t1,ivec(P4B,H5B),
	// 			  v,ivec(H5B,H7B,H1B,P4B),-1.0);
         m_t1_2_3 = ops[5].mult;

        /* i1 ( h7 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 h7 p3 p4 )_v */
        // m_t1_2_4 = Multiplication(i1_2,ivec(H7B,H1B),
	// 			  t2,ivec(P3B,P4B,H1B,H5B),
	// 			  v,ivec(H5B,H7B,P3B,P4B),-0.5);
         m_t1_2_4 = ops[6].mult;

        /* i0 ( p2 h1 )_tf + = -1 * Sum ( h7 ) * t ( p2 h7 )_t * i1 ( h7 h1 )_f */
        // m_t1_2 = Multiplication(i0,ivec(P2B,H1B),
	// 			t1,ivec(P2B,H7B),
	// 			i1_2,ivec(H7B,H1B),-1.0);
         m_t1_2 = ops[7].mult;

        /* i1 ( p2 p3 )_f + = 1 * f ( p2 p3 )_f */
        //a_t1_3_1 = Assignment(i1_3,f,1.0,ivec(P2B,P3B),ivec(P2B,P3B));
         a_t1_3_1 = ops[8].add;

        /* i1 ( p2 p3 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 p2 p3 p4 )_v */
        // m_t1_3_2 = Multiplication(i1_3,ivec(P2B,P3B),
	// 			  t1,ivec(P4B,H5B),
	// 			  v,ivec(H5B,P2B,P3B,P4B),-1.0);
        m_t1_3_2 = ops[9].mult;

        /* i0 ( p2 h1 )_tf + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i1 ( p2 p3 )_f */
        //m_t1_3 = Multiplication(i0,ivec(P2B,H1B),t1,ivec(P3B,H1B),i1_3,ivec(P2B,P3B),1.0);
        m_t1_3 = ops[10].mult;

        /* i0 ( p2 h1 )_vt + = -1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 p2 h1 p3 )_v */
        //m_t1_4 = Multiplication(i0,ivec(P2B,H1B),t1,ivec(P3B,H4B),v,ivec(H4B,P2B,H1B,P3B),-1.0);
        m_t1_4 = ops[11].mult;

        /* i1 ( h8 p7 )_f + = 1 * f ( h8 p7 )_f */
        //a_t1_5_1 = Assignment(i1_5,f,1.0,ivec(H8B,P7B), ivec(H8B,P7B));
        a_t1_5_1 = ops[12].add;

        /* i1 ( h8 p7 )_vt + = 1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h8 p5 p7 )_v */
        //m_t1_5_2 = Multiplication(i1_5,ivec(H8B,P7B),t1,ivec(P5B,H6B),v,ivec(H6B,H8B,P5B,P7B),1.0);
        m_t1_5_2 = ops[13].mult;

        /* i0 ( p2 h1 )_tf + = 1 * Sum ( p7 h8 ) * t ( p2 p7 h1 h8 )_t * i1 ( h8 p7 )_f */
        //m_t1_5 = Multiplication(i0,ivec(P2B,H1B),t2,ivec(P2B,P7B,H1B,H8B),i1_5,ivec(H8B,P7B),1.0);
        m_t1_5 = ops[14].mult;

        /* i1 ( h4 h5 h1 p3 )_v + = 1 * v ( h4 h5 h1 p3 )_v */
        //a_t1_6_1 = Assignment(i1_6,v,1.0, ivec(H4B,H5B,H1B,P3B), ivec(H4B,H5B,H1B,P3B));
        a_t1_6_1 = ops[15].add;

        /* i1 ( h4 h5 h1 p3 )_vt + = -1 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h4 h5 p3 p6 )_v */
        // m_t1_6_2 = Multiplication(i1_6,ivec(H4B,H5B,H1B,P3B),
	// 			  t1,ivec(P6B,H1B),v,ivec(H4B,H5B,P3B,P6B),-1.0);
        m_t1_6_2 = ops[16].mult;

        /* i0 ( p2 h1 )_vt + = -1/2 * Sum ( h4 h5 p3 ) * t ( p2 p3 h4 h5 )_t * i1 ( h4 h5 h1 p3 )_v */
        //m_t1_6 = Multiplication(i0,ivec(P2B,H1B),t2,ivec(P2B,P3B,H4B,H5B),i1_6,ivec(H4B,H5B,H1B,P3B),-0.5);
        m_t1_6 = ops[17].mult;

        /* i0 ( p2 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 p2 p3 p4 )_v */
        //m_t1_7 = Multiplication(i0,ivec(P2B,H1B),t2,ivec(P3B,P4B,H1B,H5B),v,ivec(H5B,P2B,P3B,P4B),-0.5);
        m_t1_7 = ops[18].mult;

        set_t1 = false;
        
      }
    }

    void ccsd_t1_2_1_createfile_cxx_(Integer *k_i1_offset, Integer *d_i1, Integer *size_i1) {
      i1_2.create(k_i1_offset, d_i1, size_i1);
    }

    void ccsd_t1_2_1_deletefile_cxx_() {
      i1_2.destroy();
    }

    void ccsd_t1_2_2_1_createfile_cxx_(Integer *k_i1_offset, Integer *d_i1, Integer *size_i1) {
      i1_2_2.create(k_i1_offset, d_i1, size_i1);
    }

    void ccsd_t1_2_2_1_deletefile_cxx_() {
      i1_2_2.destroy();
    }

    void ccsd_t1_3_1_createfile_cxx_(Integer *k_i1_offset, Integer *d_i1, Integer *size_i1) {
      i1_3.create(k_i1_offset, d_i1, size_i1);
    }

    void ccsd_t1_3_1_deletefile_cxx_() {
      i1_3.destroy();
    }

    void ccsd_t1_5_1_createfile_cxx_(Integer *k_i1_offset, Integer *d_i1, Integer *size_i1) {
      i1_5.create(k_i1_offset, d_i1, size_i1);
    }

    void ccsd_t1_5_1_deletefile_cxx_() {
      i1_5.destroy();
    }

    void ccsd_t1_6_1_createfile_cxx_(Integer *k_i1_offset, Integer *d_i1, Integer *size_i1) {
      i1_6.create(k_i1_offset, d_i1, size_i1);
    }

    void ccsd_t1_6_1_deletefile_cxx_() {
      i1_6.destroy();
    }

    void ccsd_t1_1_cxx_(Integer *d_a, Integer *k_a_offset, 
        Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t1_1);
    } // t1_1

    void ccsd_t1_2_1_cxx_(Integer *d_a, Integer *k_a_offset, 
        Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t1_2_1);
    } // t1_2_1

    void ccsd_t1_2_2_1_cxx_(Integer *d_a, Integer *k_a_offset, 
        Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t1_2_2_1);
    } // t1_2_2_1

    void ccsd_t1_2_2_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_2_2_2);
    } // t1_2_2_2

    void ccsd_t1_2_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_2_2);
    } // t1_2_2

    void ccsd_t1_2_3_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_2_3);
    } // t1_2_3

    void ccsd_t1_2_4_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_2_4);
    } // t1_2_4

    void ccsd_t1_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_2);
    } // t1_2

    void ccsd_t1_3_1_cxx_(Integer *d_a, Integer *k_a_offset, 
        Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t1_3_1);
    } // t1_3_1

    void ccsd_t1_3_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_3_2);
    } // t1_3_2

    void ccsd_t1_3_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_3);
    } // t1_3

    void ccsd_t1_4_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_4);
    } // t1_4

    void ccsd_t1_5_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t1_5_1);
    } // t1_5_1

    void ccsd_t1_5_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_5_2);
    } // t1_5_2

    void ccsd_t1_5_cxx_(Integer *d_a, Integer *k_a_offset, 
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_5);
    } // t1_5

    void ccsd_t1_6_1_cxx_(Integer *d_a, Integer *k_a_offset, 
        Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t1_6_1);
    } // t1_6_1

    void ccsd_t1_6_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_6_2);
    } // t1_6_2

    void ccsd_t1_6_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_6);
    } // t1_6

    void ccsd_t1_7_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_7);
    } // t1_7

  } // extern C

}; // namespace ctce

