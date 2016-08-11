extern "C" {
#include "ctce_parser.h"
};


#include "variables.h"
#include <iostream>
#include "tensor.h"
#include "t_mult.h"
#include "t_assign.h"
#include "input.h"
#include "corf.h"
#include "equations.h"

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


namespace ctce {

  static RangeEntry ranges[] = {
    {"O"},
    {"V"},
    {"N"}
  };

  static int O=0, V=1, N=2;

  static IndexEntry indices[] = {
    {"p1", 1},
    {"p2", 1},
    {"p3", 1},
    {"p4", 1},
    {"p5", 1},
    {"p6", 1},
    {"p7", 1},
    {"p8", 1},
    {"p9", 1},
    {"p10", 1},
    {"p11", 1},
    {"p12", 1},
    {"h1", 0},
    {"h2", 0},
    {"h3", 0},
    {"h4", 0},
    {"h5", 0},
    {"h6", 0},
    {"h7", 0},
    {"h8", 0},
    {"h9", 0},
    {"h10", 0},
    {"h11", 0},
    {"h12", 0},
  };

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

  static int ti0 = 0;
  static int tf = 1;
  static int tv = 2;
  static int tt1 = 3;
  static int tt2 = 4;
  static int ti1_2 = 5;
  static int ti1_2_2 = 6;
  static int ti1_3 = 7;
  static int ti1_5 = 8;
  static int ti1_6 = 9;

  static IndexName p1 = P1B;
  static IndexName p2 = P2B;
  static IndexName p3 = P3B;
  static IndexName p4 = P4B;
  static IndexName p5 = P5B;
  static IndexName p6 = P6B;
  static IndexName p7 = P7B;
  static IndexName p8 = P8B;
  static IndexName p9 = P9B;
  static IndexName p10 = P10B;
  static IndexName p11 = P11B;
  static IndexName p12 = P12B;
  static IndexName h1 = H1B;
  static IndexName h2 = H2B;
  static IndexName h3 = H3B;
  static IndexName h4 = H4B;
  static IndexName h5 = H5B;
  static IndexName h6 = H6B;
  static IndexName h7 = H7B;
  static IndexName h8 = H8B;
  static IndexName h9 = H9B;
  static IndexName h10 = H10B;
  static IndexName h11 = H11B;
  static IndexName h12 = H12B;

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

  static OpEntry ops[] = {
#if 1
    { OpTypeAdd, addops[0], MultOp()},    // t1_1
    { OpTypeAdd, addops[1], MultOp()},    // t1_2_1
    { OpTypeAdd, addops[2], MultOp()},    // t1_2_2_1
    { OpTypeMult, AddOp(), multops[0]},  // t1_2_2_2
    { OpTypeMult, AddOp(), multops[1]},  // t1_2_2
    { OpTypeMult, AddOp(), multops[2]},  // t1_2_3
    { OpTypeMult, AddOp(), multops[3]},  // t1_2_4
    { OpTypeMult, AddOp(), multops[4]},  // t1_2
    { OpTypeAdd, addops[3], MultOp()},    // t1_3_1
    { OpTypeMult, AddOp(), multops[5]},  // t1_3_2
    { OpTypeMult, AddOp(), multops[6]},  // t1_3
    { OpTypeMult, AddOp(), multops[7]},  // t1_4
    { OpTypeAdd, addops[4]},    // t1_5_1
    { OpTypeMult, AddOp(), multops[8]},  // t1_5_2
    { OpTypeMult, AddOp(), multops[9]},  // t1_5
#endif
    { OpTypeAdd, addops[5]},    // t1_6_1
    { OpTypeMult, AddOp(), multops[10]}, // t1_6_2
    { OpTypeMult, AddOp(), multops[11]}, // t1_6
    { OpTypeMult, AddOp(), multops[12]} // t1_7
  };

  static int num_ranges = sizeof(ranges)/sizeof(ranges[0]);
  static int num_indices = sizeof(indices)/sizeof(indices[0]);
  static int num_tensors = sizeof(tensors)/sizeof(tensors[0]);
  static int num_operations = sizeof(ops) / sizeof(ops[0]);

#define CTCE_EQ_PATH "/home/sriram/code/ctce/tensor/eqs/"

  void ccsd_t1_equations(ctce::Equations &eqs) {
    ::Equations peqs;
    ctce_parser("../ctce_parser/transform_input/generated/ccsd_t1.eq.lvl", &peqs);
    parser_eqs_to_ctce_eqs(&peqs, eqs);
  }

  void ccsd_t1_new_equations(Equations &eqs) {
    eqs.range_entries.insert(eqs.range_entries.begin(), ranges, ranges+num_ranges);
    eqs.index_entries.insert(eqs.index_entries.begin(), indices, indices+num_indices);
    eqs.tensor_entries.insert(eqs.tensor_entries.begin(), tensors, tensors+num_tensors);
    eqs.op_entries.insert(eqs.op_entries.begin(), ops, ops+num_operations);
  }
}; /*ctce*/

