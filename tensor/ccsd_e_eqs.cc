#include "variables.h"
#include <iostream>
#include "tensor.h"
#include "t_mult.h"
#include "t_assign.h"
#include "input.h"
#include "corf.h"
#include "equations.h"

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
    {"i0", {}, 0, 0},
    {"f", {N, N}, 2, 1},
    {"v", {N, N, N, N}, 4, 2},
    {"t1", {V, O}, 2, 1},
    {"t2", {V, V, O, O}, 4, 2},
    {"i1_1", {O, V}, 2, 1},
  };

  static int ti0 = 0;
  static int tf = 1;
  static int tv = 2;
  static int tt1 = 3;
  static int tt2 = 4;
  static int ti1_1 = 5;

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
    {ti1_1, tf, 1.0, {h6, p5}, {h6, p5}},             //e_1_1
  };

  static MultOp multops[] = {
    {ti1_1, tt1, tv, 0.5, {h6,p5}, {p3,h4}, {h4,h6,p3,p5}},     //e_1_2
    {ti0, tt1, ti1_1, 1.0, {}, {p5,h6}, {h6,p5}},               //e_1
    {ti0, tt2, tv, 0.25, {}, {p1,p2,h3,h4}, {h3,h4,p1,p2}},     //e_2
  };

  static OpEntry ops[] = {
    { OpTypeAdd, addops[0], MultOp()},   // e_1_1
    { OpTypeMult, AddOp(), multops[0]},  // e_1_2
    { OpTypeMult, AddOp(), multops[1]},  // e_1
    { OpTypeMult, AddOp(), multops[2]},  // e_2
  };

  static int num_ranges = sizeof(ranges)/sizeof(ranges[0]);
  static int num_indices = sizeof(indices)/sizeof(indices[0]);
  static int num_tensors = sizeof(tensors)/sizeof(tensors[0]);
  static int num_operations = sizeof(ops) / sizeof(ops[0]);

  void ccsd_e_equations(Equations &eqs) {
    eqs.range_entries.insert(eqs.range_entries.begin(), ranges, ranges+num_ranges);
    eqs.index_entries.insert(eqs.index_entries.begin(), indices, indices+num_indices);
    eqs.tensor_entries.insert(eqs.tensor_entries.begin(), tensors, tensors+num_tensors);
    eqs.op_entries.insert(eqs.op_entries.begin(), ops, ops+num_operations);
  }
}; /*ctce*/

