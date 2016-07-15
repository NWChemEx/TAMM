#include "variables.h"
#include <iostream>
#include "tensor.h"
#include "t_mult.h"
#include "t_assign.h"
#include "input.h"
#include "corf.h"
#include "equations.h"

/*
 *i0 ( p3 p4 h1 h2 )_v + = 1 * v ( p3 p4 h1 h2 )_v
 *i0 ( p3 p4 h1 h2 )_vt + = -1 * P( 2 ) * Sum ( h10 ) * t ( p3 h10 )_t * i1 ( h10 p4 h1 h2 )_v
 *    i1 ( h10 p3 h1 h2 )_v + = 1 * v ( h10 p3 h1 h2 )_v
 *    i1 ( h10 p3 h1 h2 )_vt + = 1/2 * Sum ( h11 ) * t ( p3 h11 )_t * i2 ( h10 h11 h1 h2 )_v
 *        i2 ( h10 h11 h1 h2 )_v + = -1 * v ( h10 h11 h1 h2 )_v
 *        i2 ( h10 h11 h1 h2 )_vt + = 1 * P( 2 ) * Sum ( p5 ) * t ( p5 h1 )_t * i3 ( h10 h11 h2 p5 )_v
 *            i3 ( h10 h11 h1 p5 )_v + = 1 * v ( h10 h11 h1 p5 )_v
 *            i3 ( h10 h11 h1 p5 )_vt + = -1/2 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h10 h11 p5 p6 )_v
 *        i2 ( h10 h11 h1 h2 )_vt + = -1/2 * Sum ( p7 p8 ) * t ( p7 p8 h1 h2 )_t * v ( h10 h11 p7 p8 )_v
 *    i1 ( h10 p3 h1 h2 )_vt + = -1 * P( 2 ) * Sum ( p5 ) * t ( p5 h1 )_t * i2 ( h10 p3 h2 p5 )_v
 *        i2 ( h10 p3 h1 p5 )_v + = 1 * v ( h10 p3 h1 p5 )_v
 *        i2 ( h10 p3 h1 p5 )_vt + = -1/2 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h10 p3 p5 p6 )_v
 *    i1 ( h10 p3 h1 h2 )_ft + = -1 * Sum ( p5 ) * t ( p3 p5 h1 h2 )_t * i2 ( h10 p5 )_f
 *        i2 ( h10 p5 )_f + = 1 * f ( h10 p5 )_f
 *        i2 ( h10 p5 )_vt + = -1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 h10 p5 p6 )_v
 *    i1 ( h10 p3 h1 h2 )_vt + = 1 * P( 2 ) * Sum ( h7 p9 ) * t ( p3 p9 h1 h7 )_t * i2 ( h7 h10 h2 p9 )_v
 *        i2 ( h7 h10 h1 p9 )_v + = 1 * v ( h7 h10 h1 p9 )_v
 *        i2 ( h7 h10 h1 p9 )_vt + = 1 * Sum ( p5 ) * t ( p5 h1 )_t * v ( h7 h10 p5 p9 )_v
 *    i1 ( h10 p3 h1 h2 )_vt + = 1/2 * Sum ( p5 p6 ) * t ( p5 p6 h1 h2 )_t * v ( h10 p3 p5 p6 )_v
 *i0 ( p3 p4 h1 h2 )_vt + = -1 * P( 2 ) * Sum ( p5 ) * t ( p5 h1 )_t * i1 ( p3 p4 h2 p5 )_v
 *    i1 ( p3 p4 h1 p5 )_v + = 1 * v ( p3 p4 h1 p5 )_v
 *    i1 ( p3 p4 h1 p5 )_vt + = -1/2 * Sum ( p6 ) * t ( p6 h1 )_t * v ( p3 p4 p5 p6 )_v
 *i0 ( p3 p4 h1 h2 )_tf + = -1 * P( 2 ) * Sum ( h9 ) * t ( p3 p4 h1 h9 )_t * i1 ( h9 h2 )_f
 *    i1 ( h9 h1 )_f + = 1 * f ( h9 h1 )_f
 *    i1 ( h9 h1 )_ft + = 1 * Sum ( p8 ) * t ( p8 h1 )_t * i2 ( h9 p8 )_f
 *        i2 ( h9 p8 )_f + = 1 * f ( h9 p8 )_f
 *        i2 ( h9 p8 )_vt + = 1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 h9 p6 p8 )_v
 *    i1 ( h9 h1 )_vt + = -1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 h9 h1 p6 )_v
 *    i1 ( h9 h1 )_vt + = -1/2 * Sum ( h8 p6 p7 ) * t ( p6 p7 h1 h8 )_t * v ( h8 h9 p6 p7 )_v
 *i0 ( p3 p4 h1 h2 )_tf + = 1 * P( 2 ) * Sum ( p5 ) * t ( p3 p5 h1 h2 )_t * i1 ( p4 p5 )_f
 *    i1 ( p3 p5 )_f + = 1 * f ( p3 p5 )_f
 *    i1 ( p3 p5 )_vt + = -1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 p3 p5 p6 )_v
 *    i1 ( p3 p5 )_vt + = -1/2 * Sum ( h7 h8 p6 ) * t ( p3 p6 h7 h8 )_t * v ( h7 h8 p5 p6 )_v
 *i0 ( p3 p4 h1 h2 )_vt + = -1/2 * Sum ( h11 h9 ) * t ( p3 p4 h9 h11 )_t * i1 ( h9 h11 h1 h2 )_v
 *    i1 ( h9 h11 h1 h2 )_v + = -1 * v ( h9 h11 h1 h2 )_v
 *    i1 ( h9 h11 h1 h2 )_vt + = 1 * P( 2 ) * Sum ( p8 ) * t ( p8 h1 )_t * i2 ( h9 h11 h2 p8 )_v
 *        i2 ( h9 h11 h1 p8 )_v + = 1 * v ( h9 h11 h1 p8 )_v
 *        i2 ( h9 h11 h1 p8 )_vt + = 1/2 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h9 h11 p6 p8 )_v
 *    i1 ( h9 h11 h1 h2 )_vt + = -1/2 * Sum ( p5 p6 ) * t ( p5 p6 h1 h2 )_t * v ( h9 h11 p5 p6 )_v
 *i0 ( p3 p4 h1 h2 )_vt + = -1 * P( 4 ) * Sum ( h6 p5 ) * t ( p3 p5 h1 h6 )_t * i1 ( h6 p4 h2 p5 )_v
 *    i1 ( h6 p3 h1 p5 )_v + = 1 * v ( h6 p3 h1 p5 )_v
 *    i1 ( h6 p3 h1 p5 )_vt + = -1 * Sum ( p7 ) * t ( p7 h1 )_t * v ( h6 p3 p5 p7 )_v
 *    i1 ( h6 p3 h1 p5 )_vt + = -1/2 * Sum ( h8 p7 ) * t ( p3 p7 h1 h8 )_t * v ( h6 h8 p5 p7 )_v
 *i0 ( p3 p4 h1 h2 )_vt + = 1/2 * Sum ( p5 p6 ) * t ( p5 p6 h1 h2 )_t * v ( p3 p4 p5 p6 )_v
 */

/*
 * t2_1:  i0 ( p3 p4 h1 h2 ) += 1 * v ( p3 p4 h1 h2 )
 * t2_2_1_createfile:     i1 ( h10 p3 h1 h2 )
 * t2_2_1:      t2_2_1 ( h10 p3 h1 h2 ) += 1 * v ( h10 p3 h1 h2 )
 * t2_2_2_1_createfile:     i2 ( h10 h11 h1 h2 )
 * t2_2_2_1:     t2_2_2_1 ( h10 h11 h1 h2 ) += -1 * v ( h10 h11 h1 h2 )
 * t2_2_2_2_1_createfile:     i3 ( h10 h11 h1 p5 )
 * t2_2_2_2_1:     t2_2_2_2_1 ( h10 h11 h1 p5 ) += 1 * v ( h10 h11 h1 p5 )
 * t2_2_2_2_2:     t2_2_2_2_1 ( h10 h11 h1 p5 ) += -0.5 * t ( p6 h1 ) * v ( h10 h11 p5 p6 )
 * t2_2_2_2:     t2_2_2_1 ( h10 h11 h1 h2 ) += 1 * t ( p5 h1 ) * t2_2_2_2_1 ( h10 h11 h2 p5 )
 * t2_2_2_2_1_deletefile
 * t2_2_2_3:     t2_2_2_1 ( h10 h11 h1 h2 ) += -0.5 * t ( p7 p8 h1 h2 ) * v ( h10 h11 p7 p8 )
 * t2_2_2:     t2_2_1 ( h10 p3 h1 h2 ) += 0.5 * t ( p3 h11 ) * t2_2_2_1 ( h10 h11 h1 h2 )
 * t2_2_2_1_deletefile
 * t2_2_4_1_createfile:     i2 ( h10 p5 )
 * t2_2_4_1:     t2_2_4_1 ( h10 p5 ) += 1 * f ( h10 p5 )
 * t2_2_4_2:     t2_2_4_1 ( h10 p5 ) += -1 * t ( p6 h7 ) * v ( h7 h10 p5 p6 )
 * t2_2_4:     t2_2_1 ( h10 p3 h1 h2 ) += -1 * t ( p3 p5 h1 h2 ) * t2_2_4_1 ( h10 p5 )
 * t2_2_4_1_deletefile
 * t2_2_5_1_createfile:     i2 ( h7 h10 h1 p9 )
 * t2_2_5_1:     t2_2_5_1 ( h7 h10 h1 p9 ) += 1 * v ( h7 h10 h1 p9 )
 * t2_2_5_2:     t2_2_5_1 ( h7 h10 h1 p9 ) += 1 * t ( p5 h1 ) * v ( h7 h10 p5 p9 )
 * t2_2_5:     t2_2_1 ( h10 p3 h1 h2 ) += 1 * t ( p3 p9 h1 h7 ) * t2_2_5_1 ( h7 h10 h2 p9 )
 * t2_2_5_1_deletefile
 * c2f_t2_t12: t2 ( p1 p2 h3 h4) += 0.5 * t ( p1 h3 ) * t (p2 h4)
 * t2_2_6:     t2_2_1 ( h10 p3 h1 h2 ) += 0.5 * t ( p5 p6 h1 h2 ) * v ( h10 p3 p5 p6 )
 * c2d_t2_t12: t2 ( p1 p2 h3 h4) += -0.5 * t ( p1 h3 ) * t (p2 h4)
 * t2_2:     i0 ( p3 p4 h1 h2 ) += -1 * t ( p3 h10 ) * t2_2_1 ( h10 p4 h1 h2 )
 * t2_2_1_deletefile
 * lt2_3x:     i0 ( p3 p4 h1 h2 ) += -1 * t ( p5 h1 ) * v ( p3 p4 h2 p5 )
 * OFFSET_ccsd_t2_4_1:     i1 ( h9 h1 )
 * t2_4_1:     t2_4_1 ( h9 h1 ) += 1 * f ( h9 h1 )
 * OFFSET_ccsd_t2_4_2_1: i2 ( h9 p8 )
 * t2_4_2_1:     t2_4_2_1 ( h9 p8 ) += 1 * f ( h9 p8 )
 * t2_4_2_2:     t2_4_2_1 ( h9 p8 ) += 1 * t ( p6 h7 ) * v ( h7 h9 p6 p8 )
 * t2_4_2:     t2_4_1 ( h9 h1 ) += 1 * t ( p8 h1 ) * t2_4_2_1 ( h9 p8 )
 * t2_4_3:     t2_4_1 ( h9 h1 ) += -1 * t ( p6 h7 ) * v ( h7 h9 h1 p6 )
 * t2_4_4:     t2_4_1 ( h9 h1 ) += -0.5 * t ( p6 p7 h1 h8 ) * v ( h8 h9 p6 p7 )
 * create i1_local
c    copy d_t1 ==> l_t1_local
 * t2_4:     i0 ( p3 p4 h1 h2 ) += -1 * t ( p3 p4 h1 h9 ) * t2_4_1 ( h9 h2 )
 * delete i1_local
 * DELETEFILE t2_4_1
 * t2_5_1 create:     i1 ( p3 p5 )
 * t2_5_1:     t2_5_1 ( p3 p5 ) += 1 * f ( p3 p5 )
 * t2_5_2:     t2_5_1 ( p3 p5 ) += -1 * t ( p6 h7 ) * v ( h7 p3 p5 p6 )
 * t2_5_3:     t2_5_1 ( p3 p5 ) += -0.5 * t ( p3 p6 h7 h8 ) * v ( h7 h8 p5 p6 )
 * create i1_local
c    copy d_t1 ==> l_t1_local
 * t2_5:     i0 ( p3 p4 h1 h2 ) += 1 * t ( p3 p5 h1 h2 ) * t2_5_1 ( p4 p5 )
 * delete i1_local
 * DELETEFILE t2_5_1
 * OFFSET_t2_6_1:     i1 ( h9 h11 h1 h2 )
 * t2_6_1:     t2_6_1 ( h9 h11 h1 h2 ) += -1 * v ( h9 h11 h1 h2 )
 * OFFSET_t2_6_2_1:     i2 ( h9 h11 h1 p8 )
 * t2_6_2_1:     t2_6_2_1 ( h9 h11 h1 p8 ) += 1 * v ( h9 h11 h1 p8 )
 * t2_6_2_2:     t2_6_2_1 ( h9 h11 h1 p8 ) += 0.5 * t ( p6 h1 ) * v ( h9 h11 p6 p8 )
 * t2_6_2:     t2_6_1 ( h9 h11 h1 h2 ) += 1 * t ( p8 h1 ) * t2_6_2_1 ( h9 h11 h2 p8 )
 * DELETEFILE t2_6_2_1
 * t2_6_3:     t2_6_1 ( h9 h11 h1 h2 ) += -0.5 * t ( p5 p6 h1 h2 ) * v ( h9 h11 p5 p6 )
 * t2_6:     i0 ( p3 p4 h1 h2 ) += -0.5 * t ( p3 p4 h9 h11 ) * t2_6_1 ( h9 h11 h1 h2 )
 * DELETEFILE t2_6_1
 * OFFSET_t2_7_1:     i1 ( h6 p3 h1 p5 )
 * t2_7_1:     t2_7_1 ( h6 p3 h1 p5 ) += 1 * v ( h6 p3 h1 p5 )
 * t2_7_2:     t2_7_1 ( h6 p3 h1 p5 ) += -1 * t ( p7 h1 ) * v ( h6 p3 p5 p7 )
 * t2_7_3:     t2_7_1 ( h6 p3 h1 p5 ) += -0.5 * t ( p3 p7 h1 h8 ) * v ( h6 h8 p5 p7 )
 * t2_7:     i0 ( p3 p4 h1 h2 ) += -1 * t ( p3 p5 h1 h6 ) * t2_7_1 ( h6 p4 h2 p5 )
 * DELETEFILE t2_7_1
 * vt1t1_1_createfile: C     i1 ( h5 p3 h1 h2 )
 * vt1t1_1_2:     vt1t1_1 ( h5 p3 h1 h2 ) += -2 * t ( p6 h1 ) * v ( h5 p3 h2 p6 ) 
 * vt1t1_1:     i0 ( p3 p4 h1 h2 )t += -0.5 * t ( p3 h5 ) * vt1t1_1 ( h5 p4 h1 h2 )
 * vt1t1_1_deletefile
 * c2f_t2_t12: t2 ( p1 p2 h3 h4) += 0.5 * t ( p1 h3 ) * t (p2 h4)
 * t2_8:   i0 ( p3 p4 h1 h2 )_vt + = 0.5 * t ( p5 p6 h1 h2 ) * v ( p3 p4 p5 p6 )
 * c2d_t2_t12: t2 ( p1 p2 h3 h4) += -0.5 * t ( p1 h3 ) * t (p2 h4)
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
    {"i0", {V, V, O, O}, 4, 2},
    {"f", {N, N}, 2, 1},
    {"v", {N, N, N, N}, 4, 2},
    {"t1", {V, O}, 2, 1},
    {"t2", {V, V, O, O}, 4, 2},
    {"t2_2_1", {O, V, O, O}, 4, 2},
    {"t2_2_2_1", {O, O, O, O}, 4, 2},
    {"t2_2_2_2_1", { O, O, O, V},4, 2},
    {"t2_2_4_1", {O, V}, 2, 1},
    {"t2_2_5_1", {O, O, O, V}, 4, 2},
    {"t2_4_1", {O, O}, 2, 1},
    {"t2_4_2_1", {O, V}, 2, 1},
    {"t2_5_1", {V, V}, 2, 1},
    {"t2_6_1",  {O, O, O, O}, 4, 2},
    {"t2_6_2_1",     {O,O,O,V }, 4, 2},
    {"t2_7_1",      { O,V,O,V }, 4, 2},
    {"vt1t1_1", {O, V, O, O}, 4, 2},
  };

  static int i0 = 0;
  static int f = 1;
  static int v = 2;
  static int t1 = 3;
  static int t2 = 4;
  static int t2_2_1 = 5;
  static int t2_2_2_1 = 6;
  static int t2_2_2_2_1 = 7;
  static int t2_2_4_1 = 8;
  static int t2_2_5_1 = 9;
  static int t2_4_1 = 10;
  static int t2_4_2_1 = 11;
  static int t2_5_1 = 12;
  static int t2_6_1 = 13;
  static int t2_6_2_1 = 14;
  static int t2_7_1 = 15;
  static int vt1t1_1 = 16;

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
    {i0, v, 1, {p3, p4, h1, h2}, {p3, p4, h1, h2}},              //t2_1        
    {t2_2_1, v, 1, {h10, p3, h1, h2}, {h10, p3, h1, h2}},        //t2_2_1      
    {t2_2_2_1, v, -1, {h10, h11, h1, h2}, {h10, h11, h1, h2}},   //t2_2_2_1    
    {t2_2_2_2_1, v, 1, {h10, h11, h1, p5}, {h10, h11, h1, p5}},  //t2_2_2_2_1  
    {t2_2_4_1, f, 1, {h10, p5}, {h10, p5}},                      //t2_2_4_1    
    {t2_2_5_1, v, 1, {h7, h10, h1, p9}, {h7, h10, h1, p9}},      //t2_2_5_1    
    {t2_4_1, f, 1, {h9, h1}, {h9, h1}},                          //t2_4_1      
    {t2_4_2_1, f, 1, {h9, p8}, {h9, p8}},                        //t2_4_2_1
    {t2_5_1, f, 1, {p3, p5}, {p3, p5}},                          //t2_5_1      
    {t2_6_1, v, -1, {h9, h11, h1, h2}, {h9, h11, h1, h2}},       //t2_6_1      
    {t2_6_2_1, v, 1, {h9, h11, h1, p8}, {h9, h11, h1, p8}},      //t2_6_2_1    
    {t2_7_1, v, 1, {h6, p3, h1, p5}, {h6, p3, h1, p5}},          //t2_7_1
  };

  static MultOp multops[] = {
    {t2_2_2_2_1, t1, v, -0.5, {h10, h11, h1, p5}, {p6, h1}, {h10, h11, p5, p6}},       //t2_2_2_2_2  
    {t2_2_2_1, t1, t2_2_2_2_1, 1, {h10, h11, h1, h2}, {p5, h1}, {h10, h11, h2, p5}},   //t2_2_2_2    
    {t2_2_2_1, t2, v, -0.5, {h10, h11, h1, h2}, {p7, p8, h1, h2}, {h10, h11, p7, p8}}, //t2_2_2_3    
    {t2_2_1, t1, t2_2_2_1, 0.5, {h10, p3, h1, h2}, {p3, h11}, {h10, h11, h1, h2}},     //t2_2_2      
    {t2_2_4_1, t1, v, -1, {h10, p5}, {p6, h7}, {h7, h10, p5, p6}},                     //t2_2_4_2    
    {t2_2_1, t2, t2_2_4_1, -1, {h10, p3, h1, h2}, {p3, p5, h1, h2}, {h10, p5}},        //t2_2_4      
    {t2_2_5_1, t1, v, 1, {h7, h10, h1, p9}, {p5, h1}, {h7, h10, p5, p9}},              //t2_2_5_2    
    {t2_2_1, t2, t2_2_5_1, 1, {h10, p3, h1, h2}, {p3, p9, h1, h7}, {h7, h10, h2, p9}}, //t2_2_5      
    {t2, t1, t1, 0.5, {p1, p2, h3, h4}, {p1, h3}, {p2, h4}},                           //c2f_t2_t12  
    {t2_2_1, t2, v, 0.5, {h10, p3, h1, h2}, {p5, p6, h1, h2}, {h10, p3, p5, p6}},      //t2_2_6      
    {t2, t1, t1, -0.5, {p1, p2, h3, h4}, {p1, h3}, {p2, h4}},                          //c2d_t2_t12  
    {i0, t1, t2_2_1, -1, {p3, p4, h1, h2}, {p3, h10}, {h10, p4, h1, h2}},              //t2_2        
    {i0, t1, v, -1,  {p3, p4, h1, h2}, {p5, h1}, {p3, p4, h2, p5}},                    //lt2_3x      
    {t2_4_2_1, t1, v, 1, {h9, p8}, {p6, h7}, {h7, h9, p6, p8}},                        //t2_4_2_2    
    {t2_4_1, t1, t2_4_2_1, 1, {h9, h1}, {p8, h1}, {h9, p8}},                           //t2_4_2      
    {t2_4_1, t1, v, -1, {h9, h1}, {p6, h7}, {h7, h9, h1, p6}},                         //t2_4_3      
    {t2_4_1, t2, v, -0.5, {h9, h1}, {p6, p7, h1, h8}, {h8, h9, p6, p7}},               //t2_4_4      
    {i0, t2, t2_4_1, -1, {p3, p4, h1, h2}, {p3, p4, h1, h9}, {h9, h2}},                //t2_4        
    {t2_5_1, t1, v, -1, {p3, p5}, {p6, h7}, {h7, p3, p5, p6}},                         //t2_5_2      
    {t2_5_1, t2, v, -0.5, {p3, p5}, {p3, p6, h7, h8}, {h7, h8, p5, p6}},               //t2_5_3      
    {i0, t2, t2_5_1, 1, {p3, p4, h1, h2}, {p3, p5, h1, h2}, {p4, p5}},                 //t2_5        
    {t2_6_2_1, t1, v, 0.5, {h9, h11, h1, p8}, {p6, h1}, {h9, h11, p6, p8}},            //t2_6_2_2    
    {t2_6_1, t1, t2_6_2_1, 1, {h9, h11, h1, h2}, {p8, h1}, {h9, h11, h2, p8}},         //t2_6_2      
    {t2_6_1, t2, v, -0.5, {h9, h11, h1, h2}, {p5, p6, h1, h2}, {h9, h11, p5, p6}},     //t2_6_3      
    {i0, t2, t2_6_1, -0.5, {p3, p4, h1, h2}, {p3, p4, h9, h11}, {h9, h11, h1, h2}},    //t2_6        
    {t2_7_1, t1, v, -1, {h6, p3, h1, p5}, {p7, h1}, {h6, p3, p5, p7}},                 //t2_7_2      
    {t2_7_1, t2, v, -0.5, {h6, p3, h1, p5}, {p3, p7, h1, h8}, {h6, h8, p5, p7}},       //t2_7_3      
    {i0, t2, t2_7_1, -1, {p3, p4, h1, h2}, {p3, p5, h1, h6}, {h6, p4, h2, p5}},        //t2_7        
    {vt1t1_1, t1, v, -2, {h5, p3, h1, h2}, {p6, h1}, {h5, p3, h2, p6}},                //vt1t1_1_2   
    {i0, t1, vt1t1_1, -0.5, {p3, p4, h1, h2}, {p3, h5}, {h5, p4, h1, h2}},             //vt1t1_1     
    {t2, t1, t1, 0.5, {p1, p2, h3, h4}, {p1, h3}, {p2, h4}},                           //c2f_t2_t12  
    {i0, t2, v, 0.5, {p3, p4, h1, h2}, {p5, p6, h1, h2}, {p3, p4, p5, p6}},            //t2_8        
    {t2, t1, t1, -0.5, {p1, p2, h3, h4}, {p1, h3}, {p2, h4}},                          //c2d_t2_t12  
  };

  static OpEntry ops[] = {
    {OpTypeAdd, addops[0], MultOp()},                 //t2_1       
     {OpTypeAdd, addops[1], MultOp()},                 //t2_2_1     
     {OpTypeAdd, addops[2], MultOp()},                 //t2_2_2_1   
     {OpTypeAdd, addops[3], MultOp()},                 //t2_2_2_2_1 
     {OpTypeMult, AddOp(), multops[0]},                //t2_2_2_2_2 
     {OpTypeMult, AddOp(), multops[1]},                //t2_2_2_2   
     {OpTypeMult, AddOp(), multops[2]},                //t2_2_2_3   
     {OpTypeMult, AddOp(), multops[3]},                //t2_2_2     
     {OpTypeAdd, addops[4], MultOp()},                 //t2_2_4_1   
     {OpTypeMult, AddOp(), multops[4]},                //t2_2_4_2   
     {OpTypeMult, AddOp(), multops[5]},                //t2_2_4     
     {OpTypeAdd, addops[5], MultOp()},                 //t2_2_5_1   
     {OpTypeMult, AddOp(), multops[6]},                //t2_2_5_2   
     {OpTypeMult, AddOp(), multops[7]},                //t2_2_5     
     {OpTypeMult, AddOp(), multops[8]},                //c2f_t2_t12 
     {OpTypeMult, AddOp(), multops[9]},                //t2_2_6     
     {OpTypeMult, AddOp(), multops[10]},               //c2d_t2_t12 
     {OpTypeMult, AddOp(), multops[11]},               //t2_2       
     {OpTypeMult, AddOp(), multops[12]},               //lt2_3x     
     {OpTypeAdd, addops[6], MultOp()},                 //t2_4_1     
     {OpTypeAdd, addops[7], MultOp()},                 //t2_4_2_1   
     {OpTypeMult, AddOp(), multops[13]},               //t2_4_2_2   
     {OpTypeMult, AddOp(), multops[14]},               //t2_4_2     
     {OpTypeMult, AddOp(), multops[15]},               //t2_4_3     
     {OpTypeMult, AddOp(), multops[16]},               //t2_4_4     
     {OpTypeMult, AddOp(), multops[17]},               //t2_4       
     {OpTypeAdd, addops[8], MultOp()},                 //t2_5_1     
     {OpTypeMult, AddOp(), multops[18]},               //t2_5_2     
     {OpTypeMult, AddOp(), multops[19]},               //t2_5_3     
     {OpTypeMult, AddOp(), multops[20]},               //t2_5       
     {OpTypeAdd, addops[9], MultOp()},                 //t2_6_1     
     {OpTypeAdd, addops[10], MultOp()},                //t2_6_2_1   
     {OpTypeMult, AddOp(), multops[21]},               //t2_6_2_2   
     {OpTypeMult, AddOp(), multops[22]},               //t2_6_2     
     {OpTypeMult, AddOp(), multops[23]},               //t2_6_3     
     {OpTypeMult, AddOp(), multops[24]},               //t2_6       
     {OpTypeAdd, addops[11], MultOp()},                //t2_7_1     
     {OpTypeMult, AddOp(), multops[25]},               //t2_7_2     
     {OpTypeMult, AddOp(), multops[26]},               //t2_7_3     
     {OpTypeMult, AddOp(), multops[27]},               //t2_7       
     {OpTypeMult, AddOp(), multops[28]},               //vt1t1_1_2  
     {OpTypeMult, AddOp(), multops[29]},               //vt1t1_1    
     {OpTypeMult, AddOp(), multops[30]},               //c2f_t2_t12 
     {OpTypeMult, AddOp(), multops[31]},               //t2_8       
     {OpTypeMult, AddOp(), multops[32]},               //c2d_t2_t12 
  };

  static int num_ranges = sizeof(ranges)/sizeof(ranges[0]);
  static int num_indices = sizeof(indices)/sizeof(indices[0]);
  static int num_tensors = sizeof(tensors)/sizeof(tensors[0]);
  static int num_operations = sizeof(ops) / sizeof(ops[0]);

  void ccsd_t2_equations(Equations &eqs) {
    eqs.range_entries.insert(eqs.range_entries.begin(), ranges, ranges+num_ranges);
    eqs.index_entries.insert(eqs.index_entries.begin(), indices, indices+num_indices);
    eqs.tensor_entries.insert(eqs.tensor_entries.begin(), tensors, tensors+num_tensors);
    eqs.op_entries.insert(eqs.op_entries.begin(), ops, ops+num_operations);
  }
}; /*ctce*/

