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


///////////////////////////////////////////////////////////////////////
///////////////                 CCSD_T1                 ///////////////
///////////////////////////////////////////////////////////////////////

// i0 ( p2 h1 )_f + = 1 * f ( p2 h1 )_f
TEST (CCSD_T1, t1_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {p2}, {h1}, 1.0, {p2}, {h1}));
  }

  // i1 ( h7 h1 )_f + = 1 * f ( h7 h1 )_f
TEST (CCSD_T1, t1_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h7}, {h1}, 1.0, {h7}, {h1}));
  }

  // i2 ( h7 p3 )_f + = 1 * f ( h7 p3 )_f
TEST (CCSD_T1, t1_2_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec,{h7}, {p3}, 1.0, {h7}, {p3}));
  }

  TEST (CCSD_T1, t1_2_2_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec,{h7}, {p3}, -1.0, {p5}, {h6}, {h6,h7},
            {p3,p5}));
  }
  
  TEST (CCSD_T1, t1_2_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h7}, {h1}, 1.0, {p3}, {h1}, {h7}, {p3}));
  }
  
  TEST (CCSD_T1, t1_2_3) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h7}, {h1}, -1.0, {p4}, {h5}, {h5,h7},
            {h1,p4}));
  }

  
  // i1 ( h7 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 h7 p3 p4 )_v
TEST (CCSD_T1, t1_2_4) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h7}, {h1}, -0.5, {p3,p4}, {h1,h5},
            {h5,h7}, {p3,p4}));
  }

  // i0 ( p2 h1 )_tf + = -1 * Sum ( h7 ) * t ( p2 h7 )_t * i1 ( h7 h1 )_f
TEST (CCSD_T1, t1_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {h1}, -1.0, {p2}, {h7}, {h7}, {h1}));
  }

  // i1 ( p2 p3 )_f + = 1 * f ( p2 p3 )_f
TEST (CCSD_T1, t1_3_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {p2}, {p3}, 1.0, {p2}, {p3}));
  }

  TEST (CCSD_T1, t1_3_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {p3}, -1.0, {p4}, {h5}, {h5,p2},
            {p3,p4}));
  }
  
  TEST (CCSD_T1, t1_3) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {h1}, 1.0, {p3}, {h1}, {p2}, {p3}));
  }
  
  TEST (CCSD_T1, t1_4) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {h1}, -1.0, {p3}, {h4}, {h4,p2},
            {h1,p3}));
  }

  // i1 ( h8 p7 )_f + = 1 * f ( h8 p7 )_f
TEST (CCSD_T1, t1_5_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h8}, {p7}, 1.0, {h8}, {p7}));
  }


TEST (CCSD_T1, t1_5_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h8}, {p7}, 1.0, {p5}, {h6}, {h6,h8},
            {p5,p7}));
  }
  
  TEST (CCSD_T1, t1_5) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {h1}, 1.0, {p2,p7}, {h1,h8}, {h8},
            {p7}));
  }

  // i1 ( h4 h5 h1 p3 )_v + = 1 * v ( h4 h5 h1 p3 )_v
TEST (CCSD_T1, t1_6_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h4,h5}, {h1,p3} , 1.0, {h4,h5},
            {h1,p3}));
  }

  TEST (CCSD_T1, t1_6_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h4,h5}, {h1,p3}, -1.0, {p6}, {h1}, {h4,h5},
            {p3,p6}));
  }
  
  TEST (CCSD_T1, t1_6) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {h1}, -0.5, {p2,p3}, {h4,h5},
            {h4,h5}, {h1,p3}));
  }
  
  TEST (CCSD_T1, t1_7) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {h1}, -0.5, {p3,p4}, {h1,h5},
            {h5,p2}, {p3,p4}));
  }


///////////////////////////////////////////////////////////////////////
///////////////                 CCSD_T2                 ///////////////
///////////////////////////////////////////////////////////////////////

// i0 ( p3 p4 h1 h2 )_v + = 1 * v ( p3 p4 h1 h2 )_v
TEST(CCSD_T2,t2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {p3,p4}, {h1,h2}, 1.0,
          {p3,p4}, {h1,h2}));
  }

  
  // i1 ( h10 p3 h1 h2 )_v + = 1 * v ( h10 p3 h1 h2 )_v
TEST(CCSD_T2,t2_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h10,p3}, {h1,h2}, 1.0,
            {h10,p3}, {h1,h2}));
  }

  // i2 ( h10 h11 h1 h2 )_v + = -1 * v ( h10 h11 h1 h2 )_v
TEST(CCSD_T2,t2_2_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h10,h11} ,{h1,h2}, -1.0,
            {h10,h11}, {h1,h2}));
  }

  
  // i3 ( h10 h11 h1 p5 )_v + = 1 * v ( h10 h11 h1 p5 )_v
TEST(CCSD_T2,t2_2_2_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h10,h11}, {h1,p5}, 1.0,
            {h10,h11}, {h1,p5}));
  }

  TEST(CCSD_T2,t2_2_2_2_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h10,h11}, {h1,p5}, -0.5, {p6}, {h1},
            {h10,h11}, {p5,p6}));
  }
  
  TEST(CCSD_T2,t2_2_2_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h10,h11}, {h1,h2}, 1.0, {p5}, {h1},
            {h10,h11}, {h2,p5}));
  }
  
  TEST(CCSD_T2,t2_2_2_3) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h10,h11}, {h1,h2}, -0.5, {p7,p8}, {h1,h2},
            {h10,h11}, {p7,p8}));
  }
  
  TEST(CCSD_T2,t2_2_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h10,p3}, {h1,h2}, 0.5, {p3}, {h11},
            {h10,h11}, {h1,h2}));
  }
  
  // i2 ( h10 p5 )_f + = 1 * f ( h10 p5 )_f
  TEST(CCSD_T2,t2_2_4_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h10}, {p5}, 1.0, {h10}, {p5}));
  }

  TEST(CCSD_T2,t2_2_4_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h10}, {p5}, -1.0, {p6}, {h7},
            {h7,h10}, {p5,p6}));
  }
  
  TEST(CCSD_T2,t2_2_4) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h10,p3}, {h1,h2}, -1.0, {p3,p5}, {h1,h2},
            {h10}, {p5}));
  }
  
  // i2 ( h7 h10 h1 p9 )_v + = 1 * v ( h7 h10 h1 p9 )_v
  TEST(CCSD_T2,t2_2_5_1) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, {h7,h10}, {h1,p9}, 1.0, {h7,h10}, {h1,p9}));
  }

  TEST(CCSD_T2,t2_2_5_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h7,h10}, {h1,p9}, 1.0, {p5}, {h1}, {h7,h10},
            {p5,p9}));
  }
  
  TEST(CCSD_T2,t2_2_5) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h10,p3}, {h1,h2}, 1.0, {p3,p9}, {h1,h7},
            {h7,h10}, {h2,p9}));
  }
  
  TEST(CCSD_T2,c2f_t2_t12a) {
    ASSERT_TRUE(test_mult_no_n(*g_ec,{p1,p2}, {h3,h4}, 0.5, {p1}, {h3}, {p2}, {h4}));
  }
  
  TEST(CCSD_T2,t2_2_6) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h10,p3}, {h1,h2}, 0.5, {p5,p6}, {h1,h2},
            {h10,p3}, {p5,p6}));
  }
  
  TEST(CCSD_T2,c2d_t2_t12a) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p1,p2}, {h3,h4}, -0.5, {p1}, {h3}, {p2}, {h4}));
  }
  
  TEST(CCSD_T2,t2_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p3,p4}, {h1,h2}, -1.0, {p3}, {h10}, {h10,p4},
            {h1,h2}));
  }
  
  TEST(CCSD_T2,lt2_3x) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p3,p4}, {h1,h2}, -1.0, {p5}, {h1}, {p3,p4},
            {h2,p5}));
  }
  
  // i1 ( h9 h1 )_f + = 1 * f ( h9 h1 )_f
  TEST(CCSD_T2,t2_4_1) {
      ASSERT_TRUE(test_assign_no_n(*g_ec, {h9}, {h1}, 1.0, {h9}, {h1}));
  }

  TEST(CCSD_T2,t2_4_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h9}, {p8}, 1.0, {h9}, {p8}));
  }
  
  TEST(CCSD_T2,t2_4_2_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h9}, {p8}, 1.0, {p6}, {h7}, {h7,h9},
            {p6,p8}));
  }
  
  TEST(CCSD_T2,t2_4_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h9}, {h1}, 1.0, {p8}, {h1}, {h9}, {p8}));
  }
  
  TEST(CCSD_T2,t2_4_3) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h9}, {h1}, -1.0, {p6}, {h7}, {h7,h9}, {h1,p6}));
  }
  
  TEST(CCSD_T2,t2_4_4) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h9}, {h1}, -0.5, {p6,p7}, {h1,h8}, {h8,h9},
            {p6,p7}));
  }
  
  TEST(CCSD_T2,t2_4) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p3,p4}, {h1,h2}, -1.0, {p3,p4}, {h1,h9}, {h9},
            {h2}));
  }
  
  // i1 ( p3 p5 )_f + = 1 * f ( p3 p5 )_f
  TEST(CCSD_T2,t2_5_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {p3}, {p5}, 1.0, {p3}, {p5}));
  }

  TEST(CCSD_T2,t2_5_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p3}, {p5}, -1.0, {p6}, {h7}, {h7,p3}, {p5,p6}));
  }
  
  TEST(CCSD_T2,t2_5_3) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p3}, {p5}, -0.5, {p3,p6}, {h7,h8}, {h7,h8},
            {p5,p6}));
  }
  
  TEST(CCSD_T2,t2_5) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p3,p4}, {h1,h2}, 1.0, {p3,p5}, {h1,h2}, {p4},
            {p5}));
  }
  
  // i1 ( h9 h11 h1 h2 )_v + = -1 * v ( h9 h11 h1 h2 )_v
  TEST(CCSD_T2,t2_6_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h9,h11}, {h1,h2}, -1.0, {h9,h11},
            {h1,h2}));
  }

  // i2 ( h9 h11 h1 p8 )_v + = 1 * v ( h9 h11 h1 p8 )_v
TEST(CCSD_T2,t2_6_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h9,h11}, {h1,p8}, 1.0, {h9,h11},
            {h1,p8}));
  }

  TEST(CCSD_T2,t2_6_2_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h9,h11}, {h1,p8}, 0.5, {p6}, {h1}, {h9,h11},
            {p6,p8}));
  }
  
  TEST(CCSD_T2,t2_6_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h9,h11}, {h1,h2}, 1.0, {p8}, {h1}, {h9,h11},
            {h2,p8}));
  }
  
  TEST(CCSD_T2,t2_6_3) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h9,h11}, {h1,h2}, -0.5, {p5,p6}, {h1,h2},
            {h9,h11}, {p5,p6}));
  }
  
  TEST(CCSD_T2,t2_6) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p3,p4}, {h1,h2}, -0.5, {p3,p4}, {h9,h11},
            {h9,h11}, {h1,h2}));
  }
  
  // i1 ( h6 p3 h1 p5 )_v + = 1 * v ( h6 p3 h1 p5 )_v
  TEST(CCSD_T2,t2_7_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h6,p3}, {h1,p5}, 1.0, {h6,p3}, {h1,p5}));
  }

  TEST(CCSD_T2,t2_7_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h6,p3}, {h1,p5}, -1.0, {p7}, {h1}, {h6,p3},
            {p5,p7}));
  }
  
  TEST(CCSD_T2,t2_7_3) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h6,p3}, {h1,p5}, -0.5, {p3,p7}, {h1,h8},
            {h6,h8}, {p5,p7}));
  }
  
  TEST(CCSD_T2,t2_7) {
    ASSERT_TRUE(test_mult_no_n(*g_ec,  {p3,p4}, {h1,h2}, -1.0, {p3,p5}, {h1,h6},
            {h6,p4}, {h2,p5}));
  }
  
  TEST(CCSD_T2,vt1t1_1_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h5,p3}, {h1,h2}, -2, {p6}, {h1}, {h5,p3},
            {h2,p6}));
  }
  
  TEST(CCSD_T2,vt1t1_1) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p3,p4}, {h1,h2}, -0.5, {p3}, {h5}, {h5,p4},
            {h1,h2}));
  }
  
  TEST(CCSD_T2,c2f_t2_t12b) {
    ASSERT_TRUE(test_mult_no_n(*g_ec,{p1,p2}, {h3,h4}, 0.5, {p1}, {h3}, {p2}, {h4}));
  }
  
  TEST(CCSD_T2,t2_8) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p3,p4}, {h1,h2}, 0.5, {p5,p6}, {h1,h2},
            {p3,p4}, {p5,p6}));
  }
  
  TEST(CCSD_T2,c2d_t2_t12b) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p1,p2}, {h3,h4}, -0.5, {p1}, {h3}, {p2}, {h4}));
  }
  

///////////////////////////////////////////////////////////////////////
///////////////                 CC2_T1                 ////////////////
///////////////////////////////////////////////////////////////////////

TEST (CC2_T1, t1_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {p2}, {h1}, 1.0, {p2}, {h1}));
  }
  
  TEST (CC2_T1, t1_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h7}, {h1}, 1.0, {h7}, {h1}));
  }
  
  TEST (CC2_T1, t1_2_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec,{h7}, {p3}, 1.0, {h7}, {p3}));
  }
  
  TEST (CC2_T1, t1_2_2_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec,{h7}, {p3}, -1.0, {p5}, {h6}, {h6,h7},
            {p3,p5}));
  }
  
  TEST (CC2_T1, t1_2_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h7}, {h1}, 1.0, {p3}, {h1}, {h7}, {p3}));
  }
  
  TEST (CC2_T1, t1_2_3) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h7}, {h1}, -1.0, {p4}, {h5}, {h5,h7},
            {h1,p4}));
  }
  
  TEST (CC2_T1, t1_2_4) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h7}, {h1}, -0.5, {p3,p4}, {h1,h5},
            {h5,h7}, {p3,p4}));
  }
  
  TEST (CC2_T1, t1_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {h1}, -1.0, {p2}, {h7}, {h7}, {h1}));
  }
  
  TEST (CC2_T1, t1_3_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {p2}, {p3}, 1.0, {p2}, {p3}));
  }
  
  TEST (CC2_T1, t1_3_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {p3}, -1.0, {p4}, {h5}, {h5,p2},
            {p3,p4}));
  }
  
  TEST (CC2_T1, t1_3) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {h1}, 1.0, {p3}, {h1}, {p2}, {p3}));
  }
  
  TEST (CC2_T1, t1_4) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {h1}, -1.0, {p3}, {h4}, {h4,p2},
            {h1,p3}));
  }
  
  TEST (CC2_T1, t1_5_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h8}, {p7}, 1.0, {h8}, {p7}));
  }
  
  TEST (CC2_T1, t1_5_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h8}, {p7}, 1.0, {p5}, {h6}, {h6,h8},
            {p5,p7}));
  }
  
  TEST (CC2_T1, t1_5) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {h1}, 1.0, {p2,p7}, {h1,h8}, {h8},
            {p7}));
  }
  
  TEST (CC2_T1, t1_6_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h4,h5}, {h1,p3} , 1.0, {h4,h5},
            {h1,p3}));
  }
  
  TEST (CC2_T1, t1_6_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h4,h5}, {h1,p3}, -1.0, {p6}, {h1}, {h4,h5},
            {p3,p6}));
  }
  
  TEST (CC2_T1, t1_6) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {h1}, -0.5, {p2,p3}, {h4,h5},
            {h4,h5}, {h1,p3}));
  }
  
  TEST (CC2_T1, t1_7) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {h1}, -0.5, {p3,p4}, {h1,h5},
            {h5,p2}, {p3,p4}));
  }
  

///////////////////////////////////////////////////////////////////////
///////////////                 CC2_T2                 ////////////////
///////////////////////////////////////////////////////////////////////


TEST(CC2_T2,t2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {p3,p4}, {h1,h2}, 1.0,
          {p3,p4}, {h1,h2}));
  }
  
  TEST(CC2_T2,t2_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h10,p3}, {h1,h2}, 1.0,
            {h10,p3}, {h1,h2}));
  }
  
  TEST(CC2_T2,t2_2_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h8,h10} ,{h1,h2}, -1.0,
            {h8,h10}, {h1,h2}));
  }
  
  TEST(CC2_T2,t2_2_2_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h8,h10}, {h1,p5}, 1.0,
            {h8,h10}, {h1,p5}));
  }
  
  TEST(CC2_T2,t2_2_2_2_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h8,h10}, {h1,p5}, -0.5, {p6}, {h1},
            {h8,h10}, {p5,p6}));
  }
  
  TEST(CC2_T2,t2_2_2_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h8,h10}, {h1,h2}, -1.0, {p5}, {h1},
            {h8,h10}, {h2,p5}));
  }
  
  TEST(CC2_T2,t2_2_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h10,p3}, {h1,h2}, 0.5, {p3}, {h8},
            {h8,h10}, {h1,h2}));
  }
  
  TEST(CC2_T2,t2_2_3_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h10,p3}, {h1,p5}, 1.0,
            {h10,p3}, {h1,p5}));
  }
  
  TEST(CC2_T2,t2_2_3_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h10,p3}, {h1,p5}, -0.5, {p6}, {h1},
            {h10,p3}, {p5,p6}));
  }
  
  TEST(CC2_T2,t2_2_3) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {h10,p3}, {h1,h2}, -1.0, {p5}, {h1},
            {h10,p3}, {h2,p5}));
  }
  
  TEST(CC2_T2,t2_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p3,p4}, {h1,h2}, -1.0, {p3}, {h10}, {h10,p4},
            {h1,h2}));
  }
  
  TEST(CC2_T2,t2_3_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {p3,p4}, {h1,p5}, 1.0,
            {p3,p4}, {h1,p5}));
  }
  
  TEST(CC2_T2,t2_3_2) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p3,p4}, {h1,p5}, -0.5, {p6}, {h1}, {p3,p4},
            {p5,p6}));
  }
  
  TEST(CC2_T2,t2_3) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p3,p4}, {h1,h2}, -1.0, {p5}, {h1}, {p3,p4},
            {h2,p5}));
  }
  
  TEST(CC2_T2,t2_4) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p3,p4}, {h1,h2}, -1.0, {p3,p4}, {h1,h5}, {h5},
            {h2}));
  }
  
  TEST(CC2_T2,t2_5) {
    ASSERT_TRUE(test_mult_no_n(*g_ec, {p3,p4}, {h1,h2}, 1.0, {p3,p5}, {h1,h2}, {p4},
            {p5}));
  }
  