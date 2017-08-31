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
TEST (FORT_CCSD_T1, t1_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {p2}, {h1}, 1.0, {p2}, {h1},
                ccsd_t1_1_));
  }
  
  // i1 ( h7 h1 )_f + = 1 * f ( h7 h1 )_f
  TEST (FORT_CCSD_T1, t1_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h7}, {h1}, 1.0, {h7}, {h1},
                ccsd_t1_2_1_));
  }
  
  // i2 ( h7 p3 )_f + = 1 * f ( h7 p3 )_f
  TEST (FORT_CCSD_T1, t1_2_2_1) {
        ASSERT_TRUE(test_assign_no_n(*g_ec,{h7}, {p3}, 1.0, {h7}, {p3},
                ccsd_t1_2_2_1_));
  }
  
  // i1 ( h7 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 h7 p3 p4 )_v
  
  //TEST (FORT_CCSD_T1, t1_2_4) {
  //  ASSERT_TRUE(test_mult_no_n(*g_ec, {h7}, {h1}, -0.5, {p3,p4}, {h1,h5},
  //		  {h5,h7}, {p3,p4}, ccsd_t1_2_));
  //}
  
  // i0 ( p2 h1 )_tf + = -1 * Sum ( h7 ) * t ( p2 h7 )_t * i1 ( h7 h1 )_f
  
  //TEST (FORT_CCSD_T1, t1_2) {
  //  ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {h1}, -1.0, {p2}, {h7}, {h7}, {h1},
  //		  ccsd_t1_2_));
  //}
  
  // i1 ( p2 p3 )_f + = 1 * f ( p2 p3 )_f
  
  TEST (FORT_CCSD_T1, t1_3_1) {
        ASSERT_TRUE(test_assign_no_n(*g_ec, {p2}, {p3}, 1.0, {p2}, {p3},
                ccsd_t1_3_1_));
  }
  
  // i1 ( h8 p7 )_f + = 1 * f ( h8 p7 )_f
  
  TEST (FORT_CCSD_T1, t1_5_1) {
        ASSERT_TRUE(test_assign_no_n(*g_ec, {h8}, {p7}, 1.0, {h8}, {p7},
                ccsd_t1_5_1_));
  }
  
  
  // i1 ( h4 h5 h1 p3 )_v + = 1 * v ( h4 h5 h1 p3 )_v
  
  TEST (FORT_CCSD_T1, t1_6_1) {
        ASSERT_TRUE(test_assign_no_n(*g_ec, {h4,h5}, {h1,p3} , 1.0, {h4,h5},
                {h1,p3}, ccsd_t1_6_1_));
  }
  
  

///////////////////////////////////////////////////////////////////////
///////////////                 CCSD_T2                 ///////////////
///////////////////////////////////////////////////////////////////////


// i0 ( p3 p4 h1 h2 )_v + = 1 * v ( p3 p4 h1 h2 )_v
TEST (FORT_CCSD_T2, t2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {p3,p4}, {h1,h2}, 1.0,
            {p3,p4}, {h1,h2}, ccsd_t2_1_));
  }
  
  // i1 ( h10 p3 h1 h2 )_v + = 1 * v ( h10 p3 h1 h2 )_v
  
  TEST (FORT_CCSD_T2, t2_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {p3,h10}, {h1,h2}, 1.0,
            {h10,p3}, {h1,h2}, ccsd_t2_2_1_));
  }
  
  // i2 ( h10 h11 h1 h2 )_v + = -1 * v ( h10 h11 h1 h2 )_v
  TEST (FORT_CCSD_T2, t2_2_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h10,h11} ,{h1,h2}, -1.0,
            {h10,h11}, {h1,h2}, ccsd_t2_2_2_1_));
  }
  
  // i3 ( h10 h11 h1 p5 )_v + = 1 * v ( h10 h11 h1 p5 )_v
  
  TEST (FORT_CCSD_T2, t2_2_2_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h10,h11}, {h1,p5}, 1.0,
            {h10,h11}, {h1,p5}, ccsd_t2_2_2_2_1_));
  }
  
  
  // i2 ( h10 p5 )_f + = 1 * f ( h10 p5 )_f
  TEST (FORT_CCSD_T2, t2_2_4_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h10}, {p5}, 1.0, {h10}, {p5},
            ccsd_t2_2_4_1_));
  }
  
  // i2 ( h7 h10 h1 p9 )_v + = 1 * v ( h7 h10 h1 p9 )_v
  TEST (FORT_CCSD_T2, t2_2_5_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h7,h10}, {h1,p9}, 1.0,
            {h7,h10}, {h1,p9}, ccsd_t2_2_5_1_));
  }
  
  
  // i1 ( h9 h1 )_f + = 1 * f ( h9 h1 )_f
  TEST (FORT_CCSD_T2, t2_4_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h9}, {h1}, 1.0, {h9}, {h1},
            ccsd_t2_4_1_));
  }
  
  
  // i1 ( p3 p5 )_f + = 1 * f ( p3 p5 )_f
  TEST (FORT_CCSD_T2, t2_5_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {p3}, {p5}, 1.0, {p3}, {p5},
            ccsd_t2_5_1_));
  }
  
  
  // i1 ( h9 h11 h1 h2 )_v + = -1 * v ( h9 h11 h1 h2 )_v
  TEST (FORT_CCSD_T2, t2_6_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h9,h11}, {h1,h2}, -1.0, {h9,h11},
            {h1,h2}, ccsd_t2_6_1_));
  }
  
  // i2 ( h9 h11 h1 p8 )_v + = 1 * v ( h9 h11 h1 p8 )_v
  TEST (FORT_CCSD_T2, t2_6_2_1) {
    ASSERT_TRUE(test_assign_no_n(*g_ec, {h9,h11}, {h1,p8}, 1.0, {h9,h11},
            {h1,p8}, ccsd_t2_6_2_1_));
  }
  
  
  // i1 ( h6 p3 h1 p5 )_v + = 1 * v ( h6 p3 h1 p5 )_v
  // ccsd_t2_7_1 cannot be tested independently,
  // need the whole t2_7 block together, the intermediate is produced in
  // inconsistent form (flipping of indices between upper and
  // lower not supported in tamm/tammx)
  //TEST (FORT_CCSD_T2, t2_7_1) {
  //  ASSERT_TRUE(test_assign_no_n(*g_ec, {p3,h1}, {p5,h6}, 1.0, {h6,p3},
  //		  {h1,p5}, ccsd_t2_7_1_));
  //}
  
  
///////////////////////////////////////////////////////////////////////
///////////////                 CC2_T1                 ////////////////
///////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////
///////////////                 CC2_T2                 ////////////////
///////////////////////////////////////////////////////////////////////



