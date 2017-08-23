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

TEST (CCSD_T1, t1_1) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, {p2}, {h1}, 1.0, {p2}, {h1}));
}

TEST (CCSD_T1, t1_2_1) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, {h7}, {h1}, 1.0, {h7}, {h1}));
}

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

TEST (CCSD_T1, t1_2_4) {
  ASSERT_TRUE(test_mult_no_n(*g_ec, {h7}, {h1}, -0.5, {p3,p4}, {h1,h5},
		  {h5,h7}, {p3,p4}));
}

TEST (CCSD_T1, t1_2) {
  ASSERT_TRUE(test_mult_no_n(*g_ec, {p2}, {h1}, -1.0, {p2}, {h7}, {h7}, {h1}));
}

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
