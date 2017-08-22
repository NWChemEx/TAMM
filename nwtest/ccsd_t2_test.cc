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
//#include <iostream>
//#include "gtest/gtest.h"
//
//#include "tensor/corf.h"
//#include "tammx/tammx.h"
//
//namespace {
//tammx::ExecutionContext* g_ec;
//}
//
//class TestEnvironment : public testing::Environment {
// public:
//  explicit TestEnvironment(tammx::ExecutionContext* ec) {
//    g_ec = ec;
//  }
//};
//
//const auto P1B = tamm::P1B;
//const auto P2B = tamm::P2B;
//const auto P3B = tamm::P3B;
//const auto P4B = tamm::P4B;
//const auto P5B = tamm::P5B;
//const auto P6B = tamm::P6B;
//const auto P7B = tamm::P7B;
//const auto P8B = tamm::P8B;
//const auto P10B = tamm::P10B;
//const auto P11B = tamm::P11B;
//const auto P12B = tamm::P12B;
//const auto P9B = tamm::P9B;
//const auto H1B = tamm::H1B;
//const auto H2B = tamm::H2B;
//const auto H3B = tamm::H3B;
//const auto H4B = tamm::H4B;
//const auto H5B = tamm::H5B;
//const auto H6B = tamm::H6B;
//const auto H7B = tamm::H7B;
//const auto H8B = tamm::H8B;
//const auto H9B = tamm::H9B;
//const auto H10B = tamm::H10B;
//const auto H11B = tamm::H11B;
//const auto H12B = tamm::H12B;
//
//const auto h1 = tamm::H1B;
//const auto h2 = tamm::H2B;
//const auto h3 = tamm::H3B;
//const auto h4 = tamm::H4B;
//const auto h5 = tamm::H5B;
//const auto h6 = tamm::H6B;
//const auto h7 = tamm::H7B;
//const auto h8 = tamm::H8B;
//const auto h9 = tamm::H9B;
//const auto h10 = tamm::H10B;
//const auto h11 = tamm::H11B;
//const auto h12 = tamm::H12B;
//
//const auto p1 = tamm::P1B;
//const auto p2 = tamm::P2B;
//const auto p3 = tamm::P3B;
//const auto p4 = tamm::P4B;
//const auto p5 = tamm::P5B;
//const auto p6 = tamm::P6B;
//const auto p7 = tamm::P7B;
//const auto p8 = tamm::P8B;
//const auto p9 = tamm::P9B;
//const auto p10 = tamm::P10B;
//const auto p11 = tamm::P11B;
//const auto p12 = tamm::P12B;
//
//bool test_assign_no_n(tammx::ExecutionContext& ec,
//                      const std::vector<tamm::IndexName>& cupper_labels,
//                      const std::vector<tamm::IndexName>& clower_labels,
//                      double alpha,
//                      const std::vector<tamm::IndexName>& aupper_labels,
//                      const std::vector<tamm::IndexName>& alower_labels);
//
//bool test_mult_no_n(tammx::ExecutionContext& ec,
//               const std::vector<tamm::IndexName>& cupper_labels,
//               const std::vector<tamm::IndexName>& clower_labels,
//               double alpha,
//               const std::vector<tamm::IndexName>& aupper_labels,
//               const std::vector<tamm::IndexName>& alower_labels,
//               const std::vector<tamm::IndexName>& bupper_labels,
//               const std::vector<tamm::IndexName>& blower_labels);

TEST(CCSD_T2,t2_1) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, {p3,p4}, {h1,h2}, 1.0,
		{p3,p4}, {h1,h2}));
}

TEST(CCSD_T2,t2_2_1) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, {h10,p3}, {h1,h2}, 1.0,
		  {h10,p3}, {h1,h2}));
}

TEST(CCSD_T2,t2_2_2_1) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, {h10,h11} ,{h1,h2}, -1.0,
		  {h10,h11}, {h1,h2}));
}

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

TEST(CCSD_T2,t2_6_1) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, {h9,h11}, {h1,h2}, -1.0, {h9,h11},
		  {h1,h2}));
}

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

