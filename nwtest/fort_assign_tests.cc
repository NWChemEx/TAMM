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
#include "gtest/gtest.h"

#include "tensor/corf.h"
#include "tensor/equations.h"
#include "tensor/input.h"
#include "tensor/schedulers.h"
#include "tensor/t_assign.h"
#include "tensor/t_mult.h"
#include "tensor/tensor.h"
#include "tensor/tensors_and_ops.h"
#include "tensor/variables.h"
#include "macdecls.h"

#include "tammx/tammx.h"

#include <mpi.h>
#include <ga.h>
#include <macdecls.h>

namespace {
tammx::ExecutionContext* g_ec;
}

class TestEnvironment : public testing::Environment {
 public:
  explicit TestEnvironment(tammx::ExecutionContext* ec) {
    g_ec = ec;
  }
};

extern "C" {
  
  void offset_ccsd_t1_2_1_(Integer *l_t1_2_1_offset, Integer *k_t1_2_1_offset,
                           Integer *size_t1_2_1);

  typedef void add_fn(Integer *ta, Integer *offseta, Integer *irrepa,
                      Integer *tc, Integer *offsetc, Integer *irrepc);
                       
  typedef void mult_fn(Integer *ta, Integer *offseta, Integer *irrepa,
                       Integer *tb, Integer *offsetb, Integer *irrepb,
                       Integer *tc, Integer *offsetc, Integer *irrepc);

  typedef void mult_fn_2(Integer *ta, Integer *offseta,
                       Integer *tb, Integer *offsetb,
                       Integer *tc, Integer *offsetc);

  add_fn ccsd_t1_1_, ccsd_t1_2_1_, ccsd_t1_2_2_1_, ccsd_t1_3_1_, ccsd_t1_5_1_;
  add_fn ccsd_t1_6_1_;
  mult_fn ccsd_t1_2_;
  mult_fn_2 cc2_t1_5_;
}

const auto P1B = tamm::P1B;
const auto P2B = tamm::P2B;
const auto P3B = tamm::P3B;
const auto P4B = tamm::P4B;
const auto P5B = tamm::P5B;
const auto P6B = tamm::P6B;
const auto P7B = tamm::P7B;
const auto P8B = tamm::P8B;
const auto P10B = tamm::P10B;
const auto P11B = tamm::P11B;
const auto P12B = tamm::P12B;
const auto P9B = tamm::P9B;
const auto H1B = tamm::H1B;
const auto H2B = tamm::H2B;
const auto H3B = tamm::H3B;
const auto H4B = tamm::H4B;
const auto H5B = tamm::H5B;
const auto H6B = tamm::H6B;
const auto H7B = tamm::H7B;
const auto H8B = tamm::H8B;
const auto H9B = tamm::H9B;
const auto H10B = tamm::H10B;
const auto H11B = tamm::H11B;
const auto H12B = tamm::H12B;

const auto p1 = tamm::P1B;
const auto p2 = tamm::P2B;
const auto p3 = tamm::P3B;
const auto p4 = tamm::P4B;
const auto h1 = tamm::H1B;
const auto h2 = tamm::H2B;
const auto h3 = tamm::H3B;
const auto h4 = tamm::H4B;
const auto TO = tamm::TO;
const auto TV = tamm::TV;

bool test_assign(tammx::ExecutionContext& ec,
                      double alpha,
                      const std::vector<tamm::IndexName>& cupper_labels,
                      const std::vector<tamm::IndexName>& clower_labels,
                      const std::vector<tamm::IndexName>& aupper_labels,
                      const std::vector<tamm::IndexName>& alower_labels);

bool test_assign_no_n(tammx::ExecutionContext& ec,
                      double alpha,
                      const std::vector<tamm::IndexName>& cupper_labels,
                      const std::vector<tamm::IndexName>& clower_labels,
                      const std::vector<tamm::IndexName>& aupper_labels,
                      const std::vector<tamm::IndexName>& alower_labels,
					  add_fn fortran_assign_fn);


void test_assign_ccsd_e(tammx::ExecutionContext& ec) {
	test_assign(ec, 1.0, {H6B}, {P5B},
                       {H6B}, {P5B});               // ccsd_e_copy_fock_to_t
}

void test_assign_ccsd_t1(tammx::ExecutionContext& ec) {
  test_assign_no_n(ec, 1.0, {P2B}, {H1B},
                       {P2B}, {H1B}, ccsd_t1_1_);        // ccsd_t1_1
  test_assign_no_n(ec, 1.0, {H7B}, {H1B},
                       {H7B}, {H1B}, ccsd_t1_2_1_);      // ccsd_t1_2_1
  test_assign_no_n(ec, 1.0, {H7B}, {P3B},
                       {H7B}, {P3B}, ccsd_t1_2_2_1_);    // ccsd_t1_2_2_1
  test_assign_no_n(ec, 1.0, {P2B}, {P3B},
                       {P2B}, {P3B}, ccsd_t1_3_1_);            // ccsd_t1_3_1
  test_assign_no_n(ec, 1.0, {H8B}, {P7B},
                       {H8B}, {P7B}, ccsd_t1_5_1_);            // ccsd_t1_5_1
  test_assign_no_n(ec, 1.0, {H4B, H5B}, {H1B, P3B},
                       {H4B, H5B}, {H1B, P3B}, ccsd_t1_6_1_);  // ccsd_t1_6_1
}

void test_assign_ccsd_t2(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {P3B, P4B}, {H1B, H2B},
                       {P3B, P4B}, {H1B, H2B});        // ccsd_t2_1
  test_assign(ec, 1.0, {H10B, P3B}, {H1B, H2B},
                       {H10B, P3B}, {H1B, H2B});       // ccsd_t2_2_1
  test_assign(ec, -1.0, {H10B, H11B}, {H1B, H2B},
                        {H10B, H11B}, {H1B, H2B});     // ccsd_t2_2_2_1
  test_assign(ec, 1.0, {H10B, H11B}, {H1B, P5B},
                       {H10B, H11B}, {H1B, P5B});      // ccsd_t2_2_2_2_1
  test_assign(ec, 1.0, {H10B}, {P5B}, {H10B}, {P5B});  // ccsd_t2_2_4_1
  test_assign(ec, 1.0, {H7B, H10B}, {H1B, P9B},
                       {H7B, H10B}, {H1B, P9B});       // ccsd_t2_2_5_1
  test_assign(ec, 1.0, {H9B}, {H1B}, {H9B}, {H1B});    // ccsd_t2_4_1
  test_assign(ec, 1.0, {H9B}, {P8B}, {H9B}, {P8B});    // ccsd_t2_4_2_1
  test_assign(ec, 1.0, {P3B}, {P5B}, {P3B}, {P5B});    // ccsd_t2_5_1
  test_assign(ec, -1.0, {H9B, H11B}, {H1B, H2B},
                        {H9B, H11B}, {H1B, H2B});      // ccsd_t2_6_1
  test_assign(ec, 1.0, {H9B, H11B}, {H1B, P8B},
                       {H9B, H11B}, {H1B, P8B});       // ccsd_t2_6_2_1
  test_assign(ec, 1.0, {H6B, P3B}, {H1B, P5B},
                       {H6B, P3B}, {H1B, P5B});      // ccsd_t2_7_1
}

void test_assign_cc2_t1(tammx::ExecutionContext& ec) {  // copy of ccsd_t1
  test_assign(ec, 1.0, {P2B}, {H1B},
                       {P2B}, {H1B});            // ccsd_t1_1
  test_assign(ec, 1.0, {H7B}, {H1B},
                       {H7B}, {H1B});            // ccsd_t1_1_2_1
  test_assign(ec, 1.0, {H7B}, {P3B},
                       {H7B}, {P3B});            // ccsd_t1_2_2_1
  test_assign(ec, 1.0, {P2B}, {P3B},
                       {P2B}, {P3B});            // ccsd_t1_3_1
  test_assign(ec, 1.0, {H8B}, {P7B},
                       {H8B}, {P7B});            // ccsd_t1_5_1
  test_assign(ec, 1.0, {H4B, H5B}, {H1B, P3B},
                       {H4B, H5B}, {H1B, P3B});  // ccsd_t1_6_1
}

void test_assign_cc2_t2(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {P3B, P4B}, {H1B, H2B},
                       {P3B, P4B}, {H1B, H2B});       // ccsd_t2_1
  test_assign(ec, 1.0, {H10B, P3B}, {H1B, H2B},
                       {H10B, P3B}, {H1B, H2B});      // ccsd_t2_2_1
  test_assign(ec, 1.0, {H8B, H10B}, {H1B, P5B},
                       {H8B, H10B}, {H1B, P5B});      // ccsd_t2_2_2_1
  test_assign(ec, 1.0, {H8B, H10B}, {H1B, P5B},
                       {H8B, H10B}, {H1B, P5B});      // ccsd_t2_2_2_2_1
  test_assign(ec, 1.0, {H10B, P3B}, {H1B, P5B},
                       {H10B, P3B}, {H1B, P5B});      // ccsd_t2_2_2_3
  test_assign(ec, 1.0, {P3B, P4B}, {H1B, P5B},
                       {P3B, P4B}, {H1B, P5B});       // ccsd_t2_3_1
}

void test_assign_cisd_c1(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {P2B}, {H1B},
                       {P2B}, {H1B});                 // cisd_c1_1
}

void test_assign_cisd_c2(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {P3B, P4B}, {H1B, H2B},
                       {P3B, P4B}, {H1B, H2B});       // cisd_c2_1
}

void test_assign_ccsd_lambda1(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {H2B}, {P1B},
                       {H2B}, {P1B});                 // lambda1_1
  test_assign(ec, 1.0, {H2B}, {H7B},
                       {H2B}, {H7B});                 // lambda1_2_1
  test_assign(ec, 1.0, {H2B}, {P3B},
                       {H2B}, {P3B});                 // lambda1_2_2_1
  test_assign(ec, 1.0, {P7B}, {P1B},
                       {P7B}, {P1B});                 // lambda1_3_1
  test_assign(ec, 1.0, {P9B}, {H11B},
                       {P9B}, {H11B});                // lambda1_5_1
  test_assign(ec, 1.0, {H10B}, {H11B},
                       {H10B}, {H11B});               // lambda1_5_2_1
  test_assign(ec, 1.0, {H10B}, {P3B},
                       {H10B}, {P3B});                // lambda1_5_2_2_1
  test_assign(ec, 1.0, {P9B}, {P7B},
                       {P9B}, {P7B});                 // lambda1_5_3_1
  test_assign(ec, 1.0, {H5B}, {P4B},
                       {H5B}, {P4B});                 // lambda1_5_5_1
  test_assign(ec, 1.0, {H5B, H6B}, {H11B, P4B},
                       {H5B, H6B}, {H11B, P4B});      // lambda1_5_6_1
  test_assign(ec, 1.0, {H2B, P9B}, {H11B, H12B},
                       {H2B, P9B}, {H11B, H12B});     // lambda1_6_1
  test_assign(ec, 1.0, {H2B, H7B}, {H11B, H12B},
                       {H2B, H7B}, {H11B, H12B});     // lambda1_6_2_1
  test_assign(ec, 1.0, {H2B, H7B}, {H12B, P3B},
                       {H2B, H7B}, {H12B, P3B});     // lambda1_6_2_2_1
  test_assign(ec, 1.0, {H2B, P9B}, {H12B, P3B},
                       {H2B, P9B}, {H12B, P3B});      // lambda1_6_3_1
  test_assign(ec, 1.0, {H2B}, {P5B},
                       {H2B}, {P5B});                 // lambda1_6_4_1
  test_assign(ec, 1.0, {H2B, H6B}, {H12B, P4B},
                       {H2B, H6B}, {H12B, P4B});      // lambda1_6_5_1
  test_assign(ec, -1.0, {P5B, P8B}, {H7B, P1B},
                       {P5B, P8B}, {H7B, P1B});       // lambda1_7_1
  test_assign(ec, 1.0, {P9B}, {H10B},
                       {P9B}, {H10B});                // lambda1_8_1
}

void test_assign_ccsd_lambda2(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {H3B, H4B}, {P1B, P2B},
                       {H3B, H4B}, {P1B, P2B});      // lambda2_2_1
  test_assign(ec, 1.0, {H3B}, {P1B},
                       {H3B}, {P1B});                // lambda2_2_1
  test_assign(ec, 1.0, {H3B, H4B}, {H7B, P1B},
                       {H3B, H4B}, {H7B, P1B});      // lambda2_3_1
  test_assign(ec, 1.0, {H3B}, {H9B},
                       {H3B}, {H9B});                // lambda2_5_1
  test_assign(ec, 1.0, {H3B}, {P5B},
                       {H3B}, {P5B});                // lambda2_5_2_1
  test_assign(ec, 1.0, {P10B}, {P1B},
                       {P10B}, {P1B});               // lambda2_6_1
  test_assign(ec, 1.0, {H3B, H4B}, {H9B, H10B},
                       {H3B, H4B}, {H9B, H10B});     // lambda2_7_1
  test_assign(ec, 1.0, {H3B, H4B}, {H10B, P5B},
                       {H3B, H4B}, {H10B, P5B});     // lambda2_7_2_1
  test_assign(ec, 1.0, {H3B, P7B}, {H9B, P1B},
                       {H3B, P7B}, {H9B, P1B});      // lambda2_8_1
}

void test_assign_eaccsd_x1(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {P2B}, {P6B},
                       {P2B}, {P6B});               // eaccsd_x1_1_1
  test_assign(ec, 1.0, {P6B}, {P7B},
                       {P6B}, {P7B});               // eaccsd_x1_2_1
  test_assign(ec, 1.0, {H3B}, {P7B},
                       {H3B}, {P7B});               // eaccsd_x1_4_1_1
}

void test_assign_eaccsd_x2(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {H8B}, {H1B},
                       {H8B}, {H1B});               // eaccsd_x2_2_1
  test_assign(ec, 1.0, {H8B}, {P9B},
                       {H8B}, {P9B});               // eaccsd_x2_2_2_1
  test_assign(ec, 1.0, {P3B}, {P8B},
                       {P3B}, {P8B});               // eaccsd_x2_3_1
  test_assign(ec, 1.0, {H7B, P3B}, {H1B, P8B},
                       {H7B, P3B}, {H1B, P8B});     // eaccsd_x2_4_1
  test_assign(ec, 1.0, {H9B}, {P5B},
                       {H9B}, {P5B});               // eaccsd_x2_6_2_1
  test_assign(ec, 1.0, {H8B, H9B}, {H1B, P10B},
                       {H8B, H9B}, {H1B, P10B});    // eaccsd_x2_6_3_1
  test_assign(ec, 1.0, {H5B}, {P9B},
                       {H5B}, {P9B});               // eaccsd_x2_8_1_1
}

void test_assign_icsd_t1(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {P2B}, {H1B},
                       {P2B}, {H1B});               // icsd_t1_1
  test_assign(ec, 1.0, {H7B}, {H1B},
                       {H7B}, {H1B});               // icsd_t1_2_1
  test_assign(ec, 1.0, {H7B}, {P3B},
                       {H7B}, {P3B});               // icsd_t1_2_2_1
  test_assign(ec, 1.0, {P2B}, {P3B},
                       {P2B}, {P3B});               // icsd_t1_3_1
  test_assign(ec, 1.0, {H8B}, {P7B},
                       {H8B}, {P7B});               // icsd_t1_5_1
  test_assign(ec, 1.0, {H4B, H5B}, {H1B, P3B},
                       {H4B, H5B}, {H1B, P3B});     // icsd_t1_6_1
}

void test_assign_icsd_t2(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {P3B, P4B}, {H1B, H2B},
                       {P3B, P4B}, {H1B, H2B});     // icsd_t2_1
  test_assign(ec, 1.0, {H10B, P3B}, {H1B, H2B},
                       {H10B, P3B}, {H1B, H2B});    // icsd_t2_2_1
  test_assign(ec, -1.0, {H10B, H11B}, {H1B, H2B},
                       {H10B, H11B}, {H1B, H2B});   // icsd_t2_2_2_1
  test_assign(ec, 1.0, {H10B, H11B}, {H1B, P5B},
                       {H10B, H11B}, {H1B, P5B});   // icsd_t2_2_2_2_1
  test_assign(ec, 1.0, {H10B}, {P5B},
                       {H10B}, {P5B});              // icsd_t2_2_4_1
  test_assign(ec, 1.0, {H7B, H10B}, {H1B, P9B},
                       {H7B, H10B}, {H1B, P9B});    // icsd_t2_2_5_1
  test_assign(ec, 1.0, {H9B}, {H1B},
                       {H9B}, {H1B});               // icsd_t2_4_1
  test_assign(ec, 1.0, {H9B}, {P8B},
                       {H9B}, {P8B});               // icsd_t2_4_2_1
  test_assign(ec, 1.0, {P3B}, {P5B},
                       {P3B}, {P5B});               // icsd_t2_5_1
  test_assign(ec, -1.0, {H9B, H11B}, {H1B, H2B},
                       {H9B, H11B}, {H1B, H2B});     // icsd_t2_6_1
  test_assign(ec, 1.0, {H9B, H11B}, {H1B, P8B},
                       {H9B, H11B}, {H1B, P8B});     // icsd_t2_6_2_1
  test_assign(ec, 1.0, {H6B, P3B}, {H1B, P5B},
                       {H6B, P3B}, {H1B, P5B});     // icsd_t2_7_1
}

void test_assign_ipccsd_x1(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {H6B}, {H1B},
                       {H6B}, {H1B});               // ipccsd_x1_1_1
  test_assign(ec, 1.0, {H6B}, {P7B},
                       {H6B}, {P7B});               // ipccsd_x1_1_2_1
  test_assign(ec, 1.0, {H6B}, {P7B},
                       {H6B}, {P7B});               // ipccsd_x1_2_1
  test_assign(ec, 1.0, {H6B, H8B}, {H1B, P7B},
                       {H6B, H8B}, {H1B, P7B});     // ipccsd_x1_3_1
}

void test_assign_ipccsd_x2(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {H9B, P3B}, {H1B, H2B},
                       {H9B, P3B}, {H1B, H2B});     // ipccsd_x2_1_1
  test_assign(ec, 1.0, {H9B, P3B}, {H1B, P5B},
                       {H9B, P3B}, {H1B, P5B});     // ipccsd_x2_1_2_1
  test_assign(ec, 1.0, {H9B}, {P8B},
                       {H9B}, {P8B});               // ipccsd_x1_3_1
  test_assign(ec, 1.0, {H6B, H9B}, {H1B, P5B},
                       {H6B, H9B}, {H1B, P5B});     // ipccsd_x2_1_4_1
  test_assign(ec, 1.0, {H8B}, {H1B},
                       {H8B}, {H1B});               // ipccsd_x2_2_1
  test_assign(ec, 1.0, {H8B}, {P9B},
                       {H8B}, {P9B});               // ipccsd_x2_2_2_1
  test_assign(ec, 1.0, {P3B}, {P8B},
                       {P3B}, {P8B});               // ipccsd_x2_3_1
  test_assign(ec, 1.0, {H9B, H10B}, {H1B, H2B},
                       {H9B, H10B}, {H1B, H2B});     // ipccsd_x2_4_1
  test_assign(ec, 1.0, {H9B, H10B}, {H1B, P5B},
                       {H9B, H10B}, {H1B, P5B});     // ipccsd_x2_4_2_1
  test_assign(ec, 1.0, {H7B, P3B}, {H1B, P8B},
                       {H7B, P3B}, {H1B, P8B});     // ipccsd_x2_5_1
  test_assign(ec, 1.0, {H8B, H10B}, {H1B, P5B},
                       {H8B, H10B}, {H1B, P5B});     // ipccsd_x2_6_1_2_1
  test_assign(ec, 1.0, {H10B}, {P5B},
                       {H10B}, {P5B});               // ipccsd_x2_6_2_1
  test_assign(ec, 1.0, {H8B, H10B}, {H1B, P9B},
                       {H8B, H10B}, {H1B, P9B});     // ipccsd_x2_6_3_1
}





