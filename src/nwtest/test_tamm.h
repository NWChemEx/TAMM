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
//#include "tensor/equations.h"
//#include "tensor/input.h"
//#include "tensor/schedulers.h"
//#include "tensor/t_assign.h"
//#include "tensor/t_mult.h"
//#include "tensor/tensor.h"
//#include "tensor/tensors_and_ops.h"
//#include "tensor/variables.h"
//#include "macdecls.h"
//
//#include "tammx/tammx.h"
//
//#include <mpi.h>
//#include <ga.h>
//#include <macdecls.h>

// namespace {
// tammx::ExecutionContext* g_ec;
// }

// class TestEnvironment : public testing::Environment {
//  public:
//   explicit TestEnvironment(tammx::ExecutionContext* ec) {
//     g_ec = ec;
//   }
// };

#ifndef TEST_TAMM_H
#define TEST_TAMM_H 

#include "nwtest/test_tammx.h"

extern "C" {
//
//  void offset_ccsd_t1_2_1_(Integer *l_t1_2_1_offset, Integer *k_t1_2_1_offset,
//                           Integer *size_t1_2_1);
//
//  typedef void add_fn(Integer *ta, Integer *offseta, Integer *irrepa,
//                      Integer *tc, Integer *offsetc, Integer *irrepc);
//
//  typedef void mult_fn(Integer *ta, Integer *offseta, Integer *irrepa,
//                       Integer *tb, Integer *offsetb, Integer *irrepb,
//                       Integer *tc, Integer *offsetc, Integer *irrepc);
//

typedef void mult_fn_2(Integer *ta, Integer *offseta,
  Integer *tb, Integer *offsetb,
  Integer *tc, Integer *offsetc);
  
mult_fn_2 cc2_t1_5_;

}

#if 0  

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

//const auto p1 = tamm::P1B;
//const auto p2 = tamm::P2B;
//const auto p3 = tamm::P3B;
//const auto p4 = tamm::P4B;
//const auto h1 = tamm::H1B;
//const auto h2 = tamm::H2B;
//const auto h3 = tamm::H3B;
//const auto h4 = tamm::H4B;
const auto TO = tamm::TO;
const auto TV = tamm::TV;
#endif

using namespace tammx::tensor_labels;

bool
test_assign(tammx::ExecutionContext &ec,
            double alpha,
            const tammx::IndexLabelVec &cupper_labels,
            const tammx::IndexLabelVec &clower_labels,
            const tammx::IndexLabelVec &aupper_labels,
            const tammx::IndexLabelVec &alower_labels,
            AllocationType at = AllocationType::no_n);

bool
test_assign(tammx::ExecutionContext &ec,
            const tammx::IndexLabelVec &cupper_labels,
            const tammx::IndexLabelVec &clower_labels,
            double alpha,
            const tammx::IndexLabelVec &aupper_labels,
            const tammx::IndexLabelVec &alower_labels,
            AllocationType at = AllocationType::no_n);

#if 0
bool test_assign(tammx::ExecutionContext& ec,
                      double alpha,
                      const std::vector<tamm::IndexName>& cupper_labels,
                      const std::vector<tamm::IndexName>& clower_labels,
                      const std::vector<tamm::IndexName>& aupper_labels,
                      const std::vector<tamm::IndexName>& alower_labels);

bool test_assign_no_n(tammx::ExecutionContext& ec,
                 double alpha,
                 const tammx::IndexLabelVec& cupper_labels,
                 const tammx::IndexLabelVec& clower_labels,
                 const tammx::IndexLabelVec& aupper_labels,
                 const tammx::IndexLabelVec& alower_labels);

bool test_assign_no_n(tammx::ExecutionContext& ec,
                 const tammx::IndexLabelVec& cupper_labels,
                 const tammx::IndexLabelVec& clower_labels,
                 double alpha,
                 const tammx::IndexLabelVec& aupper_labels,
                 const tammx::IndexLabelVec& alower_labels);
#endif

bool test_mult_no_n(tammx::ExecutionContext& ec,
               double alpha,
               const tammx::IndexLabelVec& cupper_labels,
               const tammx::IndexLabelVec& clower_labels,
               const tammx::IndexLabelVec& aupper_labels,
               const tammx::IndexLabelVec& alower_labels,
               const tammx::IndexLabelVec& bupper_labels,
               const tammx::IndexLabelVec& blower_labels);

bool
test_mult_no_n(tammx::ExecutionContext& ec,
               const tammx::IndexLabelVec& cupper_labels,
               const tammx::IndexLabelVec& clower_labels,
               double alpha,
               const tammx::IndexLabelVec& aupper_labels,
               const tammx::IndexLabelVec& alower_labels,
               const tammx::IndexLabelVec& bupper_labels,
               const tammx::IndexLabelVec& blower_labels);


#endif //TEST_TAMM_H 
