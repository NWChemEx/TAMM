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

namespace {
tammx::ExecutionContext* g_ec;
}

class TestEnvironment : public testing::Environment {
 public:
  explicit TestEnvironment(tammx::ExecutionContext* ec) {
    g_ec = ec;
  }
};

using namespace tammx::tensor_labels;

bool test_assign_no_n(tammx::ExecutionContext& ec,
                 double alpha,
                 const tammx::TensorLabel& cupper_labels,
                 const tammx::TensorLabel& clower_labels,
                 const tammx::TensorLabel& aupper_labels,
                 const tammx::TensorLabel& alower_labels);

bool test_assign_no_n(tammx::ExecutionContext& ec,
                 const tammx::TensorLabel& cupper_labels,
                 const tammx::TensorLabel& clower_labels,
                 double alpha,
                 const tammx::TensorLabel& aupper_labels,
                 const tammx::TensorLabel& alower_labels);

bool test_mult_no_n(tammx::ExecutionContext& ec,
               double alpha,
               const tammx::TensorLabel& cupper_labels,
               const tammx::TensorLabel& clower_labels,
               const tammx::TensorLabel& aupper_labels,
               const tammx::TensorLabel& alower_labels,
               const tammx::TensorLabel& bupper_labels,
               const tammx::TensorLabel& blower_labels);

bool
test_mult_no_n(tammx::ExecutionContext& ec,
               const tammx::TensorLabel& cupper_labels,
               const tammx::TensorLabel& clower_labels,
               double alpha,
               const tammx::TensorLabel& aupper_labels,
               const tammx::TensorLabel& alower_labels,
               const tammx::TensorLabel& bupper_labels,
               const tammx::TensorLabel& blower_labels);

