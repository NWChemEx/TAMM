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
#ifndef TAMM_TENSOR_TENSORS_AND_OPS_H_
#define TAMM_TENSOR_TENSORS_AND_OPS_H_

#include <map>
#include <string>
#include <vector>
#include "tensor/equations.h"
#include "tensor/input.h"
#include "tensor/tensor.h"

namespace tamm {

void tensors_and_ops(Equations *eqs,
                     std::map<std::string, tamm::Tensor> *tensors,
                     std::vector<Operation> *ops);
}

#endif  // TAMM_TENSOR_TENSORS_AND_OPS_H_
