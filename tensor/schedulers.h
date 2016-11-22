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
#ifndef TAMM_TENSOR_SCHEDULERS_H_
#define TAMM_TENSOR_SCHEDULERS_H_

#include <map>
#include <string>
#include <vector>
#include "tensor/input.h"
#include "tensor/tensor.h"

namespace tamm {

void schedule_linear(std::map<std::string, tamm::Tensor> *tensors,
                     std::vector<Operation> *ops);
void schedule_linear_lazy(std::map<std::string, tamm::Tensor> *tensors,
                          std::vector<Operation> *ops);
void schedule_levels(std::map<std::string, tamm::Tensor> *tensors,
                     std::vector<Operation> *ops);

} /* namespace tamm */

#endif  // TAMM_TENSOR_SCHEDULERS_H_
