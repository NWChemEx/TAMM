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
#ifndef TAMM_TENSOR_PREPROCESS_H_
#define TAMM_TENSOR_PREPROCESS_H_

#include <vector>
#include "tensor/antisymm.h"
#include "tensor/iterGroup.h"
#include "tensor/tensor.h"
#include "tensor/triangular.h"

namespace tamm {

/**
 * Generate triangluar iterator
 * @param[out] trig_itr
 * @param[in] name, group
 */
void genTrigIter(IterGroup<triangular>* trig_itr,
                 const std::vector<IndexName>& name,
                 const std::vector<int>& group);

/**
 * Generate anti-symmetry iterator
 */
void genAntiIter(const std::vector<size_t>& vtab, IterGroup<antisymm>* ext_itr,
                 const Tensor& tC, const Tensor& tA, const Tensor& tB);

} /* namespace tamm */

#endif  // TAMM_TENSOR_PREPROCESS_H_
