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
#ifndef TAMM_TENSOR_DUMMY_H_
#define TAMM_TENSOR_DUMMY_H_

#include <typesf2c.h>
#include "tensor/copyIter.h"
#include "tensor/func.h"

namespace tamm {

/* this will be replaced soon */
class Dummy {
 private:
  static CopyIter type1_; /*< perm for (2,1): (0,1,2)(2,0,1)(0,2,1) */
  static CopyIter type2_; /*< perm for (1,2): (0,1,2)(1,0,2)(1,2,0) */
  static CopyIter type3_; /*< swap permutation (0,1)(1,0) */
  static CopyIter type4_; /*< no permutation (1) */
 public:
  static void construct();
  static const CopyIter& type1() { return type1_; }
  static const CopyIter& type2() { return type2_; }
  static const CopyIter& type3() { return type3_; }
  static const CopyIter& type4() { return type4_; }
};

}  // namespace tamm

#endif  // TAMM_TENSOR_DUMMY_H_
