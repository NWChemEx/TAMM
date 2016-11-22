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
#include "tensor/dummy.h"
#include <vector>

namespace tamm {

CopyIter Dummy::type1_;
CopyIter Dummy::type2_;
CopyIter Dummy::type3_;
CopyIter Dummy::type4_;

void Dummy::construct() {
  std::vector<std::vector<size_t> > vi1;
  vi1.resize(3);
  vi1[0] = newVec<size_t>(3, 0, 1, 2);
  vi1[1] = newVec<size_t>(3, 2, 0, 1);
  vi1[2] = newVec<size_t>(3, 0, 2, 1);
  std::vector<int> s1 = newVec<int>(3, 1, 1, -1);
  CopyIter t1 = CopyIter(vi1, s1);
  type1_ = t1;
  std::vector<std::vector<size_t> > vi2;
  vi2.resize(3);

  vi2[0] = newVec<size_t>(3, 0, 1, 2);
  vi2[1] = newVec<size_t>(3, 1, 0, 2);
  vi2[2] = newVec<size_t>(3, 1, 2, 0);
  std::vector<int> s2 = newVec<int>(3, 1, -1, 1);
  CopyIter t2 = CopyIter(vi2, s2);
  type2_ = t2;

  std::vector<std::vector<size_t> > vi3;
  vi3.resize(2);
  vi3[0] = newVec<size_t>(2, 0, 1);
  vi3[1] = newVec<size_t>(2, 1, 0);
  std::vector<int> s3 = newVec<int>(2, 1, -1);
  CopyIter t3 = CopyIter(vi3, s3);
  type3_ = t3;

  std::vector<std::vector<size_t> > vi4;
  vi4.resize(1);
  vi4[0] = newVec<size_t>(1, 0);
  std::vector<int> s4 = newVec<int>(1, 1);
  CopyIter t4 = CopyIter(vi4, s4);
  type4_ = t4;
}
}  // namespace tamm
