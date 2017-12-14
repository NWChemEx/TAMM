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
#include "tensor/stats.h"
#include <iostream>

using std::cout;
using std::endl;

namespace tamm {
Profiler iterTimer, assignTimer, multTimer, getTimer, addTimer, dgemmTimer;

void printStats() {
  cout << "Iter time =" << iterTimer.time() << endl;
  cout << "Assign time =" << assignTimer.time() << endl;
  cout << "Mult time =" << multTimer.time() << endl;
  cout << "get time =" << getTimer.time() << endl;
  cout << "add time =" << addTimer.time() << endl;
  cout << "dgemm time =" << dgemmTimer.time() << endl;
}
}  // namespace tamm
