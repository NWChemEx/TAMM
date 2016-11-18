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
#ifndef TAMM_TENSOR_STATS_H_
#define TAMM_TENSOR_STATS_H_

#include <mpi.h>
#include <cassert>
#include <string>

#define TIMER() MPI_Wtime()

namespace tamm {

class Profiler {
 public:
  Profiler() {
    cnt = 0;
    ttime = 0;
    in_phase = false;
  }

  void start() {
    stime = TIMER();
    assert(!in_phase);
    in_phase = true;
  }

  void stop() {
    etime = TIMER();
    assert(in_phase);
    cnt += 1;
    ttime += etime - stime;
    in_phase = false;
  }

  ~Profiler() { assert(!in_phase); }

  int count() { return cnt; }

  double time() { return ttime; }

 private:
  int cnt;
  double ttime;
  double stime, etime;
  bool in_phase;
};

extern Profiler iterTimer, assignTimer, multTimer, getTimer, addTimer,
    dgemmTimer;
void printStats();
}  // namespace tamm

#endif  // TAMM_TENSOR_STATS_H_
