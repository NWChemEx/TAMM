#ifndef __tamm_stats_h__
#define __tamm_stats_h__

#include <cassert>
#include <mpi.h>
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

  ~Profiler() {
    assert(!in_phase);
  }

  int count() {
    return cnt;
  }
    
  double time() {
    return ttime;
  }

private:
  int cnt;
  double ttime;
  double stime, etime;
  bool in_phase;
};

extern Profiler iterTimer, assignTimer, multTimer, getTimer, addTimer, dgemmTimer;
void printStats();
}

#endif /* __tamm_stats_h__ */

