#include "stats.h"
#include <iostream>

using namespace std;
using namespace tamm;

namespace tamm {
  Profiler iterTimer, assignTimer, multTimer, getTimer, addTimer, dgemmTimer;

  void printStats() {
    cout<<"Iter time ="<<iterTimer.time()<<endl;
    cout<<"Assign time ="<<assignTimer.time()<<endl;
    cout<<"Mult time ="<<multTimer.time()<<endl;
    cout<<"get time ="<<getTimer.time()<<endl;
    cout<<"add time ="<<addTimer.time()<<endl;
    cout<<"dgemm time ="<<dgemmTimer.time()<<endl;
  }
}
