#include "tamm/tamm.hpp"
#include "macdecls.h"
#include "mpi.h"
#include "ga.h"
#undef I
namespace tamm {

int mult_counter = 0;
double multOpTime = 0;
double addOpTime = 0;
double setOpTime = 0;
double allocOpTime = 0;
double deallocOpTime = 0;
double tgetTime = 0;
double taddTime = 0;
double twaitTime = 0;
double tgemmTime = 0;
double tbarrierTime = 0;

double multOpGetTime = 0;
double multOpWaitTime = 0;
double multOpAddTime = 0;
double multOpDgemmTime = 0;
double memTime1 = 0;
double memTime2 = 0;
double memTime3 = 0;
double memTime4 = 0;
double memTime5 = 0;
double memTime6 = 0;
double memTime7 = 0;
double memTime8 = 0;
double memTime9 = 0;


void initialize(int argc, char *argv[]) {
  int flag;
  MPI_Initialized(&flag);
  if (!flag) {
    MPI_Init(&argc, &argv);
  }
  if (!GA_Initialized()) {
    GA_Initialize();
    (void)ProcGroup::self_ga_pgroup(true);
  }
  if (!MA_initialized()) {
    MA_init(MT_DBL, 8000000, 20000000);
  }
}

void finalize() {
  if (GA_Initialized()) {
    GA_Terminate();
  }
  int flag;
  MPI_Initialized(&flag);
  if (flag) {
    MPI_Finalize();
  }
}


} // namespace tamm
