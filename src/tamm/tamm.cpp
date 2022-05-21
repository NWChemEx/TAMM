#include "tamm/tamm.hpp"
#include "ga/macdecls.h"
#include "mpi.h"
#include "ga/ga.h"

namespace tamm {

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
  // if (!MA_initialized()) {
  //   MA_init(MT_DBL, 8000000, 20000000);
  // }
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

void tamm_terminate(std::string msg) {
  if(GA_Nodeid() == 0) std::cout << msg << " ... terminating program." << std::endl << std::endl;
  tamm::finalize();
  exit(0);
}

} // namespace tamm
