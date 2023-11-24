#include "tamm/tamm.hpp"

#if defined(USE_UPCXX)
upcxx::team* team_self = NULL;
#endif

namespace tamm {

void initialize(int argc, char* argv[]) {
#if defined(USE_UPCXX)
  upcxx::init();
  // Must be called with master persona
  team_self = new upcxx::team(upcxx::local_team().split(upcxx::rank_me(), 0));
  // if (upcxx::rank_me() == 0) {
  // int err = pthread_create(&abort_thread, NULL, abort_func, NULL);
  // }
#else
  int flag;
  MPI_Initialized(&flag);
  if(!flag) { MPI_Init(&argc, &argv); }
  if(!GA_Initialized()) {
    GA_Initialize();
    (void) ProcGroup::self_ga_pgroup(true);
  }
#endif
}

// MPI_THREAD_MULTIPLE
void initialize_tm(int argc, char* argv[]) {
#if defined(USE_UPCXX)
  upcxx::init();
  team_self = new upcxx::team(upcxx::local_team().split(upcxx::rank_me(), 0));
#else
  int flag;
  MPI_Initialized(&flag);
  if(!flag) {
    int prov;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &prov);
  }
  if(!GA_Initialized()) {
    GA_Initialize();
    (void) ProcGroup::self_ga_pgroup(true);
  }
#endif
}

void finalize() {
#if defined(USE_UPCXX)
  upcxx::finalize();
#else
  if(GA_Initialized()) { GA_Terminate(); }
  int flag;
  MPI_Initialized(&flag);
  if(flag) { MPI_Finalize(); }
#endif
}

void tamm_terminate(std::string msg) {
  if(ProcGroup::world_rank() == 0)
    std::cout << msg << " ... terminating program." << std::endl << std::endl;
  tamm::finalize();
  exit(0);
}

} // namespace tamm
