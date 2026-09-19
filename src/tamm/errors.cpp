
#if defined(USE_UPCXX)
#include <upcxx/upcxx.hpp>
#else
#include <ga/ga.h>
#endif

#include "tamm/errors.hpp"

namespace tamm {

namespace {
bool g_terminating = false;
}

bool tamm_terminating() { return g_terminating; }

void tamm_terminate(std::string msg) {
  // Set before exit() so destructors running from it can see the abort.
  g_terminating = true;

  int world_rank_ = 0;
#if defined(USE_UPCXX)
  world_rank_ = upcxx::rank_me();
#else
  world_rank_ = GA_Nodeid();
#endif // USE_UPCXX

  if(world_rank_ == 0) std::cout << msg << " ... terminating program." << std::endl << std::endl;

#if defined(USE_UPCXX)
  upcxx::finalize();
#else
  if(GA_Initialized()) { GA_Terminate(); }
  int flag;
  MPI_Initialized(&flag);
  if(flag) { MPI_Finalize(); }
#endif

  exit(1);
}

} // namespace tamm
