
#if defined(USE_UPCXX)
#include <upcxx/upcxx.hpp>
#else
#include <ga/ga.h>
#endif

#include "tamm/errors.hpp"

namespace tamm {

void tamm_terminate(std::string msg) {
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

  exit(0);
}

} // namespace tamm
