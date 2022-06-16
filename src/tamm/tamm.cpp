#include "tamm/tamm.hpp"
#include "tamm/tamm_impl.hpp"
#include "ga/macdecls.h"
#include "mpi.h"
#include "ga/ga.h"

#if defined(USE_UPCXX)
upcxx::team* team_self = NULL;
#endif

static volatile bool finalized = false;
static pthread_t progress_thread;
static pthread_t abort_thread;

static void *abort_func(void*) {
    auto start_time = std::chrono::high_resolution_clock::now();

    while (!finalized) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>((end_time - start_time)).count();
        if (elapsed_time > 40 * 60) {
            fprintf(stderr, "Aborting due to time out!\n");
            abort();
        }
    }
}

namespace tamm {

void initialize(int argc, char *argv[]) {
#if defined(USE_UPCXX)
  upcxx::init();

  // Must be called with master persona
  team_self = new upcxx::team(upcxx::local_team().split(upcxx::rank_me(),0));
  // if (upcxx::rank_me() == 0) {
  // int err = pthread_create(&abort_thread, NULL, abort_func, NULL);
  // if (err != 0) {
  //     fprintf(stderr, "Error launching abort thread\n");
  //     abort();
  // }
  // }
#else
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
#endif

#if !defined(USE_TALSH)  && (defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
  auto& tammInst = tamm::TAMM::getInstance();
#endif
}

void finalize() {
  finalized = true;

#if defined(USE_UPCXX)
  upcxx::finalize();
#else
  if (GA_Initialized()) {
    GA_Terminate();
  }
  int flag;
  MPI_Initialized(&flag);
  if (flag) {
    MPI_Finalize();
  }
#endif
}

void tamm_terminate(std::string msg) {
  if(ProcGroup::world_rank()==0)
    std::cout << msg << " ... terminating program." << std::endl << std::endl;
  tamm::finalize();
  exit(0);
}

} // namespace tamm
