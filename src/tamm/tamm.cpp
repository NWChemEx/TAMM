#include "tamm/tamm.hpp"
#include "ga/macdecls.h"
#include "mpi.h"
#include "ga/ga.h"
#include "pthread.h"
#include <mutex>
#undef I

int in_kernel = 0;
upcxx::team* team_self = NULL;
int id_counter = 0;

static volatile bool finalized = false;
// static pthread_t progress_thread;
static pthread_t abort_thread;
// upcxx::persona *transmit_persona = NULL;
// upcxx::persona_scope *transmit_persona_scope = NULL;
// std::mutex master_mtx;

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

// static void *progress_func(void *) {
//     while (!finalized) {
//         upcxx::persona_scope master_scope(master_mtx,
//                 upcxx::master_persona());
//         upcxx::progress();
//     }
// }

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
  upcxx::init();

  // Must be called with master persona
  team_self = new upcxx::team(upcxx::local_team().split(upcxx::rank_me(),0));

  // upcxx::liberate_master_persona();
  // transmit_persona = new upcxx::persona();
  // transmit_persona_scope = new upcxx::persona_scope(*transmit_persona);

  // int err = pthread_create(&progress_thread, NULL, progress_func, NULL);
  // if (err != 0) {
  //     fprintf(stderr, "Error launching progress thread\n");
  //     abort();
  // }
  // if (upcxx::rank_me() == 0) {
  // int err = pthread_create(&abort_thread, NULL, abort_func, NULL);
  // if (err != 0) {
  //     fprintf(stderr, "Error launching abort thread\n");
  //     abort();
  // }
  // }
}

void finalize() {
  finalized = true;

  // delete transmit_persona_scope;
  // delete transmit_persona;

  // int err = pthread_join(progress_thread, NULL);
  // if (err != 0) {
  //     fprintf(stderr, "Error joining progress thread\n");
  //     abort();
  // }

  // {
  //     // Reacquire master so that we can finalize
  //     upcxx::persona_scope master_scope(master_mtx,
  //             upcxx::master_persona());
  //     upcxx::finalize();
  // }
    upcxx::finalize();
}

void tamm_terminate(std::string msg) {
    std::cerr << msg << " ... terminating program." << std::endl << std::endl;
    abort();
}

} // namespace tamm
