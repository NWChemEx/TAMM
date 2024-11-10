#include "tamm/tamm.hpp"

#if defined(USE_UPCXX)
upcxx::team* team_self = NULL;
#endif

namespace tamm {

void initialize(int argc, char* argv[], bool is_mpi_tm) {
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
  if(!flag) {
    if(!is_mpi_tm) MPI_Init(&argc, &argv);
    else {
      int prov;
      MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &prov);
    }
  }
  if(!GA_Initialized()) {
    GA_Initialize();
    (void) ProcGroup::self_ga_pgroup(true);
  }
#endif

  // Round-robin assign of GPUs with MPI-ranks for multi-GPUs
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  int ngpu_{0};
  tamm::getDeviceCount(&ngpu_);
  if(ngpu_ == 0) { tamm_terminate("ERROR: GPU devices not available!"); }

  int dev_id_{0};
#if defined(USE_UPCXX)
  int ranks_pn_ = upcxx::local_team().rank_n();
  dev_id_       = ((upcxx::rank_me() % ranks_pn_) % ngpu_);
#else
  int ranks_pn_ = GA_Cluster_nprocs(GA_Cluster_nodeid());
  dev_id_       = ((GA_Nodeid() % ranks_pn_) % ngpu_);
#endif

  // Set the active devices such that the singleton pools like
  // GPU-streampool, GPU-memorypool uses the appropriate dev-id
  tamm::gpuSetDevice(dev_id_);
  /* auto& stream_pool  = */ tamm::GPUStreamPool::getInstance();
  /* auto& rmm_mem_pool = */ tamm::RMMMemoryManager::getInstance();

#endif // USE_CUDA, USE_HIP, USE_DPCPP
}

void finalize(bool tamm_mpi_finalize) {
#if defined(USE_UPCXX)
  upcxx::finalize();
#else
  if(GA_Initialized()) { GA_Terminate(); }
  if(tamm_mpi_finalize) {
    int flag;
    MPI_Initialized(&flag);
    if(flag) { MPI_Finalize(); }
  }
#endif
}

} // namespace tamm
