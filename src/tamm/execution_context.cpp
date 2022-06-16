#include "ga/ga.h"
#include <mpi.h>

#include "distribution.hpp"
#include "execution_context.hpp"
#include "labeled_tensor.hpp"
#include "memory_manager.hpp"
#include "proc_group.hpp"
#include "runtime_engine.hpp"

namespace tamm {
ExecutionContext::ExecutionContext(ProcGroup pg, DistributionKind default_dist_kind,
                                   MemoryManagerKind default_memory_manager_kind,
                                   RuntimeEngine*    re):
  pg_{pg},
  distribution_kind_{default_dist_kind},
  memory_manager_kind_{default_memory_manager_kind}, ac_{IndexedAC{nullptr, 0}} {
  if(re == nullptr) { re_.reset(runtime_ptr()); }
  else {
    re_.reset(re, [](auto) {});
  }

#if defined(USE_UPCXX)
  pg_self_ = ProcGroup{team_self};

#if defined(USE_UPCXX_DISTARRAY)
  hint_ = pg.size().value();
#endif

#else
  pg_self_  = ProcGroup{MPI_COMM_SELF, ProcGroup::self_ga_pgroup()};
#endif

  ngpu_ = 0;
  has_gpu_ = false;
  exhw_     = ExecutionHW::CPU;

// step1 : get total number of ranks per MPI
#if defined(USE_UPCXX)
  ranks_pn_ = upcxx::local_team().rank_n();
#else
  ranks_pn_ = GA_Cluster_nprocs(GA_Cluster_proc_nodeid(pg.rank().value()));
#endif
  nnodes_ = pg.size().value() / ranks_pn_;

// step2 : check the number of GPUs per node
#if defined(USE_TALSH)
  int errc = talshDeviceCount(DEV_NVIDIA_GPU, &ngpu_);
  assert(!errc);

  if(ranks_pn_ != ngpu_) {
    if(pg.rank() == 0) {
      std::string msg = "#ranks per node(" + std::to_string(ranks_pn_) + ") > #gpus(" +
                        std::to_string(ngpu_) + ") per node ... terminating program.";
      std::cout << msg << std::endl << std::endl;
    }
    GA_Terminate();
    MPI_Finalize();
    exit(0);
  }
#else
  #if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  tamm::getDeviceCount(&ngpu_);
  has_gpu_ = true;
  exhw_ = ExecutionHW::GPU;
  #endif
#endif

// step3 :bind the GPUs to MPI-rank as appropriate (only required for TALSH)
#if defined(USE_TALSH)
#if defined(USE_UPCXX)
  talsh_dev_id_ = upcxx::rank_me() % ngpu_;
  has_gpu_ = true;
  exhw_ = ExecutionHW::GPU;
#else
  talsh_dev_id_ = ((pg.rank().value() % ranks_pn_) % ngpu_);
  if(ngpu_ == 1) talsh_dev_id_ = 0;
  if((pg.rank().value() % ranks_pn_) < ngpu_) {
    has_gpu_ = true;
    exhw_    = ExecutionHW::GPU;
  }
#endif

  auto& pool = tamm::GPUStreamPool::getInstance();
  pool.set_device(talsh_dev_id_);
#endif // #if defined(USE_TALSH)
}

ExecutionContext::ExecutionContext(ProcGroup pg, Distribution* default_distribution,
                                   MemoryManager* default_memory_manager, RuntimeEngine* re):
  ExecutionContext{
    pg, default_distribution != nullptr ? default_distribution->kind() : DistributionKind::invalid,
    default_memory_manager != nullptr ? default_memory_manager->kind() : MemoryManagerKind::invalid,
    re} {}

void ExecutionContext::set_distribution(Distribution* distribution) {
  if(distribution) { distribution_kind_ = distribution->kind(); }
  else { distribution_kind_ = DistributionKind::invalid; }
}

void ExecutionContext::set_re(RuntimeEngine* re) { re_.reset(re); }

} // namespace tamm
