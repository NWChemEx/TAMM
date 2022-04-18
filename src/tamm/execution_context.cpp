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
  pg_self_ = ProcGroup{team_self};
#ifdef UPCXX_DISTARRAY
  hint_ = pg.size().value();
#endif
  ngpu_ = 0;
  has_gpu_ = false;
  exhw_     = ExecutionHW::CPU;

#ifdef USE_UPCXX
  nnodes_ = -1;
  if (getenv("SLURM_JOB_NUM_NODES")) {
      nnodes_ = atoi(getenv("SLURM_JOB_NUM_NODES"));
  } else if (getenv("LSB_DJOB_NUMPROC")) {
      nnodes_ = (atoi(getenv("LSB_DJOB_NUMPROC")) - 1) / 42;
  } else {
      throw std::runtime_error("unable to determine # nodes");
  }
  ranks_pn_ = upcxx::rank_n() / nnodes_;
#else
  ranks_pn_ = GA_Cluster_nprocs(GA_Cluster_proc_nodeid(pg.rank().value()));
  // nnodes_ = {GA_Cluster_nnodes()};
  nnodes_ = pg.size().value() / ranks_pn_;
#endif

#ifdef USE_TALSH
  int errc = talshDeviceCount(DEV_NVIDIA_GPU, &ngpu_);
  assert(!errc);

#if 0
  /*
   * Max Grossman (July 8 2020): This logic causes problems when running
   * multiple resource sets per node on Summit, as the # of GPUs in a node
   * differs from the # of GPUs visible to each rank. It can lead to some
   * ranks not thinking they have a GPU available, and reverting to
   * running on the CPU.
   */
  dev_id_ = ((pg.rank().value() % ranks_pn_) % ngpu_);
  if (ngpu_ == 1) dev_id_ = 0;
  if ((pg.rank().value() % ranks_pn_) < ngpu_) {
    has_gpu_ = true;
    exhw_ = ExecutionHW::GPU;
  }
  if(ranks_pn_ > ngpu_) {
    if(pg.rank()==0) {
      std::string msg = "#ranks per node(" + std::to_string(ranks_pn_) + 
        ") > #gpus(" + std::to_string(ngpu_) + ") per node ... terminating program.";
      std::cout << msg << std::endl << std::endl;
    }
    GA_Terminate();
    MPI_Finalize();
    exit(0);
  }
#else
  dev_id_ = upcxx::rank_me() % ngpu_;
  has_gpu_ = true;
  exhw_ = ExecutionHW::GPU;
#endif
#endif // USE_TALSH

  // GPUStreamPool as singleton object
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  auto& pool = tamm::GPUStreamPool::getInstance();
  pool.set_device(dev_id_);
#endif
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
