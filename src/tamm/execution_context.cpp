#include <ga.h>
#include <memory>
#include <mpi.h>

#include "labeled_tensor.hpp"
#include "distribution.hpp"
#include "execution_context.hpp"
#include "proc_group.hpp"
#include "runtime_engine.hpp"
#include "memory_manager.hpp"

namespace tamm {
ExecutionContext::ExecutionContext(ProcGroup pg, DistributionKind default_dist_kind,
                                   MemoryManagerKind default_memory_manager_kind,
                                   RuntimeEngine* re)
    : pg_{pg},
      distribution_kind_{default_dist_kind},
      memory_manager_kind_{default_memory_manager_kind},
      ac_{IndexedAC{nullptr, 0}} {
  if (re == nullptr) {
    re_.reset(runtime_ptr());
  } else {
    re_.reset(re, [](auto) {});
  }
  pg_self_ = ProcGroup{MPI_COMM_SELF, ProcGroup::self_ga_pgroup()};
  ngpu_ = 0;
  has_gpu_ = false;
  ranks_pn_ = GA_Cluster_nprocs(GA_Cluster_proc_nodeid(pg.rank().value()));
  // nnodes_ = {GA_Cluster_nnodes()};
  nnodes_ = pg.size().value() / ranks_pn_;

#ifdef USE_TALSH
  int errc = talshDeviceCount(DEV_NVIDIA_GPU, &ngpu_);
  assert(!errc);
  dev_id_ = ((pg.rank().value() % ranks_pn_) % ngpu_);
  if (ngpu_ == 1) dev_id_ = 0;
  if ((pg.rank().value() % ranks_pn_) < ngpu_) has_gpu_ = true;
#endif
  // memory_manager_local_ = MemoryManagerLocal::create_coll(pg_self_);
}

ExecutionContext::ExecutionContext(ProcGroup pg,
                                   Distribution* default_distribution,
                                   MemoryManager* default_memory_manager,
                                   RuntimeEngine* re)
    : ExecutionContext{
          pg,
          default_distribution != nullptr ? default_distribution->kind()
                                          : DistributionKind::invalid,
          default_memory_manager != nullptr ? default_memory_manager->kind()
                                            : MemoryManagerKind::invalid,
          re} {}

void ExecutionContext::set_distribution(Distribution* distribution) {
    if(distribution) {
        distribution_kind_ = distribution->kind();
    } else {
        distribution_kind_ = DistributionKind::invalid;
    }
}
} // namespace tamm
