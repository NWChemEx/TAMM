#ifndef TAMMX_EXECUTION_CONTEXT_H_
#define TAMMX_EXECUTION_CONTEXT_H_

#include "tammx/types.h"
#include "tammx/proc_group.h"
#include "tammx/memory_manager.h"
#include "tammx/distribution.h"
#include "tammx/ops.h"
#include "tammx/scheduler.h"

namespace tammx {
class ExecutionContext {
 public:
  ExecutionContext(ProcGroup pg, Distribution* default_distribution,
                   MemoryManager* default_memory_manager,
                   Irrep default_irrep, bool default_spin_restricted)
      : pg_{pg},
        default_distribution_{default_distribution},
        default_memory_manager_{default_memory_manager},
        default_irrep_{default_irrep},
        default_spin_restricted_{default_spin_restricted} {
          pg_self_ = ProcGroup{MPI_COMM_SELF};
        }

  Scheduler scheduler() {
    return Scheduler{pg_,
          default_distribution_,
          default_memory_manager_,
          default_irrep_,
          default_spin_restricted_};
  }

  void allocate() {
    //no-op
  }

  template<typename T, typename ...Args>
  void allocate(Tensor<T>& tensor, Args& ... tensor_list) {
    tensor.alloc(default_distribution_, default_memory_manager_);
    allocate(tensor_list...);
  }


  static void deallocate() {
    //no-op
  }

  template<typename T, typename ...Args>
  static void deallocate(Tensor<T>& tensor, Args& ... tensor_list) {
    tensor.dealloc();
    deallocate(tensor_list...);
  }

  ProcGroup pg() const {
    return pg_;
  }

  Distribution* distribution() const {
    return default_distribution_;
  }

  MemoryManager* memory_manager() const {
    return default_memory_manager_;
  }

  Irrep irrep() const {
    return  default_irrep_;
  }

  bool is_spin_restricted() const {
    return default_spin_restricted_;
  }
  
 private:
  ProcGroup pg_;
  ProcGroup pg_self_;
  Distribution* default_distribution_;
  MemoryManager* default_memory_manager_;
  Irrep default_irrep_;
  bool default_spin_restricted_;
}; // class ExecutionContext

} // namespace tammx

#endif // TAMMX_EXECUTION_CONTEXT_H_

