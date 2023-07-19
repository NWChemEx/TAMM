#pragma once

#include "tamm/proc_group.hpp"
//#include "tamm/tensor_impl.hpp"
#include "tamm/atomic_counter.hpp"
#include "tamm/memory_manager_ga.hpp"
#include "tamm/memory_manager_local.hpp"
//#include "tamm/distribution.hpp"
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#include "tamm/gpu_streams.hpp"
#endif
#include "tamm/types.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>

#if defined(USE_UPCXX)
extern upcxx::team* team_self;
#endif

namespace tamm {

struct IndexedAC {
  AtomicCounter* ac_;
  size_t         idx_;

  IndexedAC(AtomicCounter* ac, size_t idx): ac_{ac}, idx_{idx} {}
};
/**
 * @todo Create a proper forward declarations file.
 *
 */

class Distribution;
class Scheduler;
template<typename T>
class Tensor;
class RuntimeEngine;

class Distribution_NW;
class Distribution_Dense;
class Distribution_SimpleRoundRobin;

/**
 * @brief Wrapper class to hold information during execution.
 *
 * This class holds the choice of default memory manager, distribution, irrep,
 * etc.
 *
 * @todo Should spin_restricted be wrapper by this class? Or should it always
 * default to false?
 */
class RuntimeEngine;
class ExecutionContext {
public:
  ExecutionContext(): ac_{IndexedAC{nullptr, 0}} {
#if defined(USE_UPCXX)
    pg_self_ = ProcGroup{team_self};
#else
    pg_self_ = ProcGroup{MPI_COMM_SELF, ProcGroup::self_ga_pgroup()};
#endif
  };

  ExecutionContext(const ExecutionContext&)            = default;
  ExecutionContext& operator=(const ExecutionContext&) = default;

  ExecutionContext(ExecutionContext&&)            = default;
  ExecutionContext& operator=(ExecutionContext&&) = default;

  ExecutionContext(ProcGroup pg, DistributionKind default_distribution_kind,
                   MemoryManagerKind default_memory_manager_kind, RuntimeEngine* re = nullptr);

  /** @todo use shared pointers for solving GitHub issue #43*/
  ExecutionContext(ProcGroup pg, Distribution* default_distribution,
                   MemoryManager* default_memory_manager, RuntimeEngine* re = nullptr);
  // ExecutionContext(ProcGroup pg, Distribution* default_distribution,
  //                  MemoryManager* default_memory_manager, RuntimeEngine* re =nullptr) :
  //   pg_{pg},
  //   distribution_kind_{DistributionKind::invalid},
  // //   default_distribution_{nullptr},
  // //   default_distribution_{default_distribution},
  //   default_memory_manager_{default_memory_manager},
  //   ac_{IndexedAC{nullptr, 0}} {
  //     if (re == nullptr) {
  //       re_.reset(runtime_ptr());
  //     } else {
  //       re_.reset(re, [](auto){});
  //     }
  //      if(default_distribution != nullptr) {
  //        distribution_kind_ = default_distribution->kind();
  //     // //   default_distribution_.reset(default_distribution->clone(nullptr, Proc{1}));
  //     }
  //     pg_self_ = ProcGroup{MPI_COMM_SELF, ProcGroup::self_ga_pgroup()};
  //     has_gpu_ = false;
  //     ranks_pn_ = GA_Cluster_nprocs(GA_Cluster_proc_nodeid(pg.rank().value()));
  //     //nnodes_ = {GA_Cluster_nnodes()};
  //     nnodes_ = pg.size().value() / ranks_pn_;

  //     // memory_manager_local_ = MemoryManagerLocal::create_coll(pg_self_);
  // }
  RuntimeEngine* runtime_ptr();

  ~ExecutionContext() {
    // MemoryManagerLocal::destroy_coll(memory_manager_local_);
  }

  void allocate(const Distribution& distribution) {
    // no-op
  }

  /**
   * Allocate a list of tensor with default parameters (irrep, etc.)
   * @tparam T Type of element in tensor
   * @tparam Args Type of list of tensors to be allocated
   * @param tensor First tensor in the list
   * @param tensor_list Remaining tensors in the list
   */
  // template<typename T, typename... Args>
  // void allocate(const Distribution& distribution, Tensor<T>& tensor, Args&... tensor_list) {
  //     tensor.alloc(&distribution, default_memory_manager_);
  //     allocate(distribution, tensor_list...);
  // }

  template<typename T, typename... Args>
  void allocate(const Distribution& distribution, Tensor<T>& tensor, Args&... tensor_list) {
    tensor.alloc(&distribution, memory_manager_factory(memory_manager_kind_));
    allocate(distribution, tensor_list...);
  }

  template<typename T, typename... Args>
  void allocate(Tensor<T>& tensor, Args&... tensor_list) {
    std::unique_ptr<Distribution> distribution{distribution_factory(distribution_kind_)};
    allocate(*distribution.get(), tensor_list...);
  }

  void allocate_local(const Distribution&       distribution,
                      const MemoryManagerLocal& memory_manager_local) {
    // no-op
  }

  /**
   * Allocate a list of tensor with default parameters (irrep, etc.) using
   * local memory manager
   * @tparam T Type of element in tensor
   * @tparam Args Type of list of tensors to be allocated
   * @param tensor First tensor in the list
   * @param tensor_list Remaining tensors in the list
   */
  template<typename T, typename... Args>
  void allocate_local(const Distribution&       distribution,
                      const MemoryManagerLocal& memory_manager_local, Tensor<T>& tensor,
                      Args&... tensor_list) {
    tensor.alloc(distribution, memory_manager_local);
    allocate_local(tensor_list...);
  }

  template<typename T, typename... Args>
  void allocate_local(Tensor<T>& tensor, Args&... tensor_list) {
    std::unique_ptr<Distribution> distribution{distribution_factory(distribution_kind_)};
    MemoryManagerLocal            memory_manager_local{pg_self_};
    allocate_local(*distribution.get(), memory_manager_local, tensor_list...);
  }

  void deallocate() {
    // no-op
  }

  /**
   * Deallocate a list of tensors
   * @tparam T Type of element in tensor
   * @tparam Args Type of list of tensors to be allocated
   * @param tensor First tensor in the list
   * @param tensor_list Remaining tensors in the list
   */
  template<typename T, typename... Args>
  static void deallocate(Tensor<T>& tensor, Args&... tensor_list) {
    tensor.deallocate();
    deallocate(tensor_list...);
  }

  /**
   * Process group for this execution context
   * @return Underlying process group
   */
  ProcGroup pg() const { return pg_; }

  /**
   * @brief Set ProcGroup object for ExecutionContext
   *
   * @param [in] pg input ProcGroup object
   */
  void set_pg(const ProcGroup& pg) {
    pg_ = pg;
#if defined(USE_UPCXX_DISTARRAY)
    hint_ = pg.size().value();
#endif
  }

  /**
   * Get the default distribution
   * @return Default distribution
   */
  template<typename... Args>
  Distribution* distribution(Args&&... args) const {
    // return default_distribution_.get();
    return distribution_factory(distribution_kind_, std::forward<Args>(args)...)
      .release(); //@bug leak
  }

  Distribution* get_default_distribution() {
    // return default_distribution_.get();
    return distribution_factory(distribution_kind_).release(); //@bug leak
  }

  // DistributionKind::Kind distribution_kind() const { return distribution_kind_; }

  /**
   * @brief Set the default Distribution for ExecutionContext
   *
   * @todo: change raw pointer to smart pointers?
   *
   * @param [in] distribution pointer to Distribution object
   */
  void set_distribution(Distribution* distribution);
  // void set_distribution(Distribution* distribution) {
  //     //default_distribution_.reset(distribution->clone(nullptr, Proc{1}));
  //     if(distribution) {
  //         distribution_kind_ = distribution->kind();
  //     } else {
  //         distribution_kind_ = DistributionKind::invalid;
  //     }
  // }

  void set_distribution_kind(DistributionKind distribution_kind) {
    distribution_kind_ = distribution_kind;
  }

  /**
   * Get the default memory manager
   * @return Default memory manager
   */
  // MemoryManager* memory_manager() const { return default_memory_manager_; }
  template<typename... Args>
  MemoryManager* memory_manager(Args&&... args) const {
    return memory_manager_factory(memory_manager_kind_, std::forward<Args>(args)...).release();
  }

#if defined(USE_UPCXX_DISTARRAY)
  /**
   * @brief Set cache size for MemoryRegionGAs created by MemoryManagerGAs of this ExecutionContext
   *
   * @param [in] hint input Size of cache for MemoryRegionsGAs
   */
  void set_memory_manager_cache(const upcxx::intrank_t hint = 0) {
    hint_ = (hint ? hint : pg_.size().value());
  }
  /**
   * Cache size hint for this execution context
   * @return Cache size used by MemoryRegionGAs within this ExecutionContext
   */
  upcxx::intrank_t hint() const { return hint_; }
#endif

  /**
   * @brief Set the default memory manager for ExecutionContext
   *
   * @todo: change raw pointer to smart pointers?
   *
   * @param [in] memory_manager pointer to MemoryManager object
   */
  // void set_memory_manager(MemoryManager* memory_manager) {
  //     default_memory_manager_ = memory_manager;
  // }

  void set_memory_manager_kind(MemoryManagerKind memory_manager_kind) {
    memory_manager_kind_ = memory_manager_kind;
  }

  RuntimeEngine* re() const { return re_.get(); }

  void set_re(RuntimeEngine* re);
  // {
  //     re_.reset(re);
  // }

  /**
   * @brief Flush communication in this execution context, synchronize, and
   * delete any tensors allocated in this execution context that have gone
   * out of scope.
   *
   * @bug @fixme @todo Actually perform a communication/RMA fence
   *
   */
  void flush_and_sync() {
    pg_.barrier();
    std::sort(mem_regs_to_dealloc_.begin(), mem_regs_to_dealloc_.end());
    std::sort(unregistered_mem_regs_.begin(), unregistered_mem_regs_.end());
    std::vector<MemoryRegion*> result;
    std::set_difference(mem_regs_to_dealloc_.begin(), mem_regs_to_dealloc_.end(),
                        unregistered_mem_regs_.begin(), unregistered_mem_regs_.end(),
                        std::inserter(result, result.begin()));
    mem_regs_to_dealloc_.clear();
    unregistered_mem_regs_.clear();
    for(auto mem_reg: result) {
      EXPECTS(mem_reg->allocation_status() == AllocationStatus::created ||
              mem_reg->allocation_status() == AllocationStatus::orphaned);
      if(mem_reg->allocation_status() == AllocationStatus::orphaned) {
        mem_reg->dealloc_coll();
        delete mem_reg;
      }
      else { mem_regs_to_dealloc_.push_back(mem_reg); }
    }
  }

  void register_for_dealloc(MemoryRegion* mem_reg) { mem_regs_to_dealloc_.push_back(mem_reg); }

  void unregister_for_dealloc(MemoryRegion* mem_reg) { unregistered_mem_regs_.push_back(mem_reg); }

  IndexedAC ac() const { return ac_; }

  void set_ac(IndexedAC ac) { ac_ = ac; }

  bool has_gpu() const { return has_gpu_; }

  ExecutionHW exhw() const { return exhw_; }

  int nnodes() const { return nnodes_; }
  int ppn() const { return ranks_pn_; }

  struct meminfo {
    size_t gpu_mem_per_device; // single gpu mem per rank (GiB)
    size_t cpu_mem_per_node;   // cpu mem on single node (GiB)
    size_t total_cpu_mem;      // total cpu mem across all nodes (GiB)
  };

  meminfo mem_info() const { return minfo_; }

  void print_mem_info() {
    if(pg_.rank() != 0) return;
    std::cout << "Memory information" << std::endl;
    std::cout << "{" << std::endl;
    std::cout << " CPU memory per node (GiB): " << minfo_.cpu_mem_per_node << std::endl;
    std::cout << " Total CPU memory (GiB): " << minfo_.total_cpu_mem << std::endl;
    if(has_gpu_) {
      std::cout << " GPU memory per device (GiB): " << minfo_.gpu_mem_per_device << std::endl;
    }
    std::cout << "}" << std::endl;
  }

  bool print() const { return (pg_.rank() == 0); }

  std::stringstream& get_profile_data() { return profile_data_; }

  template<typename... Args>
  std::unique_ptr<Distribution> distribution_factory(DistributionKind dkind, Args&&... args) const {
    switch(dkind) {
      case DistributionKind::invalid: NOT_ALLOWED(); return nullptr;
      case DistributionKind::dense:
        return std::make_unique<Distribution_Dense>(std::forward<Args>(args)...);
        break;
      case DistributionKind::nw:
        return std::make_unique<Distribution_NW>(std::forward<Args>(args)...);
        break;
      case DistributionKind::simple_round_robin:
        return std::make_unique<Distribution_SimpleRoundRobin>(std::forward<Args>(args)...);
        break;
    }
    UNREACHABLE();
    return nullptr;
  }

  template<typename... Args>
  std::unique_ptr<MemoryManager> memory_manager_factory(MemoryManagerKind memkind,
                                                        Args&&... args) const {
    switch(memkind) {
      case MemoryManagerKind::invalid: NOT_ALLOWED(); return nullptr;
      case MemoryManagerKind::ga:
        // auto defd = get_memory_manager(memkind);
        return std::unique_ptr<MemoryManager>(new MemoryManagerGA{pg_});
        // return std::unique_ptr<MemoryManager>(new MemoryManagerGA{std::forward<Args>(args)...});
        break;
      case MemoryManagerKind::local:
        return std::unique_ptr<MemoryManager>(new MemoryManagerLocal{pg_self_});
        //   return std::unique_ptr<MemoryManager>(new
        //   MemoryManagerLocal{std::forward<Args>(args)...});
        break;
    }
    UNREACHABLE();
    return nullptr;
  }

private:
  ProcGroup        pg_;
  ProcGroup        pg_self_;
  DistributionKind distribution_kind_;
  // Distribution* default_distribution_;
  // std::unique_ptr<Distribution> default_distribution_;
  MemoryManagerKind memory_manager_kind_;
  // MemoryManager* default_memory_manager_;
  // MemoryManagerLocal* memory_manager_local_;
  IndexedAC                      ac_;
  std::shared_ptr<RuntimeEngine> re_;
  int                            nnodes_;
  int                            ranks_pn_;
  bool                           has_gpu_{false};
  ExecutionHW                    exhw_{ExecutionHW::CPU};
  meminfo                        minfo_;

  std::stringstream          profile_data_;
  std::vector<MemoryRegion*> mem_regs_to_dealloc_;
  std::vector<MemoryRegion*> unregistered_mem_regs_;

#if defined(USE_UPCXX_DISTARRAY)
  upcxx::intrank_t hint_;
#endif

}; // class ExecutionContext

} // namespace tamm
