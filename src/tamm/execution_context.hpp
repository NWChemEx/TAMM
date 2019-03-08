#ifndef TAMM_EXECUTION_CONTEXT_H_
#define TAMM_EXECUTION_CONTEXT_H_

#include "tamm/proc_group.hpp"
//#include "tamm/tensor_impl.hpp"
#include "tamm/memory_manager_ga.hpp"
#include "tamm/memory_manager_local.hpp"
#include "tamm/atomic_counter.h"

#include <algorithm>
#include <iterator>
#include <vector>

namespace tamm {

struct IndexedAC {
    AtomicCounter* ac_;
    size_t idx_;

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
/**
 * @brief Wrapper class to hold information during execution.
 *
 * This class holds the choice of default memory manager, distribution, irrep,
 * etc.
 *
 * @todo Should spin_restricted be wrapper by this class? Or should it always
 * default to false?
 */
class ExecutionContext {
public:
    ExecutionContext() : ac_{IndexedAC{nullptr, 0}} { pg_self_ = ProcGroup{MPI_COMM_SELF}; };
    // ExecutionContext(const ExecutionContext&) = default;
    // ExecutionContext(ExecutionContext&&) = default;
    // ExecutionContext& operator=(const ExecutionContext&) = default;
    // ExecutionContext& operator=(ExecutionContext&&) = default;

    /** @todo use shared pointers for solving GitHub issue #43*/
    ExecutionContext(ProcGroup pg, Distribution* default_distribution,
                     MemoryManager* default_memory_manager) :
      pg_{pg},
      default_distribution_{default_distribution},
      default_memory_manager_{default_memory_manager},
      ac_{IndexedAC{nullptr, 0}} {
        pg_self_ = ProcGroup{MPI_COMM_SELF};

        // memory_manager_local_ = MemoryManagerLocal::create_coll(pg_self_);
    }

    ~ExecutionContext() {
        // MemoryManagerLocal::destroy_coll(memory_manager_local_);
    }

    void allocate() {
        // no-op
    }

    /**
     * Allocate a list of tensor with default parameters (irrep, etc.)
     * @tparam T Type of element in tensor
     * @tparam Args Type of list of tensors to be allocated
     * @param tensor First tensor in the list
     * @param tensor_list Remaining tensors in the list
     */
    template<typename T, typename... Args>
    void allocate(Tensor<T>& tensor, Args&... tensor_list) {
        tensor.alloc(default_distribution_, default_memory_manager_);
        allocate(tensor_list...);
    }

    void allocate_local() {
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
    void allocate_local(Tensor<T>& tensor, Args&... tensor_list) {
        tensor.alloc(default_distribution_, memory_manager_local_);
        allocate(tensor_list...);
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
    void set_pg(const ProcGroup& pg) { pg_ = pg; }

    /**
     * Get the default distribution
     * @return Default distribution
     */
    Distribution* distribution() const { return default_distribution_; }

    /**
     * @brief Set the default Distribution for ExecutionContext
     *
     * @todo: change raw pointer to smart pointers?
     *
     * @param [in] distribution pointer to Distribution object
     */
    void set_distribution(Distribution* distribution) {
        default_distribution_ = distribution;
    }

    /**
     * Get the default memory manager
     * @return Default memory manager
     */
    MemoryManager* memory_manager() const { return default_memory_manager_; }

    /**
     * @brief Set the default memory manager for ExecutionContext
     *
     * @todo: change raw pointer to smart pointers?
     *
     * @param [in] memory_manager pointer to MemoryManager object
     */
    void set_memory_manager(MemoryManager* memory_manager) {
        default_memory_manager_ = memory_manager;
    }

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
        std::set_difference(
          mem_regs_to_dealloc_.begin(), mem_regs_to_dealloc_.end(),
          unregistered_mem_regs_.begin(), unregistered_mem_regs_.end(),
          std::inserter(result, result.begin()));
        mem_regs_to_dealloc_.clear();
        unregistered_mem_regs_.clear();
        for(auto mem_reg : result) {
            EXPECTS(mem_reg->allocation_status() == AllocationStatus::created ||
                    mem_reg->allocation_status() == AllocationStatus::orphaned);
            if(mem_reg->allocation_status() == AllocationStatus::orphaned) {
                mem_reg->dealloc_coll();
                delete mem_reg;
            } else {
                mem_regs_to_dealloc_.push_back(mem_reg);
            }
        }
    }

    void register_for_dealloc(MemoryRegion* mem_reg) {
        mem_regs_to_dealloc_.push_back(mem_reg);
    }

    void unregister_for_dealloc(MemoryRegion* mem_reg) {
        unregistered_mem_regs_.push_back(mem_reg);
    }

    IndexedAC ac() const { return ac_; }

    void set_ac(IndexedAC ac) { ac_ = ac; }

private:
    // RuntimeEngine re_;
    ProcGroup pg_;
    ProcGroup pg_self_;
    Distribution* default_distribution_;
    MemoryManager* default_memory_manager_;
    MemoryManagerLocal* memory_manager_local_;
    IndexedAC ac_;

    std::vector<MemoryRegion*> mem_regs_to_dealloc_;
    std::vector<MemoryRegion*> unregistered_mem_regs_;

}; // class ExecutionContext

} // namespace tamm

#endif // TAMM_EXECUTION_CONTEXT_H_
