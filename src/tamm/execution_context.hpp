#ifndef TAMM_EXECUTION_CONTEXT_H_
#define TAMM_EXECUTION_CONTEXT_H_

#include "tamm/distribution.hpp"
#include "tamm/proc_group.hpp"
//#include "tamm/tensor_impl.hpp"
#include "tamm/memory_manager_ga.hpp"
#include "tamm/memory_manager_local.hpp"

namespace tamm {

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
    ExecutionContext() { pg_self_ = ProcGroup{MPI_COMM_SELF}; };
    // ExecutionContext(const ExecutionContext&) = default;
    // ExecutionContext(ExecutionContext&&) = default;
    // ExecutionContext& operator=(const ExecutionContext&) = default;
    // ExecutionContext& operator=(ExecutionContext&&) = default;

    ExecutionContext(ProcGroup pg, Distribution* default_distribution,
                     MemoryManager* default_memory_manager) :
      pg_{pg},
      default_distribution_{default_distribution},
      default_memory_manager_{default_memory_manager} {
        pg_self_ = ProcGroup{MPI_COMM_SELF};
        /** @todo use shared pointers */
        // memory_manager_local_ = MemoryManagerLocal::create_coll(pg_self_);
    }

    ~ExecutionContext() {
        // MemoryManagerLocal::destroy_coll(memory_manager_local_);
    }

    /**
     * Construct a scheduler object
     * @return Scheduler object
     */
    // Scheduler& scheduler() {
    //   // return Scheduler{};
    // }

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

private:
    ProcGroup pg_;
    ProcGroup pg_self_;
    Distribution* default_distribution_;
    MemoryManager* default_memory_manager_;
    MemoryManagerLocal* memory_manager_local_;

}; // class ExecutionContext

} // namespace tamm

#endif // TAMM_EXECUTION_CONTEXT_H_
