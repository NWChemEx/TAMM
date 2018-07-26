#ifndef TAMM_SCHEDULER_HPP_
#define TAMM_SCHEDULER_HPP_

#include <set>

// #include "distribution.h"
// #include "memory_manager.h"
//#include "proc_group.h"
#include "tamm/dag_impl.hpp"
#include "tamm/execution_context.hpp"
#include "tamm/ops.hpp"
#include "tamm/tensor.hpp"

namespace tamm {
// class Distribution;
// class MemoryManager;

// class TensorHolder {
//     public:
//     template<typename T>
//     TensorHolder(Tensor<T> tensor) {
//         switch(tensor_element_type<T>()) {
//             case ElementType::double_precision:
//                 type_ = ElementType::double_precision;
//                 set(tensor);
//                 break;
//             default: UNREACHABLE();
//         }
//     }

//     ~TensorHolder() {
//         switch(type_) {
//             case ElementType::double_precision:
//                 tensor_double_.Tensor<double>::~Tensor();
//                 break;
//             default: UNREACHABLE(); break;
//         }
//     }
// };

//     template<typename T>
//     void set(Tensor<T> const& tensor);

//     template<typename T>
//     Tensor<T> const& tensor();

//     private:
//     Tensor<double> tensor_double_;
//     Tensor<float> tensor_float_;
//     // Tensor<std::complex<double>> tensor_complex_d_;
//     // Tensor<std::complex<float>> tensor_complex_f_;
//     ElementType type_;
// };

// template<>
// void TensorHolder::set<double>(Tensor<double> const& tensor) {
//     tensor_double_ = tensor;
// }

// template<>
// void TensorHolder::set<float>(Tensor<float> const& tensor) {
//     tensor_float_ = tensor;
// }

// template<>
// Tensor<double> const& TensorHolder::tensor<double>() {
//     return tensor_double_;
// }

// template<>
// Tensor<float> const& TensorHolder::tensor<float>() {
//     return tensor_float_;
// }
using internal::DAGImpl;
////////////////////////////////////////////////////
/**
 * @brief Scheduler to execute a list of operations.
 * @ingroup operations
 */
class Scheduler {
public:
    /**
     * @brief Allocation status of tensor.
     *
     * This is used to perform some correctness checks on the list of
     * operations.
     * @todo Can this be replaced by AllocationStatus
     */
    // enum class TensorStatus { invalid, allocated, deallocated, initialized };

    // @to-do: what is the default scheduler?
    Scheduler()                 = default;
    Scheduler(const Scheduler&) = default;
    Scheduler(Scheduler&&)      = default;
    Scheduler& operator=(const Scheduler&) = default;
    Scheduler& operator=(Scheduler&&) = default;

    Scheduler(ExecutionContext* ec) : ec_{ec} {}

    Scheduler& operator()() { return *this; }

    template<typename OpType, typename... OpTypes>
    Scheduler& operator()(const OpType& op, const OpTypes&... ops) {
        ops_.push_back(op.clone());
        return operator()(ops...);
    }

    // Scheduler& tensors() { return *this; }

    // template<typename ElementType, typename... ElementTypes>
    // Scheduler& tensors(Tensor<ElementType> tensor,
    //                    Tensor<ElementTypes>... rhs) {
    //     tensors_.push_back(TensorHolder{tensor});
    //     return tensors(rhs...);
    // }

    // Scheduler& live_in() { return *this; }

    // template<typename ElementType, typename... ElementTypes>
    // Scheduler& live_in(Tensor<ElementType> tensor,
    //                    Tensor<ElementTypes>... tensors) {
    //     live_in_tensors_.push_back(TensorHolder{tensor});
    //     return live_in(tensors...);
    // }

    // Scheduler& live_out() { return *this; }

    // template<typename ElementType, typename... ElementTypes>
    // Scheduler& live_out(Tensor<ElementType> tensor,
    //                     Tensor<ElementTypes>... tensors) {
    //     live_out_tensors_.push_back(TensorHolder{tensor});
    //     return live_out(tensors...);
    // }

    Scheduler& allocate() { return *this; }

    ExecutionContext* ec() { return ec_; }

    template<typename TensorType, typename... Args>
    Scheduler& allocate(TensorType tensor, Args&... tensors) {
        ops_.push_back(std::make_shared<AllocOp<TensorType>>(tensor, ec()));
        return allocate(tensors...);
    }

    Scheduler& deallocate() { return *this; }

    template<typename TensorType, typename... Args>
    Scheduler& deallocate(TensorType tensor, Args&... tensors) {
        ops_.push_back(std::make_shared<DeallocOp<TensorType>>(tensor));
        return deallocate(tensors...);
    }

    void execute() {
        for(auto& op : ops_) { op->execute(ec()->pg()); }
    }

    template<typename Func, typename... Args>
    static void execute(DAGImpl<Func, Args...> dag) {}

    ~Scheduler() {
        // delete ops
    }


    template<typename Func, typename LabeledTensorType>
    Scheduler& gop(LabeledTensorType lhs, Func func) {
        ops_.push_back(std::make_shared<ScanOp<Func,LabeledTensorType>>(lhs, func));
        return *this;
    }

    template<typename Func, typename LabeledTensorType, int N>
    Scheduler& gop(LabeledTensorType lhs, Func func,
                   std::array<LabeledTensorType, N> rhs,
                   ResultMode mode = ResultMode::set) {
        ops_.push_back(
          std::make_shared<MapOp<LabeledTensorType, Func, N>>(lhs, func, rhs, mode));
        return *this;
    }

private:
    ExecutionContext* ec_;
    // void validate() {
    //     // 1. every tensor used by operarions should be listed in tensors_

    //     // 2. every tensor must be initialized (part of LHS) or be an
    //     // input (ilve_in) tensor before it is used

    //     // 3. every output tensor should be allocated and set (be LHS in
    //     // at least one operation or be a live_in tensor)

    //     // 4. every tensor must be allocated before it is used

    //     // 5. every non-output (not in live_out) tensor must be
    //     // deallocated
    // }

    // Distribution* default_distribution_;
    // MemoryManager* default_memory_manager_;
    // ProcGroup pg_;
    std::vector<std::shared_ptr<Op>> ops_;
    // std::vector<TensorHolder> tensors_;
    // std::vector<TensorHolder> live_in_tensors_;
    // std::vector<TensorHolder> live_out_tensors_;

    //   ~Scheduler() {
    //     clear();
    //   }

    //   /**
    //    * @brief Check for correctness before execution.
    //    */
    //   void prepare_for_execution() {
    //     for(auto &ptensor: intermediate_tensors_) {
    //       EXPECTS(tensors_[ptensor].status == TensorStatus::deallocated);
    //     }
    //     //@todo Also check that io/output tensors are initialized
    //   }

    //   /**
    //    * @brief clear all internal state
    //    */
    //   void clear() {
    //     for(auto &ptr_op : ops_) {
    //       delete ptr_op;
    //     }
    //     for(auto &itensor : intermediate_tensors_) {
    //       delete itensor;
    //     }
    //     ops_.clear();
    //     intermediate_tensors_.clear();
    //     tensors_.clear();
    //   }

    // #if 0
    //   /**
    //    * Create a temporary tensor
    //    * @tparam T Type of elements in tensor
    //    * @param iinfo Indices in the tensor
    //    * @param irrep Irrep to be used
    //    * @param spin_restricted Is the tensor spin restricted
    //    * @return Pointer to the constructed tensor
    //    * @todo Why not return a reference?
    //    */
    //   template<typename T>
    //   Tensor<T>* tensor(const IndexInfo& iinfo, Irrep irrep, bool
    //   spin_restricted) {
    //     auto indices = std::get<0>(iinfo);
    //     TensorRank nupper_indices = std::get<1>(iinfo);
    //     Tensor<T>* ptensor = new Tensor<T>{indices, nupper_indices, irrep,
    //     spin_restricted}; tensors_[ptensor] =
    //     TensorInfo{TensorStatus::invalid};
    //     intermediate_tensors_.push_back(ptensor);
    //     return ptensor;
    //   }

    //   /**
    //    * Construct a temporary tensor with given indices and other default
    //    parameters
    //    * @tparam T Type of elements in the tensor to be created
    //    * @param iinfo Indices in the tensor
    //    * @return Pointer to constructed tensor.
    //    * @note The user does not need to destruct the created tensor
    //    * @todo Why not return a reference?
    //    */
    //   template<typename T>
    //   Tensor<T>* tensor(const IndexInfo& iinfo) {
    //     return tensor<T>(iinfo, default_irrep_, default_spin_restricted_);
    //   }
    // #endif

    //   /**
    //    * Add a SetOp object to the scheduler
    //    * @tparam T Type of RHS value
    //    * @tparam LabeledTensorType Type of LHS tensor
    //    * @param sop SetOpEntry object
    //    * @return Reference to the this scheduler
    //    */
    //   template<typename T, typename LabeledTensorType>
    //   Scheduler& operator()(SetOpEntry<T, LabeledTensorType> sop) {
    // /** \warning
    // *  totalview LD on following statement
    // *  back traced to tamm::diis<double> line 128 diis.h
    // *  back traced to ccsd_driver<double> line 426 stl_vector.h
    // *  back traced to main line 607 ccsd_driver.cc
    // */
    //     ops_.push_back(new SetOp<LabeledTensorType, T>(sop.value, sop.lhs,
    //     sop.mode)); EXPECTS(tensors_.find(&sop.lhs.tensor()) !=
    //     tensors_.end()); EXPECTS(tensors_[&sop.lhs.tensor()].status ==
    //     TensorStatus::allocated
    //             || tensors_[&sop.lhs.tensor()].status ==
    //             TensorStatus::initialized);
    //     tensors_[&sop.lhs.tensor()].status = TensorStatus::initialized;
    //     return *this;
    //   }

    //   Scheduler& io() {
    //     return *this;
    //   }

    //   /**
    //    * @brief Annotate a list of tensors as being input/output
    //    *
    //    * IO tensors are assumed to be allocated and initialized at this
    //    point.
    //    * @tparam Args Types of tensors being annotated
    //    * @param tensor First tensor in the list
    //    * @param args Rest of tensors in the list
    //    * @return Reference to this scheduler
    //    */
    //   template<typename ...Args>
    //   Scheduler& io(TensorBase &tensor, Args& ... args) {
    //     if(tensors_.find(&tensor) == tensors_.end()) {
    //       tensors_[&tensor] = TensorInfo{TensorStatus::initialized};
    //     } else {
    //       EXPECTS(tensors_[&tensor].status == TensorStatus::initialized);
    //     }
    //     return io(args...);
    //   }

    //   Scheduler& output() {
    //     return *this;
    //   }

    //   /**
    //    * @brief Annotate a list of tensors as being output
    //    *
    //    * Output tensors are assumed to be allocated (but not initialized) at
    //    this point.
    //    * Output tensors cannot be used in RHS of an operation util they are
    //    initialized.
    //    *
    //    * @tparam Args Types of tensors being annotated
    //    * @param tensor First tensor in the list
    //    * @param args Rest of tensors in the list
    //    * @return Reference to this scheduler
    //    */
    //   template<typename ...Args>
    //   Scheduler& output(TensorBase& tensor, Args& ... args) {
    //     if(tensors_.find(&tensor) == tensors_.end()) {
    //       tensors_[&tensor] = TensorInfo{TensorStatus::allocated};
    //     } else {
    //       EXPECTS(tensors_[&tensor].status == TensorStatus::allocated
    //               || tensors_[&tensor].status == TensorStatus::initialized);
    //     }
    //     return output(args...);
    //   }

    //   Scheduler& alloc() {
    //     return *this;
    //   }

    //   /**
    //    * Allocate a list of tensors
    //    * @tparam TensorType Type of the first tensor to be allocated
    //    * @tparam Args Types of the rest of tensors
    //    * @param tensor First tensor in the list to be allocated
    //    * @param args Rest of tensors to be allocated
    //    * @return Reference to this scheduler
    //    */
    //   template<typename TensorType, typename ...Args>
    //   Scheduler& alloc(TensorType& tensor, Args& ... args) {
    //     EXPECTS(tensors_.find(&tensor) != tensors_.end());
    //     EXPECTS(tensors_[&tensor].status == TensorStatus::invalid ||
    //             tensors_[&tensor].status == TensorStatus::deallocated);
    //     tensors_[&tensor].status = TensorStatus::allocated;
    //     ops_.push_back(new AllocOp<TensorType>(tensor, pg_,
    //     default_distribution_, default_memory_manager_)); return
    //     alloc(args...);
    //   }

    //   Scheduler& dealloc() {
    //     return *this;
    //   }

    //   /**
    //    * Deallocate a list of tensors
    //    * @tparam TensorType Type of the first tensor to be deallocated
    //    * @tparam Args Types of the rest of tensors
    //    * @param tensor First tensor in the list to be deallocated
    //    * @param args Rest of tensors to be deallocated
    //    * @return Reference to this scheduler
    //    */
    //   template<typename TensorType, typename ...Args>
    //   Scheduler& dealloc(TensorType& tensor, Args& ... args) {
    //     EXPECTS(tensors_.find(&tensor) != tensors_.end());
    //     EXPECTS(tensors_[&tensor].status == TensorStatus::allocated ||
    //             tensors_[&tensor].status == TensorStatus::initialized);
    //     tensors_[&tensor].status = TensorStatus::deallocated;
    //     ops_.push_back(new DeallocOp<TensorType>(&tensor));
    //     return dealloc(args...);
    //   }

    //   /**
    //    * Add an add-op entry to the list of operations to be executed by the
    //    scheduler
    //    * @tparam T Type of scalar factor
    //    * @tparam LabeledTensorType Type of labeled tensor
    //    * @param aop Add operation entry to the add to the scheduler
    //    * @return Reference to this scheduler
    //    */
    //   template<typename T, typename LabeledTensorType>
    //   Scheduler& operator()(AddOpEntry<T, LabeledTensorType> aop) {
    //     EXPECTS(tensors_.find(&aop.lhs.tensor()) != tensors_.end());
    //     EXPECTS(tensors_.find(&aop.rhs.tensor()) != tensors_.end());
    //     EXPECTS(tensors_[&aop.rhs.tensor()].status ==
    //     TensorStatus::initialized);
    //     EXPECTS(tensors_[&aop.lhs.tensor()].status ==
    //     TensorStatus::initialized
    //             || (aop.mode==ResultMode::set
    //                 &&
    //                 tensors_[&aop.lhs.tensor()].status==TensorStatus::allocated));
    //     tensors_[&aop.lhs.tensor()].status = TensorStatus::initialized;
    // /** \warning
    // *  totalview LD on following statement
    // *  back traced to ccsd_driver
    // */
    //     ops_.push_back(new AddOp<LabeledTensorType, T>(aop.alpha, aop.lhs,
    //     aop.rhs, aop.mode, aop.exec_mode, aop.fn)); return *this;
    //   }

    //   /**
    //    * @brief Execute an arbitrary function on this scheduler
    //    *
    //    * Arbitrary functions constrain the ability of the scheduler to
    //    reorder operations
    //    *
    //    * @tparam Func Type of function to be executed
    //    * @param func Function object to be executed
    //    * @return Reference to this scheduler
    //    */
    //   template<typename Func>
    //   Scheduler& operator() (Func func) {
    //     ops_.push_back(new LambdaOp<Func>{func});
    //     return *this;
    //   }

    //   /**
    //    * Add a tensor contraction operation to scheduler
    //    * @tparam T Tensor of scalar
    //    * @tparam LabeledTensorType Type of labeled tensor
    //    * @param aop Tensor contraction operation to be added
    //    * @return Reference to this scheduler
    //    * @todo Relabel @param aop to @param mop
    //    */
    //   template<typename T, typename LabeledTensorType>
    //   Scheduler& operator()(MultOpEntry<T, LabeledTensorType> aop) {
    //     EXPECTS(tensors_.find(&aop.lhs.tensor()) != tensors_.end());
    //     EXPECTS(tensors_.find(&aop.rhs1.tensor()) != tensors_.end());
    //     EXPECTS(tensors_.find(&aop.rhs2.tensor()) != tensors_.end());
    //     EXPECTS(tensors_[&aop.rhs1.tensor()].status ==
    //     TensorStatus::initialized);
    //     EXPECTS(tensors_[&aop.rhs2.tensor()].status ==
    //     TensorStatus::initialized);
    //     EXPECTS(tensors_[&aop.lhs.tensor()].status ==
    //     TensorStatus::initialized
    //             || (aop.mode==ResultMode::set
    //                 &&
    //                 tensors_[&aop.lhs.tensor()].status==TensorStatus::allocated));
    //     tensors_[&aop.lhs.tensor()].status = TensorStatus::initialized;
    //     ops_.push_back(new MultOp<LabeledTensorType, T>(aop.alpha, aop.lhs,
    //     aop.rhs1, aop.rhs2, aop.mode, aop.exec_mode, aop.fn)); return *this;
    //   }

    //   template<typename Func, typename LabeledTensorType>
    //   Scheduler& operator()(LabeledTensorType lhs, Func func, ResultMode mode
    //   = ResultMode::set) {
    //     EXPECTS(tensors_.find(&lhs.tensor()) != tensors_.end());
    //     EXPECTS(tensors_[&lhs.tensor()].status == TensorStatus::initialized
    //             || (mode==ResultMode::set
    //                 &&
    //                 tensors_[&lhs.tensor()].status==TensorStatus::allocated));
    //     tensors_[&lhs.tensor()].status = TensorStatus::initialized;
    //     // ops_.push_back(new MapOp<Func,LabeledTensorType,0,0>(lhs, func,
    //     mode)); ops_.push_back(new MapOp<LabeledTensorType, Func, 0>(lhs,
    //     func, mode)); return *this;
    //   }


    //   /**
    //    * @brief Execute the list of operations given to this scheduler
    //    */
    //   void execute() {
    //     prepare_for_execution(); // check before execution
    //     for(auto &op_ptr: ops_) {
    //       op_ptr->execute(pg_);
    //       for(auto t : op_ptr->reads()) {
    //         t->memory_region().fence();
    //       }
    //       pg_.barrier();
    //     }
    //   }

    //   /**
    //    * Access the default distribution used by this scheduler
    //    * @return Default distribution
    //    */
    //   Distribution* default_distribution() {
    //     return default_distribution_;
    //   }

    //   /**
    //    * Access the default memory manager used by this scheduler
    //    * @return Default memory manager
    //    */
    //   MemoryManager* default_memory_manager() {
    //     return default_memory_manager_;
    //   }

    //   ProcGroup pg() const {
    //     return pg_;
    //   }

    //  private:
    //   struct TensorInfo {
    //     TensorStatus status;
    //   };
    //   Distribution* default_distribution_;
    //   MemoryManager* default_memory_manager_;
    //   Distribution* default_distribution_;
    //   MemoryManager* default_memory_manager_;
    //   Irrep default_irrep_;
    //   bool default_spin_restricted_;
    //   ProcGroup pg_;
    //   std::map<TensorBase*,TensorInfo> tensors_;
    //   std::vector<Op*> ops_;
    //   std::vector<TensorBase*> intermediate_tensors_;
}; // class Scheduler

// template<typename T>
// inline void
// assert_zero(Scheduler& sch, Tensor<T>& tc, double threshold = 1.0e-12) {
//   auto lambda = [&] (auto &val) {
//     //    std::cout<<"assert_zero. val="<<val<<std::endl;
//     EXPECTS(std::abs(val) < threshold);
//   };
//   sch.io(tc)
//       .sop(tc(), lambda)
//       .execute();
// }

// template<typename LabeledTensorType, typename T>
// inline void
// assert_equal(Scheduler& sch, LabeledTensorType tc, T value, double threshold
// = 1.0e-12) {
//   auto lambda = [&] (auto &val) {
//     EXPECTS(std::abs(val - value) < threshold);
//   };
//   sch.io(tc.tensor())
//       .sop(tc, lambda)
//       .execute();
// }

// template<typename LabeledTensorType>
// inline void
// tensor_print(Scheduler& sch, LabeledTensorType ltensor) {
//   sch.io(ltensor.tensor())
//       .sop(ltensor, [] (auto &ival) {
//           std::cout<<ival<<" ";
//         })
//       .execute();
// }

} // namespace tamm

#endif // TAMM_SCHEDULER_HPP_
