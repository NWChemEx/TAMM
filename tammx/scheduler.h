#ifndef TAMMX_SCHEDULER_H_
#define TAMMX_SCHEDULER_H_

#include "tammx/types.h"
#include "tammx/proc_group.h"
#include "tammx/memory_manager.h"
#include "tammx/distribution.h"
#include "tammx/ops.h"

namespace tammx {
class Scheduler {
 public:
  enum class TensorStatus { invalid, allocated, deallocated, initialized };

  Scheduler(ProcGroup pg, Distribution* default_distribution,
            MemoryManager* default_memory_manager,
            Irrep default_irrep, bool default_spin_restricted)
      : default_distribution_{default_distribution},
        default_memory_manager_{default_memory_manager},
        default_irrep_{default_irrep},
        default_spin_restricted_{default_spin_restricted},
        pg_{pg} {}

  ~Scheduler() {
    reset_for_execution();
    clear();
  }

  void reset_for_execution() {
    for(auto &ptensor: intermediate_tensors_) {
      delete ptensor;
    }
    intermediate_tensors_.clear();
  }

  void clear() {
    for(auto &ptr_op : ops_) {
      delete ptr_op;
    }
    ops_.clear();
    tensors_.clear();
  }

  template<typename T>
  Tensor<T>* tensor(const IndexInfo& iinfo, Irrep irrep, bool spin_restricted) {
    auto indices = std::get<0>(iinfo);
    TensorRank nupper_indices = std::get<1>(iinfo);
    Tensor<T>* ptensor = new Tensor<T>{indices, nupper_indices, irrep, spin_restricted};
    tensors_[ptensor] = TensorInfo{TensorStatus::invalid};
    intermediate_tensors_.push_back(ptensor);
    return ptensor;
  }

  template<typename T>
  Tensor<T>* tensor(const IndexInfo& iinfo) {
    return tensor<T>(iinfo, default_irrep_, default_spin_restricted_);
  }

  template<typename T, typename LabeledTensorType>
  Scheduler& operator()(SetOpEntry<T, LabeledTensorType> sop) {
/** \warning
*  totalview LD on following statement
*  back traced to tammx::diis<double> in labeled_tensor.h
*  back traced to ccsd_driver
*/
    ops_.push_back(new SetOp<LabeledTensorType, T>(sop.value, sop.lhs, sop.mode));
    EXPECTS(tensors_.find(&sop.lhs.tensor()) != tensors_.end());
    EXPECTS(tensors_[&sop.lhs.tensor()].status == TensorStatus::allocated
            || tensors_[&sop.lhs.tensor()].status == TensorStatus::initialized);
    tensors_[&sop.lhs.tensor()].status = TensorStatus::initialized;
    return *this;
  }

  Scheduler& io() {
    return *this;
  }

  template<typename ...Args>
  Scheduler& io(TensorBase &tensor, Args& ... args) {
    EXPECTS(tensors_.find(&tensor) == tensors_.end());
    tensors_[&tensor] = TensorInfo{TensorStatus::initialized};
    return io(args...);
  }

  Scheduler& output() {
    return *this;
  }

  template<typename ...Args>
  Scheduler& output(TensorBase& tensor, Args& ... args) {
    EXPECTS(tensors_.find(&tensor) == tensors_.end());
    tensors_[&tensor] = TensorInfo{TensorStatus::allocated};
    return output(args...);
  }

  Scheduler& alloc() {
    return *this;
  }

  template<typename TensorType, typename ...Args>
  Scheduler& alloc(TensorType& tensor, Args& ... args) {
    EXPECTS(tensors_.find(&tensor) != tensors_.end());
    EXPECTS(tensors_[&tensor].status == TensorStatus::invalid ||
            tensors_[&tensor].status == TensorStatus::deallocated);
    tensors_[&tensor].status = TensorStatus::allocated;
    ops_.push_back(new AllocOp<TensorType>(tensor, pg_, default_distribution_, default_memory_manager_));
    return alloc(args...);
  }

  Scheduler& dealloc() {
    return *this;
  }

  template<typename TensorType, typename ...Args>
  Scheduler& dealloc(TensorType& tensor, Args& ... args) {
    EXPECTS(tensors_.find(&tensor) != tensors_.end());
    EXPECTS(tensors_[&tensor].status == TensorStatus::allocated ||
            tensors_[&tensor].status == TensorStatus::initialized);
    tensors_[&tensor].status = TensorStatus::deallocated;
    ops_.push_back(new DeallocOp<TensorType>(&tensor));
    return dealloc(args...);
  }


  template<typename T, typename LabeledTensorType>
  Scheduler& operator()(AddOpEntry<T, LabeledTensorType> aop) {
    EXPECTS(tensors_.find(&aop.lhs.tensor()) != tensors_.end());
    EXPECTS(tensors_.find(&aop.rhs.tensor()) != tensors_.end());
    EXPECTS(tensors_[&aop.rhs.tensor()].status == TensorStatus::initialized);
    EXPECTS(tensors_[&aop.lhs.tensor()].status == TensorStatus::initialized
            || (aop.mode==ResultMode::set
                && tensors_[&aop.lhs.tensor()].status==TensorStatus::allocated));
    tensors_[&aop.lhs.tensor()].status = TensorStatus::initialized;
/** \warning
*  totalview LD on following statement
*  back traced to ccsd_driver
*/
    ops_.push_back(new AddOp<LabeledTensorType, T>(aop.alpha, aop.lhs, aop.rhs, aop.mode, aop.exec_mode, aop.fn));
    return *this;
  }

  template<typename T, typename LabeledTensorType>
  Scheduler& operator()(MultOpEntry<T, LabeledTensorType> aop) {
    EXPECTS(tensors_.find(&aop.lhs.tensor()) != tensors_.end());
    EXPECTS(tensors_.find(&aop.rhs1.tensor()) != tensors_.end());
    EXPECTS(tensors_.find(&aop.rhs2.tensor()) != tensors_.end());
    EXPECTS(tensors_[&aop.rhs1.tensor()].status == TensorStatus::initialized);
    EXPECTS(tensors_[&aop.rhs2.tensor()].status == TensorStatus::initialized);
    EXPECTS(tensors_[&aop.lhs.tensor()].status == TensorStatus::initialized
            || (aop.mode==ResultMode::set
                && tensors_[&aop.lhs.tensor()].status==TensorStatus::allocated));
    tensors_[&aop.lhs.tensor()].status = TensorStatus::initialized;
    ops_.push_back(new MultOp<LabeledTensorType, T>(aop.alpha, aop.lhs, aop.rhs1, aop.rhs2, aop.mode, aop.exec_mode, aop.fn));
    return *this;
  }

  template<typename Func, typename LabeledTensorType>
  Scheduler& operator()(LabeledTensorType lhs, Func func, ResultMode mode = ResultMode::set) {
    EXPECTS(tensors_.find(&lhs.tensor()) != tensors_.end());
    EXPECTS(tensors_[&lhs.tensor()].status == TensorStatus::initialized
            || (mode==ResultMode::set
                && tensors_[&lhs.tensor()].status==TensorStatus::allocated));
    tensors_[&lhs.tensor()].status = TensorStatus::initialized;
    // ops_.push_back(new MapOp<Func,LabeledTensorType,0,0>(lhs, func, mode));
    ops_.push_back(new MapOp<LabeledTensorType, Func, 0>(lhs, func, mode));
    return *this;
  }

  template<typename Func, typename LabeledTensorType>
  Scheduler& sop(LabeledTensorType lhs, Func func) {
    EXPECTS(tensors_.find(&lhs.tensor()) != tensors_.end());
    EXPECTS(tensors_[&lhs.tensor()].status == TensorStatus::initialized);
    tensors_[&lhs.tensor()].status = TensorStatus::initialized;
    ops_.push_back(new ScanOp<Func,LabeledTensorType>(lhs, func));
    return *this;
  }

  void execute() {
    for(auto &op_ptr: ops_) {
      op_ptr->execute(pg_);
      for(auto t : op_ptr->reads()) {
        t->memory_region().fence();
      }
      pg_.barrier();
    }
  }

  Distribution* default_distribution() {
    return default_distribution_;
  }

  MemoryManager* default_memory_manager() {
    return default_memory_manager_;
  }

  ProcGroup pg() const {
    return pg_;
  }

 private:
  struct TensorInfo {
    TensorStatus status;
  };
  Distribution* default_distribution_;
  MemoryManager* default_memory_manager_;
  Irrep default_irrep_;
  bool default_spin_restricted_;
  ProcGroup pg_;
  std::map<TensorBase*,TensorInfo> tensors_;
  std::vector<Op*> ops_;
  std::vector<TensorBase*> intermediate_tensors_;
}; // class Scheduler

template<typename T>
inline void
assert_zero(Scheduler& sch, Tensor<T>& tc, double threshold = 1.0e-12) {
  auto lambda = [&] (auto &val) {
    //    std::cout<<"assert_zero. val="<<val<<std::endl;
    EXPECTS(std::abs(val) < threshold);
  };
  sch.io(tc)
      .sop(tc(), lambda)
      .execute();
}

template<typename LabeledTensorType, typename T>
inline void
assert_equal(Scheduler& sch, LabeledTensorType tc, T value, double threshold = 1.0e-12) {
  auto lambda = [&] (auto &val) {
    EXPECTS(std::abs(val - value) < threshold);
  };
  sch.io(tc.tensor())
      .sop(tc, lambda)
      .execute();
}

template<typename LabeledTensorType>
inline void
tensor_print(Scheduler& sch, LabeledTensorType ltensor) {
  sch.io(ltensor.tensor())
      .sop(ltensor, [] (auto &ival) {
          std::cout<<ival<<" ";
        })
      .execute();
}

}  // namespace tammx

#endif // TAMMX_SCHEDULER_H_
