#ifndef TAMMX_OPS_H_
#define TAMMX_OPS_H_

#include "tammx/tensor.h"
#include "tammx/labeled-block.h"
#include "tammx/labeled-tensor.h"
#include "tammx/util.h"

namespace tammx {

/////////////////////////////////////////////////////////////////////
//         operators
/////////////////////////////////////////////////////////////////////

class Op {
 public:
  virtual void execute() = 0;
  virtual ~Op() {}
};


template<typename T, typename LabeledTensorType>
struct SetOp : public Op {
  void execute() override;

  SetOp(T value, LabeledTensorType& lhs, ResultMode mode)
      : value_{value},
        lhs_{lhs},
        mode_{mode} {}

  T value_;
  LabeledTensorType lhs_;
  ResultMode mode_;
};

template<typename T, typename LabeledTensorType>
struct AddOp : public Op {
  void execute() override;

  AddOp(T alpha, const LabeledTensorType& lhs, const LabeledTensorType& rhs, ResultMode mode,
        ExecutionMode exec_mode,
        add_fn* fn)
      : alpha_{alpha},
        lhs_{lhs},
        rhs_{rhs},
        mode_{mode},
        exec_mode_{exec_mode},
        fn_{fn} { }

  T alpha_;
  LabeledTensorType lhs_, rhs_;
  ResultMode mode_;
  ExecutionMode exec_mode_;
  add_fn* fn_;
};

template<typename T, typename LabeledTensorType>
struct MultOp : public Op {
  void execute() override;

  MultOp(T alpha, const LabeledTensorType& lhs, const LabeledTensorType& rhs1,
         const LabeledTensorType& rhs2, ResultMode mode,
         ExecutionMode exec_mode,
         mult_fn* fn)
      : alpha_{alpha},
        lhs_{lhs},
        rhs1_{rhs1},
        rhs2_{rhs2},
        mode_{mode},
        exec_mode_{exec_mode},
        fn_{fn} { }

  T alpha_;
  LabeledTensorType lhs_, rhs1_, rhs2_;
  ResultMode mode_;
  ExecutionMode exec_mode_;
  mult_fn* fn_;
};

template<typename TensorType>
struct AllocOp: public Op {
  void execute() override {
    Expects(pg_.is_valid());
    tensor_->alloc(pg_, distribution_, memory_manager_);
  }

  AllocOp(TensorType& tensor, ProcGroup pg, Distribution* distribution, MemoryManager* memory_manager)
      : tensor_{&tensor},
        pg_{pg},
        distribution_{distribution},
        memory_manager_{memory_manager} {}

  TensorType *tensor_;
  ProcGroup pg_;
  Distribution* distribution_;
  MemoryManager* memory_manager_;
};

template<typename TensorType>
struct DeallocOp: public Op {
  void execute() override {
    tensor_->dealloc();
  }

  DeallocOp(TensorType* tensor)
      : tensor_{tensor} {}

  TensorType *tensor_;
};


template<typename LabeledTensorType, typename Func, int N>
struct MapOp : public Op {
  using RHS = std::array<LabeledTensorType, N>;
  using T = typename LabeledTensorType::element_type;
  using RHS_Blocks = std::array<Block<T>, N>;

  void execute() override {
    auto &lhs_tensor = *lhs_.tensor_;
    auto lambda = [&] (const TensorIndex& blockid) {
      auto size = lhs_tensor.block_size(blockid);
      if(!(lhs_tensor.nonzero(blockid) && lhs_tensor.spin_unique(blockid) && size > 0)) {
        return;
      }
      auto lblock = lhs_tensor.alloc(blockid);
      auto rhs_blocks = get_blocks(rhs_, blockid);
      std::array<T*,N> rhs_bufs;
      for(int i=0; i<N; i++) {
        rhs_bufs[i] = rhs_blocks[i].buf();
      }
      for(int i=0; i<size; i++) {
        func_(lblock.buf(), rhs_bufs, i);
      }
      if(mode_ == ResultMode::update) {
        lhs_tensor.add(lblock);
      } else if (mode_ == ResultMode::set) {
        lhs_tensor.put(lblock);
      } else {
        assert(0);
      }
    };
#if 0
    auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
#else
    auto itr_first = loop_iterator(slice_indices(lhs_tensor.tindices(), lhs_.label_));
#endif
    parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  MapOp(LabeledTensorType& lhs, Func func, RHS& rhs, ResultMode mode = ResultMode::set)
      : lhs_{lhs},
        func_{func},
        rhs_{rhs},
        mode_{mode} {}

  RHS_Blocks get_blocks(RHS& rhs, const TensorIndex& id) {
    RHS_Blocks blocks;
    for(int i=0; i<rhs.size(); i++) {
      blocks[i] = rhs[i].get(id);
    }
    return blocks;
  }

  LabeledTensorType& lhs_;
  Func func_;
  std::array<LabeledTensorType, N> rhs_;
  ResultMode mode_;
};

template<typename LabeledTensorType, typename Func, int N>
struct MapIdOp : public Op {
  using RHS = std::array<LabeledTensorType, N>;
  using T = typename LabeledTensorType::element_type;
  using RHS_Blocks = std::array<Block<T>, N>;

  void execute() override {
    auto &lhs_tensor = *lhs_.tensor_;
    auto lambda = [&] (const TensorIndex& blockid) {
      auto size = lhs_tensor.block_size(blockid);
      if(!(lhs_tensor.nonzero(blockid) && lhs_tensor.spin_unique(blockid) && size > 0)) {
        return;
      }
      auto lblock = lhs_tensor.alloc(blockid);
      auto rhs_blocks = get_blocks(rhs_, blockid);
      std::array<T*,N> rhs_bufs;
      for(int i=0; i<N; i++) {
        rhs_bufs[i] = rhs_blocks[i].buf();
      }

      auto lo = lhs_tensor.block_offset(blockid);
      auto dims = lhs_tensor.block_dims(blockid);
      auto hi = lo;
      for(int i=0; i<hi.size(); i++) {
        hi[i] += dims[i];
      }
      auto itr = lo;

      if(lo.size()==0) {
        func_(itr);
      } else if(lo.size() == 1) {
        int i=0;
        for(itr[0]=lo[0]; itr[0]<hi[0]; itr[0]++, i++) {
          func_(lblock.buf(), rhs_bufs, i, itr);
        }
      } else if(lo.size() == 2) {
        int i=0;
        for(itr[0]=lo[0]; itr[0]<hi[0]; itr[0]++) {
          for(itr[1]=lo[1]; itr[1]<hi[1]; itr[1]++, i++) {
            func_(lblock.buf(), rhs_bufs, i, itr);
          }
        }
      } else if(lo.size() == 3) {
        int i=0;
        for(itr[0]=lo[0]; itr[0]<hi[0]; itr[0]++) {
          for(itr[1]=lo[1]; itr[1]<hi[1]; itr[1]++) {
            for(itr[2]=lo[2]; itr[2]<hi[2]; itr[2]++, i++) {
              func_(lblock.buf(), rhs_bufs, i, itr);
            }
          }
        }
      } else if(lo.size() == 4) {
        int i=0;
        for(itr[0]=lo[0]; itr[0]<hi[0]; itr[0]++) {
          for(itr[1]=lo[1]; itr[1]<hi[1]; itr[1]++) {
            for(itr[2]=lo[2]; itr[2]<hi[2]; itr[2]++) {
              for(itr[3]=lo[3]; itr[3]<hi[3]; itr[3]++, i++) {
                func_(lblock.buf(), rhs_bufs, i, itr);
              }
            }
          }
        }
      } else {
        // @todo implement
        assert(0);
      }

      // for(int i=0; i<size; i++) {
      //   func_(lblock.buf(), rhs_bufs, i);
      // }
      if(mode_ == ResultMode::update) {
        lhs_tensor.add(lblock);
      } else if (mode_ == ResultMode::set) {
        lhs_tensor.put(lblock);
      } else {
        assert(0);
      }
    };
#if 0
    auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
#else
    auto itr_first = loop_iterator(slice_indices(lhs_tensor.tindices(), lhs_.label_));
#endif
    parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  MapIdOp(LabeledTensorType& lhs, Func func, RHS& rhs, ResultMode mode = ResultMode::set)
      : lhs_{lhs},
        func_{func},
        rhs_{rhs},
        mode_{mode} {}

  RHS_Blocks get_blocks(RHS& rhs, const TensorIndex& id) {
    RHS_Blocks blocks;
    for(int i=0; i<rhs.size(); i++) {
      blocks[i] = rhs[i].get(id);
    }
    return blocks;
  }

  LabeledTensorType& lhs_;
  Func func_;
  std::array<LabeledTensorType, N> rhs_;
  ResultMode mode_;
};



/////////////////////////////////////////////////////////////////////
//         scan operator
/////////////////////////////////////////////////////////////////////

/**
 * @todo Could be more general, similar to MapOp
 */
template<typename Func, typename LabeledTensorType>
struct ScanOp : public Op {
  void execute() {
    // std::cerr<<__FUNCTION__<<":"<<__LINE__<<": ScanOp\n";
    auto& tensor = *ltensor_.tensor_;
    auto lambda = [&] (const TensorIndex& blockid) {
      auto size = tensor.block_size(blockid);
      if(!(tensor.nonzero(blockid) && tensor.spin_unique(blockid) && size > 0)) {
        return;
      }
      //std::cout<<"ScanOp. blockid: "<<blockid<<std::endl;
      auto block = tensor.get(blockid);
      auto tbuf = block.buf();
      for(int i=0; i<size; i++) {
        func_(tbuf[i]);
      }
    };
#if 0
    auto itr_first = loop_iterator(slice_indices(tensor.indices(), ltensor_.label_));
#else
    auto itr_first = loop_iterator(slice_indices(tensor.tindices(), ltensor_.label_));
#endif
    parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  ScanOp(const LabeledTensorType& ltensor, Func func)
      : ltensor_{ltensor},
        func_{func} {
    Expects(ltensor.tensor_ != nullptr);
  }

  LabeledTensorType ltensor_;
  Func func_;
};


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
    for(auto &ptr_op : ops_) {
      delete ptr_op;
    }
    for(auto &ptensor: intermediate_tensors_) {
      delete ptensor;
    }
  }

  template<typename T>
  Tensor<T>* tensor(const IndexInfo& iinfo, Irrep irrep, bool spin_restricted) {
    auto indices = std::get<0>(iinfo);
    auto nupper_indices = std::get<1>(iinfo);
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
    ops_.push_back(new SetOp<LabeledTensorType, T>(sop.value, sop.lhs, sop.mode));
    Expects(tensors_.find(&sop.lhs.tensor()) != tensors_.end());
    Expects(tensors_[&sop.lhs.tensor()].status == TensorStatus::allocated
            || tensors_[&sop.lhs.tensor()].status == TensorStatus::initialized);
    tensors_[&sop.lhs.tensor()].status = TensorStatus::initialized;
    return *this;
  }

  Scheduler& io() {
    return *this;
  }

  template<typename ...Args>
  Scheduler& io(TensorBase &tensor, Args& ... args) {
    Expects(tensors_.find(&tensor) == tensors_.end());
    tensors_[&tensor] = TensorInfo{TensorStatus::initialized};
    return io(args...);
  }

  Scheduler& output() {
    return *this;
  }

  template<typename ...Args>
  Scheduler& output(TensorBase& tensor, Args& ... args) {
    Expects(tensors_.find(&tensor) == tensors_.end());
    tensors_[&tensor] = TensorInfo{TensorStatus::allocated};
    return output(args...);
  }

  Scheduler& alloc() {
    return *this;
  }

  template<typename TensorType, typename ...Args>
  Scheduler& alloc(TensorType& tensor, Args& ... args) {
    Expects(tensors_.find(&tensor) != tensors_.end());
    Expects(tensors_[&tensor].status == TensorStatus::invalid ||
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
    Expects(tensors_.find(&tensor) != tensors_.end());
    Expects(tensors_[&tensor].status == TensorStatus::allocated ||
            tensors_[&tensor].status == TensorStatus::initialized);
    tensors_[&tensor].status = TensorStatus::deallocated;
    ops_.push_back(new DeallocOp<TensorType>(&tensor));
    return dealloc(args...);
  }


  template<typename T, typename LabeledTensorType>
  Scheduler& operator()(AddOpEntry<T, LabeledTensorType> aop) {
    Expects(tensors_.find(&aop.lhs.tensor()) != tensors_.end());
    Expects(tensors_.find(&aop.rhs.tensor()) != tensors_.end());
    Expects(tensors_[&aop.rhs.tensor()].status == TensorStatus::initialized);
    Expects(tensors_[&aop.lhs.tensor()].status == TensorStatus::initialized
            || (aop.mode==ResultMode::set
                && tensors_[&aop.lhs.tensor()].status==TensorStatus::allocated));
    tensors_[&aop.lhs.tensor()].status = TensorStatus::initialized;
    ops_.push_back(new AddOp<LabeledTensorType, T>(aop.alpha, aop.lhs, aop.rhs, aop.mode, aop.exec_mode, aop.fn));
    return *this;
  }

  template<typename T, typename LabeledTensorType>
  Scheduler& operator()(MultOpEntry<T, LabeledTensorType> aop) {
    Expects(tensors_.find(&aop.lhs.tensor()) != tensors_.end());
    Expects(tensors_.find(&aop.rhs1.tensor()) != tensors_.end());
    Expects(tensors_.find(&aop.rhs2.tensor()) != tensors_.end());
    Expects(tensors_[&aop.rhs1.tensor()].status == TensorStatus::initialized);
    Expects(tensors_[&aop.rhs2.tensor()].status == TensorStatus::initialized);
    Expects(tensors_[&aop.lhs.tensor()].status == TensorStatus::initialized
            || (aop.mode==ResultMode::set
                && tensors_[&aop.lhs.tensor()].status==TensorStatus::allocated));
    tensors_[&aop.lhs.tensor()].status = TensorStatus::initialized;
    ops_.push_back(new MultOp<LabeledTensorType, T>(aop.alpha, aop.lhs, aop.rhs1, aop.rhs2, aop.mode, aop.exec_mode, aop.fn));
    return *this;
  }

  template<typename Func, typename LabeledTensorType>
  Scheduler& operator()(LabeledTensorType lhs, Func func, ResultMode mode = ResultMode::set) {
    Expects(tensors_.find(&lhs.tensor()) != tensors_.end());
    Expects(tensors_[&lhs.tensor()].status == TensorStatus::initialized
            || (mode==ResultMode::set
                && tensors_[&lhs.tensor()].status==TensorStatus::allocated));
    tensors_[&lhs.tensor()].status = TensorStatus::initialized;
    // ops_.push_back(new MapOp<Func,LabeledTensorType,0,0>(lhs, func, mode));
    ops_.push_back(new MapOp<LabeledTensorType, Func, 0>(lhs, func, mode));
    return *this;
  }

  template<typename Func, typename LabeledTensorType>
  Scheduler& sop(LabeledTensorType lhs, Func func) {
    Expects(tensors_.find(&lhs.tensor()) != tensors_.end());
    Expects(tensors_[&lhs.tensor()].status == TensorStatus::initialized);
    tensors_[&lhs.tensor()].status = TensorStatus::initialized;
    ops_.push_back(new ScanOp<Func,LabeledTensorType>(lhs, func));
    return *this;
  }

  void execute() {
    for(auto &op_ptr: ops_) {
      op_ptr->execute();
    }
  }

  void clear() {
    ops_.clear();
    tensors_.clear();
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
};


///////////////////////////////////////////////////////////////////////
// other operations
//////////////////////////////////////////////////////////////////////

template<typename T>
inline void
assert_zero(Scheduler& sch, Tensor<T>& tc, double threshold = 1.0e-12) {
  auto lambda = [&] (auto &val) {
    //    std::cout<<"assert_zero. val="<<val<<std::endl;
    Expects(std::abs(val) < threshold);
  };
  sch.io(tc)
      .sop(tc(), lambda)
      .execute();
}

template<typename LabeledTensorType, typename T>
inline void
assert_equal(Scheduler& sch, LabeledTensorType tc, T value, double threshold = 1.0e-12) {
  auto lambda = [&] (auto &val) {
    Expects(std::abs(val - value) < threshold);
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

template<typename T>
void
tensor_print(const Tensor<T>& tensor) {
  tensor.memory_manager()->print();
}



///////////////////////////////////////////////////////////////////////
// implementation of operators
//////////////////////////////////////////////////////////////////////

//-----------------------support routines


template<typename T>
inline TensorVec<TensorVec<TensorLabel>>
summation_labels(const LabeledTensor<T>& /*ltc*/,
                 const LabeledTensor<T>& lta,
                 const LabeledTensor<T>& ltb) {
  return group_partition(lta.tensor_->tindices(), lta.label_,
                         ltb.tensor_->tindices(), ltb.label_);

}


template<typename T>
inline std::pair<TensorVec<TensorSymmGroup>,TensorLabel>
summation_indices(const LabeledTensor<T>& /*ltc*/,
                  const LabeledTensor<T>& lta,
                  const LabeledTensor<T>& ltb) {
  auto aindices = flatten_range(lta.tensor_->tindices());
  //auto bindices = flatten(ltb.tensor_.indices());
  auto alabels = group_labels(lta.tensor_->tindices(), lta.label_);
  auto blabels = group_labels(ltb.tensor_->tindices(), ltb.label_);
  TensorVec<TensorSymmGroup> ret_indices;
  TensorLabel sum_labels;
  int apos = 0;
  for (auto &alg : alabels) {
    for (auto &blg : blabels) {
      TensorSymmGroup sg;
      size_t sg_size = 0;
      for (auto &a : alg) {
        int apos1 = 0;
        for (auto &b : blg) {
          if (a == b) {
            sg_size += 1;
            sum_labels.push_back(a);
          }
        }
        apos1++;
      }
      if (sg_size > 0) {
        ret_indices.push_back(TensorSymmGroup{aindices[apos], sg_size});
      }
    }
    apos += alg.size();
  }
  return {ret_indices, sum_labels};
}

inline TensorVec<TensorLabel>
group_labels(const TensorLabel& label,
             const TensorIndex& group_sizes) {
  TensorVec<TensorLabel> ret;
  int pos = 0;
  for(auto grp: group_sizes) {
    ret.push_back(TensorLabel(grp.value()));
    std::copy_n(label.begin() + pos, grp.value(), ret.back().begin());
    pos += grp.value();
  }
  return ret;
}

//-----------------------op execute routines

template<typename T, typename LabeledTensorType>
inline void
SetOp<T,LabeledTensorType>::execute() {
  using T1 = typename LabeledTensorType::element_type;
  // std::cerr<<"Calling setop :: execute"<<std::endl;
  auto& tensor = *lhs_.tensor_;
  auto lambda = [&] (const TensorIndex& blockid) {
    auto size = tensor.block_size(blockid);
    if(!(tensor.nonzero(blockid) && tensor.spin_unique(blockid) && size > 0)) {
      return;
    }
    auto block = tensor.alloc(blockid);
    auto tbuf = reinterpret_cast<T1*>(block.buf());
    auto value = static_cast<T1>(value_);
    for(int i=0; i<size; i++) {
      tbuf[i] = value;
    }
    if(mode_ == ResultMode::update) {
      tensor.add(block.blockid(), block);
    } else if (mode_ == ResultMode::set) {
      tensor.put(block.blockid(), block);
    } else {
      assert(0);
    }
  };
#if 0
  auto itr_first = loop_iterator(slice_indices(tensor.indices(), lhs_.label_));
#else
  auto itr_first = loop_iterator(slice_indices(tensor.tindices(), lhs_.label_));
#endif
  parallel_work(itr_first, itr_first.get_end(), lambda);
}

//--------------- new symmetrization routines

inline TensorLabel
comb_bv_to_label(const TensorVec<int>& comb_itr, const TensorLabel& tlabel) {
  assert(comb_itr.size() == tlabel.size());
  auto n = comb_itr.size();
  TensorLabel left, right;
  for(size_t i=0; i<n; i++) {
    if(comb_itr[i] == 0) {
      left.push_back(tlabel[i]);
    }
    else {
      right.push_back(tlabel[i]);
    }
  }
  TensorLabel ret{left};
  ret.insert_back(right.begin(), right.end());
  return ret;
}

/**
   @note Find first index of value in lst[lo,hi). Index hi is exclusive.
 */
template<typename T>
int find_first_index(const TensorVec<T> lst, size_t lo, size_t hi, T value) {
  int i= lo;
  for(; i<hi; i++) {
    if (lst[i] == value) {
      return i;
    }
  }
  return i;
}

/**
   @note Find last index of value in lst[lo,hi). Index hi is exclusive.
 */
template<typename T>
int find_last_index(const TensorVec<T> lst, size_t lo, size_t hi, T value) {
  int i = hi - 1;
  for( ;i >= lo; i--) {
    if (lst[i] == value) {
      return i;
    }
  }
  return i;
}

/**
   @note comb_itr is assumed to be a vector of 0s and 1s
 */
inline bool
is_unique_combination(const TensorVec<int>& comb_itr, const TensorIndex& lval) {
  Expects(std::is_sorted(lval.begin(), lval.end()));
  Expects(comb_itr.size() == lval.size());
  for(auto ci: comb_itr) {
    Expects(ci ==0 || ci == 1);
  }
  auto n = lval.size();
  size_t i = 0;
  while(i < n) {
    auto lo = i;
    auto hi = i+1;
    while(hi < n && lval[hi] == lval[lo]) {
      hi += 1;
    }
    // std::cout<<__FUNCTION__<<" lo="<<lo<<" hi="<<hi<<std::endl;
    auto first_one = find_first_index(comb_itr, lo, hi, 1);
    auto last_zero = find_last_index(comb_itr, lo, hi, 0);
    // std::cout<<__FUNCTION__<<" first_one="<<first_one<<" last_zero="<<last_zero<<std::endl;
    if(first_one < last_zero) {
      return false;
    }
    i = hi;
  }
  return true;
}

class SymmetrizerNew {
 public:
  using element_type = IndexLabel;
  SymmetrizerNew(const LabelMap<BlockDim>& lmap,
                 const TensorLabel& olabels,
                 size_t nsymm_indices)
      : lmap_{lmap},
        olabels_{olabels},
        nsymm_indices_{nsymm_indices},
        done_{false} {
          reset();
        }

  bool has_more() const {
    return !done_;
  }

  void reset() {
    done_ = false;
    auto n = olabels_.size();
    auto k = nsymm_indices_;
    Expects(k>=0 && k<=n);
    comb_itr_.resize(n);
    std::fill_n(comb_itr_.begin(), k, 0);
    std::fill_n(comb_itr_.begin()+k, n-k, 1);
    Expects(comb_itr_.size() == olabels_.size());
    olval_ = lmap_.get_blockid(olabels_);
  }

  TensorLabel get() const {
    Expects(comb_itr_.size() == olabels_.size());
    return comb_bv_to_label(comb_itr_, olabels_);
  }

  size_t itr_size() const {
    return comb_itr_.size();
  }

  void next() {
    do {
      done_ = !std::next_permutation(comb_itr_.begin(), comb_itr_.end());
    } while (!done_ && !is_unique_combination(comb_itr_, olval_));
    Expects(comb_itr_.size() == olabels_.size());
  }

 private:
  const LabelMap<BlockDim> lmap_;
  TensorLabel olabels_;
  TensorIndex olval_;
  TensorVec<int> comb_itr_;
  size_t nsymm_indices_;
  bool done_;
};


class CopySymmetrizerNew {
 public:
  using element_type = IndexLabel;
  CopySymmetrizerNew(const LabelMap<BlockDim>& lmap,
                     const TensorLabel& olabels,
                     const TensorIndex& cur_olval,
                     size_t nsymm_indices)
      : lmap_{lmap},
        olabels_{olabels},
        cur_olval_{cur_olval},
        nsymm_indices_{nsymm_indices},
        done_{false} {
          Expects(nsymm_indices>=0 && nsymm_indices<=olabels.size());
          reset();
        }

  bool has_more() const {
    return !done_;
  }

  void reset() {
    done_ = false;
    auto n = olabels_.size();
    auto k = nsymm_indices_;
    comb_itr_.resize(n);
    std::fill_n(comb_itr_.begin(), k, 0);
    std::fill_n(comb_itr_.begin()+k, n-k, 1);
    progress();
  }

  TensorLabel get() const {
    return cur_label_;
  }

  size_t itr_size() const {
    return comb_itr_.size();
  }

  void progress() {
    while(!done_) {
      cur_label_ = comb_bv_to_label(comb_itr_, olabels_);
      auto lval = lmap_.get_blockid(cur_label_);
      if(std::equal(cur_olval_.begin(), cur_olval_.end(),
                    lval.begin(), lval.end())) {
        break;
      }
      done_ = !std::next_permutation(comb_itr_.begin(), comb_itr_.end());
    }
  }

  void next() {
    done_ = !std::next_permutation(comb_itr_.begin(), comb_itr_.end());
    progress();
  }

 private:
  const LabelMap<BlockDim> lmap_;
  TensorLabel olabels_;
  TensorLabel cur_label_;
  TensorIndex cur_olval_;
  TensorVec<int> comb_itr_;
  size_t nsymm_indices_;
  bool done_;
};

inline NestedIterator<SymmetrizerNew>
symmetrization_iterator(LabelMap<BlockDim> lmap,
                        const std::vector<TensorLabel>& grps,
                        const std::vector<size_t> nsymm_indices) {
  Expects(grps.size() == nsymm_indices.size());
  std::vector<SymmetrizerNew> symms;
  for(size_t i=0; i<grps.size(); i++) {
    symms.emplace_back(lmap, grps[i], nsymm_indices[i]);
  }
  return {symms};
}

inline NestedIterator<CopySymmetrizerNew>
copy_symmetrization_iterator(LabelMap<BlockDim> lmap,
                             const std::vector<TensorLabel>& grps,
                             const std::vector<TensorIndex>& lvals,
                             const std::vector<size_t> nsymm_indices) {
  Expects(grps.size() == nsymm_indices.size());
  std::vector<CopySymmetrizerNew> symms;
  for(size_t i=0; i<grps.size(); i++) {
    symms.emplace_back(lmap, grps[i], lvals[i], nsymm_indices[i]);
  }
  return {symms};
}

/**
   @note assume one group in output (ltc) is atmost split into two
   groups in input (lta) and that, for given group, there is either
   symmetrization or unsymmetrization, but not both.
 */
template<typename LabeledTensorType>
inline NestedIterator<SymmetrizerNew>
symmetrization_iterator(LabelMap<BlockDim>& lmap,
                        const LabeledTensorType& ltc,
                        const LabeledTensorType& lta) {
  std::vector<TensorLabel> cgrps_vec;
  auto cgrps = group_labels(ltc.tensor_->tindices(),
                            ltc.label_);
  for(const auto& cgrp: cgrps) {
    cgrps_vec.push_back(cgrp);
  }

  auto cgrp_parts = group_partition(ltc.tensor_->tindices(),
                                    ltc.label_,
                                    lta.tensor_->tindices(),
                                    lta.label_);
  std::vector<size_t> nsymm_indices;
  for(const auto& csgp: cgrp_parts) {
    Expects(csgp.size() >=0 && csgp.size() <= 2);
    nsymm_indices.push_back(csgp[0].size());
  }
  return symmetrization_iterator(lmap, cgrps_vec, nsymm_indices);
}

template<typename LabeledTensorType>
inline NestedIterator<SymmetrizerNew>
symmetrization_iterator(LabelMap<BlockDim>& lmap,
                        const LabeledTensorType& ltc,
                        const LabeledTensorType& lta,
                        const LabeledTensorType& ltb) {
  std::vector<TensorLabel> cgrps_vec;
  auto cgrps = group_labels(ltc.tensor_->tindices(),
                            ltc.label_);
  for(const auto& cgrp: cgrps) {
    cgrps_vec.push_back(cgrp);
  }

  auto cgrp_parts = group_partition(ltc.tensor_->tindices(),
                                    ltc.label_,
                                    lta.tensor_->tindices(),
                                    lta.label_,
                                    ltb.tensor_->tindices(),
                                    ltb.label_);
  std::vector<size_t> nsymm_indices;
  for(const auto& csgp: cgrp_parts) {
    Expects(csgp.size() >=0 && csgp.size() <= 2);
    nsymm_indices.push_back(csgp[0].size());
  }
  return symmetrization_iterator(lmap, cgrps_vec, nsymm_indices);
}

/**
   @note assume one group in output (ltc) is atmost split into two
   groups in input (lta) and that, for given group, there is either
   symmetrization or unsymmetrization, but not both.
 */
template<typename LabeledTensorType>
inline NestedIterator<CopySymmetrizerNew>
copy_symmetrization_iterator(LabelMap<BlockDim>& lmap,
                             const LabeledTensorType& ltc,
                             const LabeledTensorType& lta,
                             const TensorIndex& cur_clval) {
  std::vector<TensorLabel> cgrps_vec;
  auto cgrps = group_labels(ltc.tensor_->tindices(),
                            ltc.label_);
  for(const auto& cgrp: cgrps) {
    cgrps_vec.push_back(cgrp);
  }

  auto cgrp_parts = group_partition(ltc.tensor_->tindices(),
                                    ltc.label_,
                                    lta.tensor_->tindices(),
                                    lta.label_);
  std::vector<size_t> nsymm_indices;
  for(const auto& csgp: cgrp_parts) {
    Expects(csgp.size() >=0 && csgp.size() <= 2);
    nsymm_indices.push_back(csgp[0].size());
  }
  std::vector<TensorIndex> clvals;
  int i = 0;
  for(const auto& csg: ltc.tensor_->tindices()) {
    clvals.push_back(TensorIndex{cur_clval.begin()+i,
            cur_clval.begin()+i+csg.size()});
    i += csg.size();
  }

  return copy_symmetrization_iterator(lmap, cgrps_vec, clvals, nsymm_indices);
}

template<typename LabeledTensorType>
inline NestedIterator<CopySymmetrizerNew>
copy_symmetrization_iterator(LabelMap<BlockDim>& lmap,
                             const LabeledTensorType& ltc,
                             const LabeledTensorType& lta,
                             const LabeledTensorType& ltb,
                             const TensorIndex& cur_clval) {
  std::vector<TensorLabel> cgrps_vec;
  auto cgrps = group_labels(ltc.tensor_->tindices(),
                            ltc.label_);
  for(const auto& cgrp: cgrps) {
    cgrps_vec.push_back(cgrp);
  }

  auto cgrp_parts = group_partition(ltc.tensor_->tindices(),
                                    ltc.label_,
                                    lta.tensor_->tindices(),
                                    lta.label_,
                                    ltb.tensor_->tindices(),
                                    ltb.label_);
  std::vector<size_t> nsymm_indices;
  for(const auto& csgp: cgrp_parts) {
    Expects(csgp.size() >=0 && csgp.size() <= 2);
    nsymm_indices.push_back(csgp[0].size());
  }
  std::vector<TensorIndex> clvals;
  int i = 0;
  for(const auto& csg: ltc.tensor_->tindices()) {
    clvals.push_back(TensorIndex{cur_clval.begin()+i,
            cur_clval.begin()+i+csg.size()});
    i += csg.size();
  }

  return copy_symmetrization_iterator(lmap, cgrps_vec, clvals, nsymm_indices);
}

template<typename LabeledTensorType>
double
compute_symmetrization_factor(const LabeledTensorType& ltc,
                              const LabeledTensorType& lta) {
  auto cgrp_parts = group_partition(ltc.tensor_->tindices(),
                                    ltc.label_,
                                    lta.tensor_->tindices(),
                                    lta.label_);
  Expects(cgrp_parts.size() == ltc.tensor_->tindices().size());
  for(size_t i=0; i<cgrp_parts.size(); i++) {
    Expects(cgrp_parts[i].size()  <= 2);
  }
  double ret = 1.0;
  for(const auto& csgp: cgrp_parts) {
    Expects(csgp.size() >=0 && csgp.size() <= 2);
    int n = csgp[0].size() + csgp[1].size();
    int r = csgp[0].size();
    ret *= factorial(n) / (factorial(r) * factorial(n-r));
  }
  return 1.0/ret;
}

template<typename LabeledTensorType>
double
compute_symmetrization_factor(const LabeledTensorType& ltc,
                              const LabeledTensorType& lta,
                              const LabeledTensorType& ltb) {
  auto cgrp_parts = group_partition(ltc.tensor_->tindices(),
                                     ltc.label_,
                                     lta.tensor_->tindices(),
                                     lta.label_,
                                     ltb.tensor_->tindices(),
                                     ltb.label_);
  Expects(cgrp_parts.size() == ltc.tensor_->tindices().size());
  for(size_t i=0; i<cgrp_parts.size(); i++) {
    Expects(cgrp_parts[i].size() <= 2);
  }
  double ret = 1.0;
  for(const auto& csgp: cgrp_parts) {
    Expects(csgp.size() >=0 && csgp.size() <= 2);
    int n = csgp[0].size() + csgp[1].size();
    int r = csgp[0].size();
    ret *= factorial(n) / (factorial(r) * factorial(n-r));
  }
  return 1.0/ret;
}

extern Integer *int_mb_tammx;
extern double *dbl_mb_tammx;


inline Integer*
int_mb() {
  //assert(0); //@todo implement
  return int_mb_tammx;
}

template<typename T>
std::pair<Integer, Integer *>
tensor_to_fortran_info(tammx::Tensor<T> &ttensor) {
  bool t_is_double = std::is_same<T, double>::value;
  Expects(t_is_double);
  auto adst_nw = static_cast<const tammx::Distribution_NW *>(ttensor.distribution());
  auto ahash = adst_nw->hash();
  auto length = 2 * ahash[0] + 1;
  Integer *offseta = new Integer[length];
  for (size_t i = 0; i < length; i++) {
    offseta[i] = ahash[i];
  }

  auto amgr_ga = static_cast<tammx::MemoryManagerGA *>(ttensor.memory_manager());
  Integer da = amgr_ga->ga();
  return {da, offseta};
}

// symmetrization or unsymmetrization, but not both in one symmetry group
template<typename T, typename LabeledTensorType>
inline void
AddOp<T, LabeledTensorType>::execute() {
  using T1 = typename LabeledTensorType::element_type;

  //std::cout<<"ADD_OP. C"<<lhs_.label_<<" += "<<alpha_<<" * A"<<rhs_.label_<<"\n";
  
  //tensor_print(*lhs_.tensor_);
  //tensor_print(*rhs_.tensor_);
  if(exec_mode_ == ExecutionMode::fortran) {
    bool t1_is_double = std::is_same<T1, double>::value;
    Expects(t1_is_double);
    Expects(fn_ != nullptr);

    Integer da, *offseta_map;
    Integer dc, *offsetc_map;
    std::tie(da, offseta_map) = tensor_to_fortran_info(*rhs_.tensor_);
    std::tie(dc, offsetc_map) = tensor_to_fortran_info(*lhs_.tensor_);
    Integer offseta = offseta_map - int_mb();
    Integer offsetc = offsetc_map - int_mb();

    fn_(&da, &offseta, &dc, &offsetc);

    //tensor_print(*lhs_.tensor_);

    delete[] offseta_map;
    delete[] offsetc_map;
    return;
  }
  const LabeledTensor<T1>& lta = rhs_;
  const LabeledTensor<T1>& ltc = lhs_;
  const auto &clabel = ltc.label_;
  const auto &alabel = lta.label_;
  Tensor<T1>& ta = *lta.tensor_;
  Tensor<T1>& tc = *ltc.tensor_;
  double symm_factor = compute_symmetrization_factor(ltc, lta);
  //std::cout<<"===symm factor="<<symm_factor<<std::endl;
#if 0
  auto citr = loop_iterator(slice_indices(tc.indices(), ltc.label_));
#else
  auto citr = loop_iterator(slice_indices(tc.tindices(), ltc.label_));
#endif
  auto lambda = [&] (const TensorIndex& cblockid) {
    //std::cout<<"---tammx assign. cblockid"<<cblockid<<std::endl;
    size_t dimc = tc.block_size(cblockid);
    if(!(tc.nonzero(cblockid) && tc.spin_unique(cblockid) && dimc > 0)) {
      return;
    }
    auto cbp = tc.alloc(cblockid);
    cbp() = 0;
    //std::cout<<"---tammx assign. ACTION ON cblockid"<<cblockid<<std::endl;
    auto label_map = LabelMap<BlockDim>().update(ltc.label_, cblockid);
    auto sit = symmetrization_iterator(label_map,ltc, lta);
    for(; sit.has_more(); sit.next()) {
      TensorLabel cur_clbl = sit.get();
      //std::cout<<"ACTION cur_clbl="<<cur_clbl<<std::endl;
      auto cur_cblockid = label_map.get_blockid(cur_clbl);
      //std::cout<<"---tammx assign. ACTION cur_cblockid"<<cur_cblockid<<std::endl;
      auto ablockid = LabelMap<BlockDim>().update(ltc.label_, cur_cblockid).get_blockid(alabel);
      auto abp = ta.get(ablockid);
      //std::cout<<"ACTION ablockid="<<ablockid<<std::endl;
      //std::cout<<"ACTION symm_factor="<<symm_factor<<std::endl;

      auto csbp = tc.alloc(cur_cblockid);
      csbp() = 0;
      csbp(clabel) += alpha_ * symm_factor * abp(alabel);

      auto csit = copy_symmetrization_iterator(label_map, ltc, lta, cur_cblockid);
      for(TensorLabel csym_clbl = csit.get(); csit.has_more(); csit.next(), csym_clbl = csit.get()) {
        // int csym_sign = (perm_count_inversions(perm_compute(cur_clbl, csym_clbl)) % 2) ? -1 : 1;
        int csym_sign = (perm_count_inversions(perm_compute(csym_clbl, clabel)) % 2) ? -1 : 1;
        //std::cout<<"===csym sign="<<csym_sign<<std::endl;
        //std::cout<<"===clabel="<<clabel<<" csym label="<<csym_clbl<<std::endl;
        cbp(clabel) += csym_sign * csbp(csym_clbl);
      }
    }
    if(mode_ == ResultMode::update) {
      tc.add(cblockid, cbp);
    } else {
      tc.put(cblockid, cbp);
    }
  };
  parallel_work(citr, citr.get_end(), lambda);
  //tensor_print(*lhs_.tensor_);
}


inline int
compute_symmetry_scaling_factor(const TensorVec<TensorSymmGroup>& sum_indices,
                                TensorIndex sumid) {
  int ret = 1;
  auto itr = sumid.begin();
  for(auto &sg: sum_indices) {
    auto sz = sg.size();
    Expects(sz > 0);
    std::sort(itr, itr+sz);
    auto fact = factorial(sz);
    int tsize = 1;
    for(int i=1; i<sz; i++) {
      if(itr[i] != itr[i-1]) {
        fact /= factorial(tsize);
        tsize = 0;
      }
      tsize += 1;
    }
    fact /= factorial(tsize);
    ret *= fact;
    itr += sz;
  }
  return ret;
}

template<typename T, typename LabeledTensorType>
inline void
MultOp<T, LabeledTensorType>::execute() {
  using T1 = typename LabeledTensorType::element_type;
  // std::cout<<"MULT_OP. C"<<lhs_.label_<<" += "<<alpha_
  //          <<" * A"<<rhs1_.label_
  //          <<" * B"<<rhs2_.label_
  //          <<"\n";

  //tensor_print(*lhs_.tensor_);
  //tensor_print(*rhs1_.tensor_);
  //tensor_print(*rhs2_.tensor_);

  if(exec_mode_ == ExecutionMode::fortran) {
    bool t1_is_double = std::is_same<T1, double>::value;
    Expects(t1_is_double);
    Expects(fn_ != nullptr);

    Integer da, *offseta_map;
    Integer db, *offsetb_map;
    Integer dc, *offsetc_map;
    std::tie(da, offseta_map) = tensor_to_fortran_info(*rhs1_.tensor_);
    std::tie(db, offsetb_map) = tensor_to_fortran_info(*rhs2_.tensor_);
    std::tie(dc, offsetc_map) = tensor_to_fortran_info(*lhs_.tensor_);
    Integer offseta = offseta_map - int_mb();
    Integer offsetb = offsetb_map - int_mb();
    Integer offsetc = offsetc_map - int_mb();

    //std::cout<<"---------INVOKING FORTRAN MULT----------\n";
    Integer zero = 0;
    fn_(&da, &offseta, &db, &offsetb, &dc, &offsetc);
    //tensor_print(*lhs_.tensor_);

    delete[] offseta_map;
    delete[] offsetb_map;
    delete[] offsetc_map;
    return;
  }

  //@todo @fixme MultOp based on nonsymmetrized_iterator cannot work with ResultMode::set
  //Expects(mode_ == ResultMode::update);
  LabeledTensor<T1>& lta = rhs1_;
  LabeledTensor<T1>& ltb = rhs2_;
  LabeledTensor<T1>& ltc = lhs_;
  const auto &clabel = ltc.label_;
  const auto &alabel = lta.label_;
  const auto &blabel = ltb.label_;
  Tensor<T1>& ta = *lta.tensor_;
  Tensor<T1>& tb = *ltb.tensor_;
  Tensor<T1>& tc = *ltc.tensor_;

  double symm_factor = 1; //compute_symmetrization_factor(ltc, lta, ltb);

  TensorLabel sum_labels;
  TensorVec<TensorSymmGroup> sum_indices;
  std::tie(sum_indices, sum_labels) = summation_indices(ltc, lta, ltb);
  auto lambda = [&] (const TensorIndex& cblockid) {
    auto dimc = tc.block_size(cblockid);
    if(!(tc.nonzero(cblockid) && tc.spin_unique(cblockid) && dimc > 0)) {
      // std::cout<<"MultOp. zero block "<<cblockid<<std::endl;
      // std::cout<<"MultOp. "<<cblockid<<" nonzero="<< tc.nonzero(cblockid) <<std::endl;
      // std::cout<<"MultOp. "<<cblockid<<" spin_unique="<< tc.spin_unique(cblockid) <<std::endl;
      // std::cout<<"MultOp. "<<cblockid<<" dimc="<< dimc <<std::endl;
      return;
    }
    auto cbp = tc.alloc(cblockid);
    cbp() = 0;
    //std::cout<<"MultOp. non-zero block"<<cblockid<<std::endl;
    auto label_map_outer = LabelMap<BlockDim>().update(ltc.label_, cblockid);
    auto sit = symmetrization_iterator(label_map_outer,ltc, lta, ltb);
    for(; sit.has_more(); sit.next()) {
      TensorLabel cur_clbl = sit.get();
      auto cur_cblockid = label_map_outer.get_blockid(cur_clbl);
      //std::cout<<"MultOp. cur_cblock"<<cur_cblockid<<std::endl;
      //std::cout<<"MultOp. cur_cblock size="<<dimc<<std::endl;

      auto sum_itr_first = loop_iterator(slice_indices(sum_indices, sum_labels));
      auto sum_itr_last = sum_itr_first.get_end();
      auto label_map = LabelMap<BlockDim>().update(ltc.label_, cur_cblockid);

      for(auto sitr = sum_itr_first; sitr!=sum_itr_last; ++sitr) {
        label_map.update(sum_labels, *sitr);
        auto ablockid = label_map.get_blockid(lta.label_);
        auto bblockid = label_map.get_blockid(ltb.label_);

        // std::cout<<"--summation loop. value="<<*sitr<<std::endl;

        //std::cout<<"--MultOp. ablockid"<<ablockid<<std::endl;
        //std::cout<<"--MultOp. bblockid"<<bblockid<<std::endl;
        if(!ta.nonzero(ablockid) || !tb.nonzero(bblockid)) {
          continue;
        }
        //std::cout<<"--MultOp. nonzero ablockid"<<ablockid<<std::endl;
        //std::cout<<"--MultOp. nonzero bblockid"<<bblockid<<std::endl;
        auto abp = ta.get(ablockid);
        auto bbp = tb.get(bblockid);
        // std::cout<<"--MultOp. a blocksize="<<abp.size()<<std::endl;
        // std::cout<<"--MultOp. b blocksize="<<bbp.size()<<std::endl;
        // std::cout<<"A=";
        // for(size_t i=0; i<abp.size(); i++) {
        //   std::cout<<abp.buf()[i]<<" ";
        // }
        // std::cout<<"\n";
        // std::cout<<"B=";
        // for(size_t i=0; i<bbp.size(); i++) {
        //   std::cout<<bbp.buf()[i]<<" ";
        // }
        // std::cout<<"\n";

        auto symm_scaling_factor = compute_symmetry_scaling_factor(sum_indices, *sitr);
        auto scale = alpha_ * symm_factor * symm_scaling_factor;
        //std::cout<<"--MultOp. symm_factor="<<symm_factor<<"  symm_scaling_factor="<<symm_scaling_factor<<std::endl;

        auto csbp = tc.alloc(cur_cblockid);
        csbp() = 0.0;
#if 1
      // std::cout<<"doing block-block multiply"<<std::endl;
        csbp(ltc.label_) += scale * abp(lta.label_) * bbp(ltb.label_);
#endif
        // std::cout<<"CSBP=";
        // for(size_t i=0; i<csbp.size(); i++) {
        //   std::cout<<csbp.buf()[i]<<" ";
        // }
        // std::cout<<"\n";

        auto csit = copy_symmetrization_iterator(label_map_outer, ltc, lta, ltb, cur_cblockid);
        for(; csit.has_more(); csit.next()) {
          TensorLabel csym_clbl = csit.get();
          // int csym_sign = (perm_count_inversions(perm_compute(cur_clbl, csym_clbl)) % 2) ? -1 : 1;
          // int csym_sign = (perm_count_inversions(perm_compute(csym_clbl, clabel)) % 2) ? -1 : 1;
          int csym_sign = (perm_count_inversions(perm_compute(clabel, csym_clbl)) % 2) ? -1 : 1;
          //std::cout<<"===csym sign="<<csym_sign<<std::endl;
          //std::cout<<"===clabel="<<clabel<<" csym label="<<csym_clbl<<std::endl;
          cbp(clabel) += csym_sign * csbp(csym_clbl);
          // std::cout<<"CBP=";
          // for(size_t i=0; i<cbp.size(); i++) {
          //   std::cout<<cbp.buf()[i]<<" ";
          // }
          // std::cout<<"\n";
        }
      }
    }
    if(mode_ == ResultMode::update) {
      tc.add(cblockid, cbp);
    } else {
      tc.put(cblockid, cbp);
    }
  };
#if 0
  auto citr = loop_iterator(slice_indices(tc.indices(), ltc.label_));
#else
  auto citr = loop_iterator(slice_indices(tc.tindices(), ltc.label_));
#endif
  parallel_work(citr, citr.get_end(), lambda);
  //tensor_print(*lhs_.tensor_);
}

} // namespace tammx

#endif  // TAMMX_OPS_H_
