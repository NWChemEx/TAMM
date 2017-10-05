#ifndef TAMMX_OPS_H_
#define TAMMX_OPS_H_

#include "tammx/tensor.h"
#include "tammx/labeled_block.h"
#include "tammx/labeled_tensor.h"
#include "tammx/util.h"
#include "tammx/fortran.h"

namespace tammx {

class ExecutionContext;

/////////////////////////////////////////////////////////////////////
//         operators
/////////////////////////////////////////////////////////////////////

class Op {
 public:
  virtual void execute(const ProcGroup& ec_pg) = 0;
  virtual TensorImpl* writes() const = 0;
  virtual std::vector<TensorImpl*> reads() const = 0;
  virtual ~Op() {}
};


template<typename T, typename LabeledTensorType>
struct SetOp : public Op {
  void execute(const ProcGroup& ec_pg) override;

  SetOp(T value, LabeledTensorType& lhs, ResultMode mode)
      : value_{value},
        lhs_{lhs},
        mode_{mode} {}

  TensorImpl* writes() const override {
    return lhs_.tensor_;
  }

  std::vector<TensorImpl*> reads() const {
    return {};
  }

  T value_;
  LabeledTensorType lhs_;
  ResultMode mode_;
};

template<typename T, typename LabeledTensorType>
struct AddOp : public Op {
  void execute(const ProcGroup& ec_pg) override;

  AddOp(T alpha, const LabeledTensorType& lhs, const LabeledTensorType& rhs, ResultMode mode,
        ExecutionMode exec_mode,
        add_fn* fn)
      : alpha_{alpha},
        lhs_{lhs},
        rhs_{rhs},
        mode_{mode},
        exec_mode_{exec_mode},
        fn_{fn} { }

  TensorImpl* writes() const override {
    return lhs_.tensor_;
  }

  std::vector<TensorImpl*> reads() const {
    return {rhs_.tensor_};
  }

  T alpha_;
  LabeledTensorType lhs_, rhs_;
  ResultMode mode_;
  ExecutionMode exec_mode_;
  add_fn* fn_;
};

template<typename T, typename LabeledTensorType>
struct MultOp : public Op {
  void execute(const ProcGroup& ec_pg) override;

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

  TensorImpl* writes() const override {
    return lhs_.tensor_;
  }

  std::vector<TensorImpl*> reads() const {
    return {rhs1_.tensor_, rhs2_.tensor_};
  }

  T alpha_;
  LabeledTensorType lhs_, rhs1_, rhs2_;
  ResultMode mode_;
  ExecutionMode exec_mode_;
  mult_fn* fn_;
};

template<typename TensorType>
struct AllocOp: public Op {
  void execute(const ProcGroup& ec_pg) override {
    EXPECTS(pg_.is_valid());
    tensor_->alloc(distribution_, memory_manager_);
  }

  AllocOp(TensorType& tensor, ProcGroup pg, Distribution* distribution, MemoryManager* memory_manager)
      : tensor_{&tensor},
        pg_{pg},
        distribution_{distribution},
        memory_manager_{memory_manager} {}

  TensorImpl* writes() const override {
    return tensor_;
  }

  std::vector<TensorImpl*> reads() const {
    return {};
  }

  TensorType *tensor_;
  ProcGroup pg_;
  Distribution* distribution_;
  MemoryManager* memory_manager_;
};

template<typename TensorType>
struct DeallocOp: public Op {
  void execute(const ProcGroup& ec_pg) override {
    tensor_->dealloc();
  }

  DeallocOp(TensorType* tensor)
      : tensor_{tensor} {}

  TensorImpl* writes() const override {
    return tensor_;
  }

  std::vector<TensorImpl*> reads() const {
    return {};
  }

  TensorType *tensor_;
};


template<typename LabeledTensorType, typename Func, int N>
struct MapOp : public Op {
  using RHS = std::array<LabeledTensorType, N>;
  using T = typename LabeledTensorType::element_type;
  using RHS_Blocks = std::array<Block<T>, N>;

  void execute(const ProcGroup& ec_pg) override {
    auto &lhs_tensor = *lhs_.tensor_;
    auto lambda = [&] (const BlockDimVec& blockid) {
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
    if(ec_pg.size() > lhs_tensor.tensor_->pg().size()) {
      parallel_work(lhs_tensor.tensor_->pg(), itr_first, itr_first.get_end(), lambda);
    } else {
      parallel_work(ec_pg, itr_first, itr_first.get_end(), lambda);
    }
    //parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  MapOp(LabeledTensorType& lhs, Func func, RHS& rhs, ResultMode mode = ResultMode::set)
      : lhs_{lhs},
        func_{func},
        rhs_{rhs},
        mode_{mode} {}

  RHS_Blocks get_blocks(RHS& rhs, const BlockDimVec& id) {
    RHS_Blocks blocks;
    for(int i=0; i<rhs.size(); i++) {
      blocks[i] = rhs[i].get(id);
    }
    return blocks;
  }

  TensorImpl* writes() const override {
    return lhs_.tensor_;
  }

  std::vector<TensorImpl*> reads() const {
    std::vector<TensorImpl*> ret;
    for(auto& lt: rhs_) {
      ret.push_back(lt.tensor_);
    }
    return ret;
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

  void execute(const ProcGroup& ec_pg) override {
    auto &lhs_tensor = *lhs_.tensor_;
    auto lambda = [&] (const BlockDimVec& blockid) {
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
    if(ec_pg.size() > lhs_tensor.tensor_->pg().size()) {
      parallel_work(lhs_tensor.tensor_->pg(), itr_first, itr_first.get_end(), lambda);
    } else {
      parallel_work(ec_pg, itr_first, itr_first.get_end(), lambda);
    }
    //parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  MapIdOp(LabeledTensorType& lhs, Func func, RHS& rhs, ResultMode mode = ResultMode::set)
      : lhs_{lhs},
        func_{func},
        rhs_{rhs},
        mode_{mode} {}

  RHS_Blocks get_blocks(RHS& rhs, const BlockDimVec& id) {
    RHS_Blocks blocks;
    for(int i=0; i<rhs.size(); i++) {
      blocks[i] = rhs[i].get(id);
    }
    return blocks;
  }

  TensorImpl* writes() const override {
    return lhs_.tensor_;
  }

  std::vector<TensorImpl*> reads() const {
    std::vector<TensorImpl*> ret;
    for(auto& lt: rhs_) {
      ret.push_back(lt.tensor_);
    }
    return ret;
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
  void execute(const ProcGroup& ec_pg) {
    // std::cerr<<__FUNCTION__<<":"<<__LINE__<<": ScanOp\n";
    auto& tensor = *ltensor_.tensor_;
    auto lambda = [&] (const BlockDimVec& blockid) {
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
    if(ec_pg.size() > tensor->pg().size()) {
      parallel_work(tensor->pg(), itr_first, itr_first.get_end(), lambda);
    } else {
      parallel_work(ec_pg, itr_first, itr_first.get_end(), lambda);
    }
    //parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  ScanOp(const LabeledTensorType& ltensor, Func func)
      : ltensor_{ltensor},
        func_{func} {
    EXPECTS(ltensor.tensor_ != nullptr);
  }

  TensorImpl* writes() const override {
    return ltensor_.tensor_;
  }

  std::vector<TensorImpl*> reads() const {
    return {};
  }

  LabeledTensorType ltensor_;
  Func func_;
};



///////////////////////////////////////////////////////////////////////
// other operations
//////////////////////////////////////////////////////////////////////

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
inline TensorVec<TensorVec<IndexLabelVec>>
summation_labels(const LabeledTensor<T>& /*ltc*/,
                 const LabeledTensor<T>& lta,
                 const LabeledTensor<T>& ltb) {
  return group_partition(lta.tensor_->tindices(), lta.label_,
                         ltb.tensor_->tindices(), ltb.label_);

}


template<typename T>
inline std::pair<TensorVec<TensorSymmGroup>,IndexLabelVec>
summation_indices(const LabeledTensor<T>& /*ltc*/,
                  const LabeledTensor<T>& lta,
                  const LabeledTensor<T>& ltb) {
  auto aindices = flatten_range(lta.tensor_->tindices());
  //auto bindices = flatten(ltb.tensor_.indices());
  auto alabels = group_labels(lta.tensor_->tindices(), lta.label_);
  auto blabels = group_labels(ltb.tensor_->tindices(), ltb.label_);
  TensorVec<TensorSymmGroup> ret_indices;
  IndexLabelVec sum_labels;
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

inline TensorVec<IndexLabelVec>
group_labels(const IndexLabelVec& label,
             const BlockDimVec& group_sizes) {
  TensorVec<IndexLabelVec> ret;
  int pos = 0;
  for(auto grp: group_sizes) {
    ret.push_back(IndexLabelVec(grp.value()));
    std::copy_n(label.begin() + pos, grp.value(), ret.back().begin());
    pos += grp.value();
  }
  return ret;
}

//-----------------------op execute routines

template<typename T, typename LabeledTensorType>
inline void
SetOp<T,LabeledTensorType>::execute(const ProcGroup& ec_pg) {
  using T1 = typename LabeledTensorType::element_type;
  // std::cerr<<"Calling setop :: execute"<<std::endl;
  auto& tensor = *lhs_.tensor_;
  auto lambda = [&] (const BlockDimVec& blockid) {
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
  if(ec_pg.size() > lhs_.tensor_->pg().size()) {
    parallel_work(lhs_.tensor_->pg(), itr_first, itr_first.get_end(), lambda);
  } else {
    parallel_work(ec_pg, itr_first, itr_first.get_end(), lambda);
  }
  //parallel_work(ec_pg, itr_first, itr_first.get_end(), lambda);
}

//--------------- new symmetrization routines

inline IndexLabelVec
comb_bv_to_label(const TensorVec<int>& comb_itr, const IndexLabelVec& tlabel) {
  assert(comb_itr.size() == tlabel.size());
  auto n = comb_itr.size();
  IndexLabelVec left, right;
  for(size_t i=0; i<n; i++) {
    if(comb_itr[i] == 0) {
      left.push_back(tlabel[i]);
    }
    else {
      right.push_back(tlabel[i]);
    }
  }
  IndexLabelVec ret{left};
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
int find_last_index(const TensorVec<T> lst, int lo, int hi, T value) {
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
is_unique_combination(const TensorVec<int>& comb_itr, const BlockDimVec& lval) {
  EXPECTS(std::is_sorted(lval.begin(), lval.end()));
  EXPECTS(comb_itr.size() == lval.size());
  for(auto ci: comb_itr) {
    EXPECTS(ci ==0 || ci == 1);
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
  SymmetrizerNew(const LabelMap<BlockIndex>& lmap,
                 const IndexLabelVec& olabels,
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
    EXPECTS(k>=0 && k<=n);
    comb_itr_.resize(n);
    std::fill_n(comb_itr_.begin(), k, 0);
    std::fill_n(comb_itr_.begin()+k, n-k, 1);
    EXPECTS(comb_itr_.size() == olabels_.size());
    olval_ = lmap_.get_blockid(olabels_);
  }

  IndexLabelVec get() const {
    EXPECTS(comb_itr_.size() == olabels_.size());
    return comb_bv_to_label(comb_itr_, olabels_);
  }

  size_t itr_size() const {
    return comb_itr_.size();
  }

  void next() {
    do {
      done_ = !std::next_permutation(comb_itr_.begin(), comb_itr_.end());
    } while (!done_ && !is_unique_combination(comb_itr_, olval_));
    EXPECTS(comb_itr_.size() == olabels_.size());
  }

 private:
  const LabelMap<BlockIndex> lmap_;
  IndexLabelVec olabels_;
  BlockDimVec olval_;
  TensorVec<int> comb_itr_;
  size_t nsymm_indices_;
  bool done_;
};


class CopySymmetrizerNew {
 public:
  using element_type = IndexLabel;
  CopySymmetrizerNew(const LabelMap<BlockIndex>& lmap,
                     const IndexLabelVec& olabels,
                     const BlockDimVec& cur_olval,
                     size_t nsymm_indices)
      : lmap_{lmap},
        olabels_{olabels},
        cur_olval_{cur_olval},
        nsymm_indices_{nsymm_indices},
        done_{false} {
          EXPECTS(nsymm_indices>=0 && nsymm_indices<=olabels.size());
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

  IndexLabelVec get() const {
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
  const LabelMap<BlockIndex> lmap_;
  IndexLabelVec olabels_;
  IndexLabelVec cur_label_;
  BlockDimVec cur_olval_;
  TensorVec<int> comb_itr_;
  size_t nsymm_indices_;
  bool done_;
};

inline NestedIterator<SymmetrizerNew>
symmetrization_iterator(LabelMap<BlockIndex> lmap,
                        const std::vector<IndexLabelVec>& grps,
                        const std::vector<size_t> nsymm_indices) {
  EXPECTS(grps.size() == nsymm_indices.size());
  std::vector<SymmetrizerNew> symms;
  for(size_t i=0; i<grps.size(); i++) {
    symms.emplace_back(lmap, grps[i], nsymm_indices[i]);
  }
  return {symms};
}

inline NestedIterator<CopySymmetrizerNew>
copy_symmetrization_iterator(LabelMap<BlockIndex> lmap,
                             const std::vector<IndexLabelVec>& grps,
                             const std::vector<BlockDimVec>& lvals,
                             const std::vector<size_t> nsymm_indices) {
  EXPECTS(grps.size() == nsymm_indices.size());
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
symmetrization_iterator(LabelMap<BlockIndex>& lmap,
                        const LabeledTensorType& ltc,
                        const LabeledTensorType& lta) {
  std::vector<IndexLabelVec> cgrps_vec;
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
    EXPECTS(csgp.size() >=0 && csgp.size() <= 2);
    nsymm_indices.push_back(csgp[0].size());
  }
  return symmetrization_iterator(lmap, cgrps_vec, nsymm_indices);
}

template<typename LabeledTensorType>
inline NestedIterator<SymmetrizerNew>
symmetrization_iterator(LabelMap<BlockIndex>& lmap,
                        const LabeledTensorType& ltc,
                        const LabeledTensorType& lta,
                        const LabeledTensorType& ltb) {
  std::vector<IndexLabelVec> cgrps_vec;
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
    EXPECTS(csgp.size() >=0 && csgp.size() <= 2);
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
copy_symmetrization_iterator(LabelMap<BlockIndex>& lmap,
                             const LabeledTensorType& ltc,
                             const LabeledTensorType& lta,
                             const BlockDimVec& cur_clval) {
  std::vector<IndexLabelVec> cgrps_vec;
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
    EXPECTS(csgp.size() >=0 && csgp.size() <= 2);
    nsymm_indices.push_back(csgp[0].size());
  }
  std::vector<BlockDimVec> clvals;
  int i = 0;
  for(const auto& csg: ltc.tensor_->tindices()) {
    clvals.push_back(BlockDimVec{cur_clval.begin()+i,
            cur_clval.begin()+i+csg.size()});
    i += csg.size();
  }

  return copy_symmetrization_iterator(lmap, cgrps_vec, clvals, nsymm_indices);
}

template<typename LabeledTensorType>
inline NestedIterator<CopySymmetrizerNew>
copy_symmetrization_iterator(LabelMap<BlockIndex>& lmap,
                             const LabeledTensorType& ltc,
                             const LabeledTensorType& lta,
                             const LabeledTensorType& ltb,
                             const BlockDimVec& cur_clval) {
  std::vector<IndexLabelVec> cgrps_vec;
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
    EXPECTS(csgp.size() >=0 && csgp.size() <= 2);
    nsymm_indices.push_back(csgp[0].size());
  }
  std::vector<BlockDimVec> clvals;
  int i = 0;
  for(const auto& csg: ltc.tensor_->tindices()) {
    clvals.push_back(BlockDimVec{cur_clval.begin()+i,
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
  EXPECTS(cgrp_parts.size() == ltc.tensor_->tindices().size());
  for(size_t i=0; i<cgrp_parts.size(); i++) {
    EXPECTS(cgrp_parts[i].size()  <= 2);
  }
  double ret = 1.0;
  for(const auto& csgp: cgrp_parts) {
    EXPECTS(csgp.size() >=0 && csgp.size() <= 2);
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
  EXPECTS(cgrp_parts.size() == ltc.tensor_->tindices().size());
  for(size_t i=0; i<cgrp_parts.size(); i++) {
    EXPECTS(cgrp_parts[i].size() <= 2);
  }
  double ret = 1.0;
  for(const auto& csgp: cgrp_parts) {
    EXPECTS(csgp.size() >=0 && csgp.size() <= 2);
    int n = csgp[0].size() + csgp[1].size();
    int r = csgp[0].size();
    ret *= factorial(n) / (factorial(r) * factorial(n-r));
  }
  return 1.0/ret;
}

template<typename T>
std::pair<FortranInt, FortranInt *>
tensor_to_fortran_info(tammx::Tensor<T> &ttensor) {
  bool t_is_double = std::is_same<T, double>::value;
  EXPECTS(t_is_double);
  auto adst_nw = static_cast<const tammx::Distribution_NW *>(ttensor.distribution());
  auto ahash = adst_nw->hash();
  auto length = 2 * ahash[0] + 1;
  FortranInt *offseta = new FortranInt[length];
  for (size_t i = 0; i < length; i++) {
    offseta[i] = ahash[i];
  }

  auto amp_ga = static_cast<tammx::MemoryRegionGA&>(ttensor.memory_region());
  FortranInt da = amp_ga.ga();
  return {da, offseta};
}

// symmetrization or unsymmetrization, but not both in one symmetry group
template<typename T, typename LabeledTensorType>
inline void
AddOp<T, LabeledTensorType>::execute(const ProcGroup& ec_pg) {
  using T1 = typename LabeledTensorType::element_type;

  //std::cout<<"ADD_OP. C"<<lhs_.label_<<" += "<<alpha_<<" * A"<<rhs_.label_<<"\n";

  //tensor_print(*lhs_.tensor_);
  //tensor_print(*rhs_.tensor_);
  if(exec_mode_ == ExecutionMode::fortran) {
    bool t1_is_double = std::is_same<T1, double>::value;
    EXPECTS(t1_is_double);
    EXPECTS(fn_ != nullptr);

    FortranInt da, *offseta_map;
    FortranInt dc, *offsetc_map;
    std::tie(da, offseta_map) = tensor_to_fortran_info(*rhs_.tensor_);
    std::tie(dc, offsetc_map) = tensor_to_fortran_info(*lhs_.tensor_);
    FortranInt offseta = offseta_map - MA::int_mb();
    FortranInt offsetc = offsetc_map - MA::int_mb();

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
  auto lambda = [&] (const BlockDimVec& cblockid) {
    //std::cout<<"---tammx assign. cblockid"<<cblockid<<std::endl;
    size_t dimc = tc.block_size(cblockid);
    if(!(tc.nonzero(cblockid) && tc.spin_unique(cblockid) && dimc > 0)) {
      return;
    }
    auto cbp = tc.alloc(cblockid);
    cbp() = 0;
    //std::cout<<"---tammx assign. ACTION ON cblockid"<<cblockid<<std::endl;
    auto label_map = LabelMap<BlockIndex>().update(ltc.label_, cblockid);
    auto sit = symmetrization_iterator(label_map,ltc, lta);
    for(; sit.has_more(); sit.next()) {
      IndexLabelVec cur_clbl = sit.get();
      //std::cout<<"ACTION cur_clbl="<<cur_clbl<<std::endl;
      auto cur_cblockid = label_map.get_blockid(cur_clbl);
      //std::cout<<"---tammx assign. ACTION cur_cblockid"<<cur_cblockid<<std::endl;
      auto ablockid = LabelMap<BlockIndex>().update(ltc.label_, cur_cblockid).get_blockid(alabel);
      auto abp = ta.get(ablockid);
      //std::cout<<"ACTION ablockid="<<ablockid<<std::endl;
      //std::cout<<"ACTION symm_factor="<<symm_factor<<std::endl;

      auto csbp = tc.alloc(cur_cblockid);
      csbp() = 0;
      csbp(clabel) += alpha_ * symm_factor * abp(alabel);

      auto csit = copy_symmetrization_iterator(label_map, ltc, lta, cur_cblockid);
      for(IndexLabelVec csym_clbl = csit.get(); csit.has_more(); csit.next(), csym_clbl = csit.get()) {
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
  if(ec_pg.size() > ltc.tensor_->pg().size()) {
    parallel_work(ltc.tensor_->pg(), citr, citr.get_end(), lambda);
  } else {
    parallel_work(ec_pg, citr, citr.get_end(), lambda);
  }

  //tensor_print(*lhs_.tensor_);
}


inline int
compute_symmetry_scaling_factor(const TensorVec<TensorSymmGroup>& sum_indices,
                                BlockDimVec sumid) {
  int ret = 1;
  auto itr = sumid.begin();
  for(auto &sg: sum_indices) {
    auto sz = sg.size();
    EXPECTS(sz > 0);
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
MultOp<T, LabeledTensorType>::execute(const ProcGroup& ec_pg) {
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
    EXPECTS(t1_is_double);
    EXPECTS(fn_ != nullptr);

    FortranInt da, *offseta_map;
    FortranInt db, *offsetb_map;
    FortranInt dc, *offsetc_map;
    std::tie(da, offseta_map) = tensor_to_fortran_info(*rhs1_.tensor_);
    std::tie(db, offsetb_map) = tensor_to_fortran_info(*rhs2_.tensor_);
    std::tie(dc, offsetc_map) = tensor_to_fortran_info(*lhs_.tensor_);
    FortranInt offseta = offseta_map - MA::int_mb();
    FortranInt offsetb = offsetb_map - MA::int_mb();
    FortranInt offsetc = offsetc_map - MA::int_mb();

    //std::cout<<"---------INVOKING FORTRAN MULT----------\n";
    FortranInt zero = 0;
    fn_(&da, &offseta, &db, &offsetb, &dc, &offsetc);
    //tensor_print(*lhs_.tensor_);

    delete[] offseta_map;
    delete[] offsetb_map;
    delete[] offsetc_map;
    return;
  }

  //@todo @fixme MultOp based on nonsymmetrized_iterator cannot work with ResultMode::set
  //EXPECTS(mode_ == ResultMode::update);
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

  IndexLabelVec sum_labels;
  TensorVec<TensorSymmGroup> sum_indices;
  std::tie(sum_indices, sum_labels) = summation_indices(ltc, lta, ltb);
  auto lambda = [&] (const BlockDimVec& cblockid) {
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
    auto label_map_outer = LabelMap<BlockIndex>().update(ltc.label_, cblockid);
    auto sit = symmetrization_iterator(label_map_outer,ltc, lta, ltb);
    for(; sit.has_more(); sit.next()) {
      IndexLabelVec cur_clbl = sit.get();
      auto cur_cblockid = label_map_outer.get_blockid(cur_clbl);
      //std::cout<<"MultOp. cur_cblock"<<cur_cblockid<<std::endl;
      //std::cout<<"MultOp. cur_cblock size="<<dimc<<std::endl;

      auto sum_itr_first = loop_iterator(slice_indices(sum_indices, sum_labels));
      auto sum_itr_last = sum_itr_first.get_end();
      auto label_map = LabelMap<BlockIndex>().update(ltc.label_, cur_cblockid);

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
          IndexLabelVec csym_clbl = csit.get();
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
  if(ec_pg.size() > ltc.tensor_->pg().size()) {
    parallel_work(ltc.tensor_->pg(), citr, citr.get_end(), lambda);
  } else {
    parallel_work(ec_pg, citr, citr.get_end(), lambda);
  }
  //tensor_print(*lhs_.tensor_);
}

} // namespace tammx

#endif  // TAMMX_OPS_H_
