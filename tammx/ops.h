
#ifndef TAMMX_OPS_H_
#define TAMMX_OPS_H_

#include "tammx/labeled-block.h"
#include "tammx/util.h"

namespace tammx {

class LabeledTensor;

/**
 * @todo These overloads to match tensor type and the scalear types
 */
template<typename T>
inline std::tuple<T, LabeledTensor>
operator * (T alpha, LabeledTensor block) {
  return {alpha, block};
}

template<typename T>
inline std::tuple<T, LabeledTensor>
operator * (LabeledTensor block, T alpha) {
  return {alpha, block};
}

template<typename T>
inline std::tuple<T, LabeledTensor, LabeledTensor>
operator * (const std::tuple<T, LabeledTensor>& rhs1, LabeledTensor rhs2)  {
  return std::tuple_cat(rhs1, std::make_tuple(rhs2));
}

inline std::tuple<LabeledTensor, LabeledTensor>
operator * (LabeledTensor rhs1, LabeledTensor rhs2)  {
  return std::make_tuple(rhs1, rhs2);
}

template<typename T>
inline std::tuple<T, LabeledTensor, LabeledTensor>
operator * (T alpha, std::tuple<LabeledTensor, LabeledTensor> rhs) {
  return std::tuple_cat(std::make_tuple(alpha), rhs);
}

/////////////////////////////////////////////////////////////////////
//         operators
/////////////////////////////////////////////////////////////////////

struct Op {
  virtual void execute() = 0;
};

template<typename T>
struct SetOp : public Op {
  void execute();

  SetOp(T value1, LabeledTensor& lhs1)
      : value{value1},
        lhs{lhs1} {}
  
  T value;
  LabeledTensor lhs;
};


template<typename T>
struct AddOp : public Op {
  void execute();

  AddOp(T alpha1, const LabeledTensor& lhs1, const LabeledTensor& rhs1, ResultMode mode)
      : alpha{alpha1},
        lhs{lhs1},
        rhs{rhs1},
        mode_{mode} { }
  
  T alpha;
  LabeledTensor lhs, rhs;
  ResultMode mode_;
};




/////////////////////////////////////////////////////////////////////
//         mult operator
/////////////////////////////////////////////////////////////////////

template<typename T>
struct MultOp : public Op {
  void execute();
  
  MultOp(T alpha1, const LabeledTensor& lhs1, const LabeledTensor& rhs11,
         const LabeledTensor& rhs12, ResultMode mode)
      : alpha{alpha1},
        lhs{lhs1},
        rhs1{rhs11},
        rhs2{rhs12},
        mode_{mode} { }
  
  T alpha;
  LabeledTensor lhs, rhs1, rhs2;
  ResultMode mode_;
};




/////////////////////////////////////////////////////////////////////
//         map operator
/////////////////////////////////////////////////////////////////////

template<typename Func, unsigned ndim, unsigned nrhs>
struct MapOp : public Op {
  void execute();
};

/**
 * @note Works with arbitrary dimensions
 */
template<typename Func>
struct MapOp<Func, 0, 0> : public Op {
  void execute() {
    std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
    Tensor& lhs_tensor = *lhs_.tensor_;
    auto lambda = [&] (const TensorIndex& blockid) {
      auto size = lhs_tensor.block_size(blockid);
      auto offset = TCE::offset(blockid[0]);
      if(lhs_tensor.nonzero(blockid) && size > 0) {
        auto lblock = lhs_tensor.alloc(blockid);
        //lblock.init(0);
        type_dispatch(lhs_tensor.element_type(), [&](auto type) {
            using dtype = decltype(type);
            auto ltbuf = reinterpret_cast<dtype*>(lblock.buf());
            for(int i=0; i<size; i++) {
              func_(ltbuf[i]);
            }
          });
        lhs_tensor.add(lblock);
      }
    };
    auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
    parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  MapOp(const LabeledTensor& lhs, Func func)
      : lhs_{lhs},
        func_{func} {
    Expects(lhs_.tensor_ != nullptr);
  }
  
  LabeledTensor lhs_;
  Func func_;
};

/**
 * @todo more generic ndimg and rhs versions
 *
 */ 
template<typename Func>
struct MapOp<Func, 1, 0> : public Op {
  void execute() {
    std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
    Tensor& lhs_tensor = *lhs_.tensor_;

    auto lambda = [&] (const TensorIndex& blockid) {
      auto size = lhs_tensor.block_size(blockid);
      auto offset = TCE::offset(blockid[0]);
      if(lhs_tensor.nonzero(blockid) && size > 0) {
        auto lblock = lhs_tensor.alloc(blockid);
        //lblock.init(0);
        type_dispatch(lhs_tensor.element_type(), [&](auto type) {
            using dtype = decltype(type);
            auto ltbuf = reinterpret_cast<dtype*>(lblock.buf());
            for(int i=0; i<size; i++) {
              func_(offset + i, ltbuf[i]);
            }
          });
        lhs_tensor.add(lblock);
      }
    };
    auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
    parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  MapOp(const LabeledTensor& lhs, Func func) 
      : lhs_{lhs}, func_{func} {
    Expects(lhs_.tensor_ != nullptr);
    Expects(lhs_.tensor_->rank() == 1);
  }
  
  LabeledTensor lhs_;
  Func func_;
};

template<typename Func>
struct MapOp<Func, 2, 0> : public Op {
  void execute() {
    std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
    Tensor& lhs_tensor = *lhs_.tensor_;

    auto lambda = [&] (const TensorIndex& blockid) {
      auto bdims = lhs_tensor.block_dims(blockid);
      auto isize = bdims[0].value();
      auto jsize = bdims[1].value();
      auto ioffset = TCE::offset(blockid[0]);
      auto joffset = TCE::offset(blockid[1]);
      if(lhs_tensor.nonzero(blockid) && isize*jsize > 0) {
        auto lblock = lhs_tensor.alloc(blockid);
        //lblock.init(0);
        type_dispatch(lhs_tensor.element_type(), [&](auto type) {
            using dtype = decltype(type);
            auto ltbuf = reinterpret_cast<dtype*>(lblock.buf());
            for(int i=0, c=0; i<isize; i++) {
              for(int j=0; j<jsize; j++, c++) {
                func_(ioffset+i, joffset+j, ltbuf[c]);
              }
            }
          });
        lhs_tensor.add(lblock);
      }
    };
    auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
    parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  MapOp(const LabeledTensor& lhs, Func func) 
      : lhs_{lhs}, func_{func} {
    Expects(lhs_.tensor_ != nullptr);
    Expects(lhs_.tensor_->rank() == 2);
  }
  
  LabeledTensor lhs_;
  Func func_;
};

template<typename Func>
struct MapOp<Func, 2, 1> : public Op {
  void execute() {
    std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
    Tensor& lhs_tensor = *lhs_.tensor_;
    Tensor& rhs1_tensor = *rhs1_.tensor_;

    auto lambda = [&] (const TensorIndex& blockid) {
      auto bdims = lhs_tensor.block_dims(blockid);
      auto isize = bdims[0].value();
      auto jsize = bdims[1].value();
      auto ioffset = TCE::offset(blockid[0]);
      auto joffset = TCE::offset(blockid[1]);
      if(lhs_tensor.nonzero(blockid) && isize*jsize > 0) {
        auto lblock = lhs_tensor.alloc(blockid);
        auto r1block = rhs1_tensor.get(blockid);
        //lblock.init(0);
        type_dispatch(lhs_tensor.element_type(), [&](auto type) {
            using dtype = decltype(type);
            auto ltbuf = reinterpret_cast<dtype*>(lblock.buf());
            auto r1tbuf = reinterpret_cast<dtype*>(r1block.buf());
            for(int i=0, c=0; i<isize; i++) {
              for(int j=0; j<jsize; j++, c++) {
                func_(ioffset+i, joffset+j, ltbuf[c], r1tbuf[c]);
              }
            }
          });
        lhs_tensor.add(lblock);
      }
    };
    auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
    parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  MapOp(const LabeledTensor& lhs, LabeledTensor& rhs1, Func func) 
      : lhs_{lhs},
        rhs1_{rhs1},
        func_{func} {
    Expects(lhs_.tensor_ != nullptr);
    Expects(lhs_.tensor_->rank() == 2);
  }
  
  LabeledTensor lhs_;
  LabeledTensor rhs1_;
  Func func_;
};

template<typename Func>
struct MapOp<Func, 2, 2> : public Op {
  void execute() {
    std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
    Tensor& lhs_tensor = *lhs_.tensor_;
    Tensor& rhs1_tensor = *rhs1_.tensor_;
    Tensor& rhs2_tensor = *rhs2_.tensor_;

    auto lambda = [&] (const TensorIndex& blockid) {
      auto bdims = lhs_tensor.block_dims(blockid);
      auto isize = bdims[0].value();
      auto jsize = bdims[1].value();
      auto ioffset = TCE::offset(blockid[0]);
      auto joffset = TCE::offset(blockid[1]);
      if(lhs_tensor.nonzero(blockid) && isize*jsize > 0) {
        auto lblock = lhs_tensor.alloc(blockid);
        auto r1block = rhs1_tensor.get(blockid); 
        auto r2block = rhs2_tensor.get(blockid); 
        //lblock.init(0);
        type_dispatch(lhs_tensor.element_type(), [&](auto type) {
            using dtype = decltype(type);
            auto ltbuf = reinterpret_cast<dtype*>(lblock.buf());
            auto r1tbuf = reinterpret_cast<dtype*>(r1block.buf());
            auto r2tbuf = reinterpret_cast<dtype*>(r2block.buf());
            for(int i=0, c=0; i<isize; i++) {
              for(int j=0; j<jsize; j++, c++) {
                func_(ioffset+i, joffset+j, ltbuf[c], r1tbuf[c], r2tbuf[c]);
              }
            }
          });
        lhs_tensor.add(lblock);
      }
    };
    auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
    parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  MapOp(const LabeledTensor& lhs, LabeledTensor& rhs1, LabeledTensor& rhs2, Func func) 
      : lhs_{lhs},
        rhs1_{rhs1},
        rhs2_{rhs2},
        func_{func} {
    Expects(lhs_.tensor_ != nullptr);
    Expects(lhs_.tensor_->rank() == 2);
  }
  
  LabeledTensor lhs_;
  LabeledTensor rhs1_, rhs2_;
  Func func_;
};


template<typename Func>
struct MapOp<Func, 1, 1> : public Op {
  void execute() {
    std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
    Tensor& lhs_tensor = *lhs_.tensor_;
    Tensor& rhs_tensor = *rhs_.tensor_;

    auto lambda = [&] (const TensorIndex& blockid) {
      auto size = lhs_tensor.block_size(blockid);
      auto offset = TCE::offset(blockid[0]);
      if(lhs_tensor.nonzero(blockid) && size > 0) {
        auto lblock = lhs_tensor.alloc(blockid);
        auto rblock = rhs_tensor.get(blockid);
        //lblock.init(0);
        type_dispatch(lhs_tensor.element_type(), [&](auto type) {
            using dtype = decltype(type);
            auto ltbuf = reinterpret_cast<dtype*>(lblock.buf());
            auto rtbuf = reinterpret_cast<dtype*>(rblock.buf());
            for(int i=0; i<size; i++) {
              func_(i+offset, ltbuf[i], rtbuf[i]);
            }
          });
        lhs_tensor.add(lblock);
      }
    };
    auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
    parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  MapOp(const LabeledTensor& lhs, const LabeledTensor& rhs, Func func)
      : lhs_{lhs},
        rhs_{rhs},
        func_{func} {
    Expects(lhs_.tensor_ != nullptr);
    Expects(lhs_.tensor_->rank() == 1);
    Expects(rhs_.tensor_ != nullptr);
    Expects(rhs_.tensor_->rank() == 1);
  }
  
  LabeledTensor lhs_, rhs_;
  Func func_;
};

/////////////////////////////////////////////////////////////////////
//         scan operator
/////////////////////////////////////////////////////////////////////

/**
 * @todo Could be more general, similar to MapOp
 */
template<typename Func, unsigned ndim>
struct ScanOp : public Op {
};

template<typename Func>
struct ScanOp<Func,0> : public Op {
  void execute() {
    std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
    Tensor& tensor = *ltensor_.tensor_;
    auto lambda = [&] (const TensorIndex& blockid) {
      auto size = tensor.block_size(blockid);
      if(tensor.nonzero(blockid) && size > 0) {
        auto block = tensor.get(blockid);
        type_dispatch(tensor.element_type(), [&](auto type) {
            using dtype = decltype(type);
            auto tbuf = reinterpret_cast<dtype*>(block.buf());
            for(int i=0; i<size; i++) {
              func_(tbuf[i]);
            }
          });
      }
    };
    auto itr_first = loop_iterator(slice_indices(tensor.indices(), ltensor_.label_));
    parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  ScanOp(const LabeledTensor& ltensor, Func func)
      : ltensor_{ltensor},
        func_{func} {
    Expects(ltensor.tensor_ != nullptr);
  }
  
  LabeledTensor ltensor_;
  Func func_;
};

template<typename Func>
struct ScanOp<Func,1> : public Op {
  void execute() {
    std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
    Tensor& tensor = *ltensor_.tensor_;
    auto lambda = [&] (const TensorIndex& blockid) {
      auto size = tensor.block_size(blockid);
      if(tensor.nonzero(blockid) && size > 0) {
        auto bdims = tensor.block_dims(blockid);
        auto isize = bdims[0].value();
        auto ioffset = TCE::offset(blockid[0]);
        auto block = tensor.get(blockid);
        type_dispatch(tensor.element_type(), [&](auto type) {
            using dtype = decltype(type);
            auto tbuf = reinterpret_cast<dtype*>(block.buf());
            for(int i=0; i<isize; i++) {
              func_(i+ioffset, tbuf[i]);
            }
          });
      }
    };
    auto itr_first = loop_iterator(slice_indices(tensor.indices(), ltensor_.label_));
    parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  ScanOp(const LabeledTensor& ltensor, Func func)
      : ltensor_{ltensor},
        func_{func} {
    Expects(ltensor.tensor_ != nullptr);
    Expects(ltensor.tensor_->rank() == 1);
  }
  
  LabeledTensor ltensor_;
  Func func_;
};

template<typename Func>
struct ScanOp<Func,2> : public Op {
  void execute() {
    std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
    Tensor& tensor = *ltensor_.tensor_;
    auto lambda = [&] (const TensorIndex& blockid) {
      auto size = tensor.block_size(blockid);
      if(tensor.nonzero(blockid) && size > 0) {
        auto bdims = tensor.block_dims(blockid);
        auto isize = bdims[0].value();
        auto jsize = bdims[1].value();
        auto ioffset = TCE::offset(blockid[0]);
        auto joffset = TCE::offset(blockid[1]);
        auto block = tensor.get(blockid);
        type_dispatch(tensor.element_type(), [&](auto type) {
            using dtype = decltype(type);
            auto tbuf = reinterpret_cast<dtype*>(block.buf());
            for(int i=0, c=0; i<isize; i++) {
              for(int j=0; j<jsize; j++, c++) {
                func_(i+ioffset, j+joffset, tbuf[c]);
              }
            }
          });
      }
    };
    auto itr_first = loop_iterator(slice_indices(tensor.indices(), ltensor_.label_));
    parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  ScanOp(const LabeledTensor& ltensor, Func func)
      : ltensor_{ltensor},
        func_{func} {
    Expects(ltensor.tensor_ != nullptr);
    Expects(ltensor.tensor_->rank() == 2);
  }
  
  LabeledTensor ltensor_;
  Func func_;
};

/////////////////////////////////////////////////////////////////////
//         helper functions
/////////////////////////////////////////////////////////////////////

struct OpList : public std::vector<Op*> {
  OpList() = default;

  ~OpList() {
    for(auto &ptr_op : *this) {
      delete ptr_op;
    }
    clear();
  }

  template<typename T>
  OpList& op(SetOpEntry<T> sop) {
    push_back(new SetOp<T>(sop.value, sop.lhs));
    return *this;
  }

  template<typename T>
  OpList& op(AddOpEntry<T> aop) {
    push_back(new AddOp<T>(aop.alpha, aop.lhs, aop.rhs, aop.mode));
    return *this;
  }

  template<typename T>
  OpList& op(MultOpEntry<T> aop) {
    push_back(new MultOp<T>(aop.alpha, aop.lhs, aop.rhs1, aop.rhs2, aop.mode));
    return *this;
  }

  template<unsigned ndim=0, unsigned nrhs=0, typename Func>
  OpList& op(LabeledTensor lhs, Func func) {
    push_back(new MapOp<Func,ndim,nrhs>(lhs, func));
    return *this;
  }

  template<unsigned ndim=0, typename Func>
  OpList& sop(LabeledTensor lhs, Func func) {
    push_back(new ScanOp<Func,ndim>(lhs, func));
    return *this;
  }

  template<unsigned ndim, unsigned nrhs, typename Func>
  OpList& op(LabeledTensor lhs, LabeledTensor rhs, Func func) {
    push_back(new MapOp<Func,ndim,nrhs>(lhs, rhs, func));
    return *this;
  }

  template<unsigned ndim, unsigned nrhs, typename Func>
  OpList& op(LabeledTensor lhs, LabeledTensor rhs1, LabeledTensor rhs2, Func func) {
    push_back(new MapOp<Func,ndim,nrhs>(lhs, rhs1, rhs2, func));
    return *this;
  }

  template<typename T>
  OpList& operator()(SetOpEntry<T> sop) {
    push_back(new SetOp<T>(sop.value, sop.lhs));
    return *this;
  }

  template<typename T>
  OpList& operator()(AddOpEntry<T> aop) {
    push_back(new AddOp<T>(aop.alpha, aop.lhs, aop.rhs, aop.mode));
    return *this;
  }

  template<typename T>
  OpList& operator()(MultOpEntry<T> aop) {
    push_back(new MultOp<T>(aop.alpha, aop.lhs, aop.rhs1, aop.rhs2, aop.mode));
    return *this;
  }

  template<typename Func>
  OpList& operator()(LabeledTensor lhs, Func func) {
    push_back(new MapOp<Func,0,0>(lhs, func));
    return *this;
  }

  void execute() {
    for(auto &ptr_op: *this) {
      ptr_op->execute();
    }
  }
};

///////////////////////////////////////////////////////////////////////
// other operations
//////////////////////////////////////////////////////////////////////

inline void
assert_zero(Tensor& tc, double threshold = 1.0e-12) {
  auto lambda = [&] (auto &val) { Expects(std::abs(val) < threshold); };
  auto op = ScanOp<decltype(lambda),0>(tc(), lambda);
  op.execute();
}


///////////////////////////////////////////////////////////////////////
// implementation of operators
//////////////////////////////////////////////////////////////////////

//-----------------------support routines


inline TensorVec<TensorVec<TensorLabel>>
summation_labels(const LabeledTensor& /*ltc*/,
                  const LabeledTensor& lta,
                  const LabeledTensor& ltb) {
  return group_partition(lta.tensor_->indices(), lta.label_,
                         ltb.tensor_->indices(), ltb.label_);

}

inline std::pair<TensorVec<SymmGroup>,TensorLabel>
summation_indices(const LabeledTensor& /*ltc*/,
                  const LabeledTensor& lta,
                  const LabeledTensor& ltb) {
  auto aindices = flatten(lta.tensor_->indices());
  //auto bindices = flatten(ltb.tensor_.indices());
  auto alabels = group_labels(lta.tensor_->indices(), lta.label_);
  auto blabels = group_labels(ltb.tensor_->indices(), ltb.label_);
  TensorVec<SymmGroup> ret_indices;
  TensorLabel sum_labels;
  int apos = 0;
  for (auto &alg : alabels) {
    for (auto &blg : blabels) {
      SymmGroup sg;
      for (auto &a : alg) {
        int apos1 = 0;
        for (auto &b : blg) {
          if (a == b) {
            sg.push_back(aindices[apos + apos1]);
            sum_labels.push_back(a);
          }
        }
        apos1++;
      }
      if (sg.size() > 0) {
        ret_indices.push_back(sg);
      }
    }
    apos += alg.size();
  }
  return {ret_indices, sum_labels};
}

inline TensorVec<TensorVec<TensorLabel>>
nonsymmetrized_external_labels(const LabeledTensor& ltc,
                               const LabeledTensor& lta) {
  auto ca_labels = group_partition(ltc.tensor_->indices(), ltc.label_,
                                   lta.tensor_->indices(), lta.label_);

  TensorVec<TensorVec<TensorLabel>> ret_labels;
  for(unsigned i=0; i<ca_labels.size(); i++)  {
    Expects(ca_labels[i].size() > 0);
    ret_labels.push_back(TensorVec<TensorLabel>());
    ret_labels.back().insert_back(ca_labels[i].begin(), ca_labels[i].end());
  }
  return ret_labels;
}

/**
 * @todo Specify where symmetrization is allowed and what indices in
 * the input tensors can form a symmetry group (or go to distinct
 * groups) in the output tensor.
 */
inline TensorVec<TensorVec<TensorLabel>>
nonsymmetrized_external_labels(const LabeledTensor& ltc,
                               const LabeledTensor& lta,
                               const LabeledTensor& ltb) {
  auto ca_labels = group_partition(ltc.tensor_->indices(), ltc.label_,
                                   lta.tensor_->indices(), lta.label_);
  auto cb_labels = group_partition(ltc.tensor_->indices(), ltc.label_,
                                   ltb.tensor_->indices(), ltb.label_);
  Expects(ca_labels.size() == cb_labels.size());

  TensorVec<TensorVec<TensorLabel>> ret_labels;
  for(unsigned i=0; i<ca_labels.size(); i++)  {
    Expects(ca_labels[i].size() + cb_labels[i].size() > 0);
    ret_labels.push_back(TensorVec<TensorLabel>());
    if(ca_labels[i].size() > 0) {
      ret_labels.back().insert_back(ca_labels[i].begin(), ca_labels[i].end());
    }
    if(cb_labels[i].size() > 0) {
      ret_labels.back().insert_back(cb_labels[i].begin(), cb_labels[i].end());
    }
  }
  return ret_labels;
}

inline TensorVec<TensorVec<TensorLabel>>
symmetrized_external_labels(const LabeledTensor& ltc,
                            const LabeledTensor&  /*lta*/,
                            const LabeledTensor&  /*ltb*/) {
  TensorVec<TensorLabel> ret {ltc.label_};
  return {ret};
}


inline ProductIterator<TriangleLoop>
nonsymmetrized_iterator(const LabeledTensor& ltc,
                        const LabeledTensor& lta,
                        const LabeledTensor& ltb) {
  auto part_labels = nonsymmetrized_external_labels(ltc ,lta, ltb);
  //auto flat_labels = flatten(flatten(part_labels));
  // std::map<IndexLabel, DimType> dim_of_label;

  // auto cflindices = flatten(ltc.tensor_->indices());
  // for(unsigned i=0; i<ltc.label_.size(); i++) {
  //   dim_of_label[ltc.label_[i]] = cflindices[i];
  // }
  // auto aflindices = flatten(lta.tensor_->indices());
  // for(unsigned i=0; i<lta.label_.size(); i++) {
  //   dim_of_label[lta.label_[i]] = aflindices[i];
  // }
  TensorVec<TriangleLoop> tloops, tloops_last;
  for(auto dim_grps: part_labels) {
    for(auto lbl: dim_grps) {
      if(lbl.size() > 0) {
        BlockDim lo, hi;
        std::tie(lo, hi) = tensor_index_range(lbl[0].dt);
        tloops.push_back(TriangleLoop(lbl.size(), lo, hi));
        tloops_last.push_back(tloops.back().get_end());
      }
    }
  }
  return ProductIterator<TriangleLoop>(tloops, tloops_last);
}


/**
 * @todo abstract the two copy_symmetrizer versions into one function
 * with the logic and two interfaces
 */
inline TensorVec<CopySymmetrizer>
copy_symmetrizer(const LabeledTensor& ltc,
                 const LabeledTensor& lta,
                 const LabeledTensor& ltb,
                 const LabelMap<BlockDim>& lmap) {
  auto part_labels = nonsymmetrized_external_labels(ltc ,lta, ltb);
  TensorVec<CopySymmetrizer> csv;
  for(auto lbls: part_labels) {
    Expects(lbls.size()>0 && lbls.size() <= 2);

    TensorLabel lbl(lbls[0].begin(), lbls[0].end());
    if(lbls.size() == 2 ) {
      lbl.insert_back(lbls[1].begin(), lbls[1].end());
    }

    auto size = lbl.size();
    Expects(size > 0);
    Expects(size <=2); // @todo implement other cases

    auto blockid = lmap.get_blockid(lbl);
    auto uniq_blockid{blockid};
    //find unique block
    std::sort(uniq_blockid.begin(), uniq_blockid.end());
    Expects(size > 0);
    //std::cout<<"CONSTRUCTING COPY SYMMETRIZER FOR LABELS="<<lbl<<std::endl;
    csv.push_back(CopySymmetrizer{size, lbls[0].size(), lbl, blockid, uniq_blockid});
  }
  return csv;
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

inline TensorVec<TensorVec<TensorLabel>>
compute_extra_symmetries(const TensorLabel& lhs_label,
                         const TensorIndex& lhs_group_sizes,
                         const TensorLabel& rhs_label,
                         const TensorIndex& rhs_group_sizes) {
  Expects(lhs_label.size() == lhs_group_sizes.size());
  Expects(rhs_label.size() == rhs_group_sizes.size());
  Expects(lhs_label.size() == rhs_label.size());

  auto lhs_label_groups = group_labels(lhs_label, lhs_group_sizes);
  auto rhs_label_groups = group_labels(rhs_label, rhs_group_sizes);

  TensorVec<TensorVec<TensorLabel>> ret_labels;
  for (auto &glhs : lhs_label_groups) {
    TensorVec<TensorLabel> ret_group;
    for (auto &grhs : rhs_label_groups) {
      auto lbls = intersect(glhs, grhs);
      if (lbls.size() > 0) {
        ret_group.push_back(lbls);
      }
    }
    ret_labels.push_back(ret_group);
  }
  return ret_labels;  
}


inline TensorVec<CopySymmetrizer>
copy_symmetrizer(const LabeledTensor& ltc,
                 const LabeledTensor& lta,
                 const LabelMap<BlockDim>& lmap) {
  auto part_labels = nonsymmetrized_external_labels(ltc ,lta);
  TensorVec<CopySymmetrizer> csv;
  for(auto lbls: part_labels) {
    Expects(lbls.size()>0 && lbls.size() <= 2);

    TensorLabel lbl(lbls[0].begin(), lbls[0].end());
    if(lbls.size() == 2 ) {
      lbl.insert_back(lbls[1].begin(), lbls[1].end());
    }

    auto size = lbl.size();
    Expects(size > 0);
    Expects(size <=2); // @todo implement other cases

    auto blockid = lmap.get_blockid(lbl);
    auto uniq_blockid{blockid};
    //find unique block
    std::sort(uniq_blockid.begin(), uniq_blockid.end());
    Expects(size > 0);
    //std::cout<<"CONSTRUCTING COPY SYMMETRIZER FOR LABELS="<<lbl<<std::endl;
    csv.push_back(CopySymmetrizer{size, lbls[0].size(), lbl, blockid, uniq_blockid});
  }
  return csv;
}



inline ProductIterator<CopySymmetrizer::Iterator>
copy_iterator(const TensorVec<CopySymmetrizer>& sitv) {
  TensorVec<CopySymmetrizer::Iterator> itrs_first, itrs_last;
  for(auto &sit: sitv) {
    itrs_first.push_back(sit.begin());
    itrs_last.push_back(sit.end());
    Expects(itrs_first.back().itr_size() == itrs_last.back().itr_size());
    Expects(itrs_first.back().itr_size() == sit.group_size_);
  }
  return {itrs_first, itrs_last};
}


//-----------------------op execute routines

template<typename T>
inline void
SetOp<T>::execute() {
  OpList().op<0,0>(lhs, [=](auto &ival) { ival = value; }).execute();
}

template<typename T>
inline void
AddOp<T>::execute() {
  std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
  const LabeledTensor& lta = rhs;
  const LabeledTensor& ltc = lhs;
  Tensor& ta = *lta.tensor_;
  Tensor& tc = *ltc.tensor_;
  auto aitr = loop_iterator(slice_indices(ta.indices(), lta.label_));
  auto lambda = [&] (const TensorIndex& ablockid) {
    size_t dima = ta.block_size(ablockid);
    if(ta.nonzero(ablockid) && dima>0) {
      auto label_map = LabelMap<BlockDim>()
          .update(lta.label_, ablockid);
      auto cblockid = label_map.get_blockid(ltc.label_);
      auto abp = ta.get(ablockid);
      auto csbp = tc.alloc(tc.find_unique_block(cblockid));
      csbp() = T(0);
      auto copy_symm = copy_symmetrizer(ltc, lta, label_map);      
      auto copy_itr = copy_iterator(copy_symm);
      auto copy_itr_last = copy_itr.get_end();
      auto copy_label = TensorLabel{ltc.label_};
      for(auto citr = copy_itr; citr != copy_itr_last; ++citr) {
        auto cperm_label = *citr;
        auto num_inversions = perm_count_inversions(perm_compute(copy_label, cperm_label));
        Sign sign = (num_inversions%2) ? -1 : 1;
        auto perm_comp = perm_apply(cperm_label, perm_compute(ltc.label_, lta.label_));
        csbp(copy_label) += sign * alpha * abp(perm_comp);
      }
      if(mode_ == ResultMode::update) {
        tc.add(csbp);
      } else {
        tc.put(csbp);
      }
    }
  };
  parallel_work(aitr, aitr.get_end(), lambda);
}

template<typename T>
inline void
MultOp<T>::execute() {
  std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
  LabeledTensor& lta = rhs1;
  LabeledTensor& ltb = rhs2;
  LabeledTensor& ltc = lhs;
  Tensor& ta = *lta.tensor_;
  Tensor& tb = *ltb.tensor_;
  Tensor& tc = *ltc.tensor_;

  TensorLabel sum_labels;
  TensorVec<SymmGroup> sum_indices;
  std::tie(sum_indices, sum_labels) = summation_indices(ltc, lta, ltb);
  auto lambda = [&] (const TensorIndex& cblockid) {
    auto dimc = tc.block_size(cblockid);
    if(!tc.nonzero(cblockid) || dimc == 0) {
      return;
    }
    auto sum_itr_first = loop_iterator(slice_indices(sum_indices, sum_labels));
    auto sum_itr_last = sum_itr_first.get_end();
    auto label_map = LabelMap<BlockDim>().update(ltc.label_, cblockid);
    auto cbp = tc.alloc(cblockid);
    cbp() = 0.0;
    for(auto sitr = sum_itr_first; sitr!=sum_itr_last; ++sitr) {
      label_map.update(sum_labels, *sitr);
      auto ablockid = label_map.get_blockid(lta.label_);
      auto bblockid = label_map.get_blockid(ltb.label_);

      if(!ta.nonzero(ablockid) || !tb.nonzero(bblockid)) {
        continue;
      }
      auto abp = ta.get(ablockid);
      auto bbp = tb.get(bblockid);

#if 0
      cbp(ltc.label_) += alpha * abp(lta.label_) * bbp(ltb.label_);
#endif
    }
    auto csbp = tc.alloc(tc.find_unique_block(cblockid));
    csbp() = 0.0;
    auto copy_symm = copy_symmetrizer(ltc, lta, ltb, label_map);      
    auto copy_itr = copy_iterator(copy_symm);
    auto copy_itr_last = copy_itr.get_end();
    auto copy_label = TensorLabel(ltc.label_);
    //std::sort(copy_label.begin(), copy_label.end());
    for(auto citr = copy_itr; citr != copy_itr_last; ++citr) {
      auto cperm_label = *citr;
      auto num_inversions = perm_count_inversions(perm_compute(copy_label, cperm_label));
      Sign sign = (num_inversions%2) ? -1 : 1;
      csbp(copy_label) += T(sign) * cbp(cperm_label);
    }
    if(mode_ == ResultMode::update) {
      tc.add(csbp);
    } else {
      tc.put(csbp);
    }
  };
  auto itr = nonsymmetrized_iterator(ltc, lta, ltb);
  parallel_work(itr, itr.get_end(), lambda);
}


}; // namespace tammx

#endif  // TAMMX_OPS_H_

