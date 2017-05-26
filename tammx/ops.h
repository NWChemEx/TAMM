
#ifndef TAMMX_OPS_H_
#define TAMMX_OPS_H_

#include "tammx/labeled-block.h"

namespace tammx {

class LabeledTensor;

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

struct Op {
  virtual void execute() = 0;
};


/////////////////////////////////////////////////////////////////////
//         map operator
/////////////////////////////////////////////////////////////////////

/**
 * @todo Correctly handle beta
 */
template<typename T>
struct AddOp : public Op {
  void execute() {
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
        csbp().init(T(0));      
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
        tc.add(csbp);
      }
    };
    parallel_work(aitr, aitr.get_end(), lambda);
  }

  AddOp(T alpha1, T beta1, const LabeledTensor& lhs1, const LabeledTensor& rhs1)
      : alpha{alpha1},
        beta{beta1},
        lhs{lhs1},
        rhs{rhs1} { }
  
  T alpha, beta;
  LabeledTensor lhs, rhs;
};




/////////////////////////////////////////////////////////////////////
//         mult operator
/////////////////////////////////////////////////////////////////////

/**
 * @todo Correctly handle beta
 */
template<typename T>
struct MultOp : public Op {
  void execute() {
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
      cbp().init(0.0);
      for(auto sitr = sum_itr_first; sitr!=sum_itr_last; ++sitr) {
        label_map.update(sum_labels, *sitr);
        auto ablockid = label_map.get_blockid(lta.label_);
        auto bblockid = label_map.get_blockid(ltb.label_);

        if(!ta.nonzero(ablockid) || !tb.nonzero(bblockid)) {
          continue;
        }
        auto abp = ta.get(ablockid);
        auto bbp = tb.get(bblockid);

        cbp(ltc.label_) += alpha * abp(lta.label_) * bbp(ltb.label_);
      }
      auto csbp = tc.alloc(tc.find_unique_block(cblockid));
      csbp().init(0.0);
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
      tc.add(csbp);
    };
    auto itr = nonsymmetrized_iterator(ltc, lta, ltb);
    parallel_work(itr, itr.get_end(), lambda);
  }
  
  MultOp(T alpha1, T beta1, const LabeledTensor& lhs1, const LabeledTensor& rhs11,
         const LabeledTensor& rhs12)
      : alpha{alpha1},
        beta{beta1},
        lhs{lhs1},
        rhs1{rhs11},
        rhs2{rhs12} { }
  
  T alpha, beta;
  LabeledTensor lhs, rhs1, rhs2;
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

  MapOp(const LabeledTensor& lhs)
      : lhs_{lhs} {
    Expects(lhs_.tensor_ != nullptr);
    Expects(lhs_.tensor_->rank() == 0);
  }
  
  LabeledTensor lhs_;
};

/**
 * @todo more generic ndimg and rhs versions
 *
 */ 
template<typename Func>
struct MapOp<Func, 1, 0> : public Op {
  void execute() {
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
struct MapOp<Func, 1, 1> : public Op {
  void execute() {
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
template<typename Func>
struct ScanOp : public Op {
  void execute() {
    auto tensor = *ltensor_.tensor_;
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

  ScanOp(const LabeledTensor& ltensor)
      : ltensor_{ltensor} {
    Expects(ltensor.tensor_ != nullptr);
    Expects(ltensor.tensor_->rank() == 0);
  }
  
  LabeledTensor ltensor_;
  Func func_;
};


/////////////////////////////////////////////////////////////////////
//         helper functions
/////////////////////////////////////////////////////////////////////

struct OpList {
  OpList() = default;
  
  ~OpList() {
    for(auto &ptr_op : ops_) {
      delete ptr_op;
    }
  }

  template<typename T>
  OpList& op(AddOpEntry<T> aop) {
    ops_.push_back(new AddOp<T>(aop.alpha, aop.beta, aop.lhs, aop.rhs1, aop.rhs2));
    return *this;
  }

  template<typename T>
  OpList& op(MultOpEntry<T> aop) {
    ops_.push_back(new MultOp<T>(aop.alpha, aop.beta, aop.lhs, aop.rhs1, aop.rhs2));
    return *this;
  }

  template<unsigned ndim, unsigned nrhs, typename Func>
  OpList& op(LabeledTensor&& lhs, Func func) {
    ops_.push_back(new MapOp<Func,ndim,nrhs>(lhs, func));
    return *this;
  }

   std::vector<Op*> ops_;
};

template<unsigned ndim, unsigned nrhs, typename Func>
MapOp<Func,ndim,nrhs> mapop_create(LabeledTensor&& lhs, Func func) {
  return MapOp<Func,ndim,nrhs>(lhs, func);
}

}; // namespace tammx

#endif  // TAMMX_OPS_H_

