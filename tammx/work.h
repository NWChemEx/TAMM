// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_WORK_H__
#define TAMMX_WORK_H__

#include <algorithm>
#include "tammx/labeled-tensor.h"
#include "tammx/tensor.h"

namespace tammx {

// @todo Parallelize
template<typename Itr, typename Fn>
void parallel_work(Itr first, Itr last, Fn fn) {
  for(; first != last; ++first) {
    // std::cout<<__FUNCTION__<<" invoking task function"<<std::endl;
    fn(*first);
  }
}

template<typename Itr, typename Fn>
void seq_work(Itr first, Itr last, Fn fn) {
  std::for_each(first, last, fn);
}

template<typename T, typename Lambda>
inline void
tensor_map (LabeledTensor<T> ltc, Lambda func) {
  Tensor<T>& tc = *ltc.tensor_;
  auto citr = loop_iterator(slice_indices(tc.indices(), ltc.label_));
  auto lambda = [&] (const TensorIndex& cblockid) {
    size_t dimc = tc.block_size(cblockid);
    if(tc.nonzero(cblockid) && dimc>0) {
      auto cblock = tc.alloc(cblockid);
      func(cblock);
      tc.add(cblock.blockid(), cblock);
    }
  };
  parallel_work(citr, citr.get_end(), lambda);  
}

template<typename T, typename Lambda>
inline void
block_for (LabeledTensor<T> ltc, Lambda func) {
  Tensor<T>& tc = *ltc.tensor_;
  auto citr = loop_iterator(slice_indices(tc.indices(), ltc.label_));
  auto lambda = [&] (const TensorIndex& cblockid) {
    size_t dimc = tc.block_size(cblockid);
    if(tc.nonzero(cblockid) && dimc>0) {
      //auto cblock = tc.alloc(cblockid);
      func(cblockid);
      //    tc.add(cblock);
    }
  };
  parallel_work(citr, citr.get_end(), lambda);
}

template<typename T, typename Lambda>
inline void
tensor_map (LabeledTensor<T> ltc, LabeledTensor<T> lta, Lambda func) {
  Tensor<T>& tc = *ltc.tensor_;
  Tensor<T>& ta = *lta.tensor_;
  auto citr = loop_iterator(tc.indices());
  auto lambda = [&] (const TensorIndex& cblockid) {
    size_t dimc = tc.block_size(cblockid);
    if(tc.nonzero(cblockid) && dimc>0) {
      auto cblock = tc.alloc(cblockid);
      auto ablock = ta.alloc(cblockid);
      func(cblock, ablock);
      tc.add(cblock.blockid(), cblock);
    }
  };
  parallel_work(citr, citr.get_end(), lambda);
}

}; // namespace tammx

#endif  // TAMMX_WORK_H__
