// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_WORK_H_
#define TAMMX_WORK_H_

#include <algorithm>
#include "tammx/labeled_tensor.h"
#include "tammx/tensor.h"

namespace tammx {

template<typename Itr, typename Fn>
void
parallel_work(Itr first, Itr last, Fn fn) {
  for(; first != last; ++first) {
    fn(*first);
  }
}

template<typename Itr, typename Fn>
void
seq_work(Itr first, Itr last, Fn fn) {
  std::for_each(first, last, fn);
}

template<typename T, typename Lambda>
void
tensor_map (LabeledTensor<T> ltc, Lambda func) {
  Tensor<T>& tc = *ltc.tensor_;
  auto citr = loop_iterator(slice_indices(tc.tindices(), ltc.label_));
  auto lambda = [&] (const TensorIndex& cblockid) {
    size_t dimc = tc.block_size(cblockid);
    if(tc.nonzero(cblockid) && tc.spin_unique(cblockid) && dimc>0) {
      auto cblock = tc.alloc(cblockid);
      func(cblock);
      tc.add(cblock.blockid(), cblock);
    }
  };
  parallel_work(citr, citr.get_end(), lambda);  
}

template<typename T, typename Lambda>
void
block_for (LabeledTensor<T> ltc, Lambda func) {
  Tensor<T>& tc = *ltc.tensor_;
  auto citr = loop_iterator(slice_indices(tc.tindices(), ltc.label_));
  auto lambda = [&] (const TensorIndex& cblockid) {
    size_t dimc = tc.block_size(cblockid);
    if(tc.nonzero(cblockid) && dimc>0) {
      func(cblockid);
    }
  };
  parallel_work(citr, citr.get_end(), lambda);
}

template<typename T, typename Lambda>
void
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

} // namespace tammx

#endif  // TAMMX_WORK_H_
