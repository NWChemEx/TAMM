#pragma once

#include <algorithm>
#include <chrono>
#include <memory>
#include <tuple>
#include <vector>

#include "tamm/errors.hpp"
#include "tamm/execution_context.hpp"
#include "tamm/labeled_tensor.hpp"
#include "tamm/multop.hpp"
#include "tamm/op_profiler.hpp"
#include "tamm/tensor.hpp"

namespace tamm {

/**
 * @brief Implementation of a lazy tensor: a lambda tensor whose blocks are computed as
 * alpha * X * Y by the same per-block routine MultOp uses (MultOp::compute_block).
 *
 * The lazy tensor's index spaces are those of its labels, so a block id passed to get() is
 * also the block id in label space; no translation is needed.
 */
template<typename T>
class LazyTensorImpl: public LambdaTensorImpl<T> {
public:
  using Func    = typename LambdaTensorImpl<T>::Func;
  using LTensor = LabeledTensor<T>;
  using MultOpT = MultOp<T, LTensor, LTensor, LTensor>;

  LazyTensorImpl(ExecutionContext& ec, const IndexLabelVec& labels, T alpha, LTensor rhs1,
                 LTensor rhs2):
    LambdaTensorImpl<T>(labels, make_lambda(ec.exhw(), labels, alpha, rhs1, rhs2)),
    sources_{rhs1.tensor(), rhs2.tensor()} {}

  std::vector<TensorBase*> source_tensors() const override {
    std::vector<TensorBase*> res;
    for(const auto& src: sources_) res.push_back(src.base_ptr());
    return res;
  }

private:
  std::vector<Tensor<T>> sources_;

  static void validate(const IndexLabelVec& labels, const LTensor& rhs1, const LTensor& rhs2) {
    if(rhs1.has_str_lbl() || rhs2.has_str_lbl())
      tamm_terminate("[TAMM ERROR] Lazy tensor sources must be labeled with TiledIndexLabels, "
                     "not strings");
    if(!rhs1.tensor().is_dense() || !rhs2.tensor().is_dense())
      tamm_terminate("[TAMM ERROR] Lazy tensor sources must be dense tensors");

    const auto& lbls1 = rhs1.labels();
    const auto& lbls2 = rhs2.labels();
    for(const auto& lbl: labels) {
      if(lbl.is_dependent())
        tamm_terminate("[TAMM ERROR] Lazy tensor labels cannot be dependent labels");
      if(std::find(lbls1.begin(), lbls1.end(), lbl) == lbls1.end() &&
         std::find(lbls2.begin(), lbls2.end(), lbl) == lbls2.end())
        tamm_terminate("[TAMM ERROR] Every lazy tensor label must appear in one of its sources");
    }
    for(const auto& lbl: lbls1) {
      if(lbl.is_dependent())
        tamm_terminate("[TAMM ERROR] Lazy tensor sources cannot use dependent labels");
    }
    for(const auto& lbl: lbls2) {
      if(lbl.is_dependent())
        tamm_terminate("[TAMM ERROR] Lazy tensor sources cannot use dependent labels");
    }
  }

  static Func make_lambda(ExecutionHW hw, const IndexLabelVec& labels, T alpha, LTensor rhs1,
                          LTensor rhs2) {
    validate(labels, rhs1, rhs2);

    // Unallocated tensor standing in for the lazy tensor as the MultOp's LHS: it supplies
    // block sizes and dims only and is never read or written.
    Tensor<T> lhs_shape(labels);
    // MultOp's constructor relabels, assigns the integer labels and validates the contraction.
    auto mop =
      std::make_shared<const MultOpT>(LTensor(lhs_shape, labels), alpha, rhs1, rhs2, false);
    auto maps = std::make_shared<const typename MultOpT::BufaccMaps>(mop->bufacc_maps());

    return [mop, maps, hw](const IndexVector& blockid, span<T> buff) {
      EXPECTS(buff.size() == mop->lhs().tensor().block_size(blockid));

      auto&      oprof     = OpProfiler::instance();
      const bool outermost = (oprof.lazyDepth++ == 0);
      const auto t0        = std::chrono::high_resolution_clock::now();

      std::fill(buff.begin(), buff.end(), T{0});

      // Nested get/compute times are not charged to the multOp* counters: the consuming
      // operation's get timer already covers this call, and the scheduler subtracts
      // lazyBlockTime from it.
      double                              discard = 0;
      const typename MultOpT::BlockTimers timers{&discard, &discard, &discard, &discard};
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
      auto& thandle = GPUStreamPool::getInstance().getStream();
#else
      gpuStream_t thandle{};
#endif
      mop->compute_block(*maps, hw, thandle, blockid, blockid, buff.data(), timers);

      --oprof.lazyDepth;
      if(outermost) {
        oprof.lazyBlockTime += std::chrono::duration_cast<std::chrono::duration<double>>(
                                 std::chrono::high_resolution_clock::now() - t0)
                                 .count();
        oprof.lazyBlockCount++;
      }
    };
  }
}; // class LazyTensorImpl

template<typename T>
template<typename AlphaT>
Tensor<T> Tensor<T>::lazy(ExecutionContext& ec, const IndexLabelVec& labels,
                          std::tuple<AlphaT, LabeledTensor<T>, LabeledTensor<T>> expr) {
  Tensor<T> res;
  res.impl_ = std::make_shared<LazyTensorImpl<T>>(ec, labels, static_cast<T>(std::get<0>(expr)),
                                                  std::get<1>(expr), std::get<2>(expr));
  return res;
}

template<typename T>
Tensor<T> Tensor<T>::lazy(ExecutionContext& ec, const IndexLabelVec& labels,
                          std::tuple<LabeledTensor<T>, LabeledTensor<T>> expr) {
  return lazy(ec, labels, std::make_tuple(T{1}, std::get<0>(expr), std::get<1>(expr)));
}

} // namespace tamm
