#ifndef TAMM_BLOCK_LAMBDA_PLAN_HPP_
#define TAMM_BLOCK_LAMBDA_PLAN_HPP_

#include "tamm/blockops_cpu.hpp"
#include "tamm/types.hpp"

namespace tamm {
///////////////////////////////////////////////////////////////////////////////
//
//                             BlockLambdaPlan
//
///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct LabeledBlockSpan {
  BlockSpan<T> block_span;
  IndexLabelVec labels;
};

template <typename... Labels>
class BlockLambdaPlan {
 public:
  constexpr static int N = sizeof...(Labels);
  enum class Plan { flat, ipgen_loop, invalid };

  BlockLambdaPlan(Labels&&... labels)
      : plan_{Plan::invalid}, labels_{labels...} {
    prep_flat_plan();
    if (plan_ == Plan::invalid) {
      prep_ipgen_loop_plan();
    }
    EXPECTS(plan_ != Plan::invalid);
  }

  template <typename Func, typename... BlockSpans>
  void apply(Func&& func, BlockSpans&&... blocks) {
    EXPECTS(plan_ != Plan::invalid);
    switch (plan_) {
      case Plan::flat:
        blockops::cpu::flat_lambda(func, blocks...);
        break;
      case Plan::ipgen_loop:
        ipgen_loop_plan_.update_plan(std::forward<BlockSpans>(blocks)...);
        blockops::cpu::ipgen_loop_lambda(std::forward<Func>(func),
                                         std::forward<BlockSpans>(blocks)...);
        break;
      case Plan::invalid:
        UNREACHABLE();
        break;
    }
  }

 private:
  void prep_flat_plan() {
    plan_ = Plan::flat;
    std::array<IndexLabelVec, sizeof...(Labels)> labels_list{labels_};
    for (size_t i = 1; i < labels_list.size(); i++) {
      if (labels_list[0].size() != labels_list[1].size()) {
        plan_ = Plan::invalid;
        return;
      }
    }
    if (labels_list.size() > 0 &&
        internal::unique_entries(labels_list[0]).size() !=
            labels_list[0].size()) {
      plan_ = Plan::invalid;
      return;
    }
    for (size_t i = 1; i < labels_list.size(); i++) {
      if (!std::equal(labels_list[0].begin(), labels_list[0].end(),
                      labels_list[i].begin())) {
        plan_ = Plan::invalid;
        return;
      }
    }
  }

  void prep_ipgen_loop_plan() {
    ipgen_loop_plan_ = blockops::cpu::IpGenLoopBuilder<N>{labels_};
    plan_ = Plan::ipgen_loop;
  }

  Plan plan_;
  std::array<IndexLabelVec, N> labels_;
  blockops::cpu::IpGenLoopBuilder<N> ipgen_loop_plan_;
};

// class BlockLambdaPlan<IndexLabelVec>;
// class BlockLambdaPlan<IndexLabelVec, IndexLabelVec>;
// class BlockLambdaPlan<IndexLabelVec, IndexLabelVec, IndexLabelVec>;

template <typename Func, typename... LabeledBlockSpans>
void block_lambda(Func&& func, LabeledBlockSpans&&... lbss) {
  BlockLambdaPlan blp{std::forward<Func>(func),
                      std::forward<LabeledBlockSpans>(lbss).labels...};
  blp.apply(std::forward<LabeledBlockSpans>(lbss).block_span...);
}

template <typename Func>
void block_lambda(Func&& func, LabeledBlockSpan<double>& arg1,
                  LabeledBlockSpan<double>& arg2);

}  // namespace tamm

#endif  // TAMM_BLOCK_LAMBDA_PLAN_HPP_