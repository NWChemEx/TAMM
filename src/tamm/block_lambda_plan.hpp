#pragma once

#include "tamm/blockops_cpu.hpp"
#include "tamm/types.hpp"
#include <algorithm>
#include <array>

namespace tamm {
///////////////////////////////////////////////////////////////////////////////
//
//                             BlockLambdaPlan
//
///////////////////////////////////////////////////////////////////////////////

template<typename T>
struct LabeledBlockSpan {
  BlockSpan<T>  block_span;
  IndexLabelVec labels;
};

template<typename... Labels>
class BlockLambdaPlan {
public:
  constexpr static int N = sizeof...(Labels);
  enum class Plan { flat, ipgen_loop, invalid };

  BlockLambdaPlan(Labels&&... labels): plan_{Plan::invalid}, labels_{labels...} {
    prep_flat_plan();
    if(plan_ == Plan::invalid) { prep_ipgen_loop_plan(); }
    EXPECTS(plan_ != Plan::invalid);
  }

  template<typename Func, typename... BlockSpans>
  void apply(Func&& func, BlockSpans&&... blocks) {
    EXPECTS(plan_ != Plan::invalid);
    switch(plan_) {
      case Plan::flat: blockops::cpu::flat_lambda(func, blocks...); break;
      case Plan::ipgen_loop:
        ipgen_loop_plan_.update_plan(std::forward<BlockSpans>(blocks)...);
        blockops::cpu::ipgen_loop_lambda(std::forward<Func>(func),
                                         std::forward<BlockSpans>(blocks)...);
        break;
      case Plan::invalid: UNREACHABLE(); break;
    }
  }

private:
  void prep_flat_plan() {
    plan_ = Plan::flat;
    std::array<IndexLabelVec, sizeof...(Labels)> labels_list{labels_};
    if(labels_list.empty()) return;

    // The flat plan applies only when every operand has the exact same label
    // list (same size and same labels), and those labels are unique.
    // Bug fix: the size check previously compared against labels_list[1] for
    // every i (a fixed index) instead of labels_list[i].
    const auto& ref = labels_list[0];
    const bool  all_same =
      internal::unique_entries(ref).size() == ref.size() &&
      std::ranges::all_of(labels_list, [&](const auto& l) { return std::ranges::equal(l, ref); });
    if(!all_same) { plan_ = Plan::invalid; }
  }

  void prep_ipgen_loop_plan() {
    ipgen_loop_plan_ = blockops::cpu::IpGenLoopBuilder<N>{labels_};
    plan_            = Plan::ipgen_loop;
  }

  Plan                               plan_;
  std::array<IndexLabelVec, N>       labels_;
  blockops::cpu::IpGenLoopBuilder<N> ipgen_loop_plan_;
};

// class BlockLambdaPlan<IndexLabelVec>;
// class BlockLambdaPlan<IndexLabelVec, IndexLabelVec>;
// class BlockLambdaPlan<IndexLabelVec, IndexLabelVec, IndexLabelVec>;

template<typename Func, typename... LabeledBlockSpans>
void block_lambda(Func&& func, LabeledBlockSpans&&... lbss) {
  BlockLambdaPlan blp{std::forward<Func>(func), std::forward<LabeledBlockSpans>(lbss).labels...};
  blp.apply(std::forward<LabeledBlockSpans>(lbss).block_span...);
}

template<typename Func>
void block_lambda(Func&& func, LabeledBlockSpan<double>& arg1, LabeledBlockSpan<double>& arg2);

} // namespace tamm
