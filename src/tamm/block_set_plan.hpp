#ifndef TAMM_BLOCK_SET_PLAN_HPP_
#define TAMM_BLOCK_SET_PLAN_HPP_

#include <set>
#include <vector>

#include "tamm/blockops_cpu.hpp"

namespace tamm {
///////////////////////////////////////////////////////////////////////////////
//
//                             BlockSetPlan
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief trivial plan for block set. Just to be symmetric and possibly delegate
 * to other libraries
 *
 */
class BlockSetPlan {
public:
    enum class OpType { set, update };
    BlockSetPlan() : plan_{Plan::invalid} {}

    BlockSetPlan(OpType optype) : optype_{optype} {
        plan_ = (optype == OpType::set) ? Plan::flat_set : Plan::flat_update;
    }

    template<typename T>
    BlockSetPlan(const std::vector<T>& lhs_labels, OpType optype) :
      optype_{optype}, plan_{Plan::invalid} {
        prep_flat_plan(lhs_labels, optype_);
        if(plan_ == Plan::invalid) { prep_general_plan(lhs_labels, optype_); }
        EXPECTS(plan_ != Plan::invalid);
    }

    template<typename T>
    void apply(BlockSpan<T>& bs, T value) {
        switch(plan_) {
            case Plan::flat_set: blockops::cpu::flat_set(bs, value); break;
            case Plan::flat_update:
                blockops::cpu::flat_update(bs, value);
                break;
            case Plan::general_set:
                ipgen_loop_builder_.update_plan(bs);
                blockops::cpu::ipgen_loop_set(
                  bs.buf(), ipgen_loop_builder_.u2ald()[0], value,
                  ipgen_loop_builder_.unique_label_dims());
                break;
            case Plan::general_update:
                ipgen_loop_builder_.update_plan(bs);
                blockops::cpu::ipgen_loop_update(
                  bs.buf(), ipgen_loop_builder_.u2ald()[0], value,
                  ipgen_loop_builder_.unique_label_dims());
                break;
            default: UNREACHABLE();
        }
    }

    template<typename T>
    void apply(T lscale, BlockSpan<T>& bs, T value) {
        switch(plan_) {
            case Plan::flat_set: NOT_ALLOWED(); break;
            case Plan::flat_update:
                blockops::cpu::flat_update(lscale, bs, value);
                break;
            case Plan::general_set: NOT_ALLOWED(); break;
            case Plan::general_update:
                ipgen_loop_builder_.update_plan(bs);
                blockops::cpu::ipgen_loop_update(
                  lscale, bs.buf(), ipgen_loop_builder_.u2ald()[0], value,
                  ipgen_loop_builder_.unique_label_dims());
                break;
            default: UNREACHABLE();
        }
    }

    template<typename T>
    void apply(BlockSpan<T>& bs, const Scalar& value) {
        std::visit(
          overloaded{[&](auto e) {
              if constexpr(std::is_assignable<T&, decltype(e)>::value) {
                  apply(bs, static_cast<T>(e));
              } else {
                  NOT_ALLOWED();
              }
          }},
          value.value());
    }

    template<typename T>
    void apply(Scalar lscale, BlockSpan<T>& bs, const Scalar& value) {
        std::visit(
          overloaded{[&](auto l) {
              std::visit(
                overloaded{[&](auto e) {
                    if constexpr(std::is_assignable<T&, decltype(l)>::value &&
                                 std::is_assignable<T&, decltype(e)>::value) {
                        apply(static_cast<T>(l), bs, static_cast<T>(e));
                    } else {
                        NOT_ALLOWED();
                    }
                }},
                value.value());
          }},
          lscale.value());
    }

    template<typename TSL, typename T>
    void apply(TSL lscale, BlockSpan<T>& bs, const Scalar& value) {
        apply(Scalar{lscale}, bs, value);
    }

    template<typename T, typename TV>
    void apply(const Scalar& lscale, BlockSpan<T>& bs, TV value) {
        apply(lscale, bs, Scalar{value});
    }

private:
    template<typename T>
    void prep_flat_plan(const std::vector<T>& labels, OpType optype) {
        // check for no repeated indices
        std::set<T> labels_set(labels.begin(), labels.end());
        if(labels_set.size() == labels.size()) {
            plan_ =
              (optype == OpType::set) ? Plan::flat_set : Plan::flat_update;
        }
    }

    template<typename T>
    void prep_general_plan(const std::vector<T>& labels, OpType optype) {
        // always valid
        ipgen_loop_builder_ = blockops::cpu::IpGenLoopBuilder<1>(
          std::array<std::vector<T>, 1>{labels});
        plan_ =
          (optype == OpType::set) ? Plan::general_set : Plan::general_update;
    }

    enum class Plan {
        invalid,
        flat_set,
        flat_update,
        general_set,
        general_update
    };
    OpType optype_;
    Plan plan_;
    blockops::cpu::IpGenLoopBuilder<1> ipgen_loop_builder_;
}; // class BlockSetPlan

} // namespace tamm

#endif // TAMM_BLOCK_SET_PLAN_HPP_