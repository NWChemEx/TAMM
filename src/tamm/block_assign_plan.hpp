#pragma once

#include <vector>

#include "tamm/block_span.hpp"
#include "tamm/blockops_blas.hpp"
#include "tamm/blockops_cpu.hpp"
#include "tamm/errors.hpp"
#include "tamm/ip_hptt.hpp"
#include "tamm/types.hpp"

///////////////////////////////////////////////////////////////////////////////
//
//                BlockAssignPlan
//
///////////////////////////////////////////////////////////////////////////////

namespace tamm {
/**
 * @brief Block assign plan for block assignment on Tensor Ops
 *
 * @todo HPTT calls should be fixed for mixed precision (T1 && T2) when we allow
 * those operations in make_op, currently only complex and scalar mix is allowed
 *
 */
class BlockAssignPlan {
public:
  enum class OpType { set, update };

  BlockAssignPlan(): plan_{Plan::invalid} {}

  template<typename T>
  BlockAssignPlan(const std::vector<T>& lhs_labels, const std::vector<T>& rhs_labels,
                  OpType optype):
    optype_{optype},
    // lhs_labels{lhs_labels},
    // rhs_labels{rhs_labels},
    plan_{Plan::invalid} {
    prep_flat_plan(lhs_labels, rhs_labels);
    if(plan_ == Plan::invalid) { prep_hptt(lhs_labels, rhs_labels); }
    if(plan_ == Plan::invalid) { prep_index_permute_plan(lhs_labels, rhs_labels); }
    if(plan_ == Plan::invalid) { prep_general_loop_plan(lhs_labels, rhs_labels); }
    if(plan_ == Plan::invalid) { prep_general_plan(lhs_labels, rhs_labels); }
    EXPECTS(plan_ != Plan::invalid);
  }

  // @todo In the current op creation rscale is always the same type with lhs
  /// this might need to change depending on the changes on op creation
  template<typename T1, typename T2>
  void apply_impl(BlockSpan<T1>& lhs, const BlockSpan<T2>& rhs) {
    EXPECTS(plan_ != Plan::invalid);
    EXPECTS(lhs.buf() != nullptr);
    EXPECTS(rhs.buf() != nullptr);

    T1 lscale = (optype_ == OpType::set) ? 0 : 1;
    switch(plan_) {
      case Plan::flat_assign: blockops::cpu::flat_assign(lhs, rhs); break;
      case Plan::flat_update: blockops::cpu::flat_update(lhs, rhs); break;
      case Plan::hptt:
        blockops::hptt::index_permute_hptt(lscale, lhs.buf(), 1, rhs.buf(), ip_plan_.perm_,
                                           rhs.block_dims());
        break;
      case Plan::index_permute_assign:
        blockops::cpu::index_permute_assign(lhs.buf(), rhs.buf(), ip_plan_.perm_, lhs.block_dims());
        break;
      case Plan::index_permute_update:
        blockops::cpu::index_permute_update(lhs.buf(), rhs.buf(), ip_plan_.perm_, lhs.block_dims());
        break;
      case Plan::ipgen_loop_assign:
        ipgen_loop_builder_.update_plan(lhs, rhs);
        blockops::cpu::ipgen_loop_assign(lhs.buf(), ipgen_loop_builder_.u2ald()[0], rhs.buf(),
                                         ipgen_loop_builder_.u2ald()[1],
                                         ipgen_loop_builder_.unique_label_dims());
        break;
      case Plan::ipgen_loop_update:
        ipgen_loop_builder_.update_plan(lhs, rhs);
        blockops::cpu::ipgen_loop_update(lhs.buf(), ipgen_loop_builder_.u2ald()[0], rhs.buf(),
                                         ipgen_loop_builder_.u2ald()[1],
                                         ipgen_loop_builder_.unique_label_dims());
        break;
      case Plan::general_assign:
        blockops::cpu::ipgen_assign(lhs.buf(), lhs.block_dims(), gen_plan_.dperm_map_,
                                    gen_plan_.dinv_perm_map_, rhs.buf(), rhs.block_dims(),
                                    gen_plan_.sperm_map_, gen_plan_.unique_label_count_);
        break;
      case Plan::general_update:
        blockops::cpu::ipgen_update(lhs.buf(), lhs.block_dims(), gen_plan_.dperm_map_,
                                    gen_plan_.dinv_perm_map_, rhs.buf(), rhs.block_dims(),
                                    gen_plan_.sperm_map_, gen_plan_.unique_label_count_);
        break;
      case Plan::invalid: UNREACHABLE(); break;
    }
  }

  /// @todo Have to split the actual apply to pre-process depending on the
  /// types
  template<typename T1, typename T2>
  void apply(BlockSpan<T1>& lhs, const BlockSpan<T2>& rhs) {
    if constexpr(internal::is_complex_v<T1> || internal::is_complex_v<T2>) {
      std::vector<T1> new_rhs_vec(rhs.num_elements());
      blockops::bops_blas::prep_rhs_buffer<T1>(rhs, new_rhs_vec);
      BlockSpan<T1> rhs_new{new_rhs_vec.data(), rhs.block_dims()};
      apply_impl(lhs, rhs_new);
    }
    else if constexpr(std::is_convertible_v<T2, T1>) { apply_impl(lhs, rhs); }
    else { NOT_ALLOWED(); }
  }

  // @todo In the current op creation rscale is always the same type with lhs
  /// this might need to change depending on the changes on op creation
  template<typename T1, typename T2>
  void apply_impl(BlockSpan<T1>& lhs, T1 rscale, const BlockSpan<T2>& rhs) {
    EXPECTS(plan_ != Plan::invalid);
    EXPECTS(lhs.buf() != nullptr);
    EXPECTS(rhs.buf() != nullptr);

    T1 lscale = (optype_ == OpType::set) ? 0 : 1;
    switch(plan_) {
      case Plan::flat_assign: blockops::cpu::flat_assign(lhs, rscale, rhs); break;
      case Plan::flat_update: blockops::cpu::flat_update(lhs, rscale, rhs); break;
      case Plan::hptt:
        blockops::hptt::index_permute_hptt(lscale, lhs.buf(), rscale, rhs.buf(), ip_plan_.perm_,
                                           rhs.block_dims());
        break;
      case Plan::index_permute_assign:
        blockops::cpu::index_permute_assign(lhs.buf(), rscale, rhs.buf(), ip_plan_.perm_,
                                            lhs.block_dims());
        break;
      case Plan::index_permute_update:
        blockops::cpu::index_permute_update(lhs.buf(), rscale, rhs.buf(), ip_plan_.perm_,
                                            lhs.block_dims());
        break;
      case Plan::ipgen_loop_assign:
        ipgen_loop_builder_.update_plan(lhs, rhs);
        blockops::cpu::ipgen_loop_assign(lhs.buf(), ipgen_loop_builder_.u2ald()[0], rscale,
                                         rhs.buf(), ipgen_loop_builder_.u2ald()[1],
                                         ipgen_loop_builder_.unique_label_dims());
        break;
      case Plan::ipgen_loop_update:
        ipgen_loop_builder_.update_plan(lhs, rhs);
        blockops::cpu::ipgen_loop_update(lhs.buf(), ipgen_loop_builder_.u2ald()[0], rscale,
                                         rhs.buf(), ipgen_loop_builder_.u2ald()[1],
                                         ipgen_loop_builder_.unique_label_dims());
        break;
      case Plan::general_assign:
        blockops::cpu::ipgen_assign(lhs.buf(), lhs.block_dims(), gen_plan_.dperm_map_,
                                    gen_plan_.dinv_perm_map_, rscale, rhs.buf(), rhs.block_dims(),
                                    gen_plan_.sperm_map_, gen_plan_.unique_label_count_);
        break;
      case Plan::general_update:
        blockops::cpu::ipgen_update(lhs.buf(), lhs.block_dims(), gen_plan_.dperm_map_,
                                    gen_plan_.dinv_perm_map_, rscale, rhs.buf(), rhs.block_dims(),
                                    gen_plan_.sperm_map_, gen_plan_.unique_label_count_);
        break;
      case Plan::invalid: UNREACHABLE(); break;
    }
  }

  /// @todo Have to split the actual apply to pre-process depending on the
  /// types
  template<typename T1, typename T2>
  void apply(BlockSpan<T1>& lhs, Scalar rscale, const BlockSpan<T2>& rhs) {
    // clang-format off
    std::visit(overloaded{
        [&](auto alpha) {
          if constexpr(std::is_same_v<T1, T2> &&
                       std::is_convertible_v<decltype(alpha), T1>) {
            apply_impl(lhs, static_cast<T1>(alpha), rhs);
          }
          else if constexpr(internal::is_complex_v<T1> ||
                            (internal::is_complex_v<T2> &&
                            std::is_convertible_v<decltype(alpha), T1>) 
                            ) {
            std::vector<T1> new_rhs_vec(rhs.num_elements());
            blockops::bops_blas::prep_rhs_buffer<T1>(rhs, new_rhs_vec);
            BlockSpan<T1> rhs_new{new_rhs_vec.data(), rhs.block_dims()};
            apply_impl(lhs, static_cast<T1>(alpha), rhs_new);
          }
          else if constexpr(std::is_convertible_v<T2, T1> &&
                            std::is_convertible_v<decltype(alpha), T1>) {
            apply_impl(lhs, static_cast<T1>(alpha), rhs);
          }
          else {
            std::cerr << "not here" << std::endl;
            NOT_ALLOWED();
          }
        }},
        rscale.value());
    // clang-format on
  }

  /// @todo In the current op creation {r-l}scale is always the same type with
  /// lhs this might need to change depending on the changes on op creation
  template<typename T1, typename T2>
  void apply_impl(T1 lscale, BlockSpan<T1>& lhs, T2 rscale, const BlockSpan<T2>& rhs) {
    EXPECTS(plan_ != Plan::invalid);
    EXPECTS(optype_ == OpType::update); // only update can take lscale
    EXPECTS(lhs.buf() != nullptr);
    EXPECTS(rhs.buf() != nullptr);
    switch(plan_) {
      case Plan::flat_assign: NOT_ALLOWED(); break;
      case Plan::flat_update: blockops::cpu::flat_update(lscale, lhs, rscale, rhs); break;
      case Plan::hptt:
        blockops::hptt::index_permute_hptt(lscale, lhs.buf(), rscale, rhs.buf(), ip_plan_.perm_,
                                           rhs.block_dims());
        break;
      case Plan::index_permute_assign: NOT_ALLOWED(); break;
      case Plan::index_permute_update:
        blockops::cpu::index_permute_update(lscale, lhs.buf(), rscale, rhs.buf(), ip_plan_.perm_,
                                            lhs.block_dims());
        break;
      case Plan::ipgen_loop_assign: NOT_ALLOWED(); break;
      case Plan::ipgen_loop_update:
        ipgen_loop_builder_.update_plan(lhs, rhs);
        blockops::cpu::ipgen_loop_update(lscale, lhs.buf(), ipgen_loop_builder_.u2ald()[0], rscale,
                                         rhs.buf(), ipgen_loop_builder_.u2ald()[1],
                                         ipgen_loop_builder_.unique_label_dims());
        break;
      case Plan::general_assign: NOT_ALLOWED(); break;
      case Plan::general_update:
        blockops::cpu::ipgen_update(lscale, lhs.buf(), lhs.block_dims(), gen_plan_.dperm_map_,
                                    gen_plan_.dinv_perm_map_, rscale, rhs.buf(), rhs.block_dims(),
                                    gen_plan_.sperm_map_, gen_plan_.unique_label_count_);
        break;
      case Plan::invalid: UNREACHABLE(); break;
    }
  }

  /// @todo Have to split the actual apply and the implementation to
  /// pre-process depending on the types. Note: using const_expr in the actual
  /// implementation breaks the recursive call to apply
  template<typename T1, typename T2>
  void apply(T1 lscale, BlockSpan<T1>& lhs, T2 rscale, const BlockSpan<T2>& rhs) {
    std::visit(overloaded{[&](auto beta, auto alpha) {
                 if constexpr(internal::is_complex_v<T1> ||
                              (internal::is_complex_v<T2> &&
                               std::is_convertible_v<decltype(beta), T1> &&
                               std::is_convertible_v<decltype(alpha), T2>) ) {
                   std::vector<T1> new_rhs_vec(rhs.num_elements());
                   blockops::bops_blas::prep_rhs_buffer<T1>(rhs, new_rhs_vec);
                   BlockSpan<T1> rhs_new{new_rhs_vec.data(), rhs.block_dims()};
                   apply_impl(static_cast<T1>(beta), lhs, static_cast<T2>(alpha), rhs_new);
                 }
                 else if constexpr(std::is_convertible_v<T2, T1>) {
                   apply_impl(lscale, lhs, rscale, rhs);
                 }
                 else { NOT_ALLOWED(); }
               }},
               lscale.value(), rscale.value());
  }

private:
  template<typename T>
  void prep_flat_plan(const std::vector<T>& lhs_labels, const std::vector<T>& rhs_labels) {
    plan_ = optype_ == OpType::set ? Plan::flat_assign : Plan::flat_update;
    if(lhs_labels.size() != rhs_labels.size()) {
      plan_ = Plan::invalid;
      return;
    }
    else if(internal::unique_entries(lhs_labels).size() != lhs_labels.size()) {
      plan_ = Plan::invalid;
      return;
    }
    else if(!std::equal(lhs_labels.begin(), lhs_labels.end(), rhs_labels.begin())) {
      plan_ = Plan::invalid;
      return;
    }
  }

  template<typename T>
  void prep_hptt(const std::vector<T>& lhs_labels, const std::vector<T>& rhs_labels) {
    prep_index_permute_plan(lhs_labels, rhs_labels);
    if(plan_ == Plan::invalid) {
      return; // no index permute plan, no hptt plan
    }
    plan_ = Plan::hptt;
  }

  template<typename T>
  void prep_index_permute_plan(const std::vector<T>& lhs_labels, const std::vector<T>& rhs_labels) {
    plan_ = optype_ == OpType::set ? Plan::index_permute_assign : Plan::index_permute_update;
    if(lhs_labels.size() != rhs_labels.size()) {
      plan_ = Plan::invalid;
      return;
    }
    else if(internal::unique_entries(lhs_labels).size() != lhs_labels.size()) {
      plan_ = Plan::invalid;
      return;
    }
    else if(!std::is_permutation(lhs_labels.begin(), lhs_labels.end(), rhs_labels.begin())) {
      plan_ = Plan::invalid;
      return;
    }
    ip_plan_.perm_ = internal::perm_compute(rhs_labels, lhs_labels);
  }

  template<typename T>
  void prep_general_loop_plan(const std::vector<T>& lhs_labels, const std::vector<T>& rhs_labels) {
    ipgen_loop_builder_ =
      blockops::cpu::IpGenLoopBuilder<2>(std::array<std::vector<T>, 2>{lhs_labels, rhs_labels});
    plan_ = optype_ == OpType::set ? Plan::ipgen_loop_assign : Plan::ipgen_loop_update;
  }

  template<typename T>
  void prep_general_plan(const std::vector<T>& lhs_labels, const std::vector<T>& rhs_labels) {
    const auto& dlabel = lhs_labels;
    const auto& slabel = rhs_labels;
    plan_              = optype_ == OpType::set ? Plan::general_assign : Plan::general_update;
    std::vector<T> unique_labels = internal::unique_entries(dlabel);
    // unique_labels = internal::sort_on_dependence(unique_labels);

    gen_plan_.unique_label_count_ = unique_labels.size();
    gen_plan_.dperm_map_          = internal::perm_map_compute(unique_labels, dlabel);
    gen_plan_.sperm_map_          = internal::perm_map_compute(unique_labels, slabel);
    gen_plan_.dinv_perm_map_      = internal::perm_map_compute(dlabel, unique_labels);
  }

  enum class Plan {
    flat_assign,
    flat_update,
    hptt,
    index_permute_assign,
    index_permute_update,
    ipgen_loop_assign,
    ipgen_loop_update,
    general_assign,
    general_update,
    invalid
  };
  OpType optype_;
  //   std::vector<T> lhs_labels;
  //   std::vector<T> rhs_labels;
  Plan plan_;
  struct IndexPermutePlan {
    PermVector perm_; // used by index_permute plans
  } ip_plan_;
  struct GeneralPlan {
    int        unique_label_count_;
    PermVector sperm_map_;
    PermVector dperm_map_;
    PermVector dinv_perm_map_;
  } gen_plan_;
  blockops::cpu::IpGenLoopBuilder<2> ipgen_loop_builder_;

  std::string plan_str(Plan plan) {
    switch(plan) {
      case Plan::flat_assign: return "flat_assign";
      case Plan::flat_update: return "flat_update";
      case Plan::hptt: return "hptt";
      case Plan::index_permute_assign: return "index_permute_assign";
      case Plan::index_permute_update: return "index_permute_update";
      case Plan::ipgen_loop_assign: return "ipgen_loop_assign";
      case Plan::ipgen_loop_update: return "ipgen_loop_update";
      case Plan::general_assign: return "general_assign";
      case Plan::general_update: return "general_update";
      default: return "invalid"; break;
    }
  }
}; // class BlockAssignPlan

template<typename T1, typename T2>
void block_assign(BlockSpan<T1>& lhs, const std::vector<T2>& lhs_labels, const BlockSpan<T1>& rhs,
                  const std::vector<T2>& rhs_labels, bool is_assign) {
  BlockAssignPlan::OpType optype = is_assign ? BlockAssignPlan::OpType::set
                                             : BlockAssignPlan::OpType::update;
  BlockAssignPlan{lhs_labels, rhs_labels, optype}.apply(lhs, rhs);
}

void block_assign(BlockSpan<double>& lhs, const std::vector<int>& lhs_labels,
                  const BlockSpan<double>& rhs, const std::vector<int>& rhs_labels, bool is_assign);

} // namespace tamm
