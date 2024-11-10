#pragma once

#define SETOP_LOCALIZE_LHS

#include <algorithm>
#include <memory>
#include <vector>

#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/label_translator.hpp"
#include "tamm/labeled_tensor.hpp"
#include "tamm/runtime_engine.hpp"
#include "tamm/tensor.hpp"
#include "tamm/types.hpp"
#include "tamm/utils.hpp"

#include "tamm/block_set_plan.hpp"

namespace tamm {
template<typename T, typename LabeledTensorT>
class SetOp;
}
namespace tamm::internal {

template<typename T, typename LabeledTensorT>
struct SetOpPlanBase {
  using SetOpT = SetOp<T, LabeledTensorT>;
  TensorBase* writes(const SetOpT& setop) const {
    auto ret1 = local_writes(setop);
    auto ret2 = global_writes(setop);
    ret1.insert(ret1.end(), ret2.begin(), ret2.end());
    return !ret1.empty() ? ret1[0] : nullptr;
  }

  TensorBase* accumulates(const SetOpT& setop) const {
    auto ret1 = local_accumulates(setop);
    auto ret2 = global_accumulates(setop);
    ret1.insert(ret1.end(), ret2.begin(), ret2.end());
    return !ret1.empty() ? ret1[0] : nullptr;
  }

  virtual std::vector<TensorBase*> global_writes(const SetOpT& setop) const      = 0;
  virtual std::vector<TensorBase*> global_accumulates(const SetOpT& setop) const = 0;
  virtual std::vector<TensorBase*> local_writes(const SetOpT& setop) const       = 0;
  virtual std::vector<TensorBase*> local_accumulates(const SetOpT& setop) const  = 0;
  virtual void apply(const SetOpT& setop, ExecutionContext& ec, ExecutionHW hw)  = 0;
}; // SetOpPlanBase

template<typename T, typename LabeledTensorT>
struct FlatPlan: public SetOpPlanBase<T, LabeledTensorT> {
  using SetOpT = SetOp<T, LabeledTensorT>;
  std::vector<TensorBase*> global_writes(const SetOpT& setop) const override { return {}; }
  std::vector<TensorBase*> global_accumulates(const SetOpT& setop) const override { return {}; }
  std::vector<TensorBase*> local_writes(const SetOpT& setop) const override {
    if(setop.is_assign()) { return {setop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_accumulates(const SetOpT& setop) const override {
    if(!setop.is_assign()) { return {setop.lhs().base_ptr()}; }
    else { return {}; }
  }
  void apply(const SetOpT& setop, ExecutionContext& ec, ExecutionHW hw) override;
};

template<typename T, typename LabeledTensorT>
struct LHSPlan: public SetOpPlanBase<T, LabeledTensorT> {
  using SetOpT = SetOp<T, LabeledTensorT>;
  std::vector<TensorBase*> global_writes(const SetOpT& setop) const override { return {}; }
  std::vector<TensorBase*> global_accumulates(const SetOpT& setop) const override { return {}; }
  std::vector<TensorBase*> local_writes(const SetOpT& setop) const override {
    if(setop.is_assign()) { return {setop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_accumulates(const SetOpT& setop) const override {
    if(!setop.is_assign()) { return {setop.lhs().base_ptr()}; }
    else { return {}; }
  }
  void apply(const SetOpT& setop, ExecutionContext& ec, ExecutionHW hw) override;
}; // namespace tamm::internal

template<typename T, typename LabeledTensorT>
struct GeneralFlatPlan: public SetOpPlanBase<T, LabeledTensorT> {
  using SetOpT = SetOp<T, LabeledTensorT>;
  std::vector<TensorBase*> global_writes(const SetOpT& setop) const override {
    if(setop.is_assign()) { return {setop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> global_accumulates(const SetOpT& setop) const override {
    if(!setop.is_assign()) { return {setop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_writes(const SetOpT& setop) const override {
    if(setop.is_assign()) { return {setop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_accumulates(const SetOpT& setop) const override {
    if(!setop.is_assign()) { return {setop.lhs().base_ptr()}; }
    else { return {}; }
  }
  void apply(const SetOpT& setop, ExecutionContext& ec, ExecutionHW hw) override;
}; // GeneralFlatPlan

template<typename T, typename LabeledTensorT>
struct GeneralLHSPlan: public SetOpPlanBase<T, LabeledTensorT> {
  using SetOpT = SetOp<T, LabeledTensorT>;
  std::vector<TensorBase*> global_writes(const SetOpT& setop) const override {
    if(setop.is_assign()) { return {setop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> global_accumulates(const SetOpT& setop) const override {
    if(!setop.is_assign()) { return {setop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_writes(const SetOpT& setop) const override {
    if(setop.is_assign()) { return {setop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_accumulates(const SetOpT& setop) const override {
    if(!setop.is_assign()) { return {setop.lhs().base_ptr()}; }
    else { return {}; }
  }
  void apply(const SetOpT& setop, ExecutionContext& ec, ExecutionHW hw) override;
}; // GeneralLHSPlan

} // namespace tamm::internal

namespace tamm {
template<typename T, typename LabeledTensorT>
class SetOp: public Op {
public:
  SetOp() = default;

  SetOp(LabeledTensorT lhs, T alpha, bool is_assign):
    lhs_{lhs},
    alpha_{alpha},
    is_assign_{is_assign},
    plan_obj_{nullptr},
    general_plan_obj_{nullptr} {
    if(!lhs.has_str_lbl() && !lhs.labels().empty()) {
      auto lbls = lhs.labels();
      internal::update_labels(lbls);
      lhs_.set_labels(lbls);
    }

    if(lhs.has_str_lbl()) { fillin_labels(); }

    validate();
    const auto& tensor = lhs_.tensor();
    if(!internal::is_slicing(lhs_) && !internal::has_duplicates(lhs_.labels()) &&
       tensor.kind() != TensorBase::TensorKind::view) {
      plan_             = Plan::flat;
      plan_obj_         = std::make_shared<internal::FlatPlan<T, LabeledTensorT>>();
      general_plan_obj_ = std::make_shared<internal::GeneralFlatPlan<T, LabeledTensorT>>();
    }
    else {
      plan_             = Plan::lhs;
      plan_obj_         = std::make_shared<internal::LHSPlan<T, LabeledTensorT>>();
      general_plan_obj_ = std::make_shared<internal::GeneralLHSPlan<T, LabeledTensorT>>();
    }
    EXPECTS(plan_ != Plan::invalid);
    EXPECTS(plan_obj_ != nullptr);
  }

  SetOp(const SetOp<T, LabeledTensorT>&) = default;

  T alpha() const { return alpha_; }

  LabeledTensorT lhs() const { return lhs_; }

  bool is_assign() const { return is_assign_; }

  OpList canonicalize() const override { return OpList{(*this)}; }

  std::shared_ptr<Op> clone() const override {
    return std::shared_ptr<Op>(new SetOp<T, LabeledTensorT>{*this});
  }

  OpType op_type() const override { return OpType::set; }
  void   execute(ExecutionContext& ec, ExecutionHW hw = ExecutionHW::CPU) override {
    EXPECTS(plan_ != Plan::invalid);
    if(lhs_.tensor().kind() != TensorBase::TensorKind::view &&
       lhs_.tensor().execution_context()->pg() == ec.pg()) {
        plan_obj_->apply(*this, ec, hw);
    }
    else { general_plan_obj_->apply(*this, ec, hw); }
  }

  TensorBase* writes() const { return plan_obj_->writes(*this); }

  std::vector<TensorBase*> reads() const { return {}; }

  TensorBase* accumulates() const { return plan_obj_->accumulates(*this); }

  TensorBase* writes(const ExecutionContext& ec) const {
    if(lhs_.tensor().pg() == ec.pg()) { return plan_obj_->writes(*this); }
    else { general_plan_obj_->writes(*this); }
  }

  std::vector<TensorBase*> reads(const ExecutionContext& ec) const {
    if(lhs_.tensor().pg() == ec.pg()) { return plan_obj_->reads(*this); }
    else { general_plan_obj_->reads(*this); }
  }

  TensorBase* accumulates(const ExecutionContext& ec) const {
    if(lhs_.tensor().pg() == ec.pg()) { return plan_obj_->accumulates(*this); }
    else { general_plan_obj_->accumulates(*this); }
  }

  bool is_memory_barrier() const { return false; }

protected:
  void fillin_labels() {
    using internal::fillin_tensor_label_from_map;
    using internal::update_fillin_map;
    std::map<std::string, Label> str_to_labels;
    update_fillin_map(str_to_labels, lhs_.str_map(), lhs_.str_labels(), 0);
    fillin_tensor_label_from_map(lhs_, str_to_labels);
  }

  /**
   * @brief Check if the parameters form a valid operation. The parameters
   * form a valid operation if:
   *
   * 1. Every label depended on by another label (i.e., all 'd' such that
   * there exists label 'l(d)') is bound at least once
   *
   * 2. There are no conflicting dependent label specifications. That if
   * 'a(i)' is a label in either lta or ltc, there is no label 'a(j)' (i!=j)
   * in either lta or ltc.
   *
   * @pre lhs_.validate(), rhs1_.validate() and rhs2_.validate() have been
   *  invoked
   */
  void validate() {
    IndexLabelVec ilv{lhs_.labels()};

    for(size_t i = 0; i < ilv.size(); i++) {
      for(const auto& dl: ilv[i].secondary_labels()) {
        size_t j;
        for(j = 0; j < ilv.size(); j++) {
          if(dl.tiled_index_space() == ilv[j].tiled_index_space() && dl.label() == ilv[j].label()) {
            break;
          }
        }
        EXPECTS(j < ilv.size());
      }
    }

    for(size_t i = 0; i < ilv.size(); i++) {
      const auto& ilbl = ilv[i];
      for(size_t j = i + 1; j < ilv.size(); j++) {
        const auto& jlbl = ilv[j];
        if(ilbl.tiled_index_space() == jlbl.tiled_index_space() && ilbl.label() == jlbl.label() &&
           ilbl.label_str() == jlbl.label_str()) {
          EXPECTS(ilbl == jlbl);
        }
      }
    }
  }

  LabeledTensorT lhs_;
  T              alpha_;
  bool           is_assign_;

  enum class Plan { invalid, flat, /*dense,*/ lhs, general_flat, general_lhs };
  Plan                                                        plan_ = Plan::invalid;
  std::shared_ptr<internal::SetOpPlanBase<T, LabeledTensorT>> plan_obj_;
  std::shared_ptr<internal::SetOpPlanBase<T, LabeledTensorT>> general_plan_obj_;

public:
  std::string opstr_;

}; // class SetOp

} // namespace tamm

namespace tamm::internal {

template<typename T, typename LabeledTensorT>
void FlatPlan<T, LabeledTensorT>::apply(const SetOp<T, LabeledTensorT>& setop, ExecutionContext& ec,
                                        ExecutionHW hw) {
  using LHS_ElType      = typename LabeledTensorT::element_type;
  LHS_ElType* lhs_buf   = setop.lhs().tensor().access_local_buf();
  size_t      lhs_size  = setop.lhs().tensor().local_buf_size();
  Scalar      alpha     = setop.alpha();
  bool        is_assign = setop.is_assign();
  auto        lhs_lt    = setop.lhs();

  BlockSpan<LHS_ElType> lhs_span{lhs_buf, {lhs_size}};
  BlockSetPlan::OpType  optype = is_assign ? BlockSetPlan::OpType::set
                                           : BlockSetPlan::OpType::update;

  BlockSetPlan set_plan{lhs_lt.labels(), optype};
  set_plan.apply(lhs_span, alpha);
}

/// GeneralLHSPlan Method Implementations
template<typename T, typename LabeledTensorT>
void LHSPlan<T, LabeledTensorT>::apply(const SetOp<T, LabeledTensorT>& setop, ExecutionContext& ec,
                                       ExecutionHW hw) {
  auto        lhs_lt    = setop.lhs();
  Scalar      alpha     = setop.alpha();
  bool        is_assign = setop.is_assign();
  const auto& ldist     = lhs_lt.tensor().distribution();
  Proc        me        = ec.pg().rank();

  BlockSetPlan::OpType op_type = is_assign ? BlockSetPlan::OpType::set
                                           : BlockSetPlan::OpType::update;

  BlockSetPlan set_plan{lhs_lt.labels(), op_type};

  auto lambda = [&](const IndexVector& blockid, Offset offset) {
    auto lhs_tensor = lhs_lt.tensor();
    EXPECTS(blockid.size() == lhs_lt.labels().size());
    EXPECTS(blockid.size() == lhs_tensor.num_modes());

    auto  lhs_block_size = lhs_tensor.block_size(blockid);
    auto  lhs_block_dims = lhs_tensor.block_dims(blockid);
    auto* lhs_buf        = lhs_tensor.access_local_buf() + offset.value();

    BlockSpan<T> lhs_span{lhs_buf, lhs_block_dims};

    set_plan.apply(lhs_span, alpha);
  };

  internal::LabelTranslator translator{lhs_lt.labels(), lhs_lt.tensor()().labels()};

  LabelLoopNest loop_nest{lhs_lt.labels()};

  for(const auto& blockid: loop_nest) {
    auto [translated_blockid, tlb_valid] = translator.apply(blockid);
    if(!lhs_lt.tensor().is_non_zero(translated_blockid)) { continue; }

    auto [lhs_proc, lhs_offset] = ldist.locate(translated_blockid);
    if(tlb_valid && lhs_proc == me) { lambda(translated_blockid, lhs_offset); }
  }
}

/// GeneralFlatPlan Method Implementations
template<typename T, typename LabeledTensorT>
void GeneralFlatPlan<T, LabeledTensorT>::apply(const SetOp<T, LabeledTensorT>& setop,
                                               ExecutionContext& ec, ExecutionHW hw) {
  using LHS_ElType = typename LabeledTensorT::element_type;

  auto lhs_lt     = setop.lhs();
  auto lhs_tensor = lhs_lt.tensor();

  Scalar alpha     = setop.alpha();
  auto   is_assign = setop.is_assign();

  ProcGroup pg_lhs        = lhs_tensor.execution_context()->pg();
  ProcGroup pg_ec         = ec.pg();
  Proc      proc_me_in_ec = ec.pg().rank();

  // EXPECTS(pg_lhs.size() == pg_ec.size());

  BlockSetPlan::OpType optype = is_assign ? optype = BlockSetPlan::OpType::set
                                          : BlockSetPlan::OpType::update;
  BlockSetPlan         plan{lhs_lt.labels(), optype};

  std::vector<Proc> pg_lhs_in_ec = pg_lhs.rank_translate(pg_ec);

  Proc round_robin_counter = 0;
  Proc ec_pg_size          = Proc{ec.pg().size()};

  for(size_t i = 0; i < pg_lhs_in_ec.size(); i++) {
    Proc assigned_proc;
    if(pg_lhs_in_ec[i] >= Proc{0}) {
      // bias locality to LHS block
      assigned_proc = pg_lhs_in_ec[i];
    }
    else {
      // if LHS proc is not in ec.pg(), roundrobin to assign to
      // proc in ec.pg()
      assigned_proc = round_robin_counter++ % ec_pg_size;
    }

    if(proc_me_in_ec == assigned_proc) {
      bool        alloced_lhs_buf{false};
      LHS_ElType* lhs_buf{nullptr};

      /// get total buffer size for a given Proc
      size_t lhs_size = lhs_tensor.total_buf_size(i);
      if(lhs_size <= 0) continue;
      if(proc_me_in_ec == pg_lhs_in_ec[i]) {
        lhs_buf         = lhs_tensor.access_local_buf();
        alloced_lhs_buf = false;
      }
      else {
        lhs_buf              = new LHS_ElType[lhs_size];
        alloced_lhs_buf      = true;
        auto* lhs_mem_region = lhs_tensor.memory_region();
        /// get all of lhs's buf at i-th proc to lhs_buf
        lhs_mem_region->get(Proc{i}, Offset{0}, Size{lhs_size}, lhs_buf);
      }

      std::vector<size_t> lhs_dims{lhs_size};

      BlockSpan<LHS_ElType> lhs_span{lhs_buf, lhs_dims};

      plan.apply(lhs_span, alpha);
      if(proc_me_in_ec != pg_lhs_in_ec[i]) {
        EXPECTS(lhs_buf != nullptr);
        /// put lhs_buf to LHS tensor buffer on i-th proc
        auto* lhs_mem_region = lhs_tensor.memory_region();
        lhs_mem_region->put(Proc{i}, Offset{0}, Size{lhs_size}, lhs_buf);
      }
      if(alloced_lhs_buf) { delete[] lhs_buf; }
    }
  }
}

/// GeneralLHSPlan Method Implementations
template<typename T, typename LabeledTensorT>
void GeneralLHSPlan<T, LabeledTensorT>::apply(const SetOp<T, LabeledTensorT>& setop,
                                              ExecutionContext& ec, ExecutionHW hw) {
  using LHS_ElType = typename LabeledTensorT::element_type;

  auto   lhs_lt     = setop.lhs();
  Scalar alpha      = setop.alpha();
  auto   is_assign  = setop.is_assign();
  auto   lhs_tensor = lhs_lt.tensor();

  const auto& ldist = lhs_tensor.distribution();

  ProcGroup         lhs_pg         = lhs_tensor.execution_context()->pg();
  Proc              proc_me_in_ec  = ec.pg().rank();
  std::vector<Proc> proc_lhs_to_ec = lhs_pg.rank_translate(ec.pg());

  auto lhs_alloc_lt = lhs_tensor();

  BlockSetPlan::OpType optype = is_assign ? BlockSetPlan::OpType::set
                                          : BlockSetPlan::OpType::update;

  BlockSetPlan plan{lhs_lt.labels(), optype};

  LabelLoopNest loop_nest{lhs_lt.labels()};

  auto lambda = [&](const IndexVector& l_blockid) {
    auto [lhs_proc, lhs_offset] = ldist.locate(l_blockid);
    auto        lhs_blocksize   = lhs_tensor.block_size(l_blockid);
    auto        lhs_blockdims   = lhs_tensor.block_dims(l_blockid);
    LHS_ElType* lhs_buf{nullptr};
    bool        lhs_alloced{false};

    if(proc_lhs_to_ec[lhs_proc.value()] == proc_me_in_ec &&
       lhs_tensor.kind() != TensorBase::TensorKind::view) {
      lhs_buf     = lhs_tensor.access_local_buf() + lhs_offset.value();
      lhs_alloced = false;
    }
    else {
      lhs_buf     = new LHS_ElType[lhs_blocksize];
      lhs_alloced = true;
      span<LHS_ElType> lhs_span{lhs_buf, lhs_blocksize};
      lhs_tensor.get(l_blockid, lhs_span);
    }

    BlockSpan<LHS_ElType> lhs_span{lhs_buf, lhs_blockdims};

    plan.apply(lhs_span, alpha);

    if(proc_me_in_ec != proc_lhs_to_ec[lhs_proc.value()] ||
       lhs_tensor.kind() == TensorBase::TensorKind::view) {
      span<LHS_ElType> lhs_span{lhs_buf, lhs_blocksize};
      lhs_tensor.put(l_blockid, lhs_span);
    }
    if(lhs_alloced) { delete[] lhs_buf; }
  };

  Proc                      round_robin_counter = 0;
  Proc                      ec_pg_size          = Proc{ec.pg().size()};
  internal::LabelTranslator translator{lhs_lt.labels(), lhs_tensor().labels()};
  for(const auto& blockid: loop_nest) {
    auto [translated_blockid, tlb_valid] = translator.apply(blockid);

    if(tlb_valid && lhs_lt.tensor().is_non_zero(translated_blockid)) {
      Proc assigned_proc;
      auto [lhs_proc, lhs_offset] = ldist.locate(translated_blockid);
      Proc lhs_owner_in_ec        = proc_lhs_to_ec[lhs_proc.value()];
      if(lhs_owner_in_ec >= Proc{0}) { assigned_proc = lhs_owner_in_ec; }
      else { assigned_proc = round_robin_counter++ % ec_pg_size; }

      if(assigned_proc == proc_me_in_ec) { lambda(translated_blockid); }
    }
  }
}

} // namespace tamm::internal
