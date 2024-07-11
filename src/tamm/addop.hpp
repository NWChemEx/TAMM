#pragma once

#define ADDOP_LOCALIZE_LHS

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "tamm/block_assign_plan.hpp"
#include "tamm/block_operations.hpp"
#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/kernels/assign.hpp"
#include "tamm/label_translator.hpp"
#include "tamm/labeled_tensor.hpp"
#include "tamm/op_base.hpp"
#include "tamm/runtime_engine.hpp"
#include "tamm/tensor.hpp"
#include "tamm/types.hpp"
#include "tamm/utils.hpp"
#include "tamm/work.hpp"

namespace tamm {

/**
 * @brief AddOp implementation using new BlockPlans
 *
 *
 * @todo Fix the creation of plan issue. Currently:
 *         - the writes/reads/accumulates calls are separated from plans
 *         - the creation of the plan is moved to execute method
 *       Plan is either to have pass where the allocation for each tensor is
 * made before the creation of AddOp or move the writes/reads/accumulates calls
 * out of plan design

 */
template<typename T, typename LabeledTensorT1, typename LabeledTensorT2>
class AddOp;
} // namespace tamm
namespace tamm::internal {

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2>
struct AddOpPlanBase {
  using AddOpT = AddOp<T, LabeledTensorT1, LabeledTensorT2>;
  TensorBase* writes(const AddOpT& addop) const {
    auto ret1 = local_writes(addop);
    auto ret2 = global_writes(addop);
    ret1.insert(ret1.end(), ret2.begin(), ret2.end());
    return !ret1.empty() ? ret1[0] : nullptr;
  }

  TensorBase* accumulates(const AddOpT& addop) const {
    auto ret1 = local_accumulates(addop);
    auto ret2 = global_accumulates(addop);
    ret1.insert(ret1.end(), ret2.begin(), ret2.end());
    return !ret1.empty() ? ret1[0] : nullptr;
  }

  std::vector<TensorBase*> reads(const AddOpT& addop) const {
    auto ret1 = local_reads(addop);
    auto ret2 = global_reads(addop);
    ret1.insert(ret1.end(), ret2.begin(), ret2.end());
    return !ret1.empty() ? ret1 : std::vector<TensorBase*>{};
  }

  virtual std::vector<TensorBase*> global_writes(const AddOpT& addop) const      = 0;
  virtual std::vector<TensorBase*> global_accumulates(const AddOpT& addop) const = 0;
  virtual std::vector<TensorBase*> global_reads(const AddOpT& addop) const       = 0;
  virtual std::vector<TensorBase*> local_writes(const AddOpT& addop) const       = 0;
  virtual std::vector<TensorBase*> local_accumulates(const AddOpT& addop) const  = 0;
  virtual std::vector<TensorBase*> local_reads(const AddOpT& addop) const        = 0;

  virtual void apply(const AddOpT& addop, ExecutionContext& ec, ExecutionHW hw) = 0;
}; // AddOpPlanBase

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2>
struct FlatAddPlan: public AddOpPlanBase<T, LabeledTensorT1, LabeledTensorT2> {
  using AddOpT = AddOp<T, LabeledTensorT1, LabeledTensorT2>;
  std::vector<TensorBase*> global_writes(const AddOpT& addop) const override { return {}; }
  std::vector<TensorBase*> global_accumulates(const AddOpT& addop) const override { return {}; }

  std::vector<TensorBase*> global_reads(const AddOpT& addop) const override { return {}; }

  std::vector<TensorBase*> local_writes(const AddOpT& addop) const override {
    if(addop.is_assign()) { return {addop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_accumulates(const AddOpT& addop) const override {
    if(!addop.is_assign()) { return {addop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_reads(const AddOpT& addop) const override {
    return {addop.rhs().base_ptr()};
  }
  void apply(const AddOpT& addop, ExecutionContext& ec, ExecutionHW hw) override;
}; // FlatAddPlan

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2>
struct LHSAddPlan: public AddOpPlanBase<T, LabeledTensorT1, LabeledTensorT2> {
  using AddOpT = AddOp<T, LabeledTensorT1, LabeledTensorT2>;
  std::vector<TensorBase*> global_writes(const AddOpT& addop) const override { return {}; }
  std::vector<TensorBase*> global_accumulates(const AddOpT& addop) const override { return {}; }
  std::vector<TensorBase*> global_reads(const AddOpT& addop) const override { return {}; }

  std::vector<TensorBase*> local_writes(const AddOpT& addop) const override {
    if(addop.is_assign()) { return {addop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_accumulates(const AddOpT& addop) const override {
    if(!addop.is_assign()) { return {addop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_reads(const AddOpT& addop) const override {
    return {addop.rhs().base_ptr()};
  }
  void apply(const AddOpT& addop, ExecutionContext& ec, ExecutionHW hw) override;
}; // LHSAddPlan

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2>
struct GeneralFlatAddPlan: public AddOpPlanBase<T, LabeledTensorT1, LabeledTensorT2> {
  using AddOpT = AddOp<T, LabeledTensorT1, LabeledTensorT2>;
  std::vector<TensorBase*> global_writes(const AddOpT& addop) const override {
    if(addop.is_assign()) { return {addop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> global_accumulates(const AddOpT& addop) const override {
    if(!addop.is_assign()) { return {addop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> global_reads(const AddOpT& addop) const override {
    return {addop.rhs().base_ptr()};
  }

  std::vector<TensorBase*> local_writes(const AddOpT& addop) const override {
    if(addop.is_assign()) { return {addop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_accumulates(const AddOpT& addop) const override {
    if(!addop.is_assign()) { return {addop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_reads(const AddOpT& addop) const override {
    return {addop.rhs().base_ptr()};
  }
  void apply(const AddOpT& addop, ExecutionContext& ec, ExecutionHW hw) override;
}; // GeneralFlatAddPlan

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2>
struct GeneralLHSAddPlan: public AddOpPlanBase<T, LabeledTensorT1, LabeledTensorT2> {
  using AddOpT = AddOp<T, LabeledTensorT1, LabeledTensorT2>;
  std::vector<TensorBase*> global_writes(const AddOpT& addop) const override {
    if(addop.is_assign()) { return {addop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> global_accumulates(const AddOpT& addop) const override {
    if(!addop.is_assign()) { return {addop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> global_reads(const AddOpT& addop) const override {
    return {addop.rhs().base_ptr()};
  }
  std::vector<TensorBase*> local_writes(const AddOpT& addop) const override {
    if(addop.is_assign()) { return {addop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_accumulates(const AddOpT& addop) const override {
    if(!addop.is_assign()) { return {addop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_reads(const AddOpT& addop) const override {
    return {addop.rhs().base_ptr()};
  }
  void apply(const AddOpT& addop, ExecutionContext& ec, ExecutionHW hw) override;
}; // GeneralLHSAddPlan

} // namespace tamm::internal

namespace tamm {
template<typename T, typename LabeledTensorT1, typename LabeledTensorT2>
class AddOp: public Op {
public:
  AddOp() = default;
  AddOp(LabeledTensorT1 lhs, T alpha, LabeledTensorT2 rhs, bool is_assign):
    lhs_{lhs}, alpha_{alpha}, rhs_{rhs}, is_assign_{is_assign} {
    EXPECTS(lhs.has_str_lbl() == rhs.has_str_lbl());

    if(!lhs.has_str_lbl() && !lhs.labels().empty()) {
      auto lhs_lbls = lhs.labels();
      auto rhs_lbls = rhs.labels();

      auto labels{lhs_lbls};
      labels.insert(labels.end(), rhs_lbls.begin(), rhs_lbls.end());
      internal::update_labels(labels);

      lhs_lbls = IndexLabelVec(labels.begin(), labels.begin() + lhs.labels().size());
      rhs_lbls = IndexLabelVec(labels.begin() + lhs.labels().size(),
                               labels.begin() + lhs.labels().size() + rhs.labels().size());

      lhs_.set_labels(lhs_lbls);
      rhs_.set_labels(rhs_lbls);
    }

    if(lhs.has_str_lbl()) { fillin_labels(); }

    fillin_int_labels();
    validate();
  }

  AddOp(const AddOp<T, LabeledTensorT1, LabeledTensorT2>&) = default;

  T alpha() const { return alpha_; }

  LabeledTensorT1 lhs() const { return lhs_; }

  LabeledTensorT2 rhs() const { return rhs_; }

  bool is_assign() const { return is_assign_; }

  OpType op_type() const override { return OpType::add; }

  OpList canonicalize() const override {
    OpList result{};

    if(is_assign_) {
      auto lhs{lhs_};
      auto assign_op = (lhs = 0);
      result.push_back(assign_op.clone());
      AddOp n_op{lhs_, alpha_, rhs_, false};
      result.push_back(n_op.clone());
    }
    else { result.push_back((*this).clone()); }

    return result;
  }

  std::shared_ptr<Op> clone() const override {
    return std::shared_ptr<Op>(new AddOp<T, LabeledTensorT1, LabeledTensorT2>{*this});
  }

  void execute(ExecutionContext& ec, ExecutionHW hw = ExecutionHW::CPU) override {
    EXPECTS(lhs_.tensor().execution_context() != nullptr);

    if(rhs_.tensor().kind() != TensorBase::TensorKind::lambda &&
       rhs_.tensor().kind() != TensorBase::TensorKind::view &&
       lhs_.tensor().kind() != TensorBase::TensorKind::view && lhs_.labels() == rhs_.labels() &&
       !internal::is_slicing(lhs_) && !internal::is_slicing(rhs_) &&
       lhs_.tensor().has_spin() == rhs_.tensor().has_spin() &&
       lhs_.tensor().execution_context()->pg() == rhs_.tensor().execution_context()->pg() &&
       lhs_.tensor().distribution() == rhs_.tensor().distribution()) {
      plan_     = Plan::flat;
      plan_obj_ = std::make_shared<internal::FlatAddPlan<T, LabeledTensorT1, LabeledTensorT2>>();
      general_plan_obj_ =
        std::make_shared<internal::GeneralFlatAddPlan<T, LabeledTensorT1, LabeledTensorT2>>();
    }
    else {
      plan_     = Plan::lhs;
      plan_obj_ = std::make_shared<internal::LHSAddPlan<T, LabeledTensorT1, LabeledTensorT2>>();
      general_plan_obj_ =
        std::make_shared<internal::GeneralLHSAddPlan<T, LabeledTensorT1, LabeledTensorT2>>();
    }

    EXPECTS(plan_ != Plan::invalid);
    if(lhs_.tensor().execution_context()->pg() == ec.pg() &&
       lhs_.tensor().kind() != TensorBase::TensorKind::view &&
       lhs_.tensor().kind() != TensorBase::TensorKind::dense) {
      plan_obj_->apply(*this, ec, hw);
    }
    else { general_plan_obj_->apply(*this, ec, hw); }
  }

#if 0
    TensorBase* writes() const { return plan_obj_->writes(*this); }

    TensorBase* accumulates() const { return plan_obj_->accumulates(*this); }

    std::vector<TensorBase*> reads() const { return plan_obj_->reads(*this); }
#else
  TensorBase* writes() const {
    if(is_assign()) { return lhs_.base_ptr(); }
    else { return nullptr; }
  }

  TensorBase* accumulates() const {
    if(!is_assign()) { return lhs_.base_ptr(); }
    else { return nullptr; }
  }

  std::vector<TensorBase*> reads() const {
    std::vector<TensorBase*> res;
    res.push_back(rhs_.base_ptr());

    return res;
  }
#endif

  TensorBase* writes(const ExecutionContext& ec) const {
    if(lhs_.tensor().pg() == ec.pg()) { return plan_obj_->writes(*this); }
    else { return general_plan_obj_->writes(*this); }
  }

  std::vector<TensorBase*> reads(const ExecutionContext& ec) const {
    if(lhs_.tensor().pg() == ec.pg()) { return plan_obj_->reads(*this); }
    else { return general_plan_obj_->reads(*this); }
  }

  TensorBase* accumulates(const ExecutionContext& ec) const {
    if(lhs_.tensor().pg() == ec.pg()) { return plan_obj_->accumulates(*this); }
    else { return general_plan_obj_->accumulates(*this); }
  }

  bool is_memory_barrier() const { return false; }

protected:
  void fillin_labels() {
    using internal::fillin_tensor_label_from_map;
    using internal::update_fillin_map;
    // every string in RHS is also in LHS. So number only LHS strings
    std::map<std::string, Label> str_to_labels;
    update_fillin_map(str_to_labels, lhs_.str_map(), lhs_.str_labels(), 0);
    fillin_tensor_label_from_map(lhs_, str_to_labels);
    fillin_tensor_label_from_map(rhs_, str_to_labels);
  }

  void split_block_id(IndexVector& lhs_lbls, IndexVector& rhs_lbls, size_t lhs_size,
                      size_t rhs_size, const IndexVector& full_blockid) const {
    IndexVector new_lhs, new_rhs;

    new_lhs.insert(new_lhs.end(), full_blockid.begin(), full_blockid.begin() + lhs_size);
    new_rhs.insert(new_rhs.end(), full_blockid.begin() + lhs_size, full_blockid.end());

    lhs_lbls = new_lhs;
    rhs_lbls = new_rhs;
  }

  /**
   * @brief Check if the parameters forma valid add operation. The parameters
   * (ltc, tuple(alpha,lta)) form a valid add operation if:
   *
   * 1. Every label depended on by another label (i.e., all 'd' such that
   * there exists label 'l(d)') is bound at least once
   *
   * 2. There are no conflicting dependent label specifications. That if
   * 'a(i)' is a label in either lta or ltc, there is no label 'a(j)' (i!=j)
   * in either lta or ltc.
   *
   * @tparam LabeledTensorType Type RHS labeled tensor
   * @tparam T Type of scaling factor (alpha)
   * @param ltc LHS tensor being added to
   * @param rhs RHS (scaling factor and labeled tensor)
   *
   * @pre ltc.validate() has been invoked
   * @pre lta.validate() has been invoked
   */
  void validate() {
    if(!(lhs_.tensor().base_ptr() != rhs1_.tensor().base_ptr())) {
      std::ostringstream os;
      os << "[TAMM ERROR] Self assignment is not supported in tensor operations!\n"
         << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }

    IndexLabelVec ilv{lhs_.labels()};
    ilv.insert(ilv.end(), rhs_.labels().begin(), rhs_.labels().end());

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

  void fillin_int_labels() {
    std::map<TileLabelElement, int> primary_labels_map;
    int                             cnt = -1;
    for(const auto& lbl: lhs_.labels()) { primary_labels_map[lbl.primary_label()] = --cnt; }
    for(const auto& lbl: rhs_.labels()) { primary_labels_map[lbl.primary_label()] = --cnt; }
    for(const auto& lbl: lhs_.labels()) {
      lhs_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
    }
    for(const auto& lbl: rhs_.labels()) {
      rhs_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
    }
  }

  LabeledTensorT1 lhs_;
  T               alpha_;
  LabeledTensorT2 rhs_;
  IntLabelVec     lhs_int_labels_, rhs_int_labels_;
  bool            is_assign_;

  enum class Plan { invalid, lhs, flat, general_lhs, general_flat };
  Plan plan_ = Plan::invalid;
  std::shared_ptr<internal::AddOpPlanBase<T, LabeledTensorT1, LabeledTensorT2>> plan_obj_;
  std::shared_ptr<internal::AddOpPlanBase<T, LabeledTensorT1, LabeledTensorT2>> general_plan_obj_;

public:
  std::string opstr_;
}; // class AddOp

} // namespace tamm

namespace tamm::internal {
template<typename T, typename LabeledTensorT1, typename LabeledTensorT2>
using AddOpT = AddOp<T, LabeledTensorT1, LabeledTensorT2>;

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2>
void FlatAddPlan<T, LabeledTensorT1, LabeledTensorT2>::apply(const AddOpT&     addop,
                                                             ExecutionContext& ec, ExecutionHW hw) {
  using T1         = typename LabeledTensorT1::element_type;
  using T2         = typename LabeledTensorT2::element_type;
  auto   lhs_lt    = addop.lhs();
  auto   rhs_lt    = addop.rhs();
  Scalar alpha     = addop.alpha();
  auto   is_assign = addop.is_assign();

  T1*                 lhs_buf  = lhs_lt.tensor().access_local_buf();
  size_t              lhs_size = lhs_lt.tensor().local_buf_size();
  std::vector<size_t> lhs_dims{lhs_size};

  T2*                 rhs_buf  = rhs_lt.tensor().access_local_buf();
  size_t              rhs_size = rhs_lt.tensor().local_buf_size();
  std::vector<size_t> rhs_dims{rhs_size};

  EXPECTS(rhs_size == lhs_size);

  BlockSpan<T1> lhs_span{lhs_buf, lhs_dims};
  BlockSpan<T2> rhs_span{rhs_buf, rhs_dims};

  BlockAssignPlan::OpType optype = is_assign ? BlockAssignPlan::OpType::set
                                             : BlockAssignPlan::OpType::update;

  BlockAssignPlan plan{lhs_lt.labels(), rhs_lt.labels(), optype};

  plan.apply(lhs_span, alpha, rhs_span);
}

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2>
void LHSAddPlan<T, LabeledTensorT1, LabeledTensorT2>::apply(const AddOpT&     addop,
                                                            ExecutionContext& ec, ExecutionHW hw) {
  using T1                 = typename LabeledTensorT1::element_type;
  using T2                 = typename LabeledTensorT2::element_type;
  auto        lhs_lt       = addop.lhs();
  auto        rhs_lt       = addop.rhs();
  Scalar      alpha        = addop.alpha();
  auto        is_assign    = addop.is_assign();
  auto        lhs_alloc_lt = lhs_lt.tensor()();
  auto        rhs_alloc_lt = rhs_lt.tensor()();
  const auto& ldist        = lhs_lt.tensor().distribution();
  const auto& rdist        = rhs_lt.tensor().distribution();
  Proc        me           = ec.pg().rank();

  IndexLabelVec merged_use_labels =
    internal::merge_vector<IndexLabelVec>(lhs_lt.labels(), rhs_lt.labels());

  IndexLabelVec merged_alloc_labels =
    internal::merge_vector<IndexLabelVec>(lhs_alloc_lt.labels(), rhs_alloc_lt.labels());

  LabelLoopNest loop_nest{merged_use_labels};

  BlockAssignPlan::OpType optype = is_assign ? BlockAssignPlan::OpType::set
                                             : BlockAssignPlan::OpType::update;

  BlockAssignPlan plan{lhs_lt.labels(), rhs_lt.labels(), optype};

  auto lambda = [&](const IndexVector& l_blockid, Offset l_offset, const IndexVector& r_blockid) {
    auto lhs_tensor = lhs_lt.tensor();
    auto rhs_tensor = rhs_lt.tensor();

    auto  lhs_blocksize = lhs_tensor.block_size(l_blockid);
    auto  lhs_blockdims = lhs_tensor.block_dims(l_blockid);
    auto* lhs_buf       = lhs_tensor.access_local_buf() + l_offset.value();

    auto rhs_blocksize = rhs_tensor.block_size(r_blockid);
    auto rhs_blockdims = rhs_tensor.block_dims(r_blockid);

    std::vector<T2> rhs_buf(rhs_blocksize);

    rhs_tensor.get(r_blockid, rhs_buf);

    BlockSpan<T1> lhs_span{lhs_buf, lhs_blockdims};
    BlockSpan<T2> rhs_span{rhs_buf.data(), rhs_blockdims};

    plan.apply(lhs_span, alpha, rhs_span);
  };

  internal::LabelTranslator translator{merged_use_labels, merged_alloc_labels};
  for(const auto& blockid: loop_nest) {
    auto [translated_blockid, tlb_valid] = translator.apply(blockid);
    auto [l_blockid, r_blockid]          = internal::split_vector<IndexVector, 2>(
      translated_blockid, {lhs_lt.labels().size(), rhs_lt.labels().size()});

    if(!lhs_lt.tensor().is_non_zero(l_blockid) || !rhs_lt.tensor().is_non_zero(r_blockid)) {
      continue;
    }

    auto [lhs_proc, lhs_offset] = ldist.locate(l_blockid);

    if(tlb_valid && lhs_proc == me) { lambda(l_blockid, lhs_offset, r_blockid); }
  }
}

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2>
void GeneralFlatAddPlan<T, LabeledTensorT1, LabeledTensorT2>::apply(const AddOpT&     addop,
                                                                    ExecutionContext& ec,
                                                                    ExecutionHW       hw) {
  using T1 = typename LabeledTensorT1::element_type;
  using T2 = typename LabeledTensorT2::element_type;

  auto   lhs_lt     = addop.lhs();
  auto   rhs_lt     = addop.rhs();
  auto   lhs_tensor = lhs_lt.tensor();
  auto   rhs_tensor = rhs_lt.tensor();
  Scalar alpha      = addop.alpha();
  auto   is_assign  = addop.is_assign();

  ProcGroup pg_lhs        = lhs_tensor.execution_context()->pg();
  ProcGroup pg_rhs        = rhs_tensor.execution_context()->pg();
  ProcGroup pg_ec         = ec.pg();
  Proc      proc_me_in_ec = ec.pg().rank();

  // EXPECTS(pg_lhs.size() == pg_rhs.size());
  // EXPECTS(pg_lhs.size() == pg_ec.size());

  BlockAssignPlan::OpType optype = is_assign ? optype = BlockAssignPlan::OpType::set
                                             : BlockAssignPlan::OpType::update;
  BlockAssignPlan         plan{lhs_lt.labels(), rhs_lt.labels(), optype};

  std::vector<Proc> pg_lhs_in_ec = pg_lhs.rank_translate(pg_ec);
  std::vector<Proc> pg_rhs_in_ec = pg_rhs.rank_translate(pg_ec);

  Proc round_robin_counter = 0;
  Proc ec_pg_size          = Proc{ec.pg().size()};

  for(size_t i = 0; i < pg_lhs_in_ec.size(); i++) {
    Proc assigned_proc;
    if(pg_lhs_in_ec[i] >= Proc{0}) {
      // bias locality to LHS block
      assigned_proc = pg_lhs_in_ec[i];
    }
    else if(pg_rhs_in_ec[i] >= Proc{0}) {
      // if LHS proc not in ec.pg() bias to RHS block
      assigned_proc = pg_rhs_in_ec[i];
    }
    else {
      // if neither LHS nor RHS proc in ec.pg(), roundrobin to assign to
      // proc in ec.pg()
      assigned_proc = round_robin_counter++ % ec_pg_size;
    }

    if(proc_me_in_ec == assigned_proc) {
      bool alloced_lhs_buf{false};
      bool alloced_rhs_buf{false};
      T1*  lhs_buf{nullptr};
      T2*  rhs_buf{nullptr};

      /// get total buffer size for a given Proc
      size_t lhs_size = lhs_tensor.total_buf_size(i);
      size_t rhs_size = rhs_tensor.total_buf_size(i);
      EXPECTS(lhs_size == rhs_size);
      if(lhs_size <= 0) continue;
      if(proc_me_in_ec == pg_lhs_in_ec[i]) {
        lhs_buf         = lhs_tensor.access_local_buf();
        alloced_lhs_buf = false;
      }
      else {
        lhs_buf              = new T1[lhs_size];
        alloced_lhs_buf      = true;
        auto* lhs_mem_region = lhs_tensor.memory_region();
        /// get all of lhs's buf at i-th proc to lhs_buf
        lhs_mem_region->get(Proc{i}, Offset{0}, Size{lhs_size}, lhs_buf);
      }
      if(proc_me_in_ec == pg_rhs_in_ec[i]) {
        rhs_buf         = rhs_tensor.access_local_buf();
        alloced_rhs_buf = false;
      }
      else {
        rhs_buf         = new T2[rhs_size];
        alloced_rhs_buf = true;
        EXPECTS(rhs_buf != nullptr);
        auto* rhs_mem_region = rhs_tensor.memory_region();
        /// get all of rhs's buf at i-th proc to rhs_buf
        rhs_mem_region->get(Proc{i}, Offset{0}, Size{rhs_size}, rhs_buf);
      }
      std::vector<size_t> lhs_dims{lhs_size};
      std::vector<size_t> rhs_dims{rhs_size};
      BlockSpan<T1>       lhs_span{lhs_buf, lhs_dims};
      BlockSpan<T2>       rhs_span{rhs_buf, rhs_dims};

      plan.apply(lhs_span, alpha, rhs_span);
      if(proc_me_in_ec != pg_lhs_in_ec[i]) {
        EXPECTS(lhs_buf != nullptr);
        /// put lhs_buf to LHS tensor buffer on i-th proc
        auto* lhs_mem_region = lhs_tensor.memory_region();
        lhs_mem_region->put(Proc{i}, Offset{0}, Size{lhs_size}, lhs_buf);
      }
      if(alloced_lhs_buf) { delete[] lhs_buf; }
      if(alloced_rhs_buf) { delete[] rhs_buf; }
    }
  }
}

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2>
void GeneralLHSAddPlan<T, LabeledTensorT1, LabeledTensorT2>::apply(const AddOpT&     addop,
                                                                   ExecutionContext& ec,
                                                                   ExecutionHW       hw) {
  using T1          = typename LabeledTensorT1::element_type;
  using T2          = typename LabeledTensorT2::element_type;
  auto   lhs_lt     = addop.lhs();
  auto   rhs_lt     = addop.rhs();
  Scalar alpha      = addop.alpha();
  auto   is_assign  = addop.is_assign();
  auto   lhs_tensor = lhs_lt.tensor();
  auto   rhs_tensor = rhs_lt.tensor();

  const auto& ldist = lhs_tensor.distribution();
  const auto& rdist = rhs_tensor.distribution();

  ProcGroup         lhs_pg         = lhs_tensor.execution_context()->pg();
  ProcGroup         rhs_pg         = rhs_tensor.execution_context()->pg();
  Proc              proc_me_in_ec  = ec.pg().rank();
  std::vector<Proc> proc_lhs_to_ec = lhs_pg.rank_translate(ec.pg());
  std::vector<Proc> proc_rhs_to_ec = rhs_pg.rank_translate(ec.pg());

  auto lhs_alloc_lt = lhs_tensor();
  auto rhs_alloc_lt = rhs_tensor();

  IndexLabelVec merged_use_labels =
    internal::merge_vector<IndexLabelVec>(lhs_lt.labels(), rhs_lt.labels());

  IndexLabelVec merged_alloc_labels =
    internal::merge_vector<IndexLabelVec>(lhs_alloc_lt.labels(), rhs_alloc_lt.labels());

  BlockAssignPlan::OpType optype = is_assign ? BlockAssignPlan::OpType::set
                                             : BlockAssignPlan::OpType::update;

  BlockAssignPlan plan{lhs_lt.labels(), rhs_lt.labels(), optype};

  LabelLoopNest loop_nest{merged_use_labels};

  auto lambda = [&](const IndexVector& l_blockid, const IndexVector& r_blockid) {
    auto [lhs_proc, lhs_offset] = ldist.locate(l_blockid);
    auto lhs_blocksize          = lhs_tensor.block_size(l_blockid);
    auto lhs_blockdims          = lhs_tensor.block_dims(l_blockid);
    T1*  lhs_buf{nullptr};
    bool lhs_alloced{false};
    if(proc_lhs_to_ec[lhs_proc.value()] == proc_me_in_ec &&
       lhs_tensor.kind() != TensorBase::TensorKind::view &&
       lhs_tensor.kind() != TensorBase::TensorKind::dense) {
      lhs_buf     = lhs_tensor.access_local_buf() + lhs_offset.value();
      lhs_alloced = false;
    }
    else {
      lhs_buf     = new T1[lhs_blocksize];
      lhs_alloced = true;
      span<T1> lhs_span{lhs_buf, lhs_blocksize};
      lhs_tensor.get(l_blockid, lhs_span);
    }

    auto [rhs_proc, rhs_offset] = rdist.locate(r_blockid);
    auto rhs_blocksize          = rhs_tensor.block_size(r_blockid);
    auto rhs_blockdims          = rhs_tensor.block_dims(r_blockid);
    T2*  rhs_buf{nullptr};
    bool rhs_alloced{false};
    if(proc_rhs_to_ec[rhs_proc.value()] == proc_me_in_ec &&
       rhs_tensor.kind() != TensorBase::TensorKind::view &&
       rhs_tensor.kind() != TensorBase::TensorKind::lambda &&
       rhs_tensor.kind() != TensorBase::TensorKind::dense) {
      rhs_buf     = rhs_tensor.access_local_buf() + rhs_offset.value();
      rhs_alloced = false;
    }
    else {
      rhs_buf     = new T2[rhs_blocksize];
      rhs_alloced = true;
      span<T2> rhs_span{rhs_buf, rhs_blocksize};
      rhs_tensor.get(r_blockid, rhs_span);
    }

    BlockSpan<T1> lhs_span{lhs_buf, lhs_blockdims};
    BlockSpan<T2> rhs_span{rhs_buf, rhs_blockdims};

    plan.apply(lhs_span, alpha, rhs_span);

    if(proc_me_in_ec != proc_lhs_to_ec[lhs_proc.value()] ||
       lhs_tensor.kind() == TensorBase::TensorKind::view ||
       lhs_tensor.kind() == TensorBase::TensorKind::dense) {
      span<T1> lhs_span{lhs_buf, lhs_blocksize};
      lhs_tensor.put(l_blockid, lhs_span);
    }
    if(lhs_alloced) { delete[] lhs_buf; }
    if(rhs_alloced) { delete[] rhs_buf; }
  };

  Proc round_robin_counter = 0;
  Proc ec_pg_size          = Proc{ec.pg().size()};

  internal::LabelTranslator translator{merged_use_labels, merged_alloc_labels};
  for(const auto& blockid: loop_nest) {
    auto [translated_blockid, tlb_valid] = translator.apply(blockid);
    auto [l_blockid, r_blockid]          = internal::split_vector<IndexVector, 2>(
      translated_blockid, {lhs_lt.labels().size(), rhs_lt.labels().size()});

    if(tlb_valid && lhs_lt.tensor().is_non_zero(l_blockid) &&
       rhs_lt.tensor().is_non_zero(r_blockid)) {
      Proc assigned_proc;
      auto [lhs_proc, lhs_offset] = ldist.locate(l_blockid);
      Proc lhs_owner_in_ec        = proc_lhs_to_ec[lhs_proc.value()];
      if(lhs_owner_in_ec >= Proc{0}) { assigned_proc = lhs_owner_in_ec; }
      else {
        auto [rhs_proc, rhs_offset] = rdist.locate(r_blockid);
        Proc rhs_owner_in_ec        = proc_rhs_to_ec[rhs_proc.value()];
        if(rhs_owner_in_ec >= Proc{0}) { assigned_proc = rhs_owner_in_ec; }
        else { assigned_proc = round_robin_counter++ % ec_pg_size; }
      }

      if(assigned_proc == proc_me_in_ec) { lambda(l_blockid, r_blockid); }
    }
  }
}

} // namespace tamm::internal
