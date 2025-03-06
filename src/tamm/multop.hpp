#pragma once

// #define MULTOP_PARTIAL_PARALLELIZE_RHS

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

// #include "tamm/block_operations.hpp"
#include "tamm/block_mult_plan.hpp"
#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/kernels/assign.hpp"
#include "tamm/kernels/multiply.hpp"
#include "tamm/labeled_tensor.hpp"
#include "tamm/runtime_engine.hpp"
#include "tamm/tensor.hpp"
#include "tamm/types.hpp"
#include "tamm/work.hpp"
#include "tamm/fastcc/contract.hpp"

namespace tamm {

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2, typename LabeledTensorT3>
class MultOp;
} // namespace tamm

namespace tamm::internal {

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2, typename LabeledTensorT3>
struct MultOpPlanBase {
  using MultOpT = MultOp<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3>;
  TensorBase* writes(const MultOpT& multop) const {
    auto ret1 = local_writes(multop);
    auto ret2 = global_writes(multop);
    ret1.insert(ret1.end(), ret2.begin(), ret2.end());
    return !ret1.empty() ? ret1[0] : nullptr;
  }

  TensorBase* accumulates(const MultOpT& multop) const {
    auto ret1 = local_accumulates(multop);
    auto ret2 = global_accumulates(multop);
    ret1.insert(ret1.end(), ret2.begin(), ret2.end());
    return !ret1.empty() ? ret1[0] : nullptr;
  }

  std::vector<TensorBase*> reads(const MultOpT& multop) const {
    auto ret1 = local_reads(multop);
    auto ret2 = global_reads(multop);
    ret1.insert(ret1.end(), ret2.begin(), ret2.end());
    return !ret1.empty() ? ret1 : std::vector<TensorBase*>{};
  }

  virtual std::vector<TensorBase*> global_writes(const MultOpT& multop) const      = 0;
  virtual std::vector<TensorBase*> global_accumulates(const MultOpT& multop) const = 0;
  virtual std::vector<TensorBase*> global_reads(const MultOpT& multop) const       = 0;
  virtual std::vector<TensorBase*> local_writes(const MultOpT& multop) const       = 0;
  virtual std::vector<TensorBase*> local_accumulates(const MultOpT& multop) const  = 0;
  virtual std::vector<TensorBase*> local_reads(const MultOpT& multop) const        = 0;

  virtual void apply(const MultOpT& multop, ExecutionContext& ec, ExecutionHW hw) = 0;
}; // MultOpPlanBase

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2, typename LabeledTensorT3>
struct FlatMultPlan: public MultOpPlanBase<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3> {
  using MultOpT = MultOp<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3>;
  std::vector<TensorBase*> global_writes(const MultOpT& multop) const override { return {}; }
  std::vector<TensorBase*> global_accumulates(const MultOpT& multop) const override { return {}; }

  std::vector<TensorBase*> global_reads(const MultOpT& multop) const override { return {}; }

  std::vector<TensorBase*> local_writes(const MultOpT& multop) const override {
    if(multop.is_assign()) { return {multop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_accumulates(const MultOpT& multop) const override {
    if(!multop.is_assign()) { return {multop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_reads(const MultOpT& multop) const override {
    return {multop.rhs1().base_ptr(), multop.rhs2().base_ptr()};
  }
  void apply(const MultOpT& multop, ExecutionContext& ec, ExecutionHW hw) override;
}; // FlatMultPlan

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2, typename LabeledTensorT3>
struct LHSMultPlan: public MultOpPlanBase<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3> {
  using MultOpT = MultOp<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3>;
  std::vector<TensorBase*> global_writes(const MultOpT& multop) const override { return {}; }
  std::vector<TensorBase*> global_accumulates(const MultOpT& multop) const override { return {}; }
  std::vector<TensorBase*> global_reads(const MultOpT& multop) const override { return {}; }

  std::vector<TensorBase*> local_writes(const MultOpT& multop) const override {
    if(multop.is_assign()) { return {multop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_accumulates(const MultOpT& multop) const override {
    if(!multop.is_assign()) { return {multop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_reads(const MultOpT& multop) const override {
    return {multop.rhs1().base_ptr(), multop.rhs2().base_ptr()};
  }
  void apply(const MultOpT& multop, ExecutionContext& ec, ExecutionHW hw) override;
}; // LHSMultPlan

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2, typename LabeledTensorT3>
struct GeneralFlatMultPlan:
  public MultOpPlanBase<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3> {
  using MultOpT = MultOp<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3>;
  std::vector<TensorBase*> global_writes(const MultOpT& multop) const override {
    if(multop.is_assign()) { return {multop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> global_accumulates(const MultOpT& multop) const override {
    if(!multop.is_assign()) { return {multop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> global_reads(const MultOpT& multop) const override {
    return {multop.rhs1().base_ptr(), multop.rhs2().base_ptr()};
  }

  std::vector<TensorBase*> local_writes(const MultOpT& multop) const override {
    if(multop.is_assign()) { return {multop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_accumulates(const MultOpT& multop) const override {
    if(!multop.is_assign()) { return {multop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_reads(const MultOpT& multop) const override {
    return {multop.rhs1().base_ptr(), multop.rhs2().base_ptr()};
  }
  void apply(const MultOpT& multop, ExecutionContext& ec, ExecutionHW hw) override;
}; // GeneralFlatMultPlan

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2, typename LabeledTensorT3>
struct GeneralLHSMultPlan:
  public MultOpPlanBase<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3> {
  using MultOpT = MultOp<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3>;
  std::vector<TensorBase*> global_writes(const MultOpT& multop) const override {
    if(multop.is_assign()) { return {multop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> global_accumulates(const MultOpT& multop) const override {
    if(!multop.is_assign()) { return {multop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> global_reads(const MultOpT& multop) const override {
    return {multop.rhs1().base_ptr(), multop.rhs2().base_ptr()};
  }
  std::vector<TensorBase*> local_writes(const MultOpT& multop) const override {
    if(multop.is_assign()) { return {multop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_accumulates(const MultOpT& multop) const override {
    if(!multop.is_assign()) { return {multop.lhs().base_ptr()}; }
    else { return {}; }
  }
  std::vector<TensorBase*> local_reads(const MultOpT& multop) const override {
    return {multop.rhs1().base_ptr(), multop.rhs2().base_ptr()};
  }
  void apply(const MultOpT& multop, ExecutionContext& ec, ExecutionHW hw) override;
}; // GeneralLHSMultPlan

} // namespace tamm::internal

namespace tamm {
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
template<typename T1, typename T2, typename T3>
struct AddBuf {
  AddBuf(T2* ta, T3* tb, T1* cbuf, const IndexVector& blockid):
    blockid_{blockid}, cbuf_{cbuf}, ta_{ta}, tb_{tb} {}
  ~AddBuf() {}

  T1*         cbuf_;
  T2*         abuf_;
  T3*         bbuf_;
  IndexVector blockid_;
  T2*         ta_;
  T3*         tb_;
};
#else
template<typename T1, typename T2, typename T3>
struct AddBuf {
  AddBuf(Tensor<T1> tensor, T1* cbuf, const IndexVector& blockid):
    tensor_{tensor}, blockid_{blockid}, cbuf_{cbuf} {}
  ~AddBuf() { assert(nbhdl_.getCompletionStatus() == true); }
  bool is_done() { return true; }
  void wait() {
    if(!nbhdl_.getCompletionStatus()) { nbhdl_.waitForCompletion(); }
  }

  T1*                     cbuf_;
  T2*                     abuf_;
  T3*                     bbuf_;
  IndexVector             blockid_;
  Tensor<T1>              tensor_;
  DataCommunicationHandle nbhdl_;
};
#endif
template<typename T, typename LabeledTensorT1, typename LabeledTensorT2, typename LabeledTensorT3>
class MultOp: public Op {
public:
  MultOp() = default;
  MultOp(LabeledTensorT1 lhs, T alpha, LabeledTensorT2 rhs1, LabeledTensorT3 rhs2, bool is_assign):
    lhs_{lhs}, alpha_{alpha}, rhs1_{rhs1}, rhs2_{rhs2}, is_assign_{is_assign} {
    if(lhs.tensor().num_modes() != 0)
      EXPECTS(lhs.has_str_lbl() == rhs1.has_str_lbl() && rhs1.has_str_lbl() == rhs2.has_str_lbl());
    else EXPECTS(rhs1.has_str_lbl() == rhs2.has_str_lbl());

    if(!lhs.has_str_lbl() && !lhs.labels().empty()) {
      auto lhs_lbls  = lhs.labels();
      auto rhs1_lbls = rhs1.labels();
      auto rhs2_lbls = rhs2.labels();

      auto labels{lhs_lbls};
      labels.insert(labels.end(), rhs1_lbls.begin(), rhs1_lbls.end());
      labels.insert(labels.end(), rhs2_lbls.begin(), rhs2_lbls.end());

      internal::update_labels(labels);

      lhs_lbls  = IndexLabelVec(labels.begin(), labels.begin() + lhs.labels().size());
      rhs1_lbls = IndexLabelVec(labels.begin() + lhs.labels().size(),
                                labels.begin() + lhs.labels().size() + rhs1.labels().size());
      rhs2_lbls = IndexLabelVec(labels.begin() + lhs.labels().size() + rhs1.labels().size(),
                                labels.begin() + lhs.labels().size() + rhs1.labels().size() +
                                  rhs2.labels().size());
      lhs_.set_labels(lhs_lbls);
      rhs1_.set_labels(rhs1_lbls);
      rhs2_.set_labels(rhs2_lbls);
    }

    if(lhs.has_str_lbl()) { fillin_labels(); }

    fillin_int_labels();
    validate();
  }

  MultOp(const MultOp<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3>&) = default;

  LabeledTensorT1 lhs() const { return lhs_; }

  T alpha() const { return alpha_; }

  LabeledTensorT2 rhs1() const { return rhs1_; }

  LabeledTensorT3 rhs2() const { return rhs2_; }

  bool is_assign() const { return is_assign_; }

  OpType op_type() const override { return OpType::mult; }

  OpList canonicalize() const override {
    OpList result{};
    using TensorElType1 = typename LabeledTensorT1::element_type;

    if(is_assign_) {
      auto lhs{lhs_};
      auto assign_op = (lhs = (TensorElType1) 0);
      result.push_back(assign_op.clone());
      MultOp n_op{lhs_, alpha_, rhs1_, rhs2_, false};
      result.push_back(n_op.clone());
    }
    else { result.push_back((*this).clone()); }

    return result;
  }

  std::shared_ptr<Op> clone() const override { return std::shared_ptr<Op>(new MultOp{*this}); }

  using TensorElType1 = typename LabeledTensorT1::element_type;
  using TensorElType2 = typename LabeledTensorT2::element_type;
  using TensorElType3 = typename LabeledTensorT3::element_type;

  template <typename DT> FastccTensor<DT> make_tensor(std::vector<DT> &data, std::vector<size_t> &dims) {
    FastccTensor<DT> tensor;
    std::cout<<"size of nonzeors " <<data.size()<<std::endl;
    std::cout<<"dimensionality of tensor " <<dims.size()<<std::endl;
    for(int i = 0; i < data.size(); i++) {
        std::vector<int> co_ordinates;
        int linearized_pos = i;
        for(size_t _d = 0; _d < dims.size(); _d++) {
            co_ordinates.push_back(linearized_pos % dims[_d]);
            linearized_pos = linearized_pos / dims[_d];
        }
        tensor.get_nonzeros().push_back({data[i], co_ordinates});
        //if(i %100 == 0) {
        //    std::cout<<"i is "<<i<<std::endl;
        //}
    }
    
    return tensor;
  }

  void execute_sparse(ExecutionContext& ec) {
    // TODO add constexpr enable for selected types only
    std::cout << " sparse kernel called" << std::endl;
    EXPECTS(!is_assign_);
    std::cout << "lhs labels are " << std::endl;
    for(auto& lbl: lhs_int_labels_) { std::cout << lbl << " "; }
    std::cout << std::endl;
    std::set<IntLabel> lhs_int_labels_set(lhs_int_labels_.begin(), lhs_int_labels_.end());
    std::cout << "rhs1 labels are " << std::endl;
    for(auto& lbl: rhs1_int_labels_) { std::cout << lbl << " "; }
    std::cout << std::endl;
    std::set<IntLabel> rhs1_int_labels_set(rhs1_int_labels_.begin(), rhs1_int_labels_.end());
    std::cout << "rhs2 labels are " << std::endl;
    for(auto& lbl: rhs2_int_labels_) { std::cout << lbl << " "; }
    std::cout << std::endl;
    std::set<IntLabel> rhs2_int_labels_set(rhs2_int_labels_.begin(), rhs2_int_labels_.end());
    std::cout << std::endl;
    std::cout<<"Contraction labels are "<<std::endl;
    
    std::set<IntLabel> left_and_right;
    std::set_intersection(rhs1_int_labels_set.begin(), rhs1_int_labels_set.end(),
                          rhs2_int_labels_set.begin(), rhs2_int_labels_set.end(),
                          std::inserter(left_and_right, left_and_right.begin()));
    std::set<IntLabel> contraction_labels_set;
    std::set_difference(left_and_right.begin(), left_and_right.end(),
                        lhs_int_labels_set.begin(), lhs_int_labels_set.end(),
                        std::inserter(contraction_labels_set, contraction_labels_set.begin()));
    for(auto c: contraction_labels_set) { std::cout << c << " "; }
    std::cout<<std::endl;
    std::set<IntLabel> batch_labels_set;
    std::set_intersection(left_and_right.begin(), left_and_right.end(),
                          lhs_int_labels_set.begin(), lhs_int_labels_set.end(),
                          std::inserter(batch_labels_set, batch_labels_set.begin()));
    std::cout<<"Batch labels are "<<std::endl;
    for(auto c: batch_labels_set) { std::cout << c << " "; }
    std::cout<<std::endl;
    std::set<IntLabel> left_labels_set;
    std::set_difference(rhs1_int_labels_set.begin(), rhs1_int_labels_set.end(),
                        left_and_right.begin(), left_and_right.end(),
                        std::inserter(left_labels_set, left_labels_set.begin()));
    std::cout<<"Left labels are "<<std::endl;
    for(auto c: left_labels_set) { std::cout << c << " "; }
    std::cout<<std::endl;
    std::set<IntLabel> right_labels_set;
    std::set_difference(rhs2_int_labels_set.begin(), rhs2_int_labels_set.end(),
                        left_and_right.begin(), left_and_right.end(),
                        std::inserter(right_labels_set, right_labels_set.begin()));
    std::cout<<"Right labels are "<<std::endl;
    for(auto c: right_labels_set) { std::cout << c << " "; }
    std::cout<<std::endl;

    // print shape of left tensor
    auto                lhs_tis_vec  = lhs_.tensor().tiled_index_spaces();
    auto                rhs1_tis_vec = rhs1_.tensor().tiled_index_spaces();
    auto                rhs2_tis_vec = rhs2_.tensor().tiled_index_spaces();
    std::vector<size_t> dims_sizes_lhs;
    std::vector<size_t> dims_sizes_rhs1;
    std::vector<size_t> dims_sizes_rhs2;

    for(const auto& tis: lhs_tis_vec) { dims_sizes_lhs.push_back(tis.max_num_indices()); }

    for(const auto& tis: rhs1_tis_vec) { dims_sizes_rhs1.push_back(tis.max_num_indices()); }

    for(const auto& tis: rhs2_tis_vec) { dims_sizes_rhs2.push_back(tis.max_num_indices()); }

    LabelLoopNest               loop_nest1{rhs1_.labels()};
    FastccTensor<TensorElType2> op_left;
    for(auto itval: loop_nest1) {
        const IndexVector          blockid = internal::translate_blockid(itval, rhs1_);
        size_t                     size    = rhs1_.tensor().block_size(blockid);
        std::vector<TensorElType2> buf(size);
        rhs1_.tensor().get(blockid, buf);
        op_left = make_tensor(buf, dims_sizes_rhs1);
    }
    std::cout << "number of nonzeros in op_left is " << op_left.get_nonzeros().size() << std::endl;

    LabelLoopNest               loop_nest2{rhs2_.labels()};
    FastccTensor<TensorElType3> op_right;
    for(auto itval: loop_nest2) {
        const IndexVector          blockid = internal::translate_blockid(itval, rhs1_);
        size_t                     size    = rhs2_.tensor().block_size(blockid);
        std::vector<TensorElType3> buf(size);
        rhs2_.tensor().get(blockid, buf);
        op_right = make_tensor(buf, dims_sizes_rhs2);
    }
    std::cout << "number of nonzeros in op_right is " << op_right.get_nonzeros().size()
              << std::endl;
    std::vector<int> left_batch, right_batch, left_contr, right_contr, left_ex, right_ex;
    for(auto c: batch_labels_set) {
        int left = std::find(rhs1_int_labels_.begin(), rhs1_int_labels_.end(), c) - rhs1_int_labels_.begin();
        int right = std::find(rhs2_int_labels_.begin(), rhs2_int_labels_.end(), c) - rhs2_int_labels_.begin();
        left_batch.push_back(left);
        right_batch.push_back(right);
    }
    for(auto c: contraction_labels_set) {
        int left = std::find(rhs1_int_labels_.begin(), rhs1_int_labels_.end(), c) - rhs1_int_labels_.begin();
        int right = std::find(rhs2_int_labels_.begin(), rhs2_int_labels_.end(), c) - rhs2_int_labels_.begin();
        left_contr.push_back(left);
        right_contr.push_back(right);
    }
    for(auto l: left_labels_set) {
        int left = std::find(rhs1_int_labels_.begin(), rhs1_int_labels_.end(), l) - rhs1_int_labels_.begin();
        left_ex.push_back(left);
    }
    for(auto r: right_labels_set) {
        int right = std::find(rhs2_int_labels_.begin(), rhs2_int_labels_.end(), r) - rhs2_int_labels_.begin();
        right_ex.push_back(right);
    }
    std::cout<<"left contractions are "<<std::endl;
    for(auto c: left_contr) { std::cout << c << " "; }
    std::cout<<std::endl;
    std::cout<<"right contractions are "<<std::endl;
    for(auto c: right_contr) { std::cout << c << " "; }
    std::cout<<std::endl;
    std::cout<<"left batch are "<<std::endl;
    for(auto c: left_batch) { std::cout << c << " "; }
    std::cout<<std::endl;
    std::cout<<"right batch are "<<std::endl;
    for(auto c: right_batch) { std::cout << c << " "; }
    std::cout<<std::endl;
    std::cout<<"left ex are "<<std::endl;
    for(auto c: left_ex) { std::cout << c << " "; }
    std::cout<<std::endl;
    std::cout<<"right ex are "<<std::endl;
    for(auto c: right_ex) { std::cout << c << " "; }
    std::cout<<std::endl;
    op_left._infer_dimensionality();
    op_left._infer_shape();
    op_right._infer_dimensionality();
    op_right._infer_shape();

    ListTensor<TensorElType1> result = op_left. template multiply_3d<double>(
      op_right, left_batch, left_contr, left_ex, right_batch, right_contr, right_ex);
    std::cout << "number of nonzeros in result is " << result.run_through_nnz() << std::endl;

    return;
  }

  void execute(ExecutionContext& ec, ExecutionHW hw = ExecutionHW::CPU) override {
      if (hw == ExecutionHW::CPU_SPARSE){
          std::cout<<"howdy do from CPU_SPARSE"<<std::endl;
          return this->execute_sparse(ec);

      }
    EXPECTS(!is_assign_);
    auto& oprof = tamm::OpProfiler::instance();

    // using TensorElType = typename LabeledTensorT1::element_type;
    // determine set of all labels
    IndexLabelVec all_labels{lhs_.labels()};
    all_labels.insert(all_labels.end(), rhs1_.labels().begin(), rhs1_.labels().end());
    all_labels.insert(all_labels.end(), rhs2_.labels().begin(), rhs2_.labels().end());
    LabelLoopNest loop_nest{all_labels};

    std::vector<AddBuf<TensorElType1, TensorElType2, TensorElType3>*> add_bufs;
    // function to compute one block
    auto lambda = [=, &oprof, &add_bufs, &loop_nest, &ec](const IndexVector itval) {
      auto ctensor = lhs_.tensor();
      auto atensor = rhs1_.tensor();
      auto btensor = rhs2_.tensor();
      // compute blockids from the loop indices. itval is the loop index

#if 1
      auto        it = itval.begin();
      IndexVector citval{it, it + lhs_.labels().size()};
      it += lhs_.labels().size();
      IndexVector aitval{it, it + rhs1_.labels().size()};
      it += rhs1_.labels().size();
      IndexVector bitval{it, it + rhs2_.labels().size()};

      auto [extracted_clabels, cblockid] = internal::extract_blockid_and_label(
        loop_nest.sorted_unique_labels(), citval, lhs_.labels());

      auto [extracted_alabels, ablockid] = internal::extract_blockid_and_label(
        loop_nest.sorted_unique_labels(), aitval, rhs1_.labels());

      auto [extracted_blabels, bblockid] = internal::extract_blockid_and_label(
        loop_nest.sorted_unique_labels(), bitval, rhs2_.labels());

      EXPECTS(lhs_.labels().size() == extracted_clabels.size());
      EXPECTS(rhs1_.labels().size() == extracted_alabels.size());
      EXPECTS(rhs2_.labels().size() == extracted_blabels.size());

      for(size_t i = 0; i < extracted_clabels.size(); i++) {
        EXPECTS(extracted_clabels[i].tiled_index_space().is_compatible_with(
          lhs_.labels()[i].tiled_index_space()));
      }

      for(size_t i = 0; i < extracted_alabels.size(); i++) {
        EXPECTS(extracted_alabels[i].tiled_index_space().is_compatible_with(
          rhs1_.labels()[i].tiled_index_space()));
      }

      for(size_t i = 0; i < extracted_blabels.size(); i++) {
        EXPECTS(extracted_blabels[i].tiled_index_space().is_compatible_with(
          rhs2_.labels()[i].tiled_index_space()));
      }

      for(size_t i = 0; i < extracted_clabels.size(); i++) {
        EXPECTS(extracted_clabels[i].tiled_index_space().is_compatible_with(
          lhs_.tensor()().labels()[i].tiled_index_space()));
      }

      for(size_t i = 0; i < extracted_alabels.size(); i++) {
        EXPECTS(extracted_alabels[i].tiled_index_space().is_compatible_with(
          rhs1_.tensor()().labels()[i].tiled_index_space()));
      }

      for(size_t i = 0; i < extracted_blabels.size(); i++) {
        EXPECTS(extracted_blabels[i].tiled_index_space().is_compatible_with(
          rhs2_.tensor()().labels()[i].tiled_index_space()));
      }

      IndexVector blockid{cblockid};
      blockid.insert(blockid.end(), ablockid.begin(), ablockid.end());
      blockid.insert(blockid.end(), bblockid.begin(), bblockid.end());

      IndexLabelVec extracted_lbls{extracted_clabels};
      extracted_lbls.insert(extracted_lbls.end(), extracted_alabels.begin(),
                            extracted_alabels.end());
      extracted_lbls.insert(extracted_lbls.end(), extracted_blabels.begin(),
                            extracted_blabels.end());

      auto tc_lbls = lhs_.tensor()().labels();
      auto ta_lbls = rhs1_.tensor()().labels();
      auto tb_lbls = rhs2_.tensor()().labels();

      IndexLabelVec tensor_lbls{tc_lbls};
      tensor_lbls.insert(tensor_lbls.end(), ta_lbls.begin(), ta_lbls.end());
      tensor_lbls.insert(tensor_lbls.end(), tb_lbls.begin(), tb_lbls.end());

      IndexVector translated_blockid;
      bool        tb_valid;

      std::tie(translated_blockid, tb_valid) =
        internal::translate_blockid_if_possible(blockid, extracted_lbls, tensor_lbls);

      auto        id_it = translated_blockid.begin();
      IndexVector translated_cblockid{id_it, id_it + lhs_.labels().size()};
      id_it += lhs_.labels().size();
      IndexVector translated_ablockid{id_it, id_it + rhs1_.labels().size()};
      id_it += rhs1_.labels().size();
      IndexVector translated_bblockid{id_it, id_it + rhs2_.labels().size()};

      for(const auto id: translated_cblockid) {
        if(id == -1) return;
      }
      for(const auto id: translated_ablockid) {
        if(id == -1) return;
      }
      for(const auto id: translated_bblockid) {
        if(id == -1) return;
      }

#else
      const auto  translated_cblockid = internal::translate_blockid(cblockid, lhs_);
      const auto  translated_ablockid = internal::translate_blockid(ablockid, rhs1_);
      const auto  translated_bblockid = internal::translate_blockid(bblockid, rhs2_);

#endif
      if(!ctensor.is_non_zero(translated_cblockid) || !atensor.is_non_zero(translated_ablockid) ||
         !btensor.is_non_zero(translated_bblockid))
        return;

      auto& memHostPool = RMMMemoryManager::getInstance().getHostMemoryPool();
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
      auto& memDevicePool = RMMMemoryManager::getInstance().getDeviceMemoryPool();
      auto& thandle       = GPUStreamPool::getInstance().getStream();
#else
      gpuStream_t thandle{};
#endif

      {
        // determine set of all labels

        // compute block size and allocate buffers
        const size_t csize = ctensor.block_size(translated_cblockid);
        const size_t asize = atensor.block_size(translated_ablockid);
        const size_t bsize = btensor.block_size(translated_bblockid);

        TensorElType1* cbuf{nullptr};
        TensorElType2* abuf{nullptr};
        TensorElType3* bbuf{nullptr};
        cbuf = static_cast<TensorElType1*>(memHostPool.allocate(csize * sizeof(TensorElType1)));
        abuf = static_cast<TensorElType2*>(memHostPool.allocate(asize * sizeof(TensorElType2)));
        bbuf = static_cast<TensorElType3*>(memHostPool.allocate(bsize * sizeof(TensorElType3)));
        std::memset(static_cast<void*>(cbuf), 0, csize * sizeof(TensorElType1));

        // get inputs
#ifdef DO_NB
        DataCommunicationHandle a_nbhandle, b_nbhandle, c_nbhandle;

        {
          TimerGuard tg_get{&oprof.multOpGetTime};
          atensor.nb_get(translated_ablockid, {abuf, asize}, &a_nbhandle);
          btensor.nb_get(translated_bblockid, {bbuf, bsize}, &b_nbhandle);
        }
        {
          TimerGuard tg_wait{&oprof.multOpWaitTime};
          if(!a_nbhandle.getCompletionStatus()) a_nbhandle.waitForCompletion();
          if(!b_nbhandle.getCompletionStatus()) b_nbhandle.waitForCompletion();
        }
#else
        {
          TimerGuard tg_get{&oprof.multOpGetTime};
          atensor.get(translated_ablockid, {abuf, asize});
        }
        {
          TimerGuard tg_get{&oprof.multOpGetTime};
          btensor.get(translated_bblockid, {bbuf, bsize});
        }
#endif
        const auto& cdims = ctensor.block_dims(translated_cblockid);
        const auto& adims = atensor.block_dims(translated_ablockid);
        const auto& bdims = btensor.block_dims(translated_bblockid);
        // double cscale = is_assign_ ? 0 : 1;
        T cscale{0};

        SizeVec adims_sz, bdims_sz, cdims_sz;
        for(const auto v: adims) { adims_sz.push_back(v); }
        for(const auto v: bdims) { bdims_sz.push_back(v); }
        for(const auto v: cdims) { cdims_sz.push_back(v); }

        AddBuf<TensorElType1, TensorElType2, TensorElType3>* ab{nullptr};
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
        TensorElType2* th_a{nullptr};
        TensorElType3* th_b{nullptr};
        th_a = static_cast<TensorElType2*>(memDevicePool.allocate(asize * sizeof(TensorElType2)));
        th_b = static_cast<TensorElType3*>(memDevicePool.allocate(bsize * sizeof(TensorElType3)));

        ab = new AddBuf<TensorElType1, TensorElType2, TensorElType3>{th_a, th_b, cbuf,
                                                                     translated_cblockid};
#else
        ab = new AddBuf<TensorElType1, TensorElType2, TensorElType3>{ctensor, cbuf,
                                                                     translated_cblockid};
#endif
        add_bufs.push_back(ab);

        {
          TimerGuard     tg_bc{&oprof.multOpBCTime};
          TensorElType1* cbuf_dev_ptr{nullptr};
          TensorElType1* cbuf_tmp_dev_ptr{nullptr};
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
          if(hw == ExecutionHW::GPU) {
            cbuf_dev_ptr =
              static_cast<TensorElType1*>(memDevicePool.allocate(csize * sizeof(TensorElType1)));
            cbuf_tmp_dev_ptr =
              static_cast<TensorElType1*>(memDevicePool.allocate(csize * sizeof(TensorElType1)));

            gpuMemsetAsync(reinterpret_cast<void*&>(cbuf_dev_ptr), csize * sizeof(TensorElType1),
                           thandle);
            gpuMemsetAsync(reinterpret_cast<void*&>(cbuf_tmp_dev_ptr),
                           csize * sizeof(TensorElType1), thandle);
          }
#endif
          kernels::block_multiply<T, TensorElType1, TensorElType2, TensorElType3>(
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
            th_a, th_b,
#endif
            thandle, alpha_, abuf, adims_sz, rhs1_int_labels_, bbuf, bdims_sz, rhs2_int_labels_,
            cscale, ab->cbuf_, cdims_sz, lhs_int_labels_, hw, true, cbuf_dev_ptr, cbuf_tmp_dev_ptr);

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
          if(hw == ExecutionHW::GPU) {
            TensorElType1* cbuf_tmp{nullptr};
            cbuf_tmp =
              static_cast<TensorElType1*>(memHostPool.allocate(csize * sizeof(TensorElType1)));
            std::memset(cbuf_tmp, 0, csize * sizeof(TensorElType1));
            {
              TimerGuard tg_copy{&oprof.multOpCopyTime};
              gpuMemcpyAsync<TensorElType1>(cbuf_tmp, cbuf_dev_ptr, csize, gpuMemcpyDeviceToHost,
                                            thandle);
              gpuStreamSynchronize(thandle);
            }
            // cbuf+=cbuf_tmp
            blas::axpy(csize, TensorElType1{1}, cbuf_tmp, 1, ab->cbuf_, 1);

            memDevicePool.deallocate(cbuf_dev_ptr, csize * sizeof(TensorElType1));
            memDevicePool.deallocate(cbuf_tmp_dev_ptr, csize * sizeof(TensorElType1));

            memHostPool.deallocate(cbuf_tmp, csize * sizeof(TensorElType1));
          }
#endif
        }

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
        memDevicePool.deallocate(th_a, asize * sizeof(TensorElType2));
        memDevicePool.deallocate(th_b, bsize * sizeof(TensorElType3));
#endif
        memHostPool.deallocate(cbuf, csize * sizeof(TensorElType1));
        memHostPool.deallocate(abuf, asize * sizeof(TensorElType2));
        memHostPool.deallocate(bbuf, bsize * sizeof(TensorElType3));

#ifndef DO_NB
        {
          TimerGuard tg_get{&oprof.multOpAddTime};
          // add the computed update to the tensor
          ctensor.add(translated_cblockid, {ab->cbuf_, csize});
        }
        delete ab;
        add_bufs.clear();
#endif
      }
    };

    //@todo use a scheduler
    //@todo make parallel
    // do_work(ec, loop_nest, lambda);

    bool has_sparse_labels = false;
    for(auto& lbl: all_labels) {
      if(lbl.is_dependent()) {
        has_sparse_labels = true;
        break;
      }
    }

    if(1 && (lhs_.tensor().is_dense() /* && !lhs_.tensor().has_spin() */) &&
       (rhs1_.tensor().is_dense() /* && !rhs1_.tensor().has_spin() */) &&
       (rhs2_.tensor().is_dense() /* && !rhs2_.tensor().has_spin() */) && !has_sparse_labels &&
       !lhs_.labels().empty() && lhs_.tensor().execution_context()->pg() == ec.pg()
       //    rhs1_.tensor().execution_context()->pg() == ec.pg() &&
       //    rhs2_.tensor().execution_context()->pg() == ec.pg()
    ) {
      execute_bufacc(ec, hw);
    }
    else { do_work(ec, loop_nest, lambda); }

#ifdef DO_NB
    {
      TimerGuard tg_add{&multOpAddTime};
      for(auto& ab: add_bufs) {
        (ab->tensor_).nb_add(ab->blockid_, ab->cbuf_, &(ab->nbhdl_));
        ab->wait();
        delete ab;
      }
      add_bufs.clear();
    }

#endif
  }

  void execute_bufacc(ExecutionContext& ec, ExecutionHW hw = ExecutionHW::CPU) {
    EXPECTS(!is_assign_);
    auto& oprof         = tamm::OpProfiler::instance();
    using TensorElType1 = typename LabeledTensorT1::element_type;
    using TensorElType2 = typename LabeledTensorT2::element_type;
    using TensorElType3 = typename LabeledTensorT3::element_type;
    // determine set of all labels
    /* std::set<TiledIndexLabel> lhs_set{lhs_.labels()};
    std::set<TiledIndexLabel> rhs_set{rhs1_.labels()};
    rhs_set.insert(rhs_set.end(), rhs2_.labels().begin(), rhs_2.labels().end()); */

    IndexLabelVec lhs_labels{lhs_.labels()};
    IndexLabelVec rhs1_labels{rhs1_.labels()};
    IndexLabelVec rhs2_labels{rhs2_.labels()};
    IndexLabelVec all_rhs_labels{rhs1_.labels()};
    all_rhs_labels.insert(all_rhs_labels.end(), rhs2_.labels().begin(), rhs2_.labels().end());

    LabelLoopNest lhs_loop_nest{lhs_.labels()};

    // compute the reduction labels
    std::sort(lhs_labels.begin(), lhs_labels.end());
    auto unique_labels = internal::unique_entries_by_primary_label(all_rhs_labels);
    std::sort(unique_labels.begin(), unique_labels.end());
    IndexLabelVec reduction_labels; //{reduction.begin(), reduction.end()};
    std::set_difference(unique_labels.begin(), unique_labels.end(), lhs_labels.begin(),
                        lhs_labels.end(), std::back_inserter(reduction_labels));

    std::vector<int> rhs1_map_output;
    std::vector<int> rhs2_map_output;
    std::vector<int> rhs1_map_reduction;
    std::vector<int> rhs2_map_reduction;
    const auto&      lhs_lbls = lhs_.labels();
    for(auto& lbl: rhs1_labels) {
      auto it_out = std::find(lhs_lbls.begin(), lhs_lbls.end(), lbl);
      if(it_out != lhs_lbls.end()) rhs1_map_output.push_back(it_out - lhs_lbls.begin());
      else rhs1_map_output.push_back(-1);

      // auto it_red = std::find(reduction.begin(), reduction.end(), lbl);
      auto it_red = std::find(reduction_labels.begin(), reduction_labels.end(), lbl);
      if(it_red != reduction_labels.end())
        rhs1_map_reduction.push_back(it_red - reduction_labels.begin());
      else rhs1_map_reduction.push_back(-1);
    }

    for(auto& lbl: rhs2_labels) {
      auto it_out = std::find(lhs_lbls.begin(), lhs_lbls.end(), lbl);
      if(it_out != lhs_lbls.end()) rhs2_map_output.push_back(it_out - lhs_lbls.begin());
      else rhs2_map_output.push_back(-1);

      auto it_red = std::find(reduction_labels.begin(), reduction_labels.end(), lbl);
      if(it_red != reduction_labels.end())
        rhs2_map_reduction.push_back(it_red - reduction_labels.begin());
      else rhs2_map_reduction.push_back(-1);
    }

    // IndexLabelVec reduction_lbls{reduction.begin(), reduction.end()};
#if defined(MULTOP_PARTIAL_PARALLELIZE_RHS)
    int64_t n_lhs_blocks, nranks_per_lhs_block, lhs_counter;
#endif

    auto& memHostPool = RMMMemoryManager::getInstance().getHostMemoryPool();
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
    auto& memDevicePool = RMMMemoryManager::getInstance().getDeviceMemoryPool();
    auto& thandle       = GPUStreamPool::getInstance().getStream();
#else
    gpuStream_t thandle{};
#endif

    // function to compute one block
    auto lambda = [&](const IndexVector itval) { // i, j
      auto ctensor = lhs_.tensor();
      auto atensor = rhs1_.tensor();
      auto btensor = rhs2_.tensor();
      // compute blockids from the loop indices. itval is the loop index

      IndexVector c_block_id{itval};
      const auto  translated_cblockid = internal::translate_blockid(c_block_id, lhs_);
      if(!ctensor.is_non_zero(translated_cblockid)) return;

      // compute block size and allocate buffers for cbuf
      const size_t   csize = ctensor.block_size(translated_cblockid);
      TensorElType1* cbuf{nullptr};
      cbuf = static_cast<TensorElType1*>(memHostPool.allocate(csize * sizeof(TensorElType1)));
      std::memset(static_cast<void*>(cbuf), 0, static_cast<size_t>(csize * sizeof(TensorElType1)));
      const auto& cdims = ctensor.block_dims(translated_cblockid);

      SizeVec cdims_sz;
      for(const auto v: cdims) { cdims_sz.push_back(v); }

      AddBuf<TensorElType1, TensorElType2, TensorElType3>* ab{nullptr};
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
      TensorElType2* th_a{nullptr};
      TensorElType3* th_b{nullptr};

      ab = new AddBuf<TensorElType1, TensorElType2, TensorElType3>{
        th_a, th_b, {}, translated_cblockid};
#else
      ab =
        new AddBuf<TensorElType1, TensorElType2, TensorElType3>{ctensor, {}, translated_cblockid};
#endif

      {
        // LabelLoopNest inner_loop{reduction_lbls};
        LabelLoopNest inner_loop{reduction_labels};

#if defined(MULTOP_PARTIAL_PARALLELIZE_RHS)
        int loop_counter     = 0;
        nranks_per_lhs_block = (ec.pg().size().value() / n_lhs_blocks) + 1 -
                               (lhs_counter >= (ec.pg().size().value() % n_lhs_blocks));
#endif

        TensorElType1* cbuf_dev_ptr{nullptr};
        TensorElType1* cbuf_tmp_dev_ptr{nullptr};
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
        if(hw == ExecutionHW::GPU) {
          cbuf_dev_ptr =
            static_cast<TensorElType1*>(memDevicePool.allocate(csize * sizeof(TensorElType1)));
          cbuf_tmp_dev_ptr =
            static_cast<TensorElType1*>(memDevicePool.allocate(csize * sizeof(TensorElType1)));

          gpuMemsetAsync(reinterpret_cast<void*&>(cbuf_dev_ptr), csize * sizeof(TensorElType1),
                         thandle);
          gpuMemsetAsync(reinterpret_cast<void*&>(cbuf_tmp_dev_ptr), csize * sizeof(TensorElType1),
                         thandle);
        }
#endif

        int slc = 0;
        for(const auto& inner_it_val: inner_loop) { // k

          IndexVector a_block_id(rhs1_.labels().size());

          for(size_t i = 0; i < rhs1_map_output.size(); i++) {
            if(rhs1_map_output[i] != -1) { a_block_id[i] = itval[rhs1_map_output[i]]; }
          }

          for(size_t i = 0; i < rhs1_map_reduction.size(); i++) {
            if(rhs1_map_reduction[i] != -1) { a_block_id[i] = inner_it_val[rhs1_map_reduction[i]]; }
          }

          const auto translated_ablockid = internal::translate_blockid(a_block_id, rhs1_);
          if(!atensor.is_non_zero(translated_ablockid)) continue;

          IndexVector b_block_id(rhs2_.labels().size());

          for(size_t i = 0; i < rhs2_map_output.size(); i++) {
            if(rhs2_map_output[i] != -1) { b_block_id[i] = itval[rhs2_map_output[i]]; }
          }

          for(size_t i = 0; i < rhs2_map_reduction.size(); i++) {
            if(rhs2_map_reduction[i] != -1) { b_block_id[i] = inner_it_val[rhs2_map_reduction[i]]; }
          }

          const auto translated_bblockid = internal::translate_blockid(b_block_id, rhs2_);
          if(!btensor.is_non_zero(translated_bblockid)) continue;

#if defined(MULTOP_PARTIAL_PARALLELIZE_RHS)
          if(!(loop_counter++ % nranks_per_lhs_block == ec.pg().rank().value() / n_lhs_blocks)) {
            continue;
          }
#endif
          // compute block size and allocate buffers for abuf and bbuf
          const size_t asize = atensor.block_size(translated_ablockid);
          const size_t bsize = btensor.block_size(translated_bblockid);

          TensorElType2* abuf{nullptr};
          TensorElType3* bbuf{nullptr};
          abuf = static_cast<TensorElType2*>(memHostPool.allocate(asize * sizeof(TensorElType2)));
          bbuf = static_cast<TensorElType3*>(memHostPool.allocate(bsize * sizeof(TensorElType3)));

#ifdef DO_NB_GET
          DataCommunicationHandle a_nbhandle, b_nbhandle;

          {
            TimerGuard tg_get{&oprof.multOpGetTime};
            atensor.nb_get(translated_ablockid, {abuf, asize}, &a_nbhandle);
            btensor.nb_get(translated_bblockid, {bbuf, bsize}, &b_nbhandle);
          }
          {
            TimerGuard tg_wait{&multOpWaitTime};
            if(!a_nbhandle.getCompletionStatus()) a_nbhandle.waitForCompletion();
            if(!b_nbhandle.getCompletionStatus()) b_nbhandle.waitForCompletion();
          }
#else
          {
            TimerGuard tg_get{&oprof.multOpGetTime};
            atensor.get(translated_ablockid, {abuf, asize});
          }
          {
            TimerGuard tg_get{&oprof.multOpGetTime};
            btensor.get(translated_bblockid, {bbuf, bsize});
          }
#endif
          const auto& adims = atensor.block_dims(translated_ablockid);
          const auto& bdims = btensor.block_dims(translated_bblockid);

          // changed cscale from 0 to 1 to aggregate on cbuf
          T cscale{1};

          SizeVec adims_sz, bdims_sz;
          for(const auto v: adims) { adims_sz.push_back(v); }
          for(const auto v: bdims) { bdims_sz.push_back(v); }

          // A*B
          {
            TimerGuard tg_bc{&oprof.multOpBCTime};
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
            TensorElType2* abuf_dev{nullptr};
            TensorElType3* bbuf_dev{nullptr};
            if(hw == ExecutionHW::GPU) {
              abuf_dev =
                static_cast<TensorElType2*>(memDevicePool.allocate(asize * sizeof(TensorElType2)));
              bbuf_dev =
                static_cast<TensorElType3*>(memDevicePool.allocate(bsize * sizeof(TensorElType3)));
            }
#endif

            kernels::block_multiply<T, TensorElType1, TensorElType2, TensorElType3>(
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
              abuf_dev, bbuf_dev,
#endif
              thandle, alpha_, abuf, adims_sz, rhs1_int_labels_, bbuf, bdims_sz, rhs2_int_labels_,
              cscale, cbuf, cdims_sz, lhs_int_labels_, hw, false, cbuf_dev_ptr, cbuf_tmp_dev_ptr);

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
            if(hw == ExecutionHW::GPU) {
              memDevicePool.deallocate(abuf_dev, asize * sizeof(TensorElType2));
              memDevicePool.deallocate(bbuf_dev, bsize * sizeof(TensorElType3));
            }
#endif
          } // A * B
          slc++;

          memHostPool.deallocate(abuf, asize * sizeof(TensorElType2));
          memHostPool.deallocate(bbuf, bsize * sizeof(TensorElType3));
        } // end of reduction loop

        // add the computed update to the tensor
        {
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
          // copy to host
          if(hw == ExecutionHW::GPU) {
            TimerGuard     tg_bc{&oprof.multOpBCTime};
            TensorElType1* cbuf_tmp{nullptr};
            cbuf_tmp =
              static_cast<TensorElType1*>(memHostPool.allocate(csize * sizeof(TensorElType1)));
            std::memset(cbuf_tmp, 0, csize * sizeof(TensorElType1));
            {
              TimerGuard tg_copy{&oprof.multOpCopyTime};
              gpuMemcpyAsync<TensorElType1>(cbuf_tmp, cbuf_dev_ptr, csize, gpuMemcpyDeviceToHost,
                                            thandle);
              gpuStreamSynchronize(thandle);
            }
            // cbuf+=cbuf_tmp
            blas::axpy(csize, TensorElType1{1}, cbuf_tmp, 1, cbuf, 1);

            memHostPool.deallocate(cbuf_tmp, csize * sizeof(TensorElType1));
          }
#endif
          {
            TimerGuard tg_add{&oprof.multOpAddTime};
            ctensor.add(translated_cblockid, {cbuf, csize});
          }
        }

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
        if(hw == ExecutionHW::GPU) {
          memDevicePool.deallocate(cbuf_dev_ptr, csize * sizeof(TensorElType1));
          memDevicePool.deallocate(cbuf_tmp_dev_ptr, csize * sizeof(TensorElType1));
        }
#endif
      } // multoptime

      delete ab;
      memHostPool.deallocate(cbuf, csize * sizeof(TensorElType1));
    };
    //@todo use a scheduler
    //@todo make parallel

#if 0 && 6
            do_work(ec, lhs_loop_nest, lambda);
#elif defined(MULTOP_PARTIAL_PARALLELIZE_RHS)
    {
      const auto& ldist        = lhs_.tensor().distribution();
      int         me           = ec.pg().rank().value();
      int         nranks       = ec.pg().rank().value();
      int         n_lhs_blocks = 0;
      for(const auto& lblockid: lhs_loop_nest) {
        if(lhs_.tensor().is_non_zero(lblockid)) { n_lhs_blocks += 1; }
      }
      int lhs_counter = 0;

      for(const auto& lblockid: lhs_loop_nest) {
        if(!lhs_.tensor().is_non_zero(lblockid)) { continue; }
        lhs_counter += 1;
        if(std::get<0>(ldist.locate(lblockid)) == me % n_lhs_blocks) {
          nranks_per_lhs_block =
            (nranks / n_lhs_blocks) + 1 - (lhs_counter >= (nranks % n_lhs_blocks));
          lambda(lblockid);
          // multOpGetTime += 1;
        }
      }
    }
#else
    {
      const auto& ldist = lhs_.tensor().distribution();
      Proc        me    = ec.pg().rank();

      for(const auto& lblockid: lhs_loop_nest) {
        const auto translated_lblockid = internal::translate_blockid(lblockid, lhs_);
        if(lhs_.tensor().is_non_zero(translated_lblockid) &&
           std::get<0>(ldist.locate(translated_lblockid)) == me) {
          lambda(lblockid);
        }
      }
    }
#endif
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
    res.push_back(rhs1_.base_ptr());
    res.push_back(rhs2_.base_ptr());

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
    std::map<std::string, Label> str_to_labels;
    const size_t                 lsize  = lhs_.labels().size();
    const size_t                 r1size = rhs1_.labels().size();
    // const size_t                 r2size = rhs2_.labels().size();

    update_fillin_map(str_to_labels, lhs_.str_map(), lhs_.str_labels(), 0);
    update_fillin_map(str_to_labels, rhs1_.str_map(), rhs1_.str_labels(), lsize);
    update_fillin_map(str_to_labels, rhs2_.str_map(), rhs2_.str_labels(), lsize + r1size);
    fillin_tensor_label_from_map(lhs_, str_to_labels);
    fillin_tensor_label_from_map(rhs1_, str_to_labels);
    fillin_tensor_label_from_map(rhs2_, str_to_labels);
  }

  void fillin_int_labels() {
    std::map<TileLabelElement, int> primary_labels_map;
    int                             cnt = -1;
    for(const auto& lbl: lhs_.labels()) { primary_labels_map[lbl.primary_label()] = --cnt; }
    for(const auto& lbl: rhs1_.labels()) { primary_labels_map[lbl.primary_label()] = --cnt; }
    for(const auto& lbl: rhs2_.labels()) { primary_labels_map[lbl.primary_label()] = --cnt; }
    for(const auto& lbl: lhs_.labels()) {
      lhs_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
    }
    for(const auto& lbl: rhs1_.labels()) {
      rhs1_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
    }
    for(const auto& lbl: rhs2_.labels()) {
      rhs2_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
    }
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
   * @pre lhs_.validate(), rhs1_.validate() and rhs2_.validate() have been
   *  invoked
   */
  void validate() {
    if(!(lhs_.tensor().base_ptr() != rhs1_.tensor().base_ptr() &&
         lhs_.tensor().base_ptr() != rhs2_.tensor().base_ptr())) {
      std::ostringstream os;
      os << "[TAMM ERROR] Self assignment is not supported in tensor operations!\n"
         << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }

    IndexLabelVec ilv{lhs_.labels()};
    ilv.insert(ilv.end(), rhs1_.labels().begin(), rhs1_.labels().end());
    ilv.insert(ilv.end(), rhs2_.labels().begin(), rhs2_.labels().end());

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

  LabeledTensorT1 lhs_;
  T               alpha_;
  LabeledTensorT2 rhs1_;
  LabeledTensorT3 rhs2_;
  IntLabelVec     lhs_int_labels_;
  IntLabelVec     rhs1_int_labels_;
  IntLabelVec     rhs2_int_labels_;
  bool            is_assign_;

public:
  std::string opstr_;

  enum class Plan { invalid, lhs, flat, general_lhs, general_flat };
  Plan plan_ = Plan::invalid;
  std::shared_ptr<internal::MultOpPlanBase<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3>>
    plan_obj_;
  std::shared_ptr<internal::MultOpPlanBase<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3>>
    general_plan_obj_;

}; // class MultOp

} // namespace tamm

namespace tamm::internal {

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2, typename LabeledTensorT3>
using MultOpT = MultOp<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3>;

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2, typename LabeledTensorT3>
void FlatMultPlan<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3>::apply(
  const MultOpT& multop, ExecutionContext& ec, ExecutionHW hw) {
  using T1 = typename LabeledTensorT1::element_type;
  using T2 = typename LabeledTensorT2::element_type;
  using T3 = typename LabeledTensorT3::element_type;

  auto lhs_lt  = multop.lhs();
  auto rhs1_lt = multop.rhs1();
  auto rhs2_lt = multop.rhs2();

  Scalar alpha     = multop.alpha();
  auto   is_assign = multop.is_assign();
  ///@todo: we don't have a way to give beta in an op so it will be set to 1
  /// if it is update 0 if it is a set operation
  Scalar beta = is_assign ? 0 : 1;

  T1*                 lhs_buf  = lhs_lt.tensor().access_local_buf();
  size_t              lhs_size = lhs_lt.tensor().local_buf_size();
  std::vector<size_t> lhs_dims{lhs_size};

  T2*                 rhs1_buf  = rhs1_lt.tensor().access_local_buf();
  size_t              rhs1_size = rhs1_lt.tensor().local_buf_size();
  std::vector<size_t> rhs1_dims{rhs1_size};

  T3*                 rhs2_buf  = rhs2_lt.tensor().access_local_buf();
  size_t              rhs2_size = rhs2_lt.tensor().local_buf_size();
  std::vector<size_t> rhs2_dims{rhs2_size};

  EXPECTS(rhs1_size == lhs_size);
  EXPECTS(rhs2_size == lhs_size);

  BlockSpan<T1> lhs_span{lhs_buf, lhs_dims};
  BlockSpan<T2> rhs1_span{rhs1_buf, rhs1_dims};
  BlockSpan<T3> rhs2_span{rhs2_buf, rhs2_dims};

  BlockMultPlan::OpType optype = is_assign ? BlockMultPlan::OpType::set
                                           : BlockMultPlan::OpType::update;

  BlockMultPlan plan{lhs_lt.labels(), rhs1_lt.labels(), rhs2_lt.labels(), optype};

  plan.apply(beta, lhs_span, alpha, rhs1_span, rhs2_span);
}

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2, typename LabeledTensorT3>
void LHSMultPlan<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3>::apply(const MultOpT& multop,
                                                                              ExecutionContext& ec,
                                                                              ExecutionHW hw) {
  using T1 = typename LabeledTensorT1::element_type;
  using T2 = typename LabeledTensorT2::element_type;
  using T3 = typename LabeledTensorT3::element_type;

  auto lhs_lt  = multop.lhs();
  auto rhs1_lt = multop.rhs1();
  auto rhs2_lt = multop.rhs2();

  Scalar alpha         = multop.alpha();
  auto   is_assign     = multop.is_assign();
  Scalar beta          = is_assign ? T1{0} : T1{1};
  auto   lhs_alloc_lt  = lhs_lt.tensor()();
  auto   rhs1_alloc_lt = rhs1_lt.tensor()();
  auto   rhs2_alloc_lt = rhs2_lt.tensor()();

  const auto& ldist  = lhs_lt.tensor().distribution();
  const auto& r1dist = rhs1_lt.tensor().distribution();
  const auto& r2dist = rhs2_lt.tensor().distribution();

  Proc me = ec.pg().rank();

  auto merged_use_labels =
    internal::merge_vector<IndexLabelVec>(lhs_lt.labels(), rhs1_lt.labels(), rhs2_lt.labels());

  auto merged_alloc_labels = internal::merge_vector<IndexLabelVec>(
    lhs_alloc_lt.labels(), rhs1_alloc_lt.labels(), rhs2_alloc_lt.labels());

  LabelLoopNest loop_nest{merged_use_labels};

  BlockMultPlan::OpType optype = is_assign ? BlockMultPlan::OpType::set
                                           : BlockMultPlan::OpType::update;

  BlockMultPlan plan{lhs_lt.labels(), rhs1_lt.labels(), rhs2_lt.labels(), optype};

  auto lambda = [&](const IndexVector& l_blockid, Offset l_offset, const IndexVector& r1_blockid,
                    const IndexVector& r2_blockid) {
    auto lhs_tensor  = lhs_lt.tensor();
    auto rhs1_tensor = rhs1_lt.tensor();
    auto rhs2_tensor = rhs2_lt.tensor();

    auto  lhs_blocksize = lhs_tensor.block_size(l_blockid);
    auto  lhs_blockdims = lhs_tensor.block_dims(l_blockid);
    auto* lhs_buf       = lhs_tensor.access_local_buf() + l_offset.value();

    auto rhs1_blocksize = rhs1_tensor.block_size(r1_blockid);
    auto rhs1_blockdims = rhs1_tensor.block_dims(r1_blockid);

    auto rhs2_blocksize = rhs2_tensor.block_size(r2_blockid);
    auto rhs2_blockdims = rhs2_tensor.block_dims(r2_blockid);

    std::vector<T2> rhs1_buf(rhs1_blocksize);
    std::vector<T3> rhs2_buf(rhs2_blocksize);

    rhs1_tensor.get(r1_blockid, rhs1_buf);
    rhs2_tensor.get(r2_blockid, rhs2_buf);

    BlockSpan<T1> lhs_span{lhs_buf, lhs_blockdims};
    BlockSpan<T2> rhs1_span{rhs1_buf.data(), rhs1_blockdims};
    BlockSpan<T2> rhs2_span{rhs2_buf.data(), rhs2_blockdims};

    plan.apply(beta, lhs_span, alpha, rhs1_span, rhs2_span);
  };

  internal::LabelTranslator translator{merged_use_labels, merged_alloc_labels};
  for(const auto& blockid: loop_nest) {
    auto [translated_blockid, tlb_valid] = translator.apply(blockid);

    auto [l_blockid, r1_blockid, r2_blockid] = internal::split_vector<IndexVector, 3>(
      translated_blockid,
      {lhs_lt.labels().size(), rhs1_lt.labels().size(), rhs2_lt.labels().size()});

    if(!lhs_lt.tensor().is_non_zero(l_blockid) || !rhs1_lt.tensor().is_non_zero(r1_blockid) ||
       !rhs2_lt.tensor().is_non_zero(r2_blockid)) {
      continue;
    }

    auto [lhs_proc, lhs_offset]   = ldist.locate(l_blockid);
    auto [rhs1_proc, rhs1_offset] = r1dist.locate(r1_blockid);
    auto [rhs2_proc, rhs2_offset] = r2dist.locate(r2_blockid);

    if(tlb_valid && lhs_proc == me) { lambda(l_blockid, lhs_offset, r1_blockid, r2_blockid); }
  }
}

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2, typename LabeledTensorT3>
void GeneralFlatMultPlan<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3>::apply(
  const MultOpT& multop, ExecutionContext& ec, ExecutionHW hw) {}

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2, typename LabeledTensorT3>
void GeneralLHSMultPlan<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3>::apply(
  const MultOpT& multop, ExecutionContext& ec, ExecutionHW hw) {}

} // namespace tamm::internal
