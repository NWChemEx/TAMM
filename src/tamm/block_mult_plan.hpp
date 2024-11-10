#pragma once

#include "tamm/block_assign_plan.hpp"
#include "tamm/block_span.hpp"
#include "tamm/blockops_blas.hpp"
#include "tamm/errors.hpp"
#include "tamm/tiled_index_space.hpp"
#include "tamm/types.hpp"

#include <algorithm>

/**
 * @brief Block multiply plan selection logic.
 *
 * - Terms:
 *   - Reduction index: an index in RHS but not in LHS. e.g., j in A(i) +=
 * B(i,j)
 *   - Hadamard index: an index in LHS and both RHS tensors. e.g., l in A(l,i,j)
 * += B(l,i,k) * C(l,k,j)
 *
 * - Choose FLAT plan if:
 *   - rhs are scalars or 1d
 *   - lhs is 1d
 *   - lhs and non-scalar rhs have the same label
 *
 * - else, choose LOOP GEMM plan if:
 *   - No repeated labels in any labeled tensor
 *   - No reduction labels
 *   - Hadamard labels (if any) are outermost in all tensors
 *
 * - else, choose LOOP TTGT plan if:
 *   - No repeated labels in any labeled tensors
 *   - No reduction labels
 *
 * - else, choose general plan if:
 *   - No repeated labels in any labeled tensors
 *
 * - else:
 *   - NOT_IMPLEMENTED() for now
 *
 */

namespace tamm::internal {

class FlatBlockMultPlan {
public:
  FlatBlockMultPlan(): valid_{false} {}
  FlatBlockMultPlan(const FlatBlockMultPlan&)            = default;
  FlatBlockMultPlan& operator=(const FlatBlockMultPlan&) = default;
  FlatBlockMultPlan& operator=(FlatBlockMultPlan&&)      = default;
  ~FlatBlockMultPlan()                                   = default;
  FlatBlockMultPlan(const IndexLabelVec& lhs_labels, const IndexLabelVec& rhs1_labels,
                    const IndexLabelVec& rhs2_labels):
    lhs_labels_{lhs_labels}, rhs1_labels_{rhs1_labels}, rhs2_labels_{rhs2_labels} {
    prep();
  }

  template<typename T>
  void apply_assign(BlockSpan<T>& lhs, const BlockSpan<T>& rhs1, const BlockSpan<T>& rhs2) {
    switch(op_type_) {
      case FlatOpType::scalar_scalar: scalar_mult_assign(lhs, rhs1, rhs2); break;
      case FlatOpType::scalar_vector: scalar_vec_mult_assign(lhs, rhs1, rhs2); break;
      case FlatOpType::vector_vector:
        vec_vec_mult_assign(lhs, rhs1, rhs2);
        break;
        /// @bug: clang doen't like to have defaults when enum type is
        /// used in switch cases
        // default:
        //   break;
    }
  }

  template<typename T>
  void apply_assign(BlockSpan<T>& lhs, T alpha, const BlockSpan<T>& rhs1,
                    const BlockSpan<T>& rhs2) {
    switch(op_type_) {
      case FlatOpType::scalar_scalar: scalar_mult_assign(lhs, alpha, rhs1, rhs2); break;
      case FlatOpType::scalar_vector: scalar_vec_mult_assign(lhs, alpha, rhs1, rhs2); break;
      case FlatOpType::vector_vector:
        vec_vec_mult_assign(lhs, alpha, rhs1, rhs2);
        break;
        /// @bug: clang doen't like to have defaults when enum type is
        /// used in switch cases
        // default:
        //   break;
    }
  }

  template<typename T>
  void apply_update(BlockSpan<T>& lhs, const BlockSpan<T>& rhs1, const BlockSpan<T>& rhs2) {
    switch(op_type_) {
      case FlatOpType::scalar_scalar: scalar_mult_update(lhs, rhs1, rhs2); break;
      case FlatOpType::scalar_vector: scalar_vec_mult_update(lhs, rhs1, rhs2); break;
      case FlatOpType::vector_vector:
        vec_vec_mult_update(lhs, rhs1, rhs2);
        break;
        /// @bug: clang doen't like to have defaults when enum type is
        /// used in switch cases
        // default:
        //   break;
    }
  }

  template<typename T>
  void apply_update(BlockSpan<T>& lhs, T alpha, const BlockSpan<T>& rhs1,
                    const BlockSpan<T>& rhs2) {
    switch(op_type_) {
      case FlatOpType::scalar_scalar: scalar_mult_update(lhs, alpha, rhs1, rhs2); break;
      case FlatOpType::scalar_vector: scalar_vec_mult_update(lhs, alpha, rhs1, rhs2); break;
      case FlatOpType::vector_vector:
        vec_vec_mult_update(lhs, alpha, rhs1, rhs2);
        break;
        /// @bug: clang doen't like to have defaults when enum type is
        /// used in switch cases
        // default:
        //   break;
    }
  }

  template<typename T>
  void apply_update(T beta, BlockSpan<T>& lhs, T alpha, const BlockSpan<T>& rhs1,
                    const BlockSpan<T>& rhs2) {
    switch(op_type_) {
      case FlatOpType::scalar_scalar: scalar_mult_update(beta, lhs, alpha, rhs1, rhs2); break;
      case FlatOpType::scalar_vector: scalar_vec_mult_update(beta, lhs, alpha, rhs1, rhs2); break;
      case FlatOpType::vector_vector:
        vec_vec_mult_update(beta, lhs, alpha, rhs1, rhs2);
        break;
        /// @bug: clang doen't like to have defaults when enum type is
        /// used in switch cases
        // default:
        //   break;
    }
  }

private:
  enum class FlatOpType {
    scalar_scalar,
    scalar_vector,
    vector_vector,
  };

  void prep() {
    EXPECTS(lhs_labels_.size() < 2);
    EXPECTS(rhs1_labels_.size() < 2);
    EXPECTS(rhs2_labels_.size() < 2);

    if(rhs1_labels_.size() == 0 && rhs2_labels_.size() == 0 && lhs_labels_.size() == 0) {
      op_type_ = FlatOpType::scalar_scalar;
      valid_   = true;
      return;
    }
    else if(rhs1_labels_.size() == 1 && rhs2_labels_.size() == 1 && lhs_labels_.size() == 1) {
      op_type_ = FlatOpType::vector_vector;
      valid_   = true;
      return;
    }
    else if((lhs_labels_.size() == 1 && rhs1_labels_.size() == 1) || rhs2_labels_.size() == 1) {
      op_type_ = FlatOpType::scalar_vector;
      valid_   = true;
    }
    else { valid_ = false; }
  }

  template<typename T>
  void scalar_mult_update(T beta, BlockSpan<T>& lhs, T alpha, const BlockSpan<T>& rhs1,
                          const BlockSpan<T>& rhs2) {
    lhs[0] = (beta * lhs[0]) + (alpha * rhs1[0] * rhs2[0]);
  }

  template<typename T>
  void scalar_mult_update(BlockSpan<T>& lhs, T alpha, const BlockSpan<T>& rhs1,
                          const BlockSpan<T>& rhs2) {
    lhs[0] += alpha * rhs1[0] * rhs2[0];
  }

  template<typename T>
  void scalar_mult_update(BlockSpan<T>& lhs, const BlockSpan<T>& rhs1, const BlockSpan<T>& rhs2) {
    lhs[0] += rhs1[0] * rhs2[0];
  }

  template<typename T>
  void scalar_mult_assign(BlockSpan<T>& lhs, T alpha, const BlockSpan<T>& rhs1,
                          const BlockSpan<T>& rhs2) {
    lhs[0] = alpha * rhs1[0] * rhs2[0];
  }

  template<typename T>
  void scalar_mult_assign(BlockSpan<T>& lhs, const BlockSpan<T>& rhs1, const BlockSpan<T>& rhs2) {
    lhs[0] = rhs1[0] * rhs2[0];
  }

  template<typename T>
  void scalar_vec_mult_update(T beta, BlockSpan<T>& lhs_vec, T alpha,
                              const BlockSpan<T>& rhs1_scalar, const BlockSpan<T>& rhs2_vec) {
    EXPECTS(lhs_vec.num_elements() == rhs2_vec.num_elements());
    for(size_t i = 0; i < lhs_vec.num_elements(); i++) {
      auto new_alpha = alpha * rhs1_scalar[0];
      blockops::cpu::flat_update(beta, lhs_vec, new_alpha, rhs2_vec);
    }
  }

  template<typename T>
  void scalar_vec_mult_update(BlockSpan<T>& lhs_vec, T alpha, const BlockSpan<T>& rhs1_scalar,
                              const BlockSpan<T>& rhs2_vec) {
    EXPECTS(lhs_vec.num_elements() == rhs2_vec.num_elements());
    for(size_t i = 0; i < lhs_vec.num_elements(); i++) {
      auto new_alpha = alpha * rhs1_scalar[0];
      blockops::cpu::flat_update(lhs_vec, new_alpha, rhs2_vec);
    }
  }

  template<typename T>
  void scalar_vec_mult_assign(BlockSpan<T>& lhs_vec, T alpha, const BlockSpan<T>& rhs1_scalar,
                              const BlockSpan<T>& rhs2_vec) {
    EXPECTS(lhs_vec.num_elements() == rhs2_vec.num_elements());
    for(size_t i = 0; i < lhs_vec.num_elements(); i++) {
      auto new_alpha = alpha * rhs1_scalar[0];
      blockops::cpu::flat_assign(lhs_vec, new_alpha, rhs2_vec);
    }
  }

  template<typename T>
  void vec_vec_mult_update(T beta, BlockSpan<T>& lhs_vec, T alpha, const BlockSpan<T>& rhs1_vec,
                           const BlockSpan<T>& rhs2_vec) {
    for(size_t i = 0; i < lhs_vec.num_elements(); i++) {
      lhs_vec[i] = (beta * lhs_vec[i]) + (alpha * rhs1_vec[i] * rhs2_vec[i]);
    }
  }

  template<typename T>
  void vec_vec_mult_update(BlockSpan<T>& lhs_vec, T alpha, const BlockSpan<T>& rhs1_vec,
                           const BlockSpan<T>& rhs2_vec) {
    for(size_t i = 0; i < lhs_vec.num_elements(); i++) {
      lhs_vec[i] += alpha * rhs1_vec[i] * rhs2_vec[i];
    }
  }

  template<typename T>
  void vec_vec_mult_update(BlockSpan<T>& lhs_vec, const BlockSpan<T>& rhs1_vec,
                           const BlockSpan<T>& rhs2_vec) {
    for(size_t i = 0; i < lhs_vec.num_elements(); i++) { lhs_vec[i] += rhs1_vec[i] * rhs2_vec[i]; }
  }

  template<typename T>
  void vec_vec_mult_assign(BlockSpan<T>& lhs_vec, T alpha, const BlockSpan<T>& rhs1_vec,
                           const BlockSpan<T>& rhs2_vec) {
    for(size_t i = 0; i < lhs_vec.num_elements(); i++) {
      lhs_vec[i] = alpha * rhs1_vec[i] * rhs2_vec[i];
    }
  }

  template<typename T>
  void vec_vec_mult_assign(BlockSpan<T>& lhs_vec, const BlockSpan<T>& rhs1_vec,
                           const BlockSpan<T>& rhs2_vec) {
    for(size_t i = 0; i < lhs_vec.num_elements(); i++) { lhs_vec[i] = rhs1_vec[i] * rhs2_vec[i]; }
  }

  IndexLabelVec lhs_labels_;
  IndexLabelVec rhs1_labels_;
  IndexLabelVec rhs2_labels_;
  FlatOpType    op_type_;
  bool          valid_;
}; // class FlatBlockMultPlan
/**
 * @brief basic ttgt plan
 *
 * @todo
 * - Loop ttgt
 * - Avoid transpose when not needed
 * - Separate prep() from apply()
 *
 */
class TTGTPlan {
public:
  TTGTPlan(): valid_{false} {}
  TTGTPlan(const TTGTPlan&)            = default;
  TTGTPlan& operator=(const TTGTPlan&) = default;
  TTGTPlan& operator=(TTGTPlan&&)      = default;
  ~TTGTPlan()                          = default;
  TTGTPlan(const IndexLabelVec& lhs_labels, const IndexLabelVec& rhs1_labels,
           const IndexLabelVec& rhs2_labels) {
    prep(lhs_labels, rhs1_labels, rhs2_labels);
  }

  template<typename T>
  void apply(T beta, BlockSpan<T>& lhs, T alpha, const BlockSpan<T>& rhs1,
             const BlockSpan<T>& rhs2) {
    NOT_IMPLEMENTED();
  }

  bool is_valid() const { return valid_; }

private:
  void prep(const IndexLabelVec& lhs_labels, const IndexLabelVec& rhs1_labels,
            const IndexLabelVec& rhs2_labels) {
    lhs_labels_  = lhs_labels;
    rhs1_labels_ = rhs1_labels;
    rhs2_labels_ = rhs2_labels;
  }
  IndexLabelVec lhs_labels_;
  IndexLabelVec rhs1_labels_;
  IndexLabelVec rhs2_labels_;
  bool          valid_;
}; // class TTGTPlan

class GemmPlan {
public:
  GemmPlan(): valid_{false} {}

  GemmPlan(const GemmPlan&)            = default;
  GemmPlan& operator=(const GemmPlan&) = default;

  GemmPlan(const IndexLabelVec& lhs_labels, const IndexLabelVec& rhs1_labels,
           const IndexLabelVec& rhs2_labels) {
    if(has_repeated_indices(lhs_labels) || has_repeated_indices(rhs1_labels) ||
       has_repeated_indices(rhs2_labels)) {
      return;
    }

    auto find = [](const auto& collection, const auto& element) {
      return std::find(collection.begin(), collection.end(), element);
    };

    auto has = [find](const auto& collection, const auto& element) {
      return find(collection, element) != collection.end();
    };

    bool                 rhs1_is_arg1 = (lhs_labels.size() == 0 || has(rhs1_labels, lhs_labels[0]));
    const IndexLabelVec& arg1_labels  = (rhs1_is_arg1 ? rhs1_labels : rhs2_labels);
    const IndexLabelVec& arg2_labels  = (rhs1_is_arg1 ? rhs2_labels : rhs1_labels);
    if(rhs1_labels.size() + rhs2_labels.size() < lhs_labels.size()) { return; }
    size_t num_kindices = (rhs1_labels.size() + rhs2_labels.size() - lhs_labels.size()) / 2;
    if(arg1_labels.size() < num_kindices || arg2_labels.size() < num_kindices) { return; }
    size_t num_mindices = arg1_labels.size() - num_kindices;
    size_t num_nindices = arg2_labels.size() - num_kindices;
    if(num_mindices < 0 || num_nindices < 0 || num_kindices < 0) { return; }
    if((lhs_labels.size() != num_mindices + num_nindices) ||
       (arg1_labels.size() != num_mindices + num_kindices) ||
       (arg2_labels.size() != num_kindices + num_nindices)) {
      return;
    }
    bool transpose_arg1 = (num_mindices > 0 && lhs_labels[0] != arg1_labels[0]);
    bool transpose_arg2 = (num_nindices > 0 && lhs_labels[num_mindices] == arg2_labels[0]);

    int lhs_mpos  = 0;
    int lhs_npos  = num_mindices;
    int arg1_mpos = transpose_arg1 ? num_kindices : 0;
    int arg1_kpos = transpose_arg1 ? 0 : num_mindices;
    int arg2_kpos = transpose_arg2 ? num_nindices : 0;
    int arg2_npos = transpose_arg2 ? 0 : num_kindices;
    if(!std::equal(lhs_labels.begin() + lhs_mpos, lhs_labels.begin() + lhs_mpos + num_mindices,
                   arg1_labels.begin() + arg1_mpos)) {
      return;
    }
    if(!std::equal(lhs_labels.begin() + lhs_npos, lhs_labels.begin() + lhs_npos + num_nindices,
                   arg2_labels.begin() + arg2_npos)) {
      return;
    }
    if(!std::equal(arg1_labels.begin() + arg1_kpos, arg1_labels.begin() + arg1_kpos + num_kindices,
                   arg2_labels.begin() + arg2_kpos)) {
      return;
    }

    num_mindices_   = num_mindices;
    num_nindices_   = num_nindices;
    num_kindices_   = num_kindices;
    rhs1_is_arg1_   = rhs1_is_arg1;
    transpose_arg1_ = transpose_arg1;
    transpose_arg2_ = transpose_arg2;
    valid_          = true;
  }

  bool is_valid() const { return valid_; }

  template<typename T0, typename T1, typename T2, typename T3, typename T4>
  void apply(T0 beta, BlockSpan<T1>& lhs, T1 alpha, BlockSpan<T2>& rhs1, BlockSpan<T3>& rhs2) {
    EXPECTS(valid_);
    update_plan(lhs, rhs1, rhs2);
    size_t loff = 0, r1off = 0, r2off = 0;
    for(size_t i = 0; i < num_batches_; i++) {
      auto TransA = transpose_arg1_ ? blas::Op::Trans : blas::Op::NoTrans;
      auto TransB = transpose_arg2_ ? blas::Op::Trans : blas::Op::NoTrans;
      int  lda    = !transpose_arg1_ ? K_ : M_;
      int  ldb    = !transpose_arg2_ ? N_ : K_;
      int  ldc    = N_;

      auto* bufa = rhs1_is_arg1_ ? reinterpret_cast<T2*>(&bufa_) : reinterpret_cast<T3*>(&bufa_);
      auto* bufb = rhs1_is_arg1_ ? reinterpret_cast<T3*>(&bufb_) : reinterpret_cast<T2*>(&bufb_);
      auto* bufc = reinterpret_cast<T1*>(&bufc_);

      blas::gemm(blas::Layout::RowMajor, TransA, TransB, M_, N_, K_, alpha, &bufa[r1off], lda,
                 &bufb[r2off], ldb, beta, &bufc[loff], ldc);
      loff += M_ * N_;
      r1off += M_ * K_;
      r2off += K_ * N_;
    }
  }

private:
  bool has_repeated_indices(const IndexLabelVec& labels) {
    return labels.size() != internal::unique_entries(labels).size();
  }

  template<typename T, typename T1, typename T2>
  void update_plan(const BlockSpan<T>& lhs, const BlockSpan<T1>& rhs1, const BlockSpan<T2>& rhs2) {
    M_ = 1;
    N_ = 1;
    K_ = 1;

    const std::vector<size_t>& lhs_bdims = lhs.block_dims();
    for(size_t i = 0; i < num_mindices_; i++) { M_ *= lhs_bdims[num_batch_indices_ + i]; }
    for(size_t i = num_mindices_; i < lhs_bdims.size(); i++) {
      N_ *= lhs_bdims[num_batch_indices_ + i];
    }
    size_t      kstart_idx = (transpose_arg1_ ? 0 : num_mindices_);
    const auto& arg1_bdims = (rhs1_is_arg1_ ? rhs1.block_dims() : rhs2.block_dims());
    for(int k = 0; k < num_kindices_; k++) {
      K_ *= arg1_bdims[num_batch_indices_ + kstart_idx + k];
    }
    bufc_ = &lhs.buf();
    bufa_ = rhs1_is_arg1_ ? &rhs1.buf() : &rhs2.buf();
    bufb_ = rhs1_is_arg1_ ? &rhs2.buf() : &rhs1.buf();

    int num_batches_ = 1;
    for(int i = 0; i < num_batch_indices_; i++) { num_batches_ *= lhs_bdims[i]; }
  }

  bool valid_;
  int  num_batch_indices_;
  int  num_mindices_;
  int  num_nindices_;
  int  num_kindices_; // number of summation indices
  bool rhs1_is_arg1_; // true if LHS = RHS1()*RHS2(). False if LHS =
                      // RHS2()*RHS1()
  bool transpose_arg1_;
  bool transpose_arg2_;

  int   M_, N_, K_;
  int   num_batches_;
  void *bufa_, *bufb_, *bufc_;
}; // class GemmPlan

/**
 * @brief General mult plan with repeated indices. This is not designed for
 * sparse blocks (labels with dependent indices).
 *
 * @todo
 * - Avoid block assign plans when not needed
 * - There could bt multiple intermediate, one here and one in TTGT. Incorporate
 * TTGT functionality here to minimize copies.
 *
 */
class GeneralMultPlan {
public:
  GeneralMultPlan(): valid_{false} {}
  GeneralMultPlan(const GeneralMultPlan&)            = default;
  GeneralMultPlan& operator=(const GeneralMultPlan&) = default;

  bool is_valid() const { return valid_; }

  GeneralMultPlan(const IndexLabelVec& lhs_labels, const IndexLabelVec& rhs1_labels,
                  const IndexLabelVec& rhs2_labels):
    valid_{true}, lhs_labels_{lhs_labels}, rhs1_labels_{rhs1_labels}, rhs2_labels_{rhs2_labels} {
    for(const auto& lbl: lhs_labels) {
      if(std::find(lhs_inter_labels_.begin(), lhs_inter_labels_.end(), lbl) ==
         lhs_inter_labels_.end()) {
        lhs_inter_labels_.push_back(lbl);
      }
    }
    for(const auto& lbl: rhs1_labels) {
      if(std::find(rhs1_inter_labels_.begin(), rhs1_inter_labels_.end(), lbl) ==
         rhs1_inter_labels_.end()) {
        rhs1_inter_labels_.push_back(lbl);
      }
    }
    for(const auto& lbl: rhs2_labels) {
      if(std::find(rhs2_inter_labels_.begin(), rhs2_inter_labels_.end(), lbl) ==
         rhs2_inter_labels_.end()) {
        rhs2_inter_labels_.push_back(lbl);
      }
    }
    lhs_ba_plan_  = BlockAssignPlan{lhs_labels_, lhs_inter_labels_, BlockAssignPlan::OpType::set};
    rhs1_ba_plan_ = BlockAssignPlan{rhs1_inter_labels_, rhs1_labels_, BlockAssignPlan::OpType::set};
    rhs2_ba_plan_ = BlockAssignPlan{rhs2_inter_labels_, rhs2_labels_, BlockAssignPlan::OpType::set};
    linter_to_l_perm_   = perm_map_compute(lhs_inter_labels_, lhs_labels_);
    r1_to_r1inter_perm_ = perm_map_compute(rhs1_labels_, rhs1_inter_labels_);
    r2_to_r2inter_perm_ = perm_map_compute(rhs2_labels_, rhs2_inter_labels_);
    ttgt_plan_          = TTGTPlan{lhs_inter_labels_, rhs1_inter_labels_, rhs2_inter_labels_};
  }

  template<typename T0, typename T1, typename T2, typename T3, typename T4>
  void apply(T0 beta, BlockSpan<T1>& lhs, T1 alpha, BlockSpan<T2>& rhs1, BlockSpan<T3>& rhs2) {
    EXPECTS(valid_);
    std::vector<T1> linter_buf(lhs.num_elements());
    std::vector<T1> r1inter_buf(rhs1.num_elements());
    std::vector<T1> r2inter_buf(rhs2.num_elements());

    const auto& linter_dims  = perm_map_apply(lhs.block_dims(), linter_to_l_perm_);
    const auto& r1inter_dims = perm_map_apply(rhs1.block_dims(), r1_to_r1inter_perm_);
    const auto& r2inter_dims = perm_map_apply(rhs2.block_dims(), r2_to_r2inter_perm_);

    BlockSpan<T1> lhs_inter{linter_buf.data(), linter_dims};
    BlockSpan<T2> rhs1_inter{r1inter_buf.data(), r1inter_dims};
    BlockSpan<T3> rhs2_inter{r2inter_buf.data(), r2inter_dims};

    rhs1_ba_plan_.apply(rhs1_inter, rhs1);
    rhs2_ba_plan_.apply(rhs2_inter, rhs2);
    ttgt_plan_.apply(beta, lhs_inter, alpha, rhs1_inter, rhs2_inter);
    lhs_ba_plan_.apply(lhs, lhs_inter);
  }

private:
  bool            valid_;
  IndexLabelVec   lhs_labels_;
  IndexLabelVec   rhs1_labels_;
  IndexLabelVec   rhs2_labels_;
  PermVector      linter_to_l_perm_;
  PermVector      r1_to_r1inter_perm_;
  PermVector      r2_to_r2inter_perm_;
  IndexLabelVec   lhs_inter_labels_;
  IndexLabelVec   rhs1_inter_labels_;
  IndexLabelVec   rhs2_inter_labels_;
  BlockAssignPlan rhs1_ba_plan_;
  BlockAssignPlan rhs2_ba_plan_;
  BlockAssignPlan lhs_ba_plan_;
  TTGTPlan        ttgt_plan_;
}; // class GeneralMultPlan

} // namespace tamm::internal

namespace tamm {
class BlockMultPlan {
public:
  enum class OpType { set, update };
  BlockMultPlan(const IndexLabelVec& lhs_labels, const IndexLabelVec& rhs1_labels,
                const IndexLabelVec& rhs2_labels, OpType optype):
    lhs_labels_{lhs_labels},
    rhs1_labels_{rhs1_labels},
    rhs2_labels_{rhs2_labels},
    optype_{optype},
    plan_{Plan::invalid},
    has_repeated_index_{has_repeated_index()},
    has_reduction_index_{has_reduction_index()},
    has_hadamard_index_{has_hadamard_index()} {
    prep_flat_plan();
    if(plan_ == Plan::invalid) { prep_loop_gemm_plan(); }
    if(plan_ == Plan::invalid) { prep_loop_ttgt_plan(); }
    if(plan_ == Plan::invalid) { prep_general_plan(); }
    if(plan_ == Plan::invalid) { NOT_IMPLEMENTED(); }
    EXPECTS(plan_ != Plan::invalid);
  }

  template<typename T1, typename T2, typename T3>
  void apply_impl(T1 lscale, BlockSpan<T1>& lhs, T1 rscale, BlockSpan<T1>& rhs1,
                  BlockSpan<T1>& rhs2) {
    switch(plan_) {
      case Plan::invalid: {
        NOT_ALLOWED();
        break;
      }
      case Plan::flat_assign: {
        internal::FlatBlockMultPlan flat_plan{lhs_labels_, rhs1_labels_, rhs2_labels_};
        flat_plan.apply_assign(lscale, lhs, rscale, rhs1, rhs2, optype_);
        break;
      }
      case Plan::loop_gemm: {
        internal::GemmPlan gemm_plan{lhs_labels_, rhs1_labels_, rhs2_labels_};
        gemm_plan.apply(lscale, lhs, rscale, rhs1, rhs2);
        break;
      }
      case Plan::loop_ttgt: {
        internal::TTGTPlan ttgt_plan{lhs_labels_, rhs1_labels_, rhs2_labels_};
        ttgt_plan.apply(lscale, lhs, rscale, rhs1, rhs2);
        break;
      }
      case Plan::general: {
        internal::GeneralMultPlan general_plan{lhs_labels_, rhs1_labels_, rhs2_labels_};
        general_plan.apply(lscale, lhs, rscale, rhs1, rhs2);
        break;
      }
      default: UNREACHABLE();
    }
  }

  template<typename T1, typename T2, typename T3>
  void apply(Scalar lscale, BlockSpan<T1>& lhs, Scalar rscale, BlockSpan<T2>& rhs1,
             BlockSpan<T3>& rhs2) {}

private:
  bool has_reduction_index() {
    std::set<TiledIndexLabel> lhs_labels(lhs_labels_.begin(), lhs_labels_.end());
    std::set<TiledIndexLabel> rhs_labels(rhs1_labels_.begin(), rhs1_labels_.end());
    for(auto lbl: rhs2_labels_) { rhs_labels.insert(lbl); }

    IndexLabelVec reduction_lbls;

    std::set_difference(rhs_labels.begin(), rhs_labels.end(), lhs_labels.begin(), lhs_labels.end(),
                        std::back_inserter(reduction_lbls));

    return reduction_lbls.size() != 0;
  }

  bool has_hadamard_index() {
    std::set<TiledIndexLabel> rhs1_labels(rhs1_labels_.begin(), rhs1_labels_.end());
    std::set<TiledIndexLabel> rhs2_labels(rhs2_labels_.begin(), rhs2_labels_.end());
    for(const auto& lbl: lhs_labels_) {
      if(rhs1_labels.count(lbl) != 0 && rhs2_labels.count(lbl) != 0) { return true; }
    }
    return false;
  }

  bool has_repeated_index() {
    return internal::has_repeated_elements(rhs1_labels_) ||
           internal::has_repeated_elements(rhs2_labels_) ||
           internal::has_repeated_elements(lhs_labels_);
  }

  IndexLabelVec get_hadamard_labels() {
    if(!has_hadamard_index_) { return {}; }
    IndexLabelVec result;

    std::set<TiledIndexLabel> rhs1_labels(rhs1_labels_.begin(), rhs1_labels_.end());
    std::set<TiledIndexLabel> rhs2_labels(rhs2_labels_.begin(), rhs2_labels_.end());

    for(const auto& lbl: lhs_labels_) {
      if(rhs1_labels.count(lbl) != 0 && rhs2_labels.count(lbl) != 0) { result.push_back(lbl); }
    }

    return result;
  }

  /**
   * @brief Choose FLAT plan if:
   *   - rhs are scalars or 1d
   *   - lhs is 1d
   *   - lhs and non-scalar rhs have the same label
   */
  void prep_flat_plan() {
    if(rhs1_labels_.size() == 0 && rhs2_labels_.size() == 0 && lhs_labels_.size() == 0) {
      plan_ = optype_ == OpType::set ? Plan::flat_assign : Plan::flat_update;
      return;
    }
    else if(rhs1_labels_.size() == 1 && rhs2_labels_.size() == 1 && lhs_labels_.size() == 1) {
      if(lhs_labels_ == rhs1_labels_ && lhs_labels_ == rhs1_labels_) {
        plan_ = optype_ == OpType::set ? Plan::flat_assign : Plan::flat_update;
        return;
      }
    }
    else if(rhs1_labels_.size() == 1 || rhs1_labels_.size() == 0) {
      if(rhs2_labels_.size() == 1 || rhs2_labels_.size() == 0) {
        size_t rhs_size = std::max(rhs1_labels_.size(), rhs2_labels_.size());
        if(lhs_labels_.size() == rhs_size) {
          if((rhs1_labels_.size() == 1 && lhs_labels_ == rhs1_labels_) ||
             (rhs2_labels_.size() == 1 && lhs_labels_ == rhs2_labels_)) {
            plan_ = optype_ == OpType::set ? Plan::flat_assign : Plan::flat_update;
          }
        }
      }
    }
  }

  /**
   * @brief choose LOOP GEMM plan if:
   *   - No repeated labels in any labeled tensor
   *   - No reduction labels
   *   - Hadamard labels (if any) are outermost in all tensors
   */
  void prep_loop_gemm_plan() {
    if(!has_repeated_index_ && !has_reduction_index_) {
      if(has_hadamard_index_) {
        auto            hadamard_labels = get_hadamard_labels();
        const ptrdiff_t hlabels         = hadamard_labels.size();
        for(size_t i = 0; i < hadamard_labels.size(); i++) {
          auto lbl = lhs_labels_[i];
          auto rhs1_pos =
            std::find(rhs1_labels_.begin(), rhs2_labels_.end(), lbl) - rhs1_labels_.begin();
          auto rhs2_pos =
            std::find(rhs1_labels_.begin(), rhs2_labels_.end(), lbl) - rhs1_labels_.begin();
          if(rhs1_pos >= hlabels || rhs2_pos >= hlabels) { return; }
        }
        plan_ = Plan::loop_gemm;
      }
      else { plan_ = Plan::loop_gemm; }
    }
  }

  /**
   * @brief choose LOOP TTGT plan if:
   *   - No repeated labels in any labeled tensors
   *   - No reduction labels
   */
  void prep_loop_ttgt_plan() {
    if(!has_repeated_index_ && !has_reduction_index_) { plan_ = Plan::loop_ttgt; }
  }

  /**
   * @brief choose general plan if:
   *   - No repeated labels in any labeled tensors
   */
  void prep_general_plan() {
    if(!has_repeated_index_) { plan_ = Plan::general; }
  }

  enum class Plan {
    flat_assign,
    flat_update,
    loop_gemm,
    loop_ttgt,
    general,
    invalid,
  };

  IndexLabelVec lhs_labels_;
  IndexLabelVec rhs1_labels_;
  IndexLabelVec rhs2_labels_;
  OpType        optype_;
  Plan          plan_;

  bool has_repeated_index_;
  bool has_reduction_index_;
  bool has_hadamard_index_;
}; // class BlockMultPlan
} // namespace tamm
