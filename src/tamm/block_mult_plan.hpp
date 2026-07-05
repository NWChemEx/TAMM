#pragma once

#include "tamm/block_assign_plan.hpp"
#include "tamm/block_span.hpp"
#include "tamm/blockops_blas.hpp"
#include "tamm/errors.hpp"
#include "tamm/scalar.hpp"
#include "tamm/tiled_index_space.hpp"
#include "tamm/types.hpp"

#include <algorithm>
#include <cstddef> // std::byte
#include <variant>

/**
 * @brief Block multiply plan selection and dispatch.
 *
 * C++20 / perf design:
 *  - BlockMultPlan stores a cached std::variant of the selected sub-plan,
 *    built once in the constructor.  apply_impl() is a zero-copy std::visit
 *    dispatch — no per-call plan reconstruction or label-vector copies.
 *  - GeneralMultPlan caches permuted block dims (recomputed only on shape
 *    change) and intermediate byte-buffers (resized only on growth),
 *    eliminating all hot-path allocations in the CCSD contraction loop.
 *  - GemmPlan stores raw data pointers (void*) directly, not
 *    pointer-to-pointer, avoiding the dangling-address UB of &lhs.buf().
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
      case FlatOpType::vector_vector: vec_vec_mult_assign(lhs, rhs1, rhs2); break;
    }
  }

  template<typename T>
  void apply_assign(BlockSpan<T>& lhs, T alpha, const BlockSpan<T>& rhs1,
                    const BlockSpan<T>& rhs2) {
    switch(op_type_) {
      case FlatOpType::scalar_scalar: scalar_mult_assign(lhs, alpha, rhs1, rhs2); break;
      case FlatOpType::scalar_vector: scalar_vec_mult_assign(lhs, alpha, rhs1, rhs2); break;
      case FlatOpType::vector_vector: vec_vec_mult_assign(lhs, alpha, rhs1, rhs2); break;
    }
  }

  template<typename T>
  void apply_update(BlockSpan<T>& lhs, const BlockSpan<T>& rhs1, const BlockSpan<T>& rhs2) {
    switch(op_type_) {
      case FlatOpType::scalar_scalar: scalar_mult_update(lhs, rhs1, rhs2); break;
      case FlatOpType::scalar_vector: scalar_vec_mult_update(lhs, rhs1, rhs2); break;
      case FlatOpType::vector_vector: vec_vec_mult_update(lhs, rhs1, rhs2); break;
    }
  }

  template<typename T>
  void apply_update(BlockSpan<T>& lhs, T alpha, const BlockSpan<T>& rhs1,
                    const BlockSpan<T>& rhs2) {
    switch(op_type_) {
      case FlatOpType::scalar_scalar: scalar_mult_update(lhs, alpha, rhs1, rhs2); break;
      case FlatOpType::scalar_vector: scalar_vec_mult_update(lhs, alpha, rhs1, rhs2); break;
      case FlatOpType::vector_vector: vec_vec_mult_update(lhs, alpha, rhs1, rhs2); break;
    }
  }

  template<typename T>
  void apply_update(T beta, BlockSpan<T>& lhs, T alpha, const BlockSpan<T>& rhs1,
                    const BlockSpan<T>& rhs2) {
    switch(op_type_) {
      case FlatOpType::scalar_scalar: scalar_mult_update(beta, lhs, alpha, rhs1, rhs2); break;
      case FlatOpType::scalar_vector: scalar_vec_mult_update(beta, lhs, alpha, rhs1, rhs2); break;
      case FlatOpType::vector_vector: vec_vec_mult_update(beta, lhs, alpha, rhs1, rhs2); break;
    }
  }

private:
  enum class FlatOpType { scalar_scalar, scalar_vector, vector_vector };

  void prep() {
    EXPECTS(lhs_labels_.size() < 2);
    EXPECTS(rhs1_labels_.size() < 2);
    EXPECTS(rhs2_labels_.size() < 2);

    if(rhs1_labels_.size() == 0 && rhs2_labels_.size() == 0 && lhs_labels_.size() == 0) {
      op_type_ = FlatOpType::scalar_scalar; valid_ = true; return;
    }
    else if(rhs1_labels_.size() == 1 && rhs2_labels_.size() == 1 && lhs_labels_.size() == 1) {
      op_type_ = FlatOpType::vector_vector; valid_ = true; return;
    }
    else if((lhs_labels_.size() == 1 && rhs1_labels_.size() == 1) || rhs2_labels_.size() == 1) {
      op_type_ = FlatOpType::scalar_vector; valid_ = true;
    }
    else { valid_ = false; }
  }

  template<typename T>
  void scalar_mult_update(T beta, BlockSpan<T>& lhs, T alpha,
                          const BlockSpan<T>& rhs1, const BlockSpan<T>& rhs2) {
    lhs[0] = (beta * lhs[0]) + (alpha * rhs1[0] * rhs2[0]);
  }
  template<typename T>
  void scalar_mult_update(BlockSpan<T>& lhs, T alpha,
                          const BlockSpan<T>& rhs1, const BlockSpan<T>& rhs2) {
    lhs[0] += alpha * rhs1[0] * rhs2[0];
  }
  template<typename T>
  void scalar_mult_update(BlockSpan<T>& lhs,
                          const BlockSpan<T>& rhs1, const BlockSpan<T>& rhs2) {
    lhs[0] += rhs1[0] * rhs2[0];
  }
  template<typename T>
  void scalar_mult_assign(BlockSpan<T>& lhs, T alpha,
                          const BlockSpan<T>& rhs1, const BlockSpan<T>& rhs2) {
    lhs[0] = alpha * rhs1[0] * rhs2[0];
  }
  template<typename T>
  void scalar_mult_assign(BlockSpan<T>& lhs,
                          const BlockSpan<T>& rhs1, const BlockSpan<T>& rhs2) {
    lhs[0] = rhs1[0] * rhs2[0];
  }
  template<typename T>
  void scalar_vec_mult_update(T beta, BlockSpan<T>& lhs_vec, T alpha,
                              const BlockSpan<T>& rhs1_scalar, const BlockSpan<T>& rhs2_vec) {
    EXPECTS(lhs_vec.num_elements() == rhs2_vec.num_elements());
    blockops::cpu::flat_update(beta, lhs_vec, alpha * rhs1_scalar[0], rhs2_vec);
  }
  template<typename T>
  void scalar_vec_mult_update(BlockSpan<T>& lhs_vec, T alpha, const BlockSpan<T>& rhs1_scalar,
                              const BlockSpan<T>& rhs2_vec) {
    EXPECTS(lhs_vec.num_elements() == rhs2_vec.num_elements());
    blockops::cpu::flat_update(lhs_vec, alpha * rhs1_scalar[0], rhs2_vec);
  }
  template<typename T>
  void scalar_vec_mult_assign(BlockSpan<T>& lhs_vec, T alpha, const BlockSpan<T>& rhs1_scalar,
                              const BlockSpan<T>& rhs2_vec) {
    EXPECTS(lhs_vec.num_elements() == rhs2_vec.num_elements());
    blockops::cpu::flat_assign(lhs_vec, alpha * rhs1_scalar[0], rhs2_vec);
  }
  template<typename T>
  void scalar_vec_mult_update(BlockSpan<T>& lhs_vec, const BlockSpan<T>& rhs1_scalar,
                              const BlockSpan<T>& rhs2_vec) {
    EXPECTS(lhs_vec.num_elements() == rhs2_vec.num_elements());
    blockops::cpu::flat_update(lhs_vec, rhs1_scalar[0], rhs2_vec);
  }
  template<typename T>
  void scalar_vec_mult_assign(BlockSpan<T>& lhs_vec, const BlockSpan<T>& rhs1_scalar,
                              const BlockSpan<T>& rhs2_vec) {
    EXPECTS(lhs_vec.num_elements() == rhs2_vec.num_elements());
    blockops::cpu::flat_assign(lhs_vec, rhs1_scalar[0], rhs2_vec);
  }
  template<typename T>
  void vec_vec_mult_update(T beta, BlockSpan<T>& lhs_vec, T alpha,
                           const BlockSpan<T>& rhs1_vec,
                           const BlockSpan<T>& rhs2_vec) {
    for(size_t i = 0; i < lhs_vec.num_elements(); i++)
      lhs_vec[i] = (beta * lhs_vec[i]) + (alpha * rhs1_vec[i] * rhs2_vec[i]);
  }
  template<typename T>
  void vec_vec_mult_update(BlockSpan<T>& lhs_vec, T alpha,
                           const BlockSpan<T>& rhs1_vec,
                           const BlockSpan<T>& rhs2_vec) {
    for(size_t i = 0; i < lhs_vec.num_elements(); i++)
      lhs_vec[i] += alpha * rhs1_vec[i] * rhs2_vec[i];
  }
  template<typename T>
  void vec_vec_mult_update(BlockSpan<T>& lhs_vec,
                           const BlockSpan<T>& rhs1_vec,
                           const BlockSpan<T>& rhs2_vec) {
    for(size_t i = 0; i < lhs_vec.num_elements(); i++) lhs_vec[i] += rhs1_vec[i] * rhs2_vec[i];
  }
  template<typename T>
  void vec_vec_mult_assign(BlockSpan<T>& lhs_vec, T alpha,
                           const BlockSpan<T>& rhs1_vec,
                           const BlockSpan<T>& rhs2_vec) {
    for(size_t i = 0; i < lhs_vec.num_elements(); i++)
      lhs_vec[i] = alpha * rhs1_vec[i] * rhs2_vec[i];
  }
  template<typename T>
  void vec_vec_mult_assign(BlockSpan<T>& lhs_vec,
                           const BlockSpan<T>& rhs1_vec,
                           const BlockSpan<T>& rhs2_vec) {
    for(size_t i = 0; i < lhs_vec.num_elements(); i++) lhs_vec[i] = rhs1_vec[i] * rhs2_vec[i];
  }

  IndexLabelVec lhs_labels_;
  IndexLabelVec rhs1_labels_;
  IndexLabelVec rhs2_labels_;
  FlatOpType    op_type_;
  bool          valid_;
};

// ---------------------------------------------------------------------------

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
  void apply(T beta, BlockSpan<T>& lhs, T alpha,
             const BlockSpan<T>& rhs1, const BlockSpan<T>& rhs2) {
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
};

// ---------------------------------------------------------------------------

class GemmPlan {
public:
  GemmPlan(): valid_{false} {}
  GemmPlan(const GemmPlan&)            = default;
  GemmPlan& operator=(const GemmPlan&) = default;

  GemmPlan(const IndexLabelVec& lhs_labels, const IndexLabelVec& rhs1_labels,
           const IndexLabelVec& rhs2_labels) {
    if(has_repeated_indices(lhs_labels) || has_repeated_indices(rhs1_labels) ||
       has_repeated_indices(rhs2_labels)) { return; }

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
       (arg2_labels.size() != num_kindices + num_nindices)) { return; }
    bool transpose_arg1 = (num_mindices > 0 && lhs_labels[0] != arg1_labels[0]);
    bool transpose_arg2 = (num_nindices > 0 && lhs_labels[num_mindices] == arg2_labels[0]);

    int lhs_mpos  = 0;
    int lhs_npos  = num_mindices;
    int arg1_mpos = transpose_arg1 ? num_kindices : 0;
    int arg1_kpos = transpose_arg1 ? 0 : num_mindices;
    int arg2_kpos = transpose_arg2 ? num_nindices : 0;
    int arg2_npos = transpose_arg2 ? 0 : num_kindices;
    if(!std::equal(lhs_labels.begin() + lhs_mpos,
                   lhs_labels.begin() + lhs_mpos + num_mindices,
                   arg1_labels.begin() + arg1_mpos)) { return; }
    if(!std::equal(lhs_labels.begin() + lhs_npos,
                   lhs_labels.begin() + lhs_npos + num_nindices,
                   arg2_labels.begin() + arg2_npos)) { return; }
    if(!std::equal(arg1_labels.begin() + arg1_kpos,
                   arg1_labels.begin() + arg1_kpos + num_kindices,
                   arg2_labels.begin() + arg2_kpos)) { return; }

    num_mindices_   = num_mindices;
    num_nindices_   = num_nindices;
    num_kindices_   = num_kindices;
    rhs1_is_arg1_   = rhs1_is_arg1;
    transpose_arg1_ = transpose_arg1;
    transpose_arg2_ = transpose_arg2;
    valid_          = true;
  }

  bool is_valid() const { return valid_; }

  template<typename T0, typename T1, typename T2, typename T3>
  void apply(T0 beta, BlockSpan<T1>& lhs, T1 alpha, BlockSpan<T2>& rhs1, BlockSpan<T3>& rhs2) {
    EXPECTS(valid_);
    update_plan(lhs, rhs1, rhs2);
    size_t loff = 0, r1off = 0, r2off = 0;
    for(size_t i = 0; i < static_cast<size_t>(num_batches_); i++) {
      auto TransA = transpose_arg1_ ? blas::Op::Trans : blas::Op::NoTrans;
      auto TransB = transpose_arg2_ ? blas::Op::Trans : blas::Op::NoTrans;
      int  lda    = !transpose_arg1_ ? K_ : M_;
      int  ldb    = !transpose_arg2_ ? N_ : K_;
      int  ldc    = N_;
      // Fix: store and use raw data pointers directly (not pointer-to-pointer).
      auto* bufa = static_cast<T2*>(bufa_);
      auto* bufb = static_cast<T3*>(bufb_);
      auto* bufc = static_cast<T1*>(bufc_);
      blas::gemm(blas::Layout::RowMajor, TransA, TransB, M_, N_, K_, alpha, bufa + r1off, lda,
                 bufb + r2off, ldb, beta, bufc + loff, ldc);
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
    M_                    = 1;
    N_                    = 1;
    K_                    = 1;
    const auto& lhs_bdims = lhs.block_dims(); // const& — no copy
    for(size_t i = 0; i < static_cast<size_t>(num_mindices_); i++)
      M_ *= lhs_bdims[num_batch_indices_ + i];
    for(size_t i = static_cast<size_t>(num_mindices_); i < lhs_bdims.size(); i++)
      N_ *= lhs_bdims[num_batch_indices_ + i];
    size_t      kstart_idx = (transpose_arg1_ ? 0 : static_cast<size_t>(num_mindices_));
    const auto& arg1_bdims = (rhs1_is_arg1_ ? rhs1.block_dims() : rhs2.block_dims());
    for(int k = 0; k < num_kindices_; k++) K_ *= arg1_bdims[num_batch_indices_ + kstart_idx + k];
    // Fix Bug 2: store the data pointer VALUE (void*), not the address of
    // the T* temporary returned by buf().  Previously '&lhs.buf()' took
    // the address of a prvalue, which is a dangling pointer / UB.
    bufc_        = static_cast<void*>(lhs.buf());
    bufa_        = static_cast<void*>(rhs1_is_arg1_ ? rhs1.buf() : rhs2.buf());
    bufb_        = static_cast<void*>(rhs1_is_arg1_ ? rhs2.buf() : rhs1.buf());
    num_batches_ = 1;
    for(int i = 0; i < num_batch_indices_; i++) num_batches_ *= lhs_bdims[i];
  }

  bool valid_{false};
  int  num_batch_indices_{0}; // in-class init: was uninitialized (UB) before commit 3ac53c2
  int  num_mindices_{};
  int  num_nindices_{};
  int  num_kindices_{};
  bool rhs1_is_arg1_{};
  bool transpose_arg1_{};
  bool transpose_arg2_{};
  int  M_{}, N_{}, K_{};
  int  num_batches_{};
  // Fix Bug 2: plain void* storing the data pointer value directly.
  void* bufa_{};
  void* bufb_{};
  void* bufc_{};
};

// ---------------------------------------------------------------------------

class GeneralMultPlan {
public:
  GeneralMultPlan(): valid_{false} {}
  GeneralMultPlan(const GeneralMultPlan&)            = default;
  GeneralMultPlan& operator=(const GeneralMultPlan&) = default;

  bool is_valid() const { return valid_; }

  GeneralMultPlan(const IndexLabelVec& lhs_labels,
                  const IndexLabelVec& rhs1_labels,
                  const IndexLabelVec& rhs2_labels):
    valid_{true},
    lhs_labels_{lhs_labels},
    rhs1_labels_{rhs1_labels},
    rhs2_labels_{rhs2_labels} {
    for(const auto& lbl: lhs_labels) {
      if(std::find(lhs_inter_labels_.begin(), lhs_inter_labels_.end(), lbl) ==
         lhs_inter_labels_.end())
        lhs_inter_labels_.push_back(lbl);
    }
    for(const auto& lbl: rhs1_labels) {
      if(std::find(rhs1_inter_labels_.begin(), rhs1_inter_labels_.end(), lbl) ==
         rhs1_inter_labels_.end())
        rhs1_inter_labels_.push_back(lbl);
    }
    for(const auto& lbl: rhs2_labels) {
      if(std::find(rhs2_inter_labels_.begin(), rhs2_inter_labels_.end(), lbl) ==
         rhs2_inter_labels_.end())
        rhs2_inter_labels_.push_back(lbl);
    }
    lhs_ba_plan_  = BlockAssignPlan{lhs_labels_, lhs_inter_labels_,
                                    BlockAssignPlan::OpType::set};
    rhs1_ba_plan_ = BlockAssignPlan{rhs1_inter_labels_, rhs1_labels_,
                                    BlockAssignPlan::OpType::set};
    rhs2_ba_plan_ = BlockAssignPlan{rhs2_inter_labels_, rhs2_labels_,
                                    BlockAssignPlan::OpType::set};
    linter_to_l_perm_   = perm_map_compute(lhs_inter_labels_,  lhs_labels_);
    r1_to_r1inter_perm_ = perm_map_compute(rhs1_labels_, rhs1_inter_labels_);
    r2_to_r2inter_perm_ = perm_map_compute(rhs2_labels_, rhs2_inter_labels_);
    ttgt_plan_ = TTGTPlan{lhs_inter_labels_, rhs1_inter_labels_, rhs2_inter_labels_};
  }

  template<typename T0, typename T1, typename T2, typename T3>
  void apply(T0 beta, BlockSpan<T1>& lhs, T1 alpha, BlockSpan<T2>& rhs1, BlockSpan<T3>& rhs2) {
    EXPECTS(valid_);

    const size_t lhs_nelems  = lhs.num_elements();
    const size_t rhs1_nelems = rhs1.num_elements();
    const size_t rhs2_nelems = rhs2.num_elements();

    // Fix Bug 1: intermediate buffers are std::vector<std::byte> sized in
    // bytes, not std::vector<double>.  This handles float, double, and
    // complex types uniformly without type-punning or memory corruption.
    if(linter_buf_.size() < lhs_nelems * sizeof(T1)) linter_buf_.resize(lhs_nelems * sizeof(T1));
    if(r1inter_buf_.size() < rhs1_nelems * sizeof(T2))
      r1inter_buf_.resize(rhs1_nelems * sizeof(T2));
    if(r2inter_buf_.size() < rhs2_nelems * sizeof(T3))
      r2inter_buf_.resize(rhs2_nelems * sizeof(T3));

    // Cache permuted dims — recompute only when block shape changes.
    const auto& lhs_dims  = lhs.block_dims(); // const& — no copy
    const auto& rhs1_dims = rhs1.block_dims();
    const auto& rhs2_dims = rhs2.block_dims();

    if(linter_dims_.empty() ||
       !std::equal(lhs_dims.begin(), lhs_dims.end(), linter_dims_src_.begin())) {
      linter_dims_src_.assign(lhs_dims.begin(), lhs_dims.end());
      linter_dims_ = perm_map_apply(linter_dims_src_, linter_to_l_perm_);
    }
    if(r1inter_dims_.empty() ||
       !std::equal(rhs1_dims.begin(), rhs1_dims.end(), r1inter_dims_src_.begin())) {
      r1inter_dims_src_.assign(rhs1_dims.begin(), rhs1_dims.end());
      r1inter_dims_ = perm_map_apply(r1inter_dims_src_, r1_to_r1inter_perm_);
    }
    if(r2inter_dims_.empty() ||
       !std::equal(rhs2_dims.begin(), rhs2_dims.end(), r2inter_dims_src_.begin())) {
      r2inter_dims_src_.assign(rhs2_dims.begin(), rhs2_dims.end());
      r2inter_dims_ = perm_map_apply(r2inter_dims_src_, r2_to_r2inter_perm_);
    }

    // Reinterpret byte buffers as the correct element type.
    BlockSpan<T1> lhs_inter{reinterpret_cast<T1*>(linter_buf_.data()), linter_dims_};
    BlockSpan<T2> rhs1_inter{reinterpret_cast<T2*>(r1inter_buf_.data()), r1inter_dims_};
    BlockSpan<T3> rhs2_inter{reinterpret_cast<T3*>(r2inter_buf_.data()), r2inter_dims_};

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
  // Fix Bug 1: byte buffers — correct for any element type T1/T2/T3.
  std::vector<std::byte> linter_buf_;
  std::vector<std::byte> r1inter_buf_;
  std::vector<std::byte> r2inter_buf_;
  // Cached permuted dims (recomputed only when block shape changes).
  std::vector<size_t> linter_dims_src_, linter_dims_;
  std::vector<size_t> r1inter_dims_src_, r1inter_dims_;
  std::vector<size_t> r2inter_dims_src_, r2inter_dims_;
};

} // namespace tamm::internal

// ---------------------------------------------------------------------------

namespace tamm {

/**
 * @brief BlockMultPlan — select and cache the best contraction sub-plan.
 *
 * The sub-plan is selected once in the constructor; apply_impl() dispatches
 * via std::visit with zero label-vector copies at call time.
 *
 * The public apply() overload (Scalar lscale/rscale) now correctly forwards
 * to apply_impl() instead of silently doing nothing.
 */
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
    if(plan_ == Plan::invalid) prep_loop_gemm_plan();
    if(plan_ == Plan::invalid) prep_loop_ttgt_plan();
    if(plan_ == Plan::invalid) prep_general_plan();
    if(plan_ == Plan::invalid) { NOT_IMPLEMENTED(); }
    EXPECTS(plan_ != Plan::invalid);
    build_cached_plan();
  }

  // Primary hot-path dispatch: typed scalars, all types known at compile time.
  template<typename T1>
  void apply_impl(T1 lscale, BlockSpan<T1>& lhs, T1 rscale, BlockSpan<T1>& rhs1,
                  BlockSpan<T1>& rhs2) {
    std::visit(
      [&](auto& p) {
        using PT = std::decay_t<decltype(p)>;
        if constexpr(std::is_same_v<PT, internal::FlatBlockMultPlan>) {
          if(optype_ == OpType::set) p.apply_assign(lhs, rscale, rhs1, rhs2);
          else p.apply_update(lscale, lhs, rscale, rhs1, rhs2);
        }
        else if constexpr(std::is_same_v<PT, internal::GemmPlan>) {
          p.apply(lscale, lhs, rscale, rhs1, rhs2);
        }
        else if constexpr(std::is_same_v<PT, internal::TTGTPlan>) {
          p.apply(lscale, lhs, rscale, rhs1, rhs2);
        }
        else if constexpr(std::is_same_v<PT, internal::GeneralMultPlan>) {
          p.apply(lscale, lhs, rscale, rhs1, rhs2);
        }
        else { NOT_ALLOWED(); }
      },
      cached_plan_);
  }

  // Public Scalar overload forwards to the typed dispatch.
  //
  // This plan is the single-element-type contraction path: lhs, rhs1 and rhs2
  // must all share the same element type.  Mixed-type contractions (e.g.
  // real x complex) are handled by kernels::block_multiply in multop.hpp, not
  // by BlockMultPlan.  We constrain the overload to T1==T2==T3 so the previous
  // reinterpret_cast<BlockSpan<T1>&>(rhs) UB (valid only when the types were
  // already identical) can no longer be reached with incompatible types.
  template<typename T1, typename T2, typename T3>
    requires(std::is_same_v<T1, T2> && std::is_same_v<T1, T3>)
  void apply(Scalar lscale, BlockSpan<T1>& lhs, Scalar rscale, BlockSpan<T2>& rhs1,
             BlockSpan<T3>& rhs2) {
    apply_impl(lscale.template get<T1>(), lhs, rscale.template get<T1>(), rhs1, rhs2);
  }

private:
  using CachedPlan = std::variant<internal::FlatBlockMultPlan, internal::GemmPlan,
                                  internal::TTGTPlan, internal::GeneralMultPlan>;

  void build_cached_plan() {
    switch(plan_) {
      case Plan::flat_assign:
      case Plan::flat_update:
        cached_plan_ = internal::FlatBlockMultPlan{lhs_labels_, rhs1_labels_, rhs2_labels_};
        break;
      case Plan::loop_gemm:
        cached_plan_ = internal::GemmPlan{lhs_labels_, rhs1_labels_, rhs2_labels_};
        break;
      case Plan::loop_ttgt:
        cached_plan_ = internal::TTGTPlan{lhs_labels_, rhs1_labels_, rhs2_labels_};
        break;
      case Plan::general:
        cached_plan_ = internal::GeneralMultPlan{lhs_labels_, rhs1_labels_, rhs2_labels_};
        break;
      default: UNREACHABLE();
    }
  }

  bool has_reduction_index() {
    std::set<TiledIndexLabel> lhs_set(lhs_labels_.begin(), lhs_labels_.end());
    std::set<TiledIndexLabel> rhs_set(rhs1_labels_.begin(), rhs1_labels_.end());
    for(auto lbl: rhs2_labels_) rhs_set.insert(lbl);
    IndexLabelVec reduction_lbls;
    std::set_difference(rhs_set.begin(), rhs_set.end(), lhs_set.begin(), lhs_set.end(),
                        std::back_inserter(reduction_lbls));
    return !reduction_lbls.empty();
  }

  bool has_hadamard_index() {
    std::set<TiledIndexLabel> rhs1_set(rhs1_labels_.begin(), rhs1_labels_.end());
    std::set<TiledIndexLabel> rhs2_set(rhs2_labels_.begin(), rhs2_labels_.end());
    for(const auto& lbl: lhs_labels_)
      if(rhs1_set.count(lbl) && rhs2_set.count(lbl)) return true;
    return false;
  }

  bool has_repeated_index() {
    return internal::has_repeated_elements(rhs1_labels_) ||
           internal::has_repeated_elements(rhs2_labels_) ||
           internal::has_repeated_elements(lhs_labels_);
  }

  IndexLabelVec get_hadamard_labels() {
    if(!has_hadamard_index_) return {};
    IndexLabelVec             result;
    std::set<TiledIndexLabel> rhs1_set(rhs1_labels_.begin(), rhs1_labels_.end());
    std::set<TiledIndexLabel> rhs2_set(rhs2_labels_.begin(), rhs2_labels_.end());
    for(const auto& lbl: lhs_labels_)
      if(rhs1_set.count(lbl) && rhs2_set.count(lbl)) result.push_back(lbl);
    return result;
  }

  void prep_flat_plan() {
    if(rhs1_labels_.size() == 0 && rhs2_labels_.size() == 0 && lhs_labels_.size() == 0) {
      plan_ = optype_ == OpType::set ? Plan::flat_assign : Plan::flat_update; return;
    }
    else if(rhs1_labels_.size() == 1 && rhs2_labels_.size() == 1 && lhs_labels_.size() == 1) {
      if(lhs_labels_ == rhs1_labels_ && lhs_labels_ == rhs2_labels_) {
        plan_ = optype_ == OpType::set ? Plan::flat_assign : Plan::flat_update;
        return;
      }
    }
    else if(rhs1_labels_.size() == 1 || rhs1_labels_.size() == 0) {
      if(rhs2_labels_.size() == 1 || rhs2_labels_.size() == 0) {
        size_t rhs_size = std::max(rhs1_labels_.size(), rhs2_labels_.size());
        if(lhs_labels_.size() == rhs_size) {
          if((rhs1_labels_.size() == 1 && lhs_labels_ == rhs1_labels_) ||
             (rhs2_labels_.size() == 1 && lhs_labels_ == rhs2_labels_))
            plan_ = optype_ == OpType::set ? Plan::flat_assign : Plan::flat_update;
        }
      }
    }
  }

  void prep_loop_gemm_plan() {
    if(!has_repeated_index_ && !has_reduction_index_) {
      if(has_hadamard_index_) {
        auto            hadamard_labels = get_hadamard_labels();
        const ptrdiff_t hlabels         = static_cast<ptrdiff_t>(hadamard_labels.size());
        for(size_t i = 0; i < hadamard_labels.size(); i++) {
          auto lbl = lhs_labels_[i];
          auto rhs1_pos =
            std::find(rhs1_labels_.begin(), rhs1_labels_.end(), lbl) - rhs1_labels_.begin();
          auto rhs2_pos =
            std::find(rhs2_labels_.begin(), rhs2_labels_.end(), lbl) - rhs2_labels_.begin();
          if(rhs1_pos >= hlabels || rhs2_pos >= hlabels) return;
        }
        plan_ = Plan::loop_gemm;
      } else {
        plan_ = Plan::loop_gemm;
      }
    }
  }

  void prep_loop_ttgt_plan() {
    if(!has_repeated_index_ && !has_reduction_index_) plan_ = Plan::loop_ttgt;
  }

  void prep_general_plan() {
    if(!has_repeated_index_) plan_ = Plan::general;
  }

  enum class Plan { flat_assign, flat_update, loop_gemm, loop_ttgt, general, invalid };

  IndexLabelVec lhs_labels_;
  IndexLabelVec rhs1_labels_;
  IndexLabelVec rhs2_labels_;
  OpType        optype_;
  Plan          plan_;
  CachedPlan    cached_plan_; ///< built once in ctor; zero-copy std::visit dispatch

  bool has_repeated_index_;
  bool has_reduction_index_;
  bool has_hadamard_index_;
};

} // namespace tamm
