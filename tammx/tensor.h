#ifndef TAMMX_TENSOR_H_
#define TAMMX_TENSOR_H_

#include "tammx/types.h"
#include "tammx/tce.h"
#include "tammx/triangle_loop.h"
#include "tammx/product_iterator.h"
#include "tammx/util.h"
#include "tammx/block.h"
#include "tammx/tensor_base.h"
#include "tammx/distribution.h"
#include "tammx/memory_manager.h"

namespace tammx {

template<typename T>
class LabeledTensor;

template<typename T>
class Tensor : public TensorBase {
 public:
  using element_type = T;

  Tensor() = delete;
  Tensor(Tensor<T>&&) = delete;
  Tensor(const Tensor<T>&) = delete;
  Tensor<T>& operator = (const Tensor<T>&) = delete;
  Tensor<T>& operator = (Tensor<T>&& tensor) = delete;

  // Tensor(Tensor<T>&& tensor)
  //     : TensorBase{tensor},
  //       pg_{tensor.pg_},
  //       allocation_status_{AllocationStatus::invalid},
  //       mgr_{std::move(tensor.mgr_)},
  //       distribution_{std::move(tensor.distribution_)} {}

  Tensor(const TensorVec<SymmGroup> &indices,
         TensorRank nupper_indices,
         Irrep irrep,
         bool spin_restricted)
      : allocation_status_{AllocationStatus::invalid},
        TensorBase{indices, nupper_indices, irrep, spin_restricted},
        mgr_{nullptr},
        distribution_{nullptr} {}

  Tensor(const IndexInfo& iinfo,
         Irrep irrep,
         bool spin_restricted)
      : Tensor{std::get<0>(iinfo), std::get<1>(iinfo), irrep, spin_restricted} {}

  Tensor(ProcGroup pg,
         const TensorVec<SymmGroup> &indices,
         TensorRank nupper_indices,
         Irrep irrep,
         bool spin_restricted,
         const Distribution* distribution,
         const MemoryManager*  mgr)
      : pg_{pg},
        allocation_status_{AllocationStatus::invalid},
        TensorBase{indices, nupper_indices, irrep, spin_restricted} {
          if(mgr) {
            mgr_ = mgr->clone(pg);
          }
          if(distribution) {
            distribution_ = DistributionFactory::make_distribution(*distribution, this, pg.size());
          }
        }

  // Tensor<T>& operator = (Tensor<T>&& tensor) {
  //   TensorBase::operator = (tensor);
  //   pg_ = tensor.pg_;
  //   allocation_status_ = tensor.allocation_status_;
  //   mgr_ = std::move(tensor.mgr_);
  //   distribution_ = std::move(tensor.distribution_);
  // }

  ProcGroup pg() const {
    return pg_;
  }

  //@todo implement the factory
  void alloc(ProcGroup pg, Distribution* distribution=nullptr, MemoryManager* memory_manager=nullptr) {
    pg_ = pg;
    if(distribution) {
      // distribution_ = DistributionFactory::make_distribution(*distribution, this, pg.size());
      distribution_ = std::shared_ptr<Distribution>(distribution->clone(this, pg_.size()));
    }
    if(memory_manager) {
      mgr_ = std::unique_ptr<MemoryManager>(memory_manager->clone(pg_));
    }
    Expects(mgr_.get() != nullptr);
    Expects(distribution_.get() != nullptr);
    auto rank = pg_.rank();
    auto buf_size = distribution_->buf_size(rank);
    auto eltype = tensor_element_type<element_type>;
    Expects(buf_size >=0 );
    mgr_->alloc(pg_, eltype, buf_size);
    allocation_status_ = AllocationStatus::created;
  }

  void dealloc() {
    mgr_->dealloc();
    allocation_status_ = AllocationStatus::invalid;
  }

  void attach(std::unique_ptr<MemoryManager> mgr) {
    mgr_ = mgr;
  }

  Block<T> alloc(const TensorIndex& blockid) {
    return {*this, blockid};
  }

  Block<T> alloc(const TensorIndex& blockid,
                 const TensorPerm& layout,
                 Sign sign) {
    return {*this, blockid, layout, sign};
  }

  Block<T> get(const TensorIndex& blockid) {
    Expects(constructed());
    Offset offset;
    Proc proc;
    auto sblockid = find_spin_unique_block(blockid);
    auto uniq_blockid = find_unique_block(sblockid);
    TensorPerm layout;
    Sign sign;
    std::tie(layout, sign) = compute_sign_from_unique_block(sblockid);
    auto size = block_size(blockid);
    auto block {alloc(blockid, layout, sign)};
    std::tie(proc, offset) = distribution_->locate(uniq_blockid);
    mgr_->get(proc, offset, Size{size}, block.buf());
    return std::move(block);
  }

  void put(const TensorIndex& blockid, const Block<T>& block) {
    Expects(constructed());
    Expects(find_spin_unique_block(blockid) == blockid);
    Expects(find_unique_block(blockid) == blockid);
    Offset offset;
    Proc proc;
    auto size = block_size(blockid);
    std::tie(proc, offset) = distribution_->locate(blockid);
    mgr_->put(proc, offset, Size{size}, block.buf());
  }

  void add(const TensorIndex& blockid, const Block<T>& block) {
    Expects(constructed());
    Expects(find_spin_unique_block(blockid) == blockid);
    Expects(find_unique_block(blockid) == blockid);
    Offset offset;
    Proc proc;
    auto size = block_size(blockid);
    std::tie(proc, offset) = distribution_->locate(blockid);
    mgr_->add(proc, offset, Size{size}, block.buf());
  }

  LabeledTensor<T> operator () (const TensorLabel& label) {
    return {this, label};
  }

  LabeledTensor<T> operator () () {
    TensorLabel label;
    for(int i=0; i<rank(); i++) {
      label.push_back({i, flindices()[i]});
    }
    return (*this)(label);
  }

  template<typename ...Args>
  LabeledTensor<T> operator () (IndexLabel ilbl, Args... rest) {
    TensorLabel label{ilbl};
    pack(label, rest...);
    return (*this)(label);
  }

  const Distribution* distribution() const {
    return distribution_.get();
  }

  const MemoryManager* memory_manager() const {
    return mgr_.get();
  }

  static void allocate(ProcGroup pg, Distribution* distribution, MemoryManager* memory_manager) {
    //no-op
  }

  template<typename ...Args>
  static void allocate(ProcGroup pg, Distribution* distribution, MemoryManager* memory_manager, Tensor<T>& tensor, Args& ... tensor_list) {
    tensor.alloc(pg, distribution, memory_manager);
    allocate(pg, distribution, memory_manager, tensor_list...);
  }


  static void deallocate() {
    //no-op
  }

  template<typename ...Args>
  static void deallocate(Tensor<T>& tensor, Args& ... tensor_list) {
    tensor.dealloc();
    deallocate(tensor_list...);
  }

  bool constructed() const {
    return allocation_status_ != AllocationStatus::invalid;
  }
  
  //TensorBuilder<Tensor<T>> builder() const;

 protected:
  void pack(TensorLabel& label) {}

  template<typename ...Args>
  void pack(TensorLabel& label, IndexLabel ilbl, Args... rest) {
    label.push_back(ilbl);
    pack(label, rest...);
  }

  ProcGroup pg_;
  enum class AllocationStatus { invalid, created, attached };
  AllocationStatus allocation_status_;
  std::unique_ptr<MemoryManager> mgr_;
  std::shared_ptr<Distribution> distribution_;
}; // class Tensor


template<typename TensorType>
class TensorBuilder {
 public:
  TensorBuilder<TensorType> indices(const TensorVec<SymmGroup>& ind) {
    indices_ = ind;
    return *this;
  }

  TensorBuilder<TensorType> indices(const IndexInfo& iinfo) {
    indices_ = std::get<0>(iinfo);
    nupper_indices_ = std::get<1>(iinfo);
    return *this;
  }

  TensorBuilder<TensorType> irrep(Irrep irr) {
    irrep_ = irr;
    return *this;
  }

  TensorBuilder<TensorType> nupper_indices(int nup) {
    nupper_indices_ = nup;
    return *this;
  }

  TensorBuilder<TensorType> pg(ProcGroup pg1) {
    pg_ = pg1;
    return *this;
  }

  TensorBuilder<TensorType> spin_restricted(bool spinr) {
    spin_restricted_ = spinr;
    return *this;
  }

  template<typename T = typename TensorType::element_type>
  Tensor<T> build() {
    return Tensor<T>{pg_, indices_, nupper_indices_, irrep_, spin_restricted_,
          tensor_.distribution(), tensor_.memory_manager()};
  }

  TensorBuilder(const TensorType& tensor)
      : tensor_{tensor},
        indices_{tensor.indices()},
        nupper_indices_{tensor.nupper_indices()},
        pg_{tensor.pg()},
        irrep_{tensor.irrep()},
        spin_restricted_{tensor.spin_restricted()} {}

  const TensorType& tensor_;
  TensorVec<SymmGroup> indices_;
  int nupper_indices_;
  ProcGroup pg_;
  Irrep irrep_;
  bool spin_restricted_;

  template<typename T>
  friend class Tensor;
};

template<typename T>
class Scalar : public Tensor<T> {
 public:
  Scalar()
      : Tensor<T>({}, 0, Irrep{0}, false) {}

  T value() {
    return *reinterpret_cast<T*>(Tensor<T>::mgr_->access(Offset{0}));
  }
};

}  // namespace tammx

#endif  // TAMMX_TENSOR_H_
