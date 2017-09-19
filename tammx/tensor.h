#ifndef TAMMX_TENSOR_H_
#define TAMMX_TENSOR_H_

#include "tammx/errors.h"
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

  Tensor(const TensorVec<TensorSymmGroup> &indices,
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
/** \warning
*  totalview LD on following statement
*  back traced to ccsd_driver line 607 main
*/
      : Tensor{std::get<0>(iinfo), static_cast<TensorRank>(std::get<1>(iinfo)), irrep, spin_restricted} {}

  Tensor(ProcGroup pg,
         const TensorVec<TensorSymmGroup> &indices,
         TensorRank nupper_indices,
         Irrep irrep,
         bool spin_restricted,
         const Distribution* distribution,
         const MemoryManager*  mgr)
      : pg_{pg},
        allocation_status_{AllocationStatus::invalid},
        TensorBase{indices, nupper_indices, irrep, spin_restricted} {
          EXPECTS(pg.is_valid());
          if(mgr) {
            mgr_ = mgr->clone(pg);
          }
          if(distribution) {
            distribution_ = DistributionFactory::make_distribution(*distribution, this, pg.size());
          }
        }
  ProcGroup pg() const {
    return pg_;
  }

  //@todo implement the factory
  void alloc(ProcGroup pg, Distribution* distribution=nullptr, MemoryManager* memory_manager=nullptr) {
    pg_ = pg;
    EXPECTS(pg.is_valid());
    if(distribution) {
      // distribution_ = DistributionFactory::make_distribution(*distribution, this, pg.size());
      distribution_ = std::shared_ptr<Distribution>(distribution->clone(this, pg_.size()));
    }
    if(memory_manager) {
      mgr_ = std::shared_ptr<MemoryManager>(memory_manager->clone(pg_));
    }
    EXPECTS(mgr_.get() != nullptr);
    EXPECTS(distribution_.get() != nullptr);
    auto rank = pg_.rank();
    auto buf_size = distribution_->buf_size(rank);
    auto eltype = tensor_element_type<element_type>();
    EXPECTS(buf_size >=0 );
    mgr_->alloc(eltype, buf_size);
/** \warning
*  totalview LD on following statement
*  forward traced to tammx::MemoryManagerGA::alloc 
*  line 91 memory_manager_ga.h
*  bach traced to line 37 execution_context.h
*  back traced to main line 607 ccsd_driver
*/
    allocation_status_ = AllocationStatus::created;
  }

  void dealloc() {
    EXPECTS(allocation_status_ == AllocationStatus::created);
    mgr_->dealloc();
    allocation_status_ = AllocationStatus::invalid;
  }

  void attach(Distribution* distribution, std::shared_ptr<MemoryManager> mgr) {
    pg_ = mgr->proc_group();
    EXPECTS(pg_.is_valid());
    distribution_ = std::shared_ptr<Distribution>(distribution->clone(this, pg_.size()));
    mgr_ = mgr;
    allocation_status_ = AllocationStatus::attached;
  }

  void detach() {
    mgr_ = nullptr;
    allocation_status_ = AllocationStatus::invalid;
  }

  Block<T> alloc(const BlockDimVec& blockid) {
    return {*this, blockid};
  }

  Block<T> alloc(const BlockDimVec& blockid,
                 const PermVec& layout,
                 Sign sign) {
    return {*this, blockid, layout, sign};
  }

  Block<T> get(const BlockDimVec& blockid) {
    EXPECTS(constructed());
    Offset offset;
    Proc proc;
    auto sblockid = find_spin_unique_block(blockid);
    auto uniq_blockid = find_unique_block(sblockid);
    PermVec layout;
    Sign sign;
    std::tie(layout, sign) = compute_sign_from_unique_block(sblockid);
    auto size = block_size(blockid);
    auto block {alloc(blockid, layout, sign)};
    std::tie(proc, offset) = distribution_->locate(uniq_blockid);
    mgr_->get(proc, offset, Size{size}, block.buf());
    return std::move(block);
  }

  void put(const BlockDimVec& blockid, const Block<T>& block) {
    EXPECTS(constructed());
    EXPECTS(find_spin_unique_block(blockid) == blockid);
    EXPECTS(find_unique_block(blockid) == blockid);
    Offset offset;
    Proc proc;
    auto size = block_size(blockid);
    std::tie(proc, offset) = distribution_->locate(blockid);
    mgr_->put(proc, offset, Size{size}, block.buf());
  }

  void add(const BlockDimVec& blockid, const Block<T>& block) {
    EXPECTS(constructed());
    EXPECTS(find_spin_unique_block(blockid) == blockid);
    EXPECTS(find_unique_block(blockid) == blockid);
    Offset offset;
    Proc proc;
    auto size = block_size(blockid);
    std::tie(proc, offset) = distribution_->locate(blockid);
    mgr_->add(proc, offset, Size{size}, block.buf());
  }

  LabeledTensor<T> operator () (const IndexLabelVec& label) {
    return {this, label};
  }

  LabeledTensor<T> operator () () {
    IndexLabelVec label;
    for(int i=0; i<rank(); i++) {
      label.push_back({i, flindices()[i]});
    }
    return (*this)(label);
  }

  template<typename ...Args>
  LabeledTensor<T> operator () (IndexLabel ilbl, Args... rest) {
    IndexLabelVec label{ilbl};
    pack(label, rest...);
    return (*this)(label);
  }

  const Distribution* distribution() const {
    return distribution_.get();
  }

  const MemoryManager* memory_manager() const {
    return mgr_.get();
  }

  MemoryManager* memory_manager() {
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

 protected:
  void pack(IndexLabelVec& label) {}

  template<typename ...Args>
  void pack(IndexLabelVec& label, IndexLabel ilbl, Args... rest) {
    label.push_back(ilbl);
    pack(label, rest...);
  }

  ProcGroup pg_;
  AllocationStatus allocation_status_;
  std::shared_ptr<MemoryManager> mgr_;
  std::shared_ptr<Distribution> distribution_;
}; // class Tensor


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
