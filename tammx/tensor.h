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

/**
 * @ingroup tensors
 */
class TensorImpl : public TensorBase {
 public:
  using TensorBase::TensorBase;
  virtual ~TensorImpl() {}
  virtual MemoryRegion& memory_region() = 0;
  virtual const MemoryRegion& memory_region() const = 0;
};


/**
 * @ingroup tensors
 * @tparam T Type of elements in the tensor
 */
template<typename T>
class Tensor : public TensorImpl {
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
      : TensorImpl{indices, nupper_indices, irrep, spin_restricted},
        mpb_{nullptr},
        distribution_{nullptr} {}

  Tensor(const IndexInfo& iinfo,
         Irrep irrep,
         bool spin_restricted)
/** \warning
*  totalview LD on following statement
*  back traced to ccsd_driver line 607 main
*/
      : Tensor{std::get<0>(iinfo), static_cast<TensorRank>(std::get<1>(iinfo)), irrep, spin_restricted} {}

  ProcGroup pg() const {
    EXPECTS(mpb_ != nullptr);
    return mpb_->mgr().pg();
  }

  //@todo implement the factory
  void alloc(Distribution* distribution, MemoryManager* memory_manager) {
    EXPECTS(distribution != nullptr);
    EXPECTS(memory_manager != nullptr);
    // distribution_ = DistributionFactory::make_distribution(*distribution, this, pg.size());
    distribution_ = std::shared_ptr<Distribution>(
        distribution->clone(this,
                            memory_manager->pg().size()));
    auto rank = memory_manager->pg().rank();
    auto buf_size = distribution_->buf_size(rank);
    auto eltype = tensor_element_type<element_type>();
    EXPECTS(buf_size >=0 );
    mpb_ = std::unique_ptr<MemoryRegion>{memory_manager->alloc_coll(eltype, buf_size)};
  }

  void dealloc() {
    EXPECTS(mpb_);
    mpb_->dealloc_coll();
  }

  void attach(Distribution* distribution, MemoryRegion* mpb) {
    EXPECTS(distribution != nullptr);
    EXPECTS(mpb != nullptr);
    distribution_ = std::shared_ptr<Distribution>(distribution->clone(this, pg_.size()));
    mpb_ = mpb->mgr().attach_coll(*mpb);
  }

  void detach() {
    EXPECTS(mpb_);
    mpb_->detach_coll();
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
    mpb_->get(proc, offset, Size{size}, block.buf());
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
    mpb_->put(proc, offset, Size{size}, block.buf());
  }

  void add(const BlockDimVec& blockid, const Block<T>& block) {
    EXPECTS(constructed());
    EXPECTS(find_spin_unique_block(blockid) == blockid);
    EXPECTS(find_unique_block(blockid) == blockid);
    Offset offset;
    Proc proc;
    auto size = block_size(blockid);
    std::tie(proc, offset) = distribution_->locate(blockid);
    mpb_->add(proc, offset, Size{size}, block.buf());
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

  const MemoryManager& memory_manager() const {
    return mpb_->mgr();
  }

  MemoryManager& memory_manager() {
    return mpb_->mgr();
  }

  const MemoryRegion& memory_region() const override {
    return *mpb_.get();
  }
  
  MemoryRegion& memory_region() override {
    return *mpb_.get();
  }
  
  static void allocate(Distribution* distribution, MemoryManager* memory_manager) {
    //no-op
  }

  template<typename ...Args>
  static void allocate(Distribution* distribution, MemoryManager* memory_manager, Tensor<T>& tensor, Args& ... tensor_list) {
    tensor.alloc(distribution, memory_manager);
    allocate(distribution, memory_manager, tensor_list...);
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
    return mpb_ != nullptr
        && mpb_->allocation_status() != AllocationStatus::invalid;
  }

 protected:
  void pack(IndexLabelVec& label) {}

  template<typename ...Args>
  void pack(IndexLabelVec& label, IndexLabel ilbl, Args... rest) {
    label.push_back(ilbl);
    pack(label, rest...);
  }

  ProcGroup pg_;
  std::unique_ptr<MemoryRegion> mpb_;
  std::shared_ptr<Distribution> distribution_;
}; // class Tensor

/**
 * @ingroup tensors
 * @tparam T Type of scalar element
 */
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
