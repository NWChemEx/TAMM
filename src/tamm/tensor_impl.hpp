
#pragma once

#include "ga/ga.h"
#include "tamm/blockops_cpu.hpp"
#include "tamm/distribution.hpp"
#include "tamm/execution_context.hpp"
#include "tamm/index_loop_nest.hpp"
#include "tamm/index_space.hpp"
#include "tamm/memory_manager_local.hpp"
#include "tamm/tensor_base.hpp"
#include <functional>
#include <gsl/span>
#include <type_traits>
#if defined(USE_UPCXX)
#include <upcxx/upcxx.hpp>
#endif

namespace tamm {

using gsl::span;

#if defined(USE_UPCXX)
class TensorTile {
public:
  int64_t lo[2];
  int64_t dim[2];
  int64_t rank;
  int64_t offset;

  TensorTile(int64_t lo_0, int64_t lo_1, int64_t dim_0, int64_t dim_1, int64_t _rank,
             int64_t _offset) {
    lo[0]  = lo_0;
    lo[1]  = lo_1;
    dim[0] = dim_0;
    dim[1] = dim_1;
    rank   = _rank;
    offset = _offset;
  }

  bool contains(int64_t i0, int64_t i1) {
    return i0 >= lo[0] && i1 >= lo[1] && i0 < lo[0] + dim[0] && i1 < lo[1] + dim[1];
  }
};
#endif

/**
 * @ingroup tensors
 * @brief Implementation class for TensorBase class
 *
 * @tparam T Element type of Tensor
 */
template<typename T>
class TensorImpl: public TensorBase {
public:
  using TensorBase::TensorBase;
  // Ctors
  TensorImpl() = default;

  /**
   * @brief Construct a new TensorImpl object using a vector of
   * TiledIndexSpace objects for each mode of the tensor
   *
   * @param [in] tis vector of TiledIndexSpace objects for each
   * mode used to construct the tensor
   */
  TensorImpl(const TiledIndexSpaceVec& tis): TensorBase{tis} {
    has_spin_symmetry_    = false;
    has_spatial_symmetry_ = false;
    kind_                 = TensorBase::TensorKind::normal;
  }

  /**
   * @brief Construct a new TensorImpl object using a vector of
   * TiledIndexSpace objects for each mode of the tensor
   *
   * @param [in] lbls vector of tiled index labels used for extracting
   * corresponding TiledIndexSpace objects for each mode used to construct
   * the tensor
   */
  TensorImpl(const std::vector<TiledIndexLabel>& lbls): TensorBase{lbls} {
    has_spin_symmetry_    = false;
    has_spatial_symmetry_ = false;
    kind_                 = TensorBase::TensorKind::normal;
  }

  /**
   * @brief Construct a new TensorBase object recursively with a set of
   * TiledIndexSpace objects followed by a lambda expression
   *
   * @tparam Ts variadic template for rest of the arguments
   * @param [in] tis TiledIndexSpace object used as a mode
   * @param [in] rest remaining part of the arguments
   */
  template<class... Ts>
  TensorImpl(const TiledIndexSpace& tis, Ts... rest): TensorBase{tis, rest...} {
    num_modes_ = block_indices_.size();
    construct_dep_map();
    has_spin_symmetry_    = false;
    has_spatial_symmetry_ = false;
    kind_                 = TensorBase::TensorKind::normal;
  }

  // SpinTensor related constructors
  /**
   * @brief Construct a new SpinTensorImpl object using set of TiledIndexSpace
   * objects and Spin attribute mask
   *
   * @param [in] t_spaces
   * @param [in] spin_mask
   */
  TensorImpl(TiledIndexSpaceVec t_spaces, SpinMask spin_mask): TensorBase(t_spaces) {
    EXPECTS(t_spaces.size() == spin_mask.size());

    // for(const auto& tis : t_spaces) { EXPECTS(tis.has_spin()); }

    spin_mask_            = spin_mask;
    has_spin_symmetry_    = true;
    has_spatial_symmetry_ = false;
    // spin_total_        = calculate_spin();
    kind_ = TensorBase::TensorKind::spin;
  }

  /**
   * @brief Construct a new SpinTensorImpl object using set of TiledIndexLabel
   * objects and Spin attribute mask
   *
   * @param [in] t_labels
   * @param [in] spin_mask
   */
  TensorImpl(IndexLabelVec t_labels, SpinMask spin_mask): TensorBase(t_labels) {
    EXPECTS(t_labels.size() == spin_mask.size());
    // for(const auto& tlbl : t_labels) {
    //     EXPECTS(tlbl.tiled_index_space().has_spin());
    // }
    spin_mask_            = spin_mask;
    has_spin_symmetry_    = true;
    has_spatial_symmetry_ = false;
    // spin_total_        = calculate_spin();
    kind_ = TensorBase::TensorKind::spin;
  }

  /**
   * @brief Construct a new SpinTensorImpl object using set of TiledIndexSpace
   * objects and Spin attribute mask
   *
   * @param [in] t_spaces
   * @param [in] spin_mask
   */
  TensorImpl(TiledIndexSpaceVec t_spaces, std::vector<size_t> spin_sizes): TensorBase(t_spaces) {
    // EXPECTS(t_spaces.size() == spin_mask.size());
    int spin_size_sum = std::accumulate(spin_sizes.begin(), spin_sizes.end(), 0);
    EXPECTS(t_spaces.size() >= spin_size_sum);

    EXPECTS(spin_sizes.size() > 0);
    // for(const auto& tis : t_spaces) { EXPECTS(tis.has_spin()); }
    SpinMask spin_mask;
    size_t   upper  = spin_sizes[0];
    size_t   lower  = spin_sizes.size() > 1 ? spin_sizes[1] : t_spaces.size() - upper;
    size_t   ignore = spin_sizes.size() > 2 ? spin_sizes[1] : t_spaces.size() - (upper + lower);

    for(size_t i = 0; i < upper; i++) { spin_mask.push_back(SpinPosition::upper); }

    for(size_t i = 0; i < lower; i++) { spin_mask.push_back(SpinPosition::lower); }

    for(size_t i = 0; i < ignore; i++) { spin_mask.push_back(SpinPosition::ignore); }

    spin_mask_            = spin_mask;
    has_spin_symmetry_    = true;
    has_spatial_symmetry_ = false;
    // spin_total_        = calculate_spin();
    kind_ = TensorBase::TensorKind::spin;
  }

  /**
   * @brief Construct a new SpinTensorImpl object using set of TiledIndexLabel
   * objects and Spin attribute mask
   *
   * @param [in] t_labels
   * @param [in] spin_mask
   */
  TensorImpl(IndexLabelVec t_labels, std::vector<size_t> spin_sizes): TensorBase(t_labels) {
    // EXPECTS(t_labels.size() == spin_mask.size());
    int spin_size_sum = std::accumulate(spin_sizes.begin(), spin_sizes.end(), 0);
    EXPECTS(t_labels.size() >= spin_size_sum);
    EXPECTS(spin_sizes.size() > 0);
    // for(const auto& tlbl : t_labels) {
    //     EXPECTS(tlbl.tiled_index_space().has_spin());
    // }

    SpinMask spin_mask;
    size_t   upper  = spin_sizes[0];
    size_t   lower  = spin_sizes.size() > 1 ? spin_sizes[1] : t_labels.size() - upper;
    size_t   ignore = spin_sizes.size() > 2 ? spin_sizes[1] : t_labels.size() - (upper + lower);

    for(size_t i = 0; i < upper; i++) { spin_mask.push_back(SpinPosition::upper); }

    for(size_t i = 0; i < lower; i++) { spin_mask.push_back(SpinPosition::lower); }

    for(size_t i = 0; i < ignore; i++) { spin_mask.push_back(SpinPosition::ignore); }

    spin_mask_            = spin_mask;
    has_spin_symmetry_    = true;
    has_spatial_symmetry_ = false;
    // spin_total_        = calculate_spin();
    kind_ = TensorBase::TensorKind::spin;
  }

  // Copy/Move Ctors and Assignment Operators
  TensorImpl(TensorImpl&&)                 = default;
  TensorImpl(const TensorImpl&)            = delete;
  TensorImpl& operator=(TensorImpl&&)      = default;
  TensorImpl& operator=(const TensorImpl&) = delete;

  // Dtor
  ~TensorImpl() {
    if(mpb_ != nullptr) { mpb_->allocation_status_ = AllocationStatus::orphaned; }
  }

  /**
   * @brief Virtual method for deallocating a Tensor
   *
   */
  virtual void deallocate() {
    EXPECTS(allocation_status_ == AllocationStatus::created);
    EXPECTS(mpb_);
    ec_->unregister_for_dealloc(mpb_);
    mpb_->dealloc_coll();
    delete mpb_;
    mpb_ = nullptr;
    update_status(AllocationStatus::deallocated);
  }

  /**
   * @brief Virtual method for allocating a Tensor using an ExecutionContext
   *
   * @param [in] ec ExecutionContext to be used for allocation
   */
  virtual void allocate(ExecutionContext* ec) {
    {
      EXPECTS(allocation_status_ == AllocationStatus::invalid);
      auto          defd = ec->get_default_distribution();
      Distribution* distribution =
        ec->distribution(defd->get_tensor_base(), defd->get_dist_proc()); // defd->kind());
      // Distribution* distribution    =
      // ec->distribution(defd.tensor_base(), nproc );
#if defined(USE_UPCXX_DISTARRAY) && defined(USE_UPCXX)
      MemoryManager* memory_manager = ec->memory_manager(ec->hint());
#else
      MemoryManager* memory_manager = ec->memory_manager();
#endif
      EXPECTS(distribution != nullptr);
      EXPECTS(memory_manager != nullptr);
      ec_ = ec;
      // distribution_ =
      // DistributionFactory::make_distribution(*distribution, this,
      // pg.size());
      {
        distribution_ =
          std::shared_ptr<Distribution>(distribution->clone(this, memory_manager->pg().size()));
      }
#if 0
        auto rank = memory_manager->pg().rank();
        auto buf_size = distribution_->buf_size(rank);
        auto eltype = tensor_element_type<T>();
        EXPECTS(buf_size >= 0);
        mpb_ = memory_manager->alloc_coll(eltype, buf_size);
#else
      auto           eltype         = tensor_element_type<T>();
      if(proc_list_.size() > 0)
        mpb_ = memory_manager->alloc_coll_balanced(eltype, distribution_->max_proc_buf_size(),
                                                   proc_list_);
      else mpb_ = memory_manager->alloc_coll_balanced(eltype, distribution_->max_proc_buf_size());

#endif
      EXPECTS(mpb_ != nullptr);
      ec_->register_for_dealloc(mpb_);
      update_status(AllocationStatus::created);
    }
  }

  virtual const Distribution& distribution() const { return *distribution_.get(); }

  // Tensor Accessors
  /**
   * @brief Tensor accessor method for getting values from a set of
   * indices to specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to fetch the values
   * @param [in] buff_span memory span where to put the fetched values
   */

  virtual void get(const IndexVector& idx_vec, span<T> buff_span) const {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);

    if(!is_non_zero(idx_vec)) {
      Size size = block_size(idx_vec);
      EXPECTS(size <= buff_span.size());
      for(size_t i = 0; i < size; i++) { buff_span[i] = (T) 0; }
      return;
    }

    Proc   proc;
    Offset offset;
    std::tie(proc, offset) = distribution_->locate(idx_vec);
    Size size              = block_size(idx_vec);
    EXPECTS(size <= buff_span.size());
    mpb_->mgr().get(*mpb_, proc, offset, Size{size}, buff_span.data());
  }

  /**
   * @brief Tensor accessor method for getting values in nonblocking fashion
   * from a set of indices to specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to fetch the values
   * @param [in] buff_span memory span where to put the fetched values
   */

  virtual void nb_get(const IndexVector& idx_vec, span<T> buff_span,
                      DataCommunicationHandlePtr data_comm_handle) const {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);

    if(!is_non_zero(idx_vec)) {
      Size size = block_size(idx_vec);
      EXPECTS(size <= buff_span.size());
      for(size_t i = 0; i < size; i++) { buff_span[i] = (T) 0; }
      return;
    }

    Proc   proc;
    Offset offset;
    std::tie(proc, offset) = distribution_->locate(idx_vec);
    Size size              = block_size(idx_vec);
    EXPECTS(size <= buff_span.size());
    mpb_->mgr().nb_get(*mpb_, proc, offset, Size{size}, buff_span.data(), data_comm_handle);
  }

  /**
   * @brief Tensor accessor method for putting values to a set of indices
   * with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  virtual void put(const IndexVector& idx_vec, span<T> buff_span) {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);

    if(!is_non_zero(idx_vec)) { return; }

    Proc   proc;
    Offset offset;
    std::tie(proc, offset) = distribution_->locate(idx_vec);
    Size size              = block_size(idx_vec);
    EXPECTS(size <= buff_span.size());
    mpb_->mgr().put(*mpb_, proc, offset, Size{size}, buff_span.data());
  }

  /**
   * @brief Tensor accessor method for putting values in nonblocking fashion
   * to a set of indices with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  virtual void nb_put(const IndexVector& idx_vec, span<T> buff_span,
                      DataCommunicationHandlePtr data_comm_handle) {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);

    if(!is_non_zero(idx_vec)) { return; }

    Proc   proc;
    Offset offset;
    std::tie(proc, offset) = distribution_->locate(idx_vec);
    Size size              = block_size(idx_vec);
    EXPECTS(size <= buff_span.size());
    mpb_->mgr().nb_put(*mpb_, proc, offset, Size{size}, buff_span.data(), data_comm_handle);
  }

  /**
   * @brief Tensor accessor method for adding svalues to a set of indices
   * with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  virtual void add(const IndexVector& idx_vec, span<T> buff_span) {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);

    if(!is_non_zero(idx_vec)) { return; }

    Proc   proc;
    Offset offset;
    std::tie(proc, offset) = distribution_->locate(idx_vec);
    Size size              = block_size(idx_vec);
    EXPECTS(size <= buff_span.size());
    mpb_->mgr().add(*mpb_, proc, offset, Size{size}, buff_span.data());
  }

  /**
   * @brief Tensor accessor method for adding svalues in nonblocking fashion
   * to a set of indices with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  virtual void nb_add(const IndexVector& idx_vec, span<T> buff_span,
                      DataCommunicationHandlePtr data_comm_handle) {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);

    if(!is_non_zero(idx_vec)) { return; }

    Proc   proc;
    Offset offset;
    std::tie(proc, offset) = distribution_->locate(idx_vec);
    Size size              = block_size(idx_vec);
    EXPECTS(size <= buff_span.size());
    mpb_->mgr().nb_add(*mpb_, proc, offset, Size{size}, buff_span.data(), data_comm_handle);
  }

  /**
   * @brief Virtual method for getting the diagonal values in a Tensor
   *
   * @returns a vector with the diagonal values
   * @warning available for tensors with 2 modes
   */
  // virtual std::vector<T> diagonal() {
  //     EXPECTS(num_modes() == 2);
  //     std::vector<T> dest;
  //     for(const IndexVector& blockid : loop_nest()) {
  //         if(blockid[0] == blockid[1]) {
  //             const TAMM_SIZE size = block_size(blockid);
  //             std::vector<T> buf(size);
  //             get(blockid, buf);
  //             auto block_dims1  = block_dims(blockid);
  //             auto block_offset = block_offsets(blockid);
  //             auto dim          = block_dims1[0];
  //             auto offset       = block_offset[0];
  //             size_t i          = 0;
  //             for(auto p = offset; p < offset + dim; p++, i++) {
  //                 dest.push_back(buf[i * dim + i]);
  //             }
  //         }
  //     }
  //     return dest;
  // }

#if !defined(USE_UPCXX)
  virtual int ga_handle() {
    const MemoryRegionGA& mr = static_cast<const MemoryRegionGA&>(*mpb_);
    return mr.ga();
  }
#endif

  virtual void put_raw(int64_t* lo, int64_t* hi, void* buf, int64_t* buf_ld) const { abort(); }

  virtual void get_raw(int64_t* lo, int64_t* hi, void* buf, int64_t* buf_ld) const { abort(); }

  virtual T* access_local_buf() {
    EXPECTS(mpb_);
    return static_cast<T*>(mpb_->mgr().access(*mpb_, Offset{0}));
  }

  virtual const T* access_local_buf() const {
    EXPECTS(mpb_);
    return static_cast<T*>(mpb_->mgr().access(*mpb_, Offset{0}));
  }

  virtual size_t local_buf_size() const {
    EXPECTS(mpb_);
    return mpb_->local_nelements().value();
  }

  virtual size_t total_buf_size(Proc proc) const {
    EXPECTS(distribution_);
    return distribution_->buf_size(proc).value();
  }

  MemoryRegion* memory_region() const {
    EXPECTS(mpb_);
    return mpb_;
  }

  virtual int64_t size() const {
    EXPECTS(distribution_ != nullptr);
    return distribution_->total_size().value();
  }

  bool is_allocated() const { return (allocation_status_ == AllocationStatus::created); }

  void set_proc_list(const ProcList& proc_list) { proc_list_ = proc_list; }

  virtual bool is_block_cyclic() { return false; }

protected:
  std::shared_ptr<Distribution> distribution_;  /**< shared pointer to associated Distribution */
  MemoryRegion*                 mpb_ = nullptr; /**< Raw pointer memory region (default null) */
  ProcList                      proc_list_ = {};

}; // TensorImpl

/**
 * @ingroup tensors
 * @brief Implementation class for TensorBase with Lambda function construction
 *
 * @tparam T Element type of Tensor
 */
template<typename T>
class LambdaTensorImpl: public TensorImpl<T> {
public:
  using TensorBase::block_indices_;
  using TensorBase::setKind;

  /// @brief Function signature for the Lambda method
  using Func = std::function<void(const IndexVector&, span<T>)>;
  // Ctors
  LambdaTensorImpl() = default;

  // Copy/Move Ctors and Assignment Operators
  LambdaTensorImpl(LambdaTensorImpl&&)                 = default;
  LambdaTensorImpl(const LambdaTensorImpl&)            = delete;
  LambdaTensorImpl& operator=(LambdaTensorImpl&&)      = default;
  LambdaTensorImpl& operator=(const LambdaTensorImpl&) = delete;

  /**
   * @brief Construct a new LambdaTensorImpl object using a Lambda function
   *
   * @param [in] tis_vec vector of TiledIndexSpace objects for each mode of
   * the Tensor
   * @param [in] lambda a function for constructing the Tensor
   */
  LambdaTensorImpl(const TiledIndexSpaceVec& tis_vec, Func lambda):
    TensorImpl<T>(tis_vec), lambda_{lambda} {
    setKind(TensorBase::TensorKind::lambda);
  }

  LambdaTensorImpl(const IndexLabelVec& til_vec, Func lambda):
    TensorImpl<T>(til_vec), lambda_{lambda} {
    setKind(TensorBase::TensorKind::lambda);
  }

  LambdaTensorImpl(const IndexLabelVec& til_vec, Tensor<T> ref_tensor,
                   const IndexLabelVec& ref_labels, const IndexLabelVec& use_labels,
                   const std::vector<TranslateFunc>& translate_func_vec):
    TensorImpl<T>(til_vec),
    lambda_{construct_lambda(til_vec, ref_tensor, ref_labels, use_labels, translate_func_vec)} {}

  // Dtor
  ~LambdaTensorImpl() = default;

  void deallocate() override {
    // NO_OP();
  }

  void allocate(ExecutionContext* ec) override {
    // NO_OP();
  }

  void get(const IndexVector& idx_vec, span<T> buff_span) const override {
    lambda_(idx_vec, buff_span);
  }

  void put(const IndexVector& idx_vec, span<T> buff_span) override { NOT_ALLOWED(); }

  void add(const IndexVector& idx_vec, span<T> buff_span) override { NOT_ALLOWED(); }

  T* access_local_buf() override { NOT_ALLOWED(); }

  const T* access_local_buf() const override { NOT_ALLOWED(); }

  size_t local_buf_size() const override { NOT_ALLOWED(); }

  int64_t size() const override {
    int64_t res = 1;
    for(const auto& tis: block_indices_) { res *= tis.max_num_indices(); }
    return res;
  }

protected:
  Func lambda_; /**< Lambda function for the Tensor */

  Func construct_lambda(const IndexLabelVec& til_vec, Tensor<T> ref_tensor,
                        const IndexLabelVec& ref_labels, const IndexLabelVec& use_labels,
                        const std::vector<TranslateFunc>& translate_func_vec) {
    auto ip_gen_loop_builder_ptr = std::make_shared<blockops::cpu::IpGenLoopBuilder<2>>(
      std::array<IndexLabelVec, 2>{use_labels, ref_labels});

    auto perm_dup_map_compute = [](const IndexLabelVec& from, const IndexLabelVec& to) {
      std::vector<int> ret(from.size(), -1);
      for(size_t i = 0; i < from.size(); i++) {
        auto it = std::find(to.begin(), to.end(), from[i]);
        if(it != to.end()) { ret[i] = it - to.begin(); }
      }
      return ret;
    };

    auto perm_map = perm_dup_map_compute(use_labels, ref_labels);
    EXPECTS(perm_map.size() == use_labels.size());

    auto lambda = [til_vec, ref_tensor, ip_gen_loop_builder_ptr, translate_func_vec,
                   perm_map](const IndexVector& blockid, span<T> buff) {
      EXPECTS(blockid.size() == til_vec.size());
      EXPECTS(translate_func_vec.size() == blockid.size());

      IndexVector ref_blockid;
      for(size_t i = 0; i < blockid.size(); i++) {
        auto translated_id = translate_func_vec[i](blockid[i]);
        if(translated_id != -1) ref_blockid.push_back(translated_id);
      }

      auto           ref_buff_size  = ref_tensor.block_size(ref_blockid);
      auto           ref_block_dims = ref_tensor.block_dims(ref_blockid);
      std::vector<T> ref_buf(ref_buff_size);
      BlockSpan      ref_span(ref_buf.data(), ref_block_dims);

      std::vector<size_t> use_block_dims;
      for(size_t i = 0; i < perm_map.size(); i++) {
        use_block_dims.push_back(ref_block_dims[perm_map[i]]);
      }

      BlockSpan lhs_span(buff.data(), use_block_dims);
      EXPECTS(lhs_span.num_elements() == buff.size());

      ref_tensor.get(ref_blockid, ref_buf);

      blockops::cpu::flat_set(lhs_span, 0);

      ip_gen_loop_builder_ptr->update_plan(lhs_span, ref_span);
      blockops::cpu::ipgen_loop_assign(lhs_span.buf(), ip_gen_loop_builder_ptr->u2ald()[0],
                                       ref_span.buf(), ip_gen_loop_builder_ptr->u2ald()[1],
                                       ip_gen_loop_builder_ptr->unique_label_dims());
    };

    return lambda;
  }
}; // class LambdaTensorImpl

/**
 * @ingroup tensors
 * @brief Implementation class for TensorBase with dense multidimensional GA
 *
 * @tparam T Element type of Tensor
 */
template<typename T>
class DenseTensorImpl: public TensorImpl<T> {
public:
  using TensorImpl<T>::TensorBase::ec_;
  using TensorImpl<T>::TensorBase::setKind;
  using TensorImpl<T>::TensorBase::block_indices_;
  using TensorImpl<T>::TensorBase::allocation_status_;

  using TensorImpl<T>::block_size;
  using TensorImpl<T>::block_dims;
  using TensorImpl<T>::block_offsets;
  using TensorImpl<T>::proc_list_;
  using TensorImpl<T>::distribution_;
  using TensorImpl<T>::TensorBase::tindices;
  using TensorImpl<T>::TensorBase::num_modes;
  using TensorImpl<T>::TensorBase::update_status;

  // Ctors
  DenseTensorImpl() = default;

  // Copy/Move Ctors and Assignment Operators
  DenseTensorImpl(DenseTensorImpl&&)                 = default;
  DenseTensorImpl(const DenseTensorImpl&)            = delete;
  DenseTensorImpl& operator=(DenseTensorImpl&&)      = default;
  DenseTensorImpl& operator=(const DenseTensorImpl&) = delete;

  /**
   * @brief Construct a new LambdaTensorImpl object using a Lambda function
   *
   * @param [in] tis_vec vector of TiledIndexSpace objects for each mode of
   * the Tensor
   * @param [in] lambda a function for constructing the Tensor
   */
  DenseTensorImpl(const TiledIndexSpaceVec& tis_vec, const ProcGrid proc_grid,
                  const bool is_block_cyclic = false):
    TensorImpl<T>{tis_vec}, proc_grid_{proc_grid}, is_block_cyclic_{is_block_cyclic} {
    // check no dependences
    // for(auto& tis : tis_vec) { EXPECTS(!tis.is_dependent()); }

    // check index spaces are dense
    setKind(TensorBase::TensorKind::dense);
  }

  DenseTensorImpl(const IndexLabelVec& til_vec, const ProcGrid proc_grid,
                  const bool is_block_cyclic = false):
    TensorImpl<T>{til_vec}, proc_grid_{proc_grid}, is_block_cyclic_{is_block_cyclic} {
    // check no dependences
    // check index spaces are dense
    // for(auto& til : til_vec) { EXPECTS(!til.is_dependent()); }
    setKind(TensorBase::TensorKind::dense);
  }

  // Dtor
  ~DenseTensorImpl() {
    // EXPECTS(allocation_status_ == AllocationStatus::deallocated ||
    //         allocation_status_ == AllocationStatus::invalid);
  }

  const Distribution& distribution() const override {
    // return ref_tensor_.distribution();
    return *distribution_.get();
  }

  void deallocate() {
    EXPECTS(allocation_status_ == AllocationStatus::created);
#if defined(USE_UPCXX)
    ec_->pg().barrier();
    upcxx::delete_array(local_gptr_);
    ec_->pg().barrier();
#else
    NGA_Destroy(ga_);
    ga_ = -1;
#endif
    update_status(AllocationStatus::deallocated);
  }

  void allocate(ExecutionContext* ec) {
    EXPECTS(allocation_status_ == AllocationStatus::invalid);

#if defined(USE_UPCXX)
    // Assert we are in the calling proc group
    EXPECTS(ec->pg().rank() >= 0 && ec->pg().rank() < ec->pg().size());

    // All the code below only supports 2D tensors.
    EXPECTS(num_modes() == 2);

    ec_ = ec;
#else
    ec_ = ec;
    ga_ = NGA_Create_handle();
#endif
    const int ndims = num_modes();

    auto          defd = ec->get_default_distribution();
    Distribution* distribution =
      ec->distribution(defd->get_tensor_base(), defd->get_dist_proc()); // defd->kind());
    EXPECTS(distribution != nullptr);
    if(!proc_grid_.empty()) distribution->set_proc_grid(proc_grid_);
    distribution_ = std::shared_ptr<Distribution>(distribution->clone(this, ec->pg().size()));
    proc_grid_    = distribution_->proc_grid();

    auto tis_dims = tindices();

    std::vector<bool> is_irreg_tis(ndims, false);
    for(int i = 0; i < ndims; i++) is_irreg_tis[i] = !tis_dims[i].input_tile_sizes().empty();
#if defined(USE_UPCXX)
    for(auto tis: tis_dims) { tensor_dims_.push_back(tis.index_space().num_indices()); }
    eltype_             = tensor_element_type<T>();
    size_t element_size = MemoryManagerGA::get_element_size(eltype_);

    const bool is_irreg_tis1 = !tis_dims[0].input_tile_sizes().empty();
    const bool is_irreg_tis2 = !tis_dims[1].input_tile_sizes().empty();

    int nranks = ec->pg().size().value();

    std::vector<int64_t> bsize(2);
    bsize[0] = tis_dims[0].input_tile_size();
    bsize[1] = tis_dims[1].input_tile_size();

#else
    std::vector<int64_t> dims;
    for(auto tis: tis_dims) dims.push_back(tis.index_space().num_indices());

    NGA_Set_data64(ga_, ndims, &dims[0], ga_eltype_);

    if(proc_list_.size() > 0) {
      int nproc = proc_list_.size();
      int proclist_c[nproc];
      std::copy(proc_list_.begin(), proc_list_.end(), proclist_c);
      GA_Set_restricted(ga_, proclist_c, nproc);
    }
#endif

    if(is_block_cyclic_) {
      /*
       * Implement a block-cyclic distribution of tiles with size
       * (tis_dims[0].input_tile_size(), tis_dims[1].input_tile_size()) over a
       * processor grid with size (proc_grid_[0], proc_grid_[1]).
       */

      // EXPECTS(ndims == 2);
      std::vector<int64_t> bsize(2);
      std::vector<int64_t> pgrid(2);
#if defined(USE_UPCXX)
      std::vector<int64_t> ntiles(2);
#endif
      {
        // Cannot provide list of irreg tile sizes
        EXPECTS(!is_irreg_tis[0] && !is_irreg_tis[1]);

        bsize[0] = tis_dims[0].input_tile_size();
        bsize[1] = tis_dims[1].input_tile_size();

        pgrid[0] = proc_grid_[0].value();
        pgrid[1] = proc_grid_[1].value();
      }
#if defined(USE_UPCXX)
      int64_t total_n_procs = pgrid[0] * pgrid[1];

      ntiles[0]              = (tensor_dims_[0] + bsize[0] - 1) / bsize[0];
      ntiles[1]              = (tensor_dims_[1] + bsize[1] - 1) / bsize[1];
      int64_t total_n_tiles  = ntiles[0] * ntiles[1];
      int64_t tiles_per_proc = (total_n_tiles + total_n_procs - 1) / total_n_procs;

      size_t tile_size_in_bytes = bsize[0] * bsize[1] * element_size;

      local_gptr_ = upcxx::new_array<uint8_t>(tiles_per_proc * tile_size_in_bytes);
      memset(local_gptr_.local(), 0x00, tiles_per_proc * tile_size_in_bytes);

      int64_t  owning_rank  = 0;
      int64_t* tile_offsets = new int64_t[nranks];
      memset(tile_offsets, 0x00, nranks * sizeof(*tile_offsets));
      for(int64_t tile_row = 0; tile_row < tensor_dims_[0]; tile_row += bsize[0]) {
        for(int64_t tile_col = 0; tile_col < tensor_dims_[1]; tile_col += bsize[1]) {
          tiles.push_back(TensorTile(tile_row, tile_col, bsize[0], bsize[1], owning_rank,
                                     tile_offsets[owning_rank]));
          tile_offsets[owning_rank] += (bsize[0] * bsize[1]);
          owning_rank = (owning_rank + 1) % nranks;
        }
      }
      delete tile_offsets;
#else
      // blocks_ = block sizes for scalapack distribution
      NGA_Set_block_cyclic_proc_grid64(ga_, &bsize[0], &pgrid[0]);
#endif
    }
    else {
      // only needed when irreg tile sizes are provided
      const bool is_irreg_tens =
        std::any_of(is_irreg_tis.begin(), is_irreg_tis.end(), [](bool v) { return v; });
      if(is_irreg_tens) {
#if defined(USE_UPCXX)
        /*
         * Not supporting this on UPC++ for now, but
         * can be added if a motivating use case arises.
         */
        throw std::runtime_error("upcxx: irregular tile sizes not supported");
#else
        std::vector<std::vector<Tile>> new_tiles(ndims);
        for(int i = 0; i < ndims; i++) {
          new_tiles[i] = is_irreg_tis[i] ? tis_dims[i].input_tile_sizes()
                                         : std::vector<Tile>{tis_dims[i].input_tile_size()};
        }

        int64_t size_map;
        int64_t pgrid[ndims];
        int64_t nblock[ndims];
        std::vector<std::vector<Tile>> tiles(ndims);

        for(int i = 0; i < ndims; i++) pgrid[i] = proc_grid_[i].value();

        for(int i = 0; i < new_tiles.size(); i++) {
          int64_t dimc = 0;
          for(int j = 0; j < new_tiles[i].size(); j++) {
            tiles[i].push_back(new_tiles[i][j]);
            dimc += new_tiles[i][j];
            if(dimc >= dims[i]) break;
          }
        }

        for(int i = 0; i < ndims; i++) {
          nblock[i] = is_irreg_tis[i] ? tiles[i].size() : std::ceil(dims[i] * 1.0 / tiles[i][0]);
          // assert nblock[i] >= pgrid[i], if not, restrict ga to subset of procs
          if(pgrid[i] > nblock[i]) {
            pgrid[i] = nblock[i];
            proc_grid_[i] = nblock[i];
          }
        }
        distribution->set_proc_grid(proc_grid_);

        {
          int nproc_restricted =
            std::accumulate(pgrid, pgrid + ndims, (int) 1, std::multiplies<int>());
          int proclist_c[nproc_restricted];
          std::iota(proclist_c, proclist_c + nproc_restricted, 0);
          GA_Set_restricted(ga_, proclist_c, nproc_restricted);
        }

        size_map = std::accumulate(nblock, nblock + ndims, (int64_t) 0);

        // create map
        std::vector<int64_t> k_map(size_map);
        {
          auto mi = 0;
          for(auto idim = 0; idim < ndims; idim++) {
            auto size_blk = (dims[idim] / nblock[idim]);
            // regular tile size
            for(auto i = 0; i < nblock[idim]; i++) {
              k_map[mi] = size_blk * i;
              mi++;
            }
          }
        }
        NGA_Set_tiled_irreg_proc_grid64(ga_, &k_map[0], nblock, pgrid);
#endif
      }
      else {
        // fixed tilesize for both dims
#if defined(USE_UPCXX)
        int64_t chunk[2]          = {tis_dims[0].input_tile_size(), tis_dims[1].input_tile_size()};
        int64_t chunks_per_dim[2] = {(tensor_dims_[0] + chunk[0] - 1) / chunk[0],
                                     (tensor_dims_[1] + chunk[1] - 1) / chunk[1]};
        int64_t total_chunks      = chunks_per_dim[0] * chunks_per_dim[1];
        int64_t chunks_per_proc   = (total_chunks + nranks - 1) / nranks;
        local_gptr_ =
          upcxx::new_array<uint8_t>(chunks_per_proc * chunk[0] * chunk[1] * element_size);
        memset(local_gptr_.local(), 0x00, chunks_per_proc * chunk[0] * chunk[1] * element_size);

        int64_t  tile_index   = 0;
        int64_t* tile_offsets = new int64_t[nranks];
        memset(tile_offsets, 0x00, nranks * sizeof(*tile_offsets));
        for(int64_t tile_row = 0; tile_row < tensor_dims_[0]; tile_row += bsize[0]) {
          for(int64_t tile_col = 0; tile_col < tensor_dims_[1]; tile_col += bsize[1]) {
            const int owning_rank = tile_index / chunks_per_proc;
            tiles.push_back(TensorTile(tile_row, tile_col, bsize[0], bsize[1], owning_rank,
                                       tile_offsets[owning_rank]));
            tile_offsets[owning_rank] += (bsize[0] * bsize[1]);
            tile_index++;
          }
        }
        delete tile_offsets;
#else
        int64_t chunk[ndims];
        for(int i = 0; i < ndims; i++) chunk[i] = tis_dims[i].input_tile_size();
        GA_Set_chunk64(ga_, chunk);
#endif
      }
    }

#if defined(USE_UPCXX)
    gptrs_.resize(nranks);
    upcxx::dist_object<upcxx::global_ptr<uint8_t>>* dobj =
      new upcxx::dist_object<upcxx::global_ptr<uint8_t>>(local_gptr_, *ec->pg().team());
    ec->pg().barrier();
    for(int r = 0; r < nranks; r++) { gptrs_[r] = dobj->fetch(r).wait(); }
    ec->pg().barrier();
#else
    NGA_Set_pgroup(ga_, ec->pg().ga_pg());
    NGA_Allocate(ga_);
    distribution_->set_ga_handle(ga_);
#endif

    update_status(AllocationStatus::created);
  }

#if defined(USE_UPCXX)
  TensorTile find_tile(int64_t row, int64_t col) const {
    for(auto i = tiles.begin(), e = tiles.end(); i != e; i++) {
      TensorTile t = *i;
      if(t.contains(row, col)) { return t; }
    }
    abort();
  }
#endif

  void get(const IndexVector& blockid, span<T> buff_span) const {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);
    std::vector<int64_t> lo = compute_lo(blockid);
    std::vector<int64_t> hi = compute_hi(blockid);
    std::vector<int64_t> ld = compute_ld(blockid);
    EXPECTS(block_size(blockid) <= buff_span.size());
#if defined(USE_UPCXX)
    get_raw(&lo[0], &hi[0], buff_span.data(), &ld[0]);
#else
    NGA_Get64(ga_, &lo[0], &hi[0], buff_span.data(), &ld[0]);
#endif
  }

  void put(const IndexVector& blockid, span<T> buff_span) {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);
    std::vector<int64_t> lo = compute_lo(blockid);
    std::vector<int64_t> hi = compute_hi(blockid);
    std::vector<int64_t> ld = compute_ld(blockid);

    EXPECTS(block_size(blockid) <= buff_span.size());

#if defined(USE_UPCXX)
    put_raw(&lo[0], &hi[0], buff_span.data(), &ld[0]);
#else
    NGA_Put64(ga_, &lo[0], &hi[0], buff_span.data(), &ld[0]);
#endif
  }

  void add(const IndexVector& blockid, span<T> buff_span) {
#if defined(USE_UPCXX)
    throw std::runtime_error("upcxx: dense tensor - add unsupported");
#else
    EXPECTS(allocation_status_ != AllocationStatus::invalid);
    std::vector<int64_t> lo = compute_lo(blockid);
    std::vector<int64_t> hi = compute_hi(blockid);
    std::vector<int64_t> ld = compute_ld(blockid);
    EXPECTS(block_size(blockid) <= buff_span.size());
    void* alpha;
    switch(from_ga_eltype(ga_eltype_)) {
      case ElementType::single_precision: alpha = reinterpret_cast<void*>(&sp_alpha); break;
      case ElementType::double_precision: alpha = reinterpret_cast<void*>(&dp_alpha); break;
      case ElementType::single_complex: alpha = reinterpret_cast<void*>(&scp_alpha); break;
      case ElementType::double_complex: alpha = reinterpret_cast<void*>(&dcp_alpha); break;
      case ElementType::invalid:
      default: UNREACHABLE();
    }
    NGA_Acc64(ga_, &lo[0], &hi[0], reinterpret_cast<void*>(buff_span.data()), &ld[0], alpha);
#endif
  }

#ifndef USE_UPCXX
  int ga_handle() override { return ga_; }
#endif

  bool is_block_cyclic() override { return is_block_cyclic_; }

  /// @todo Should this be GA_Nodeid() or GA_Proup_nodeid(GA_Get_pgroup(ga_))
  T* access_local_buf() override {
#if defined(USE_UPCXX)
    throw std::runtime_error("upcxx: dense tensor - access_local_buf unsupported");
#else
    EXPECTS(allocation_status_ != AllocationStatus::invalid);
    T* ptr;
    int64_t len;
    NGA_Access_block_segment64(ga_, GA_Pgroup_nodeid(GA_Get_pgroup(ga_)),
                               reinterpret_cast<void*>(&ptr), &len);
    return ptr;
#endif
  }

  /// @todo Should this be GA_Nodeid() or GA_Proup_nodeid(GA_Get_pgroup(ga_))
  const T* access_local_buf() const override {
#if defined(USE_UPCXX)
    throw std::runtime_error("upcxx: dense tensor - access_local_buf unsupported");
#else
    EXPECTS(allocation_status_ != AllocationStatus::invalid);
    T* ptr;
    int64_t len;
    NGA_Access_block_segment64(ga_, GA_Pgroup_nodeid(GA_Get_pgroup(ga_)),
                               reinterpret_cast<void*>(&ptr), &len);
    return ptr;
#endif
  }

#if defined(USE_UPCXX)
  void put_raw(int64_t* lo, int64_t* hi, void* buf, int64_t* buf_ld) const {
    const auto elem_sz = MemoryManagerGA::get_element_size(eltype_);
    int        i       = 0;
    for(int64_t row = lo[0]; row <= hi[0]; row++) {
      for(int64_t col = lo[1]; col <= hi[1]; col++) {
        TensorTile t = find_tile(row, col);

        int64_t row_offset = row - t.lo[0];
        int64_t col_offset = col - t.lo[1];

        int64_t                    tile_offset = row_offset * t.dim[1] + col_offset;
        upcxx::global_ptr<uint8_t> target      = gptrs_[t.rank];
        upcxx::global_ptr<uint8_t> remote_addr =
          target + (t.offset * elem_sz) + (tile_offset * elem_sz);

        int64_t  buff_offset = row * buf_ld[0] + col;
        uint8_t* local_addr  = ((uint8_t*) buf) + (i++ * elem_sz);

        upcxx::rput(local_addr, remote_addr, elem_sz).wait();
      }
    }
  }

  void get_raw(int64_t* lo, int64_t* hi, void* buf, int64_t* buf_ld) const {
    const auto elem_sz = MemoryManagerGA::get_element_size(eltype_);
    int        i       = 0;
    for(int64_t row = lo[0]; row <= hi[0]; row++) {
      for(int64_t col = lo[1]; col <= hi[1]; col++) {
        TensorTile t = find_tile(row, col);

        int64_t row_offset = row - t.lo[0];
        int64_t col_offset = col - t.lo[1];

        int64_t                    tile_offset = row_offset * t.dim[1] + col_offset;
        upcxx::global_ptr<uint8_t> target      = gptrs_[t.rank];
        upcxx::global_ptr<uint8_t> remote_addr =
          target + (t.offset * elem_sz) + (tile_offset * elem_sz);

        int64_t  buff_offset = row * buf_ld[0] + col;
        uint8_t* local_addr  = ((uint8_t*) buf) + (i++ * elem_sz);

        upcxx::rget(remote_addr, local_addr, elem_sz).wait();
      }
    }
  }
#endif

  /// @todo Check for a GA method to get the local buf size?
  size_t local_buf_size() const override {
#if defined(USE_UPCXX)
    throw std::runtime_error("upcxx: dense tensor - local_buf_size unsupported");
#else
    EXPECTS(allocation_status_ != AllocationStatus::invalid);
    T* ptr;
    int64_t len;
    NGA_Access_block_segment64(ga_, GA_Pgroup_nodeid(GA_Get_pgroup(ga_)),
                               reinterpret_cast<void*>(&ptr), &len);
    size_t res = (size_t) len;
    return res;
#endif
  }

  /// @todo implement accordingly
  int64_t size() const override {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);
    int64_t res = 1;
    for(const auto& tis: block_indices_) { res *= tis.max_num_indices(); }
    return res;
  }

protected:
  std::vector<int64_t> compute_lo(const IndexVector& blockid) const {
    std::vector<int64_t> retv;
    std::vector<size_t>  off = block_offsets(blockid);
    for(const auto& i: off) { retv.push_back(static_cast<int64_t>(i)); }
    return retv;
  }

  std::vector<int64_t> compute_hi(const IndexVector& blockid) const {
    std::vector<int64_t> retv;
    std::vector<size_t>  boff  = block_offsets(blockid);
    std::vector<size_t>  bdims = block_dims(blockid);
    for(size_t i = 0; i < boff.size(); i++) {
      retv.push_back(static_cast<int64_t>(boff[i] + bdims[i] - 1));
    }
    return retv;
  }

  std::vector<int64_t> compute_ld(const IndexVector& blockid) const {
    std::vector<size_t>  bdims = block_dims(blockid);
    std::vector<int64_t> retv(bdims.size() - 1, 1);
    for(size_t i = 1; i < bdims.size(); i++) retv[i - 1] = (int64_t) (bdims[i]);
    return retv;
  }

#if defined(USE_UPCXX)
  upcxx::global_ptr<uint8_t>              local_gptr_;
  std::vector<upcxx::global_ptr<uint8_t>> gptrs_;
  ProcGrid                                proc_grid_;
  std::vector<int64_t>                    tensor_dims_;
  ElementType                             eltype_;
  std::vector<TensorTile>                 tiles;
#else
  int ga_;
  ProcGrid proc_grid_;
#endif

  bool is_block_cyclic_ = false;

  // constants for NGA_Acc call
  float         sp_alpha  = 1.0;
  double        dp_alpha  = 1.0;
  SingleComplex scp_alpha = {1, 0};
  DoubleComplex dcp_alpha = {1, 0};

  int ga_eltype_ = to_ga_eltype(tensor_element_type<T>());

}; // class DenseTensorImpl

template<typename T>
class ViewTensorImpl: public TensorImpl<T> {
public:
  using TensorImpl<T>::distribution_;
  using TensorImpl<T>::TensorBase::update_status;
  using TensorImpl<T>::TensorBase::setKind;
  using TensorImpl<T>::TensorBase::block_size;
  using TensorImpl<T>::TensorBase::block_dims;
  using TensorImpl<T>::TensorBase::ec_;
  using Func     = std::function<IndexVector(const IndexVector&)>;
  using CopyFunc = std::function<void(const BlockSpan<T>&, BlockSpan<T>&, const IndexVector&)>;

  // Ctors
  ViewTensorImpl() = default;
  ViewTensorImpl(Tensor<T> ref_tensor, const TiledIndexSpaceVec& tis_vec, Func ref_map_func):
    TensorImpl<T>(tis_vec), ref_tensor_{ref_tensor}, ref_map_func_{ref_map_func} {
    setKind(TensorBase::TensorKind::view);
    if(ref_tensor_.is_allocated()) {
      distribution_ =
        std::make_shared<ViewDistribution>(&ref_tensor_.distribution(), ref_map_func_);
      update_status(AllocationStatus::created);
    }
    EXPECTS(ref_tensor_.is_allocated());
    ec_ = ref_tensor.execution_context();
  }

  ViewTensorImpl(Tensor<T> ref_tensor, const IndexLabelVec& labels, Func ref_map_func):
    TensorImpl<T>(labels), ref_tensor_{ref_tensor}, ref_map_func_{ref_map_func} {
    setKind(TensorBase::TensorKind::view);

    if(ref_tensor_.is_allocated()) {
      distribution_ =
        std::make_shared<ViewDistribution>(&ref_tensor_.distribution(), ref_map_func_);
      update_status(AllocationStatus::created);
    }
    EXPECTS(ref_tensor_.is_allocated());
    ec_ = ref_tensor.execution_context();
  }

  ViewTensorImpl(Tensor<T> ref_tensor, const IndexLabelVec& labels, Func ref_map_func,
                 CopyFunc get_func, CopyFunc put_func):
    TensorImpl<T>(labels),
    ref_tensor_{ref_tensor},
    ref_map_func_{ref_map_func},
    get_func_{get_func},
    put_func_{put_func} {
    setKind(TensorBase::TensorKind::view);

    if(ref_tensor_.is_allocated()) {
      distribution_ =
        std::make_shared<ViewDistribution>(&ref_tensor_.distribution(), ref_map_func_);
      update_status(AllocationStatus::created);
    }
    EXPECTS(ref_tensor_.is_allocated());
    ec_ = ref_tensor.execution_context();
  }

  // Copy/Move Ctors and Assignment Operators
  ViewTensorImpl(ViewTensorImpl&&)                 = default;
  ViewTensorImpl(const ViewTensorImpl&)            = delete;
  ViewTensorImpl& operator=(ViewTensorImpl&&)      = default;
  ViewTensorImpl& operator=(const ViewTensorImpl&) = delete;

  // Dtor
  ~ViewTensorImpl() = default;

  /**
   * @brief Virtual method for deallocating a Tensor
   *
   */
  void deallocate() override {
    // No-op
  }

  /**
   * @brief Virtual method for allocating a Tensor using an ExecutionContext
   *
   * @param [in] ec ExecutionContext to be used for allocation
   */
  void allocate(ExecutionContext* ec) override {
    // No-op
    // EXPECTS(ref_tensor_.is_allocated());
    // distribution_ = std::make_shared<ViewDistribution>(
    //     &ref_tensor_.distribution(), ref_map_func_);
    // update_status(AllocationStatus::created);
  }

  const Distribution& distribution() const override {
    // return ref_tensor_.distribution();
    return *distribution_.get();
  }

  // Tensor Accessors
  /**
   * @brief Tensor accessor method for getting values from a set of
   * indices to specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to fetch the values
   * @param [in] buff_span memory span where to put the fetched values
   */

  void get(const IndexVector& view_vec, span<T> buff_span) const override {
    // Call Reference tensors get
    const auto&     idx_vec       = ref_map_func_(view_vec);
    auto            ref_buf_size  = ref_tensor_.block_size(idx_vec);
    auto            ref_blck_dims = ref_tensor_.block_dims(idx_vec);
    std::vector<T>  ref_vec(ref_buf_size, 0);
    const BlockSpan ref_span{ref_vec.data(), ref_blck_dims};
    // @todo we might not need to do a get depending on if the tile has non-zero values
    ref_tensor_.get(idx_vec, ref_vec);

    auto      blck_dims = block_dims(view_vec);
    BlockSpan blck_span{buff_span.data(), blck_dims};

    get_func_(ref_span, blck_span, view_vec);
  }

  /**
   * @brief Tensor accessor method for getting values in nonblocking fashion
   * from a set of indices to specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to fetch the values
   * @param [in] buff_span memory span where to put the fetched values
   *
   * @bug should fix handle return
   */

  void nb_get(const IndexVector& view_vec, span<T> buff_span,
              DataCommunicationHandlePtr data_comm_handle) const override {
    // Call Reference tensors nb_get
    const auto&     idx_vec       = ref_map_func_(view_vec);
    auto            ref_buf_size  = ref_tensor_.block_size(idx_vec);
    auto            ref_blck_dims = ref_tensor_.block_dims(idx_vec);
    std::vector<T>  ref_vec(ref_buf_size, 0);
    const BlockSpan ref_span{ref_vec.data(), ref_blck_dims};

    ref_tensor_.nb_get(idx_vec, ref_vec, data_comm_handle);

    auto      blck_dims = block_dims(view_vec);
    BlockSpan blck_span{buff_span.data(), blck_dims};

    get_func_(ref_span, blck_span, view_vec);
  }

  /**
   * @brief Tensor accessor method for putting values to a set of indices
   * with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  void put(const IndexVector& view_vec, span<T> buff_span) override {
    NOT_ALLOWED();
    // Call Reference tensors put
    const auto&    idx_vec       = ref_map_func_(view_vec);
    auto           ref_buf_size  = ref_tensor_.block_size(idx_vec);
    auto           ref_blck_dims = ref_tensor_.block_dims(idx_vec);
    std::vector<T> ref_vec(ref_buf_size, 0);
    BlockSpan      ref_span{ref_vec.data(), ref_blck_dims};

    auto            blck_dims = block_dims(view_vec);
    const BlockSpan blck_span{buff_span.data(), blck_dims};

    put_func_(blck_span, ref_span, view_vec);

    ref_tensor_.put(idx_vec, ref_vec);
  }

  /**
   * @brief Tensor accessor method for putting values in nonblocking fashion
   * to a set of indices with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  void nb_put(const IndexVector& view_vec, span<T> buff_span,
              DataCommunicationHandlePtr data_comm_handle) override {
    NOT_ALLOWED();
    // Call Reference tensors nb_put
    const auto&    idx_vec       = ref_map_func_(view_vec);
    auto           ref_buf_size  = ref_tensor_.block_size(idx_vec);
    auto           ref_blck_dims = ref_tensor_.block_dims(idx_vec);
    std::vector<T> ref_vec(ref_buf_size, 0);
    BlockSpan      ref_span{ref_vec.data(), ref_blck_dims};

    auto            blck_dims = block_dims(view_vec);
    const BlockSpan blck_span{buff_span.data(), blck_dims};

    put_func_(blck_span, ref_span, view_vec);
    ref_tensor_.nb_put(idx_vec, ref_vec, data_comm_handle);
  }

  /**
   * @brief Tensor accessor method for adding svalues to a set of indices
   * with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  void add(const IndexVector& view_vec, span<T> buff_span) override {
    // Call Reference tensors nb_put
    const auto& idx_vec = ref_map_func_(view_vec);
    ref_tensor_.add(idx_vec, buff_span);
  }

  /**
   * @brief Tensor accessor method for adding svalues in nonblocking fashion
   * to a set of indices with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  void nb_add(const IndexVector& view_vec, span<T> buff_span,
              DataCommunicationHandlePtr data_comm_handle) override {
    // Call Reference tensors nb_put
    const auto& idx_vec = ref_map_func_(view_vec);
    ref_tensor_.nb_add(idx_vec, buff_span, data_comm_handle);
  }

#if !defined(USE_UPCXX)
  int ga_handle() override {
    // Call Reference tensor
    return ref_tensor_.ga_handle();
  }
#endif

  T* access_local_buf() override { NOT_ALLOWED(); }

  const T* access_local_buf() const override { NOT_ALLOWED(); }

protected:
  Func      ref_map_func_;
  CopyFunc  get_func_;
  CopyFunc  put_func_;
  Tensor<T> ref_tensor_;
}; // class ViewTensorImpl

template<typename T>
class TensorUnitTiled: public TensorImpl<T> {
public:
  using TensorImpl<T>::mpb_;
  using TensorImpl<T>::distribution_;
  using TensorImpl<T>::update_status;
  using TensorImpl<T>::is_allocated;

  using TensorImpl<T>::TensorBase::ec_;
  using TensorImpl<T>::TensorBase::setKind;
  using TensorImpl<T>::TensorBase::allocation_status_;
  using TensorImpl<T>::TensorBase::is_non_zero;
  using TensorImpl<T>::TensorBase::block_size;

  // Ctors
  TensorUnitTiled() = default;

  TensorUnitTiled(const Tensor<T>& opt_tensor, size_t unit_tis_count):
    TensorImpl<T>{construct_new_tis(opt_tensor, unit_tis_count)}, tensor_opt_{opt_tensor} {
    setKind(TensorBase::TensorKind::unit_view);
    if(tensor_opt_.is_allocated()) { allocate(tensor_opt_.execution_context()); }
  }

  TensorUnitTiled(const Tensor<T>& opt_tensor, size_t unit_tis_count,
                  const std::vector<size_t>& spin_sizes):
    TensorImpl<T>{construct_new_tis(opt_tensor, unit_tis_count), spin_sizes},
    tensor_opt_{opt_tensor} {
    setKind(TensorBase::TensorKind::unit_view);
    if(tensor_opt_.is_allocated()) { allocate(tensor_opt_.execution_context()); }
  }

  // Copy/Move Ctors and Assignment Operators
  TensorUnitTiled(TensorUnitTiled&&)                 = default;
  TensorUnitTiled(const TensorUnitTiled&)            = delete;
  TensorUnitTiled& operator=(TensorUnitTiled&&)      = default;
  TensorUnitTiled& operator=(const TensorUnitTiled&) = delete;

  // Dtor
  ~TensorUnitTiled() = default;

  /**
   * @brief Virtual method implementation for deallocating a unit tiled view tensor
   * @todo Decide on the actual behavior - no action is done for now
   */
  void deallocate() override {
    EXPECTS(allocation_status_ == AllocationStatus::created);
    EXPECTS(mpb_);
    // ec_->unregister_for_dealloc(mpb_);
    // mpb_->dealloc_coll();
    // delete mpb_;
    // mpb_ = nullptr;
    // update_status(AllocationStatus::deallocated);
  }

  /**
   * @brief Virtual method implementation for allocating a unit tiled view tensor using an
   * ExecutionContext
   *
   * @param [in] ec ExecutionContext to be used for allocation
   *
   */
  void allocate(ExecutionContext* ec) override {
    EXPECTS(tensor_opt_.is_allocated());

    if(!is_allocated()) {
      auto          defd         = ec->get_default_distribution();
      Distribution* distribution = ec->distribution(defd->get_tensor_base(), defd->get_dist_proc());
      MemoryManager* memory_manager = ec->memory_manager();
      EXPECTS(distribution != nullptr);
      EXPECTS(memory_manager != nullptr);
      ec_ = ec;

      distribution_ =
        std::shared_ptr<Distribution>(new UnitTileDistribution(this, &tensor_opt_.distribution()));

      EXPECTS(distribution_ != nullptr);

      auto eltype = tensor_element_type<T>();
      mpb_        = tensor_opt_.memory_region();
      EXPECTS(mpb_ != nullptr);
      update_status(AllocationStatus::created);
    }
  }

  const Distribution& distribution() const override { return *distribution_.get(); }

  // // Tensor Accessors
  // /**
  //  * @brief Tensor accessor method for getting values from a set of
  //  * indices to specified memory span
  //  *
  //  * @tparam T type of the values hold on the tensor object
  //  * @param [in] idx_vec a vector of indices to fetch the values
  //  * @param [in] buff_span memory span where to put the fetched values
  //  */

  // void get(const IndexVector &idx_vec, span<T> buff_span) const override {
  //   EXPECTS(allocation_status_ != AllocationStatus::invalid);

  //   if(!is_non_zero(idx_vec)) {
  //       Size size = block_size(idx_vec);
  //       EXPECTS(size <= buff_span.size());
  //       for(size_t i = 0; i < size; i++) { buff_span[i] = (T)0; }
  //       return;
  //   }

  //   Proc proc;
  //   Offset offset;
  //   std::tie(proc, offset) = distribution_->locate(idx_vec);
  //   Size size              = block_size(idx_vec);
  //   EXPECTS(size <= buff_span.size());
  //   mpb_->mgr().get(*mpb_, proc, offset, Size{size}, buff_span.data());
  // }

  //   /**
  //    * @brief Tensor accessor method for getting values in nonblocking fashion
  //    * from a set of indices to specified memory span
  //    *
  //    * @tparam T type of the values hold on the tensor object
  //    * @param [in] idx_vec a vector of indices to fetch the values
  //    * @param [in] buff_span memory span where to put the fetched values
  //    */

  //   virtual void nb_get(const IndexVector& idx_vec, span<T> buff_span,
  //                       DataCommunicationHandlePtr data_comm_handle) const {
  //       EXPECTS(allocation_status_ != AllocationStatus::invalid);

  //       if(!is_non_zero(idx_vec)) {
  //           Size size = block_size(idx_vec);
  //           EXPECTS(size <= buff_span.size());
  //           for(size_t i = 0; i < size; i++) { buff_span[i] = (T)0; }
  //           return;
  //       }

  //       Proc proc;
  //       Offset offset;
  //       std::tie(proc, offset) = distribution_->locate(idx_vec);
  //       Size size              = block_size(idx_vec);
  //       EXPECTS(size <= buff_span.size());
  //       mpb_->mgr().nb_get(*mpb_, proc, offset, Size{size}, buff_span.data(),
  //                          data_comm_handle);
  //   }

  //   /**
  //    * @brief Tensor accessor method for putting values to a set of indices
  //    * with the specified memory span
  //    *
  //    * @tparam T type of the values hold on the tensor object
  //    * @param [in] idx_vec a vector of indices to put the values
  //    * @param [in] buff_span buff_span memory span for the values to put
  //    */

  //   virtual void put(const IndexVector& idx_vec, span<T> buff_span) {
  //       EXPECTS(allocation_status_ != AllocationStatus::invalid);

  //       if(!is_non_zero(idx_vec)) { return; }

  //       Proc proc;
  //       Offset offset;
  //       std::tie(proc, offset) = distribution_->locate(idx_vec);
  //       Size size              = block_size(idx_vec);
  //       EXPECTS(size <= buff_span.size());
  //       mpb_->mgr().put(*mpb_, proc, offset, Size{size}, buff_span.data());
  //   }

  //   /**
  //    * @brief Tensor accessor method for putting values in nonblocking fashion
  //    * to a set of indices with the specified memory span
  //    *
  //    * @tparam T type of the values hold on the tensor object
  //    * @param [in] idx_vec a vector of indices to put the values
  //    * @param [in] buff_span buff_span memory span for the values to put
  //    */

  //   virtual void nb_put(const IndexVector& idx_vec, span<T> buff_span,
  //                       DataCommunicationHandlePtr data_comm_handle) {
  //       EXPECTS(allocation_status_ != AllocationStatus::invalid);

  //       if(!is_non_zero(idx_vec)) { return; }

  //       Proc proc;
  //       Offset offset;
  //       std::tie(proc, offset) = distribution_->locate(idx_vec);
  //       Size size              = block_size(idx_vec);
  //       EXPECTS(size <= buff_span.size());
  //       mpb_->mgr().nb_put(*mpb_, proc, offset, Size{size}, buff_span.data(),
  //                          data_comm_handle);
  //   }

  //   /**
  //    * @brief Tensor accessor method for adding svalues to a set of indices
  //    * with the specified memory span
  //    *
  //    * @tparam T type of the values hold on the tensor object
  //    * @param [in] idx_vec a vector of indices to put the values
  //    * @param [in] buff_span buff_span memory span for the values to put
  //    */

  //   virtual void add(const IndexVector& idx_vec, span<T> buff_span) {
  //       EXPECTS(allocation_status_ != AllocationStatus::invalid);

  //       if(!is_non_zero(idx_vec)) { return; }

  //       Proc proc;
  //       Offset offset;
  //       std::tie(proc, offset) = distribution_->locate(idx_vec);
  //       Size size              = block_size(idx_vec);
  //       EXPECTS(size <= buff_span.size());
  //       mpb_->mgr().add(*mpb_, proc, offset, Size{size}, buff_span.data());
  //   }

  //   /**
  //    * @brief Tensor accessor method for adding svalues in nonblocking fashion
  //    * to a set of indices with the specified memory span
  //    *
  //    * @tparam T type of the values hold on the tensor object
  //    * @param [in] idx_vec a vector of indices to put the values
  //    * @param [in] buff_span buff_span memory span for the values to put
  //    */

  //   virtual void nb_add(const IndexVector& idx_vec, span<T> buff_span,
  //                       DataCommunicationHandlePtr data_comm_handle) {
  //       EXPECTS(allocation_status_ != AllocationStatus::invalid);

  //       if(!is_non_zero(idx_vec)) { return; }

  //       Proc proc;
  //       Offset offset;
  //       std::tie(proc, offset) = distribution_->locate(idx_vec);
  //       Size size              = block_size(idx_vec);
  //       EXPECTS(size <= buff_span.size());
  //       mpb_->mgr().nb_add(*mpb_, proc, offset, Size{size}, buff_span.data(),
  //                          data_comm_handle);
  //   }

private:
  Tensor<T> tensor_opt_;

  TiledIndexSpaceVec construct_new_tis(const Tensor<T>& opt_tensor, size_t unit_tis_count) const {
    TiledIndexSpaceVec result_tis_list = opt_tensor.tiled_index_spaces();

    for(size_t i = 0; i < unit_tis_count; i++) {
      // get opt tiled index space
      TiledIndexSpace orig_tis = result_tis_list[i];

      // construct unit tiled index space
      TiledIndexSpace unit_tis{orig_tis.index_space()};

      // update resulting tis list
      result_tis_list[i] = unit_tis;
    }

    return result_tis_list;
  }

  bool is_unit_tiled(const TiledIndexSpace& tis) {
    return (tis.num_tiles() == tis.index_space().num_indices());
  }
}; // class TensorUnitTiled
} // namespace tamm
