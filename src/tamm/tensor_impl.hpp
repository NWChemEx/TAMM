
#ifndef TAMM_TENSOR_IMPL_HPP_
#define TAMM_TENSOR_IMPL_HPP_

#include "tamm/distribution.hpp"
#include "tamm/execution_context.hpp"
#include "tamm/index_loop_nest.hpp"
#include "tamm/index_space.hpp"
#include "tamm/labeled_tensor.hpp"
#include "tamm/memory_manager_local.hpp"
#include "tamm/tensor_base.hpp"

#include <functional>
#include <gsl/span>
#include <type_traits>
#include "ga.h"

namespace tamm {

using gsl::span;

template<typename T>
class LabeledTensor;

/**
 * @ingroup tensors
 * @brief Implementation class for TensorBase class
 *
 * @tparam T Element type of Tensor
 */
template<typename T>
class TensorImpl : public TensorBase {
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
    TensorImpl(const TiledIndexSpaceVec& tis) : TensorBase{tis} {
        has_spin_symmetry_ = false;
    }

    /**
     * @brief Construct a new TensorImpl object using a vector of
     * TiledIndexSpace objects for each mode of the tensor
     *
     * @param [in] lbls vector of tiled index labels used for extracting
     * corresponding TiledIndexSpace objects for each mode used to construct
     * the tensor
     */
    TensorImpl(const std::vector<TiledIndexLabel>& lbls) : TensorBase{lbls} {
        has_spin_symmetry_ = false;
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
    TensorImpl(const TiledIndexSpace& tis, Ts... rest) :
      TensorBase{tis, rest...} {
        num_modes_ = block_indices_.size();
        construct_dep_map();
        has_spin_symmetry_ = false;
    }

    // SpinTensor related constructors
    /**
     * @brief Construct a new SpinTensorImpl object using set of TiledIndexSpace
     * objects and Spin attribute mask
     *
     * @param [in] t_spaces
     * @param [in] spin_mask
     */
    TensorImpl(TiledIndexSpaceVec t_spaces, SpinMask spin_mask) :
      TensorBase(t_spaces) {
        EXPECTS(t_spaces.size() == spin_mask.size());

        // for(const auto& tis : t_spaces) { EXPECTS(tis.has_spin()); }

        spin_mask_         = spin_mask;
        has_spin_symmetry_ = true;
        // spin_total_        = calculate_spin();
    }

    /**
     * @brief Construct a new SpinTensorImpl object using set of TiledIndexLabel
     * objects and Spin attribute mask
     *
     * @param [in] t_labels
     * @param [in] spin_mask
     */
    TensorImpl(IndexLabelVec t_labels, SpinMask spin_mask) :
      TensorBase(t_labels) {
        EXPECTS(t_labels.size() == spin_mask.size());
        // for(const auto& tlbl : t_labels) {
        //     EXPECTS(tlbl.tiled_index_space().has_spin());
        // }
        spin_mask_         = spin_mask;
        has_spin_symmetry_ = true;
        // spin_total_        = calculate_spin();
    }

    /**
     * @brief Construct a new SpinTensorImpl object using set of TiledIndexSpace
     * objects and Spin attribute mask
     *
     * @param [in] t_spaces
     * @param [in] spin_mask
     */
    TensorImpl(TiledIndexSpaceVec t_spaces, std::vector<size_t> spin_sizes) :
      TensorBase(t_spaces) {
        // EXPECTS(t_spaces.size() == spin_mask.size());
        EXPECTS(spin_sizes.size() > 0);
        // for(const auto& tis : t_spaces) { EXPECTS(tis.has_spin()); }
        SpinMask spin_mask;
        size_t upper = spin_sizes[0];
        size_t lower =
          spin_sizes.size() > 1 ? spin_sizes[1] : t_spaces.size() - upper;
        size_t ignore = spin_sizes.size() > 2 ?
                          spin_sizes[1] :
                          t_spaces.size() - (upper + lower);

        for(size_t i = 0; i < upper; i++) {
            spin_mask.push_back(SpinPosition::upper);
        }

        for(size_t i = 0; i < lower; i++) {
            spin_mask.push_back(SpinPosition::lower);
        }

        for(size_t i = 0; i < upper; i++) {
            spin_mask.push_back(SpinPosition::ignore);
        }

        spin_mask_         = spin_mask;
        has_spin_symmetry_ = true;
        // spin_total_        = calculate_spin();
    }

    /**
     * @brief Construct a new SpinTensorImpl object using set of TiledIndexLabel
     * objects and Spin attribute mask
     *
     * @param [in] t_labels
     * @param [in] spin_mask
     */
    TensorImpl(IndexLabelVec t_labels, std::vector<size_t> spin_sizes) :
      TensorBase(t_labels) {
        // EXPECTS(t_labels.size() == spin_mask.size());
        EXPECTS(spin_sizes.size() > 0);
        // for(const auto& tlbl : t_labels) {
        //     EXPECTS(tlbl.tiled_index_space().has_spin());
        // }

        SpinMask spin_mask;
        size_t upper = spin_sizes[0];
        size_t lower =
          spin_sizes.size() > 1 ? spin_sizes[1] : t_labels.size() - upper;
        size_t ignore = spin_sizes.size() > 2 ?
                          spin_sizes[1] :
                          t_labels.size() - (upper + lower);

        for(size_t i = 0; i < upper; i++) {
            spin_mask.push_back(SpinPosition::upper);
        }

        for(size_t i = 0; i < lower; i++) {
            spin_mask.push_back(SpinPosition::lower);
        }

        for(size_t i = 0; i < upper; i++) {
            spin_mask.push_back(SpinPosition::ignore);
        }

        spin_mask_         = spin_mask;
        has_spin_symmetry_ = true;
        // spin_total_        = calculate_spin();
    }

    // Copy/Move Ctors and Assignment Operators
    TensorImpl(TensorImpl&&)      = default;
    TensorImpl(const TensorImpl&) = delete;
    TensorImpl& operator=(TensorImpl&&) = default;
    TensorImpl& operator=(const TensorImpl&) = delete;

    // Dtor
    ~TensorImpl() {
        if(mpb_ != nullptr) {
            mpb_->allocation_status_ = AllocationStatus::orphaned;
        }
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
        EXPECTS(allocation_status_ == AllocationStatus::invalid);
        Distribution* distribution    = ec->distribution();
        MemoryManager* memory_manager = ec->memory_manager();
        EXPECTS(distribution != nullptr);
        EXPECTS(memory_manager != nullptr);
        ec_ = ec;
        // distribution_ = DistributionFactory::make_distribution(*distribution,
        // this, pg.size());
        distribution_ = std::shared_ptr<Distribution>(
          distribution->clone(this, memory_manager->pg().size()));
        auto rank     = memory_manager->pg().rank();
        auto buf_size = distribution_->buf_size(rank);
        auto eltype   = tensor_element_type<T>();
        EXPECTS(buf_size >= 0);
        mpb_ = memory_manager->alloc_coll(eltype, buf_size);
        EXPECTS(mpb_ != nullptr);
        ec_->register_for_dealloc(mpb_);
        update_status(AllocationStatus::created);
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

    virtual void get(const IndexVector& idx_vec, span<T> buff_span) const {
        EXPECTS(allocation_status_ != AllocationStatus::invalid);

        if(!is_non_zero(idx_vec)) {
            Size size = block_size(idx_vec);
            EXPECTS(size <= buff_span.size());
            for(size_t i = 0; i < size; i++) { buff_span[i] = (T)0; }
            return;
        }

        Proc proc;
        Offset offset;
        std::tie(proc, offset) = distribution_->locate(idx_vec);
        Size size              = block_size(idx_vec);
        EXPECTS(size <= buff_span.size());
        mpb_->mgr().get(*mpb_, proc, offset, Size{size}, buff_span.data());
    }

    /**
     * @brief Tensor accessor method for getting values in nonblocking fashion from a set of
     * indices to specified memory span
     *
     * @tparam T type of the values hold on the tensor object
     * @param [in] idx_vec a vector of indices to fetch the values
     * @param [in] buff_span memory span where to put the fetched values
     */

    virtual void nb_get(const IndexVector& idx_vec, span<T> buff_span, DataCommunicationHandlePtr data_comm_handle) const {
        EXPECTS(allocation_status_ != AllocationStatus::invalid);

        if(!is_non_zero(idx_vec)) {
            Size size = block_size(idx_vec);
            EXPECTS(size <= buff_span.size());
            for(size_t i = 0; i < size; i++) { buff_span[i] = (T)0; }
            return;
        }

        Proc proc;
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

        Proc proc;
        Offset offset;
        std::tie(proc, offset) = distribution_->locate(idx_vec);
        Size size              = block_size(idx_vec);
        EXPECTS(size <= buff_span.size());
        mpb_->mgr().put(*mpb_, proc, offset, Size{size}, buff_span.data());
    }

    /**
     * @brief Tensor accessor method for putting values in nonblocking fashion to a set of indices
     * with the specified memory span
     *
     * @tparam T type of the values hold on the tensor object
     * @param [in] idx_vec a vector of indices to put the values
     * @param [in] buff_span buff_span memory span for the values to put
     */

    virtual void nb_put(const IndexVector& idx_vec, span<T> buff_span, DataCommunicationHandlePtr data_comm_handle) {
        EXPECTS(allocation_status_ != AllocationStatus::invalid);

        if(!is_non_zero(idx_vec)) { return; }

        Proc proc;
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

        Proc proc;
        Offset offset;
        std::tie(proc, offset) = distribution_->locate(idx_vec);
        Size size              = block_size(idx_vec);
        EXPECTS(size <= buff_span.size());
        mpb_->mgr().add(*mpb_, proc, offset, Size{size}, buff_span.data());
    }
    
    /**
     * @brief Tensor accessor method for adding svalues in nonblocking fashion to a set of indices
     * with the specified memory span
     *
     * @tparam T type of the values hold on the tensor object
     * @param [in] idx_vec a vector of indices to put the values
     * @param [in] buff_span buff_span memory span for the values to put
     */

    virtual void nb_add(const IndexVector& idx_vec, span<T> buff_span, DataCommunicationHandlePtr data_comm_handle) {
        EXPECTS(allocation_status_ != AllocationStatus::invalid);

        if(!is_non_zero(idx_vec)) { return; }

        Proc proc;
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

    virtual int ga_handle() {
        const MemoryRegionGA& mr = static_cast<const MemoryRegionGA&>(*mpb_);
        return mr.ga();
    }

protected:
    std::shared_ptr<Distribution> distribution_;    /**< shared pointer to associated Distribution */
    MemoryRegion* mpb_ = nullptr;   /**< Raw pointer memory region (default null) */
}; // TensorImpl

/**
 * @ingroup tensors
 * @brief Implementation class for TensorBase with Lambda function construction
 * 
 * @tparam T Element type of Tensor
 */
template<typename T>
class LambdaTensorImpl : public TensorImpl<T> {
public:
    /// @brief Function signature for the Lambda method
    using Func = std::function<void(const IndexVector&, span<T>)>;
    // Ctors
    LambdaTensorImpl() = default;

    // Copy/Move Ctors and Assignment Operators
    LambdaTensorImpl(LambdaTensorImpl&&)      = default;
    LambdaTensorImpl(const LambdaTensorImpl&) = delete;
    LambdaTensorImpl& operator=(LambdaTensorImpl&&) = default;
    LambdaTensorImpl& operator=(const LambdaTensorImpl&) = delete;

    /**
     * @brief Construct a new LambdaTensorImpl object using a Lambda function
     * 
     * @param [in] tis_vec vector of TiledIndexSpace objects for each mode of the Tensor 
     * @param [in] lambda a function for constructing the Tensor
     */
    LambdaTensorImpl(const TiledIndexSpaceVec& tis_vec, Func lambda) :
      TensorImpl<T>(tis_vec),
      lambda_{lambda} {}


    LambdaTensorImpl(const IndexLabelVec& til_vec, Func lambda) : 
        TensorImpl<T>(til_vec),
        lambda_{lambda} {}

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

    void put(const IndexVector& idx_vec, span<T> buff_span) override {
        NOT_ALLOWED();
    }

    void add(const IndexVector& idx_vec, span<T> buff_span) override {
        NOT_ALLOWED();
    }

protected:
    Func lambda_;   /**< Lambda function for the Tensor */
}; // class LambdaTensorImpl

/**
 * @ingroup tensors
 * @brief Implementation class for TensorBase with dense multidimensional GA
 *
 * @tparam T Element type of Tensor
 */
template <typename T>
class DenseTensorImpl : public TensorImpl<T> {
 public:
  using  TensorImpl<T>::TensorBase::ec_;
  using  TensorImpl<T>::TensorBase::allocation_status_;
  
  using TensorImpl<T>::block_size;
  using TensorImpl<T>::block_dims;
  using TensorImpl<T>::block_offsets;
  using TensorImpl<T>::TensorBase::tindices;
  using TensorImpl<T>::TensorBase::num_modes;
  using TensorImpl<T>::TensorBase::update_status;
  

  // Ctors
  DenseTensorImpl() = default;

  // Copy/Move Ctors and Assignment Operators
  DenseTensorImpl(DenseTensorImpl&&) = default;
  DenseTensorImpl(const DenseTensorImpl&) = delete;
  DenseTensorImpl& operator=(DenseTensorImpl&&) = default;
  DenseTensorImpl& operator=(const DenseTensorImpl&) = delete;

  /**
   * @brief Construct a new LambdaTensorImpl object using a Lambda function
   *
   * @param [in] tis_vec vector of TiledIndexSpace objects for each mode of the
   * Tensor
   * @param [in] lambda a function for constructing the Tensor
   */
  DenseTensorImpl(const TiledIndexSpaceVec& tis_vec,
                  const ProcGrid proc_grid, const bool scalapack_distribution=false)
      : TensorImpl<T>(tis_vec), proc_grid_{proc_grid}, scalapack_distribution_{scalapack_distribution} {
      // check no dependences
      // for(auto& tis : tis_vec) { EXPECTS(!tis.is_dependent()); }

      // check index spaces are dense
  }

  DenseTensorImpl(const IndexLabelVec& til_vec,
                  const ProcGrid proc_grid, const bool scalapack_distribution=false) 
        : TensorImpl<T>(til_vec), proc_grid_{proc_grid}, scalapack_distribution_{scalapack_distribution}   {
    // check no dependences
    // check index spaces are dense
    // for(auto& til : til_vec) { EXPECTS(!til.is_dependent()); }
    
  }

  // Dtor
  ~DenseTensorImpl() {
    // EXPECTS(allocation_status_ == AllocationStatus::deallocated ||
    //         allocation_status_ == AllocationStatus::invalid);
  }

  void deallocate() {
    NGA_Destroy(ga_);
    update_status(AllocationStatus::deallocated);
  }

  void allocate(ExecutionContext* ec) {
    EXPECTS(allocation_status_ == AllocationStatus::invalid);

    ec_ = ec;
    ga_ = NGA_Create_handle();
    const int ndims = num_modes();

    auto tis_dims = tindices();
    std::vector<int64_t> dims;
    for(auto tis: tis_dims) dims.push_back(tis.index_space().num_indices());

    NGA_Set_data64(ga_,ndims,&dims[0],ga_eltype_);

    const bool is_irreg_tis1 = !tis_dims[0].input_tile_sizes().empty();
    const bool is_irreg_tis2 = !tis_dims[1].input_tile_sizes().empty();
    //std::cout << "#dims 0,1 = " << dims[0] << "," << dims[1] << std::endl;

    if (scalapack_distribution_) {
      // EXPECTS(ndims == 2);
      std::vector<int64_t> bsize(2);
      std::vector<int64_t> pgrid(2);
      {
        //TODO: grid_factor doesnt work
        if(proc_grid_.empty()){
          int idx, idy;
          grid_factor((*ec).pg().size().value(),&idx,&idy);
          proc_grid_.push_back(idx);
          proc_grid_.push_back(idy);
        }
        
        //Cannot provide list of irreg tile sizes
        EXPECTS(!is_irreg_tis1 && !is_irreg_tis2);
        
        bsize[0] = tis_dims[0].input_tile_size();
        bsize[1] = tis_dims[1].input_tile_size();

        pgrid[0] = proc_grid_[0].value();
        pgrid[1] = proc_grid_[1].value();
      }
      //blocks_ = block sizes for scalapack distribution
      GA_Set_block_cyclic_proc_grid64(ga_, &bsize[0], &pgrid[0]);
    } 
    else {
      //only needed when irreg tile sizes are provided
      if(is_irreg_tis1 || is_irreg_tis2){
        std::vector<Tile> tiles1 = 
          is_irreg_tis1? tis_dims[0].input_tile_sizes() : std::vector<Tile>{tis_dims[0].input_tile_size()};
        std::vector<Tile> tiles2 = 
          is_irreg_tis2? tis_dims[1].input_tile_sizes() : std::vector<Tile>{tis_dims[1].input_tile_size()};

        int64_t size_map;
        int64_t nblock[ndims];

        int nranks = (*ec).pg().size().value();
        int ranks_list[nranks];
        for (int i = 0; i < nranks; i++) ranks_list[i] = i;
    
        int idx, idy;
        grid_factor((*ec).pg().size().value(),&idx,&idy);        

        nblock[0] = is_irreg_tis1? tiles1.size(): std::ceil(dims[0]*1.0/tiles1[0]);
        nblock[1] = is_irreg_tis2? tiles2.size(): std::ceil(dims[1]*1.0/tiles2[0]);

        // int max_t1 = is_irreg_tis1? *max_element(tiles1.begin(), tiles1.end()) : tiles1[0];
        // int max_t2 = is_irreg_tis2? *max_element(tiles2.begin(), tiles2.end()) : tiles2[0];
        // int new_t1 = std::ceil(nblock[0]/idx)*max_t1;
        // int new_t2 = std::ceil(nblock[1]/idy)*max_t2;
        nblock[0] = (int64_t)idx;
        nblock[1] = (int64_t)idy;

        size_map = nblock[0]+nblock[1];

        // std::cout << "#blocks 0,1 = " << nblock[0] << "," << nblock[1] << std::endl;

        //create map
        std::vector<int64_t> k_map(size_map);
        {
          auto mi=0;
          for (auto idim=0;idim<ndims;idim++){
              auto size_blk = std::ceil(1.0*dims[idim]/nblock[idim]);
              //regular tile size
              for (auto i=0;i<nblock[idim];i++){
                k_map[mi] = size_blk*i;
                mi++;
              }
          }
          //k_map[mi] = 0;
        }
        GA_Set_irreg_distr64(ga_, &k_map[0], nblock);
      }
      else{
        //fixed tilesize for both dims
        int64_t chunk[2] = {tis_dims[0].input_tile_size(),tis_dims[1].input_tile_size()};
        GA_Set_chunk64(ga_,chunk);
      }
    }
    NGA_Allocate(ga_);
    update_status(AllocationStatus::created);
  }

  void get(const IndexVector& blockid, span<T> buff_span) const {
    std::vector<int64_t> lo = compute_lo(blockid);
    std::vector<int64_t> hi = compute_hi(blockid);
    std::vector<int64_t> ld = compute_ld(blockid);
    EXPECTS(block_size(blockid) <= buff_span.size());
    NGA_Get64(ga_, &lo[0], &hi[0], buff_span.data(), &ld[0]);
  }

  void put(const IndexVector& blockid, span<T> buff_span) {
    std::vector<int64_t> lo = compute_lo(blockid);
    std::vector<int64_t> hi = compute_hi(blockid);
    std::vector<int64_t> ld = compute_ld(blockid);

    EXPECTS(block_size(blockid) <= buff_span.size());
    NGA_Put64(ga_, &lo[0], &hi[0], buff_span.data(), &ld[0]);
  }

  void add(const IndexVector& blockid, span<T> buff_span) {
    std::vector<int64_t> lo = compute_lo(blockid);
    std::vector<int64_t> hi = compute_hi(blockid);
    std::vector<int64_t> ld = compute_ld(blockid);
    EXPECTS(block_size(blockid) <= buff_span.size());
    void* alpha;
    switch (from_ga_eltype(ga_eltype_)) {
      case ElementType::single_precision:
        alpha = reinterpret_cast<void*>(&sp_alpha);
        break;
      case ElementType::double_precision:
        alpha = reinterpret_cast<void*>(&dp_alpha);
        break;
      case ElementType::single_complex:
        alpha = reinterpret_cast<void*>(&scp_alpha);
        break;
      case ElementType::double_complex:
        alpha = reinterpret_cast<void*>(&dcp_alpha);
        break;
      case ElementType::invalid:
      default:
        UNREACHABLE();
    }
    NGA_Acc64(ga_, &lo[0], &hi[0], reinterpret_cast<void*>(buff_span.data()), &ld[0], alpha);
  }

  int ga_handle() { return ga_; }

 protected:
  std::vector<int64_t> compute_lo(const IndexVector& blockid) const {
    std::vector<int64_t> retv;
    std::vector<size_t> off = block_offsets(blockid);
    for (const auto& i : off) {
      retv.push_back(static_cast<int64_t>(i));
    }
    return retv;
  }

  std::vector<int64_t> compute_hi(const IndexVector& blockid) const {
    std::vector<int64_t> retv;
    std::vector<size_t> boff = block_offsets(blockid);
    std::vector<size_t> bdims = block_dims(blockid);
    for (size_t i = 0; i < boff.size(); i++) {
      retv.push_back(static_cast<int64_t>(boff[i] + bdims[i]-1));
    }
    return retv;
  }

  std::vector<int64_t> compute_ld(const IndexVector& blockid) const {
    std::vector<size_t> bdims = block_dims(blockid);
    std::vector<int64_t> retv(bdims.size()-1,1);
    size_t ri=0;
    for(size_t i=bdims.size()-1; i>0; i--) {
        retv[ri] = (int64_t)bdims[i];
        ri++;
    }
    return retv;
  }

  int ga_;
  ProcGrid proc_grid_;
  //true only when a ProcGrid is explicity passed to Tensor constructor
  bool scalapack_distribution_ = false;

  // constants for NGA_Acc call
  float sp_alpha = 1.0;
  double dp_alpha = 1.0;
  SingleComplex scp_alpha = {1, 0};
  DoubleComplex dcp_alpha = {1, 0};

  int ga_eltype_ = to_ga_eltype(tensor_element_type<T>());

  /**
 * Factor p processors into 2D processor grid of dimensions px, py
 */
void grid_factor(int p, int *idx, int *idy) {
  int i, j;
  const int MAX_FACTOR=512;
  int ip, ifac, pmax, prime[MAX_FACTOR];
  int fac[MAX_FACTOR];
  int ix, iy, ichk;

  i = 1;
  /**
   *   factor p completely
   *   first, find all prime numbers, besides 1, less than or equal to 
   *   the square root of p
   */
  ip = (int)(sqrt((double)p))+1;
  pmax = 0;
  for (i=2; i<=ip; i++) {
    ichk = 1;
    for (j=0; j<pmax; j++) {
      if (i%prime[j] == 0) {
        ichk = 0;
        break;
      }
    }
    if (ichk) {
      pmax = pmax + 1;
      if (pmax > MAX_FACTOR) printf("Overflow in grid_factor\n");
      prime[pmax-1] = i;
    }
  }
  /**
   *   find all prime factors of p
   */
  ip = p;
  ifac = 0;
  for (i=0; i<pmax; i++) {
    while(ip%prime[i] == 0) {
      ifac = ifac + 1;
      fac[ifac-1] = prime[i];
      ip = ip/prime[i];
    }
  }
  /**
   *  p is prime
   */
  if (ifac==0) {
    ifac++;
    fac[0] = p;
  }
  /**
   *    find two factors of p of approximately the
   *    same size
   */
  *idx = 1;
  *idy = 1;
  for (i = ifac-1; i >= 0; i--) {
    ix = *idx;
    iy = *idy;
    if (ix <= iy) {
      *idx = fac[i]*(*idx);
    } else {
      *idy = fac[i]*(*idy);
    }
  }
}


};  // class DenseTensorImpl


} // namespace tamm

#endif // TENSOR_IMPL_HPP_
