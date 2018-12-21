
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

} // namespace tamm

#endif // TENSOR_IMPL_HPP_
