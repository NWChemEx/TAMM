#ifndef TAMM_TENSOR_IMPL_HPP_
#define TAMM_TENSOR_IMPL_HPP_

#include "tamm/distribution.hpp"
#include "tamm/execution_context.hpp"
#include "tamm/index_loop_nest.hpp"
#include "tamm/index_space.hpp"
#include "tamm/labeled_tensor.hpp"
#include "tamm/memory_manager_local.hpp"
#include "tamm/tensor_base.hpp"

#include <gsl/span>
#include <type_traits>

namespace tamm {

// template<typename T>
// struct span;
using gsl::span;

#if 0
namespace detail {
    template <class T>
    struct is_span_helper : std::false_type {};
    template <class T>
    struct is_span_helper<span<T>> : std::true_type {};
    template <class T>
    struct is_span : public is_span_helper<std::remove_cv_t<T>> {};
    template <class T>
    struct is_std_array_helper : std::false_type {};
    template <class T, std::size_t N>
    struct is_std_array_helper<std::array<T, N>> : std::true_type {};
    template <class T>
    struct is_std_array : public is_std_array_helper<std::remove_cv_t<T>> { };
}

/**
 * @brief A struct for mimicing use of \c gsl::span and \c std::span
 *
 * @note This class will be made compatible with C++20 \c std::span or it
 * will be replaced by it.
 *
 * @tparam T Element type; must be a complete type that is not an abstract class
 * type.
 */
template<typename T>
struct span {
public:
    /**
     * @brief Construct a new span object from a pointer and a size
     * 
     * @param ref Pointer to elements.
     * @param size The amount of elements stored.
     */
    span(T* ref, size_t size) : ref_{ref}, size_{size} {}

    /**
     * @brief Construct a new span object from a container
     *
     * @note In C++17, we would use \c std::data() and \c std::size()
     * instead of member functions
     *
     * @tparam Type of the container The container to construct the span from.  
     * @param c The container to construct the span from.
     */
    template<typename Container,
             typename = std::enable_if_t<
                        !detail::is_span<Container>::value && !detail::is_std_array<Container>::value &&
                        std::is_convertible<typename Container::pointer, T*>::value &&
                        std::is_convertible<typename Container::pointer,
                                            decltype(std::declval<Container>().data())>::value>>
    span(Container& c) : ref_{c.data()}, size_{c.size()} {}

    const T* ref() const { return ref_; }

    T* ref() { return ref_; }
    
    size_t size() const { return size_; }

private:
    T* ref_;
    size_t size_;
};
// C++17 deduction guides
// template<class Container>
// span(Container&) -> span<typename Container::value_type>;
// template<class Container>
// span(const Container&) -> span<const typename Container::value_type>;
#endif
template<typename T>
class LabeledTensor;

/**
 * @brief Implementation class for TensorBase class
 *
 */
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
    TensorImpl(const TiledIndexSpaceVec& tis) : TensorBase{tis} 
        { has_spin_symmetry_ = false; }

    /**
     * @brief Construct a new TensorImpl object using a vector of
     * TiledIndexSpace objects for each mode of the tensor
     *
     * @param [in] lbls vector of tiled index labels used for extracting
     * corresponding TiledIndexSpace objects for each mode used to construct
     * the tensor
     */
    TensorImpl(const std::vector<TiledIndexLabel>& lbls) : TensorBase{lbls} 
        { has_spin_symmetry_ = false; }

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

#if 1
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
        size_t lower = spin_sizes.size() > 1 ? spin_sizes[1] : t_spaces.size() - upper;
        size_t ignore = spin_sizes.size() > 2 ? spin_sizes[1] : t_spaces.size() - (upper + lower);

        for (size_t i = 0; i < upper; i++) {
            spin_mask.push_back(SpinPosition::upper);
        }

        for (size_t i = 0; i < lower; i++) {
            spin_mask.push_back(SpinPosition::lower);
        }

        for (size_t i = 0; i < upper; i++) {
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
        size_t lower = spin_sizes.size() > 1 ? spin_sizes[1] : t_labels.size() - upper;
        size_t ignore = spin_sizes.size() > 2 ? spin_sizes[1] : t_labels.size() - (upper + lower);

        for (size_t i = 0; i < upper; i++) {
            spin_mask.push_back(SpinPosition::upper);
        }

        for (size_t i = 0; i < lower; i++) {
            spin_mask.push_back(SpinPosition::lower);
        }

        for (size_t i = 0; i < upper; i++) {
            spin_mask.push_back(SpinPosition::ignore);
        }

        spin_mask_         = spin_mask;
        has_spin_symmetry_ = true;
        // spin_total_        = calculate_spin();
    }
#endif

    // Copy/Move Ctors and Assignment Operators
    TensorImpl(TensorImpl&&)      = default;
    TensorImpl(const TensorImpl&) = default;
    TensorImpl& operator=(TensorImpl&&) = default;
    TensorImpl& operator=(const TensorImpl&) = default;

    // Dtor
    ~TensorImpl() = default;

    void deallocate() {
        EXPECTS(allocation_status_ != AllocationStatus::invalid);
        EXPECTS(mpb_);
        mpb_->dealloc_coll();
        update_status(AllocationStatus::invalid);
    }

    template<typename T>
    void allocate(const ExecutionContext* ec) {
        EXPECTS(allocation_status_ == AllocationStatus::invalid);
        Distribution* distribution    = ec->distribution();
        MemoryManager* memory_manager = ec->memory_manager();
        EXPECTS(distribution != nullptr);
        EXPECTS(memory_manager != nullptr);
        // distribution_ = DistributionFactory::make_distribution(*distribution,
        // this, pg.size());
        distribution_ = std::shared_ptr<Distribution>(
          distribution->clone(this, memory_manager->pg().size()));
        auto rank     = memory_manager->pg().rank();
        auto buf_size = distribution_->buf_size(rank);
        auto eltype   = tensor_element_type<T>();
        EXPECTS(buf_size >= 0);
        mpb_ = std::unique_ptr<MemoryRegion>{
          memory_manager->alloc_coll(eltype, buf_size)};

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
    template<typename T>
    void get(const IndexVector& idx_vec, span<T> buff_span) const {
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
        mpb_->mgr().get(*mpb_.get(), proc, offset, Size{size},
                        buff_span.data());
    }

    /**
     * @brief Tensor accessor method for putting values to a set of indices
     * with the specified memory span
     *
     * @tparam T type of the values hold on the tensor object
     * @param [in] idx_vec a vector of indices to put the values
     * @param [in] buff_span buff_span memory span for the values to put
     */
    template<typename T>
    void put(const IndexVector& idx_vec, span<T> buff_span) {
        EXPECTS(allocation_status_ != AllocationStatus::invalid);

        if(!is_non_zero(idx_vec)) { return; }

        Proc proc;
        Offset offset;
        std::tie(proc, offset) = distribution_->locate(idx_vec);
        Size size              = block_size(idx_vec);
        EXPECTS(size <= buff_span.size());
        mpb_->mgr().put(*mpb_.get(), proc, offset, Size{size},
                        buff_span.data());
    }

    /**
     * @brief Tensor accessor method for adding svalues to a set of indices
     * with the specified memory span
     *
     * @tparam T type of the values hold on the tensor object
     * @param [in] idx_vec a vector of indices to put the values
     * @param [in] buff_span buff_span memory span for the values to put
     */
    template<typename T>
    void add(const IndexVector& idx_vec, span<T> buff_span) {
        EXPECTS(allocation_status_ != AllocationStatus::invalid);

        if(!is_non_zero(idx_vec)) { return; }
        
        Proc proc;
        Offset offset;
        std::tie(proc, offset) = distribution_->locate(idx_vec);
        Size size              = block_size(idx_vec);
        EXPECTS(size <= buff_span.size());
        mpb_->mgr().add(*mpb_.get(), proc, offset, Size{size},
                        buff_span.data());
    }

    template<typename T>
    T trace() const {
        EXPECTS(num_modes() == 2);
        T ts = 0;
        for(const IndexVector& blockid : loop_nest()) {
          if(blockid[0] == blockid[1]) {
              const TAMM_SIZE size = block_size(blockid);
              std::vector<T> buf(size);
              get<T>(blockid, buf);
              auto block_dims1   = block_dims(blockid);
              auto block_offset = block_offsets(blockid);
              auto dim          = block_dims1[0];
              auto offset       = block_offset[0];
              size_t i          = 0;
              for(auto p = offset; p < offset + dim; p++, i++) {
                  ts += buf[i * dim + i];
              }
          }
      }
      return ts;
    }

    template<typename T>
    std::vector<T> diagonal() {
        EXPECTS(num_modes() == 2);
        std::vector<T> dest;
        for(const IndexVector& blockid : loop_nest()) {
          if(blockid[0] == blockid[1]) {
              const TAMM_SIZE size = block_size(blockid);
              std::vector<T> buf(size);
              get<T>(blockid, buf);
              auto block_dims1   = block_dims(blockid);
              auto block_offset = block_offsets(blockid);
              auto dim          = block_dims1[0];
              auto offset       = block_offset[0];
              size_t i          = 0;
              for(auto p = offset; p < offset + dim; p++, i++) {
                  dest.push_back(buf[i * dim + i]);
              }
          }
      }
      return dest;
    }

protected:
    std::shared_ptr<Distribution> distribution_;
    std::unique_ptr<MemoryRegion> mpb_;
}; // TensorImpl

class SpinTensorImpl : public TensorBase {
public:
    // Ctors
    SpinTensorImpl() = default;

    /**
     * @brief Construct a new SpinTensorImpl object using set of TiledIndexSpace
     * objects and Spin attribute mask
     *
     * @param [in] t_spaces
     * @param [in] spin_mask
     */
    SpinTensorImpl(const TiledIndexSpaceVec& t_spaces,
                   const SpinMask& spin_mask) :
      TensorBase(t_spaces) {
        EXPECTS(t_spaces.size() == spin_mask.size());

        for(const auto& tis : t_spaces) { EXPECTS(tis.has_spin()); }

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
    SpinTensorImpl(const IndexLabelVec& t_labels, const SpinMask& spin_mask) :
      TensorBase(t_labels) {
        EXPECTS(t_labels.size() == spin_mask.size());
        for(const auto& tlbl : t_labels) {
            EXPECTS(tlbl.tiled_index_space().has_spin());
        }
        spin_mask_         = spin_mask;
        has_spin_symmetry_ = true;
        // spin_total_        = calculate_spin();
    }

    SpinTensorImpl(IndexLabelVec&& t_labels, SpinMask&& spin_mask);
    /**
     * @brief Construct a new SpinTensorImpl object
     *
     * @todo: implement
     *
     * @tparam Args
     * @param [in] tis
     * @param [in] rest
     */
    template<typename... Args>
    SpinTensorImpl(const TiledIndexSpace& tis, Args... rest);

    /**
     * @brief Construct a new SpinTensorImpl object
     *
     * @todo: implement
     *
     * @tparam Args
     * @param [in] tis
     * @param [in] rest
     */
    template<typename... Args>
    SpinTensorImpl(const TiledIndexLabel& tis, Args... rest);

    // Copy/Move Ctors and Assignment Operators
    SpinTensorImpl(SpinTensorImpl&&)      = default;
    SpinTensorImpl(const SpinTensorImpl&) = default;
    SpinTensorImpl& operator=(SpinTensorImpl&&) = default;
    SpinTensorImpl& operator=(const SpinTensorImpl&) = default;

    // Dtor
    ~SpinTensorImpl() = default;

    // Tensor Allocate/Deallocate
    void deallocate() {
        EXPECTS(allocation_status_ != AllocationStatus::invalid);
        EXPECTS(mpb_);
        mpb_->dealloc_coll();
        update_status(AllocationStatus::invalid);
    }

    template<typename T>
    void allocate(const ExecutionContext* ec) {
        EXPECTS(allocation_status_ == AllocationStatus::invalid);

        Distribution* distribution    = ec->distribution();
        MemoryManager* memory_manager = ec->memory_manager();
        EXPECTS(distribution != nullptr);
        EXPECTS(memory_manager != nullptr);
        // distribution_ = DistributionFactory::make_distribution(*distribution,
        // this, pg.size());
        distribution_ = std::shared_ptr<Distribution>(
          distribution->clone(this, memory_manager->pg().size()));
        auto rank     = memory_manager->pg().rank();
        auto buf_size = distribution_->buf_size(rank);
        auto eltype   = tensor_element_type<T>();
        EXPECTS(buf_size >= 0);
        mpb_ = std::unique_ptr<MemoryRegion>{
          memory_manager->alloc_coll(eltype, buf_size)};

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
    template<typename T>
    void get(const IndexVector& idx_vec, span<T> buff_span) const {
        EXPECTS(allocation_status_ != AllocationStatus::invalid);

        if(!is_non_zero(idx_vec)) {
            Size size = block_size(idx_vec);
            EXPECTS(size <= buff_span.size());
            for(size_t i = 0; i < size; i++) { buff_span[i] = T{0}; }
        }

        Proc proc;
        Offset offset;
        std::tie(proc, offset) = distribution_->locate(idx_vec);
        Size size              = block_size(idx_vec);
        EXPECTS(size <= buff_span.size());
        mpb_->mgr().get(*mpb_.get(), proc, offset, Size{size},
                        buff_span.data());
    }

    /**
     * @brief Tensor accessor method for putting values to a set of indices
     * with the specified memory span
     *
     * @tparam T type of the values hold on the tensor object
     * @param [in] idx_vec a vector of indices to put the values
     * @param [in] buff_span buff_span memory span for the values to put
     */
    template<typename T>
    void put(const IndexVector& idx_vec, span<T> buff_span) {
        EXPECTS(allocation_status_ != AllocationStatus::invalid);

        if(!is_non_zero(idx_vec)) { return; }

        Proc proc;
        Offset offset;
        std::tie(proc, offset) = distribution_->locate(idx_vec);
        Size size              = block_size(idx_vec);
        EXPECTS(size <= buff_span.size());
        mpb_->mgr().put(*mpb_.get(), proc, offset, Size{size},
                        buff_span.data());
    }

    /**
     * @brief Tensor accessor method for adding svalues to a set of indices
     * with the specified memory span
     *
     * @tparam T type of the values hold on the tensor object
     * @param [in] idx_vec a vector of indices to put the values
     * @param [in] buff_span buff_span memory span for the values to put
     */
    template<typename T>
    void add(const IndexVector& idx_vec, span<T> buff_span) {
        EXPECTS(allocation_status_ != AllocationStatus::invalid);

        if(!is_non_zero(idx_vec)) { return; }

        Proc proc;
        Offset offset;
        std::tie(proc, offset) = distribution_->locate(idx_vec);
        Size size              = block_size(idx_vec);
        EXPECTS(size <= buff_span.size());
        mpb_->mgr().add(*mpb_.get(), proc, offset, Size{size},
                        buff_span.data());
    }

protected:
    std::shared_ptr<Distribution> distribution_;
    std::unique_ptr<MemoryRegion> mpb_;

    /**
     * @brief Calculates the spin value for the given TiledIndexSpace objects
     *
     * @todo: implement
     *
     * @returns
     */
    Spin calculate_spin() {
        Spin ret;

        return ret;
    }
}; // class SpinTensorImpl

} // namespace tamm

#endif // TENSOR_IMPL_HPP_
