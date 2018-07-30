#ifndef TAMM_TENSOR_IMPL_HPP_
#define TAMM_TENSOR_IMPL_HPP_

#include "tamm/index_space.hpp"
#include "tamm/labeled_tensor.hpp"
#include "tamm/tensor_base.hpp"
#include "tamm/memory_manager_local.hpp"
#include "tamm/distribution.hpp"
#include "tamm/execution_context.hpp"
#include "tamm/index_loop_nest.hpp"

namespace tamm {

template<typename T>
/**
 * @brief A struct for mimicing use of gsl::span
 *
 */
struct span {
public:
    span(T* ref, size_t size) : ref_{ref}, size_{size} {}

    const T* ref() const { return ref_; }

    T* ref() { return ref_; }
    
    size_t size() const { return size_; }

private:
    T* ref_;
    size_t size_;
};

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
     * @param [in] block_indices vector of TiledIndexSpace objects for each
     * mode used to construct the tensor
     */
    TensorImpl(const std::vector<TiledIndexSpace>& tis) : TensorBase{tis} {}

    /**
     * @brief Construct a new TensorImpl object using a vector of
     * TiledIndexSpace objects for each mode of the tensor
     *
     * @param [in] lbls vector of tiled index labels used for extracting
     * corresponding TiledIndexSpace objects for each mode used to construct
     * the tensor
     */
    TensorImpl(const std::vector<TiledIndexLabel>& lbls) : TensorBase{lbls} {}

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
    }


    // Copy/Move Ctors and Assignment Operators
    TensorImpl(TensorImpl&&)      = default;
    TensorImpl(const TensorImpl&) = default;
    TensorImpl& operator=(TensorImpl&&) = default;
    TensorImpl& operator=(const TensorImpl&) = default;

    // Dtor
    ~TensorImpl() = default;

  void deallocate() {
    EXPECTS(mpb_);
    mpb_->dealloc_coll();
  }

  template<typename T>
  void allocate(const ExecutionContext* ec) {
    Distribution* distribution = ec->distribution();
    MemoryManager* memory_manager = ec->memory_manager();
    EXPECTS(distribution != nullptr);
    EXPECTS(memory_manager != nullptr);
    // distribution_ = DistributionFactory::make_distribution(*distribution, this, pg.size());
    distribution_ = std::shared_ptr<Distribution>(
        distribution->clone(this,memory_manager->pg().size()));
    auto rank = memory_manager->pg().rank();
    auto buf_size = distribution_->buf_size(rank);
    auto eltype = tensor_element_type<T>();
    EXPECTS(buf_size >=0 );
    mpb_ = std::unique_ptr<MemoryRegion>{memory_manager->alloc_coll(eltype, buf_size)};
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
        Proc proc;
        Offset offset;
        std::tie(proc, offset) = distribution_->locate(idx_vec);
        Size size = block_size(idx_vec);
        EXPECTS(size <= buff_span.size());
        mpb_->mgr().get(*mpb_.get(), proc, offset, Size{size}, buff_span.ref());
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
        Proc proc;
        Offset offset;
        std::tie(proc, offset) = distribution_->locate(idx_vec);
        Size size = block_size(idx_vec);
        EXPECTS(size <= buff_span.size());
        mpb_->mgr().put(*mpb_.get(), proc, offset, Size{size}, buff_span.ref());
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
        Proc proc;
        Offset offset;
        std::tie(proc, offset) = distribution_->locate(idx_vec);
        Size size = block_size(idx_vec);
        EXPECTS(size <= buff_span.size());
        mpb_->mgr().add(*mpb_.get(), proc, offset, Size{size}, buff_span.ref());
    }

protected:
    std::shared_ptr<Distribution> distribution_;
    std::unique_ptr<MemoryRegion> mpb_;
}; // TensorImpl

} // namespace tamm

#endif // TENSOR_IMPL_HPP_
