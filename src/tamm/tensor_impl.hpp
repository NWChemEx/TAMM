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
    T* ref_;
    size_t size_;

    span(T* ref, size_t size) : ref_{ref}, size_{size} {}
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

  void allocate(const ExecutionContext& ec) {
    Distribution* distribution = ec.distribution();
    MemoryManager* memory_manager = ec.memory_manager();
    EXPECTS(distribution != nullptr);
    EXPECTS(memory_manager != nullptr);
    // distribution_ = DistributionFactory::make_distribution(*distribution, this, pg.size());
    distribution_ = std::shared_ptr<Distribution>(
        distribution); //->clone(this,memory_manager->pg().size()));
    auto rank = memory_manager->pg().rank();
    auto buf_size = distribution_->buf_size(rank);
    auto eltype = tensor_element_type<double>();
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
    void get(const IndexVector& idx_vec, span<T> buff_span) const {}

    /**
     * @brief Tensor accessor method for putting values to a set of indices
     * with the specified memory span
     *
     * @tparam T type of the values hold on the tensor object
     * @param [in] idx_vec a vector of indices to put the values
     * @param [in] buff_span buff_span memory span for the values to put
     */
    template<typename T>
    void put(const IndexVector& idx_vec, span<T> buff_span) {}

    /**
     * @brief Tensor accessor method for adding svalues to a set of indices
     * with the specified memory span
     *
     * @tparam T type of the values hold on the tensor object
     * @param [in] idx_vec a vector of indices to put the values
     * @param [in] buff_span buff_span memory span for the values to put
     */
    template<typename T>
    void add(const IndexVector& idx_vec, span<T> buff_span) {}

protected:
    std::shared_ptr<Distribution> distribution_;
    std::unique_ptr<MemoryRegion> mpb_;
}; // TensorImpl

/**
 * @brief Templated Tensor class designed using PIMPL (pointer to
 * implementation) idiom. All of the implementation (except the static
 * methods) are done in TensorImpl class
 *
 * @tparam T type for the Tensor value
 */
template<typename T>
class Tensor {
public:
    using element_type = T;

    Tensor() : 
      impl_{std::make_shared<TensorImpl>()} {}

    /**
     * @brief Construct a new Tensor object from a set of TiledIndexSpace
     * objects as modes of the Tensor
     *
     * @param [in] tis set of TiledIndexSpace objects for each mode
     */
    Tensor(std::initializer_list<TiledIndexSpace> tis) :
      impl_{std::make_shared<TensorImpl>(tis)} {}

    /**
     * @brief Construct a new Tensor object from a set of TiledIndexLabel
     * objects that are used to extract TiledIndexSpace information as
     * modes of the Tensor
     *
     * @param [in] tis set of TiledIndexLabel objects for each mode
     */
    Tensor(const std::initializer_list<TiledIndexLabel>& lbls) :
      impl_{std::make_shared<TensorImpl>(lbls)} {}

    /**
     * @brief Construct a new Tensor object recursively with a set of
     * TiledIndexSpace objects followed by a lambda expression
     *
     * @tparam Ts variadic template for the input arguments
     * @param [in] tis TiledIndexSpace object for the corresponding mode of
     * the Tensor object
     * @param [in] rest remaining parts of the input arguments
     */
    template<class... Ts>
    Tensor(const TiledIndexSpace& tis, Ts... rest) :
      impl_{std::make_shared<TensorImpl>(tis, rest...)} {}

    /**
     * @brief Operator overload for constructing a LabeledTensor object
     * (main construct for Tensor operations)
     *
     * @returns a LabeledTensor object to be used in Tensor operations
     */
    // LabeledTensor<T> operator()() const { return {}; }

    /**
     * @brief Operator overload for constructing a LabeledTensor object with
     * input TiledIndexLabel objects (main construct for Tensor operations)
     *
     * @tparam Ts variadic template for set of input TiledIndexLabels
     * @param [in] inputs input TiledIndexLabels
     * @returns a LabeledTensro object created with the input arguments
     */
    template<class... Args>
     LabeledTensor<T> operator()(Args&&... rest) const; 
    //  {
    //      return LabeledTensor<T>{*this, std::forward<Args>(rest)...};
    // }

    // template <typename ...Args>
    // LabeledTensor<T> operator()(const std::string str, Args... rest) const {}
    // LabeledTensor<T> operator()(std::initializer_list<const std::string> lbl_strs) const {
    //     return LabeledTensor<T>(*this, lbl_strs);
    // }

    // Tensor Accessors
    /**
     * @brief Get method for Tensor values
     *
     * @param [in] idx_vec set of indices to get data from
     * @param [in] buff_span a memory span to write the fetched data
     */
    void get(IndexVector idx_vec, span<T> buff_span) const {
        impl_->get(idx_vec, buff_span);
    }

    /**
     * @brief Put method for Tensor values
     *
     * @param [in] idx_vec set of indices to put data to
     * @param [in] buff_span a memory span for the data to be put
     */
    void put(IndexVector idx_vec, span<T> buff_span) {
        impl_->put(idx_vec, buff_span);
    }

    /**
     * @brief Add method for Tensor values
     *
     * @param [in] idx_vec set of indices to put data to
     * @param [in] buff_span a memory span for the data to be put
     */
    void add(IndexVector idx_vec, span<T> buff_span) {
        impl_->add(idx_vec, buff_span);
    }

    IndexLoopNest loop_nest() const {
        return impl_->loop_nest();
    }

    size_t block_size(const IndexVector& blockid) const {
        return impl_->block_size(blockid);
    }

    /**
     * @brief Memory allocation method for the Tensor object
     *
     */
    static void allocate(const ExecutionContext& ec) { } // impl_->allocate(ec); 

    /**
     * @brief Memory deallocation method for the Tensor object
     *
     */
    static void deallocate() {} //impl_->deallocate(); }

    // Static methods for allocate/deallocate
    /**
     * @brief Static memory allocation method for a set of Tensors
     *
     * @tparam Args variadic template for set of Tensor objects
     * @param [in] exec input ExecutionContext object to be used for
     * allocation
     * @param [in] rest set of Tensor objects to be allocated
     */
    template<typename... Args>
    static void allocate(const ExecutionContext& ec, Tensor<T>& tensor, Args& ... rest) {
       tensor.impl_->allocate(ec);
       allocate(ec,rest...);
    }

    /**
     * @brief Static memory deallocation method for a set of Tensors
     *
     * @tparam Args variadic template for set of Tensor objects
     * @param [in] rest set of Tensor objects to be deallocated
     */
    template<typename... Args>
    static void deallocate(Tensor<T>& tensor, Args& ... rest) {
        tensor.impl_->deallocate();
        deallocate(rest...);
    }

    size_t num_modes() const {
        return impl_->num_modes();
    }

//private:
    std::shared_ptr<TensorImpl> impl_;
};

} // namespace tamm

#endif // TENSOR_IMPL_HPP_
