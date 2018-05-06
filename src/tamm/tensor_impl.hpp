#ifndef TAMM_TENSOR_IMPL_HPP_
#define TAMM_TENSOR_IMPL_HPP_

#include "tamm/execution_context.hpp"
#include "tamm/index_space.hpp"
#include "tamm/labeled_tensor.hpp"

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

/**
 * @brief Base class for a tensor
 *
 */
class TensorBase {
public:
    // Ctors
    TensorBase() = default;

    /**
     * @brief Construct a new TensorBase object using a vector of
     * TiledIndexSpace objects for each mode of the tensor
     *
     * @param [in] block_indices vector of TiledIndexSpace objects for each mode
     * used to construct the tensor
     */
    TensorBase(const std::vector<TiledIndexSpace>& block_indices) :
      block_indices_{block_indices} {}

    /**
     * @brief Construct a new TensorBase object using a vector of
     * TiledIndexSpace objects for each mode of the tensor
     *
     * @param [in] lbls vector of tiled index labels used for extracting
     * corresponding TiledIndexSpace objects for each mode used to construct the
     * tensor
     */
    TensorBase(const std::vector<TiledIndexLabel>& lbls) {
        for(const auto& lbl : lbls) {
            block_indices_.push_back(lbl.tiled_index_space());
        }
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
    TensorBase(const TiledIndexSpace& tis, Ts... rest) : TensorBase{rest...} {
        block_indices_.insert(block_indices_.begin(), tis);
    }

    /**
     * @brief Construct a new TensorBase object from a single TiledIndexSpace
     * object and a lambda expression
     *
     * @tparam Func template for lambda expression
     * @param [in] tis TiledIndexSpace object used as the mode of the tensor
     * @param [in] func lambda expression
     */
    template<typename Func>
    TensorBase(const TiledIndexSpace& tis, const Func& func) {
        block_indices_.insert(block_indices_.begin(), tis);
    }

    // Dtor
    virtual ~TensorBase(){};

    /**
     * @brief Method for getting the number of modes (rank) of the tensor
     *
     * @returns a TensorRank typed number as the rank of the tensor
     */
    virtual TensorRank rank() const = 0;

    /**
     * @brief Memory allocation method for the tensor object
     *
     */
    virtual void allocate() = 0;

    /**
     * @brief Memory deallocation method for the tensor object
     *
     */
    virtual void deallocate() = 0;

protected:
    std::vector<TiledIndexSpace> block_indices_;
    Spin spin_total_;
    bool has_spatial_symmetry_;
    bool has_spin_symmetry_;

    // std::vector<IndexPosition> ipmask_;
    // PermGroup perm_groups_;
    // Irrep irrep_;
    // std::vector<SpinMask> spin_mask_;
}; // TensorBase

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
     * @param [in] block_indices vector of TiledIndexSpace objects for each mode
     * used to construct the tensor
     */
    TensorImpl(const std::vector<TiledIndexSpace>& tis) :
      TensorBase{tis},
      rank_{tis.size()} {}

    /**
     * @brief Construct a new TensorImpl object using a vector of
     * TiledIndexSpace objects for each mode of the tensor
     *
     * @param [in] lbls vector of tiled index labels used for extracting
     * corresponding TiledIndexSpace objects for each mode used to construct the
     * tensor
     */
    TensorImpl(const std::vector<TiledIndexLabel>& lbls) :
      TensorBase{lbls},
      rank_{lbls.size()} {}

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
        rank_ = block_indices_.size();
    }

    // Copy/Move Ctors and Assignment Operators
    TensorImpl(TensorImpl&&)      = default;
    TensorImpl(const TensorImpl&) = default;
    TensorImpl& operator=(TensorImpl&&) = default;
    TensorImpl& operator=(const TensorImpl&) = default;

    // Dtor
    ~TensorImpl() = default;

    // Overriden methods
    TensorRank rank() const override { return rank_; }
    void allocate() override {}
    void deallocate() override {}

    // Tensor Accessors
    /**
     * @brief Tensor accessor method for getting values from a set of indices to
     * specified memory span
     *
     * @tparam T type of the values hold on the tensor object
     * @param [in] idx_vec a vector of indices to fetch the values
     * @param [in] buff_span memory span where to put the fetched values
     */
    template<typename T>
    void get(const IndexVector& idx_vec, span<T> buff_span) const {}

    /**
     * @brief Tensor accessor method for putting values to a set of indices with
     * the specified memory span
     *
     * @tparam T type of the values hold on the tensor object
     * @param [in] idx_vec a vector of indices to put the values
     * @param [in] buff_span buff_span memory span for the values to put
     */
    template<typename T>
    void put(const IndexVector& idx_vec, span<T> buff_span) {}

protected:
    //using TensorBase::TensorBase;

    TensorRank rank_;
}; // TensorImpl

/**
 * @brief Templated Tensor class designed using PIMPL (pointer to
 * implementation) idiom. All of the implementation (except the static methods)
 * are done in TensorImpl class
 *
 * @tparam T type for the Tensor value
 */
template<typename T>
class Tensor {
public:
    Tensor() = default;

    /**
     * @brief Construct a new Tensor object from a set of TiledIndexSpace
     * objects as modes of the Tensor
     *
     * @param [in] tis set of TiledIndexSpace objects for each mode
     */
    Tensor(const std::initializer_list<TiledIndexSpace>& tis) :
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
     * @param [in] tis TiledIndexSpace object for the corresponding mode of the
     * Tensor object
     * @param [in] rest remaining parts of the input arguments
     */
    template<class... Ts>
    Tensor(const TiledIndexSpace& tis, Ts... rest) :
      impl_{std::make_shared<TensorImpl>(tis, rest...)} {}

    /**
     * @brief Operator overload for constructing a LabeledTensor object (main
     * construct for Tensor operations)
     *
     * @returns a LabeledTensor object to be used in Tensor operations
     */
    LabeledTensor<T> operator()() const { return {}; }

    /**
     * @brief Operator overload for constructing a LabeledTensor object with
     * input TiledIndexLabel objects (main construct for Tensor operations)
     *
     * @tparam Ts variadic template for set of input TiledIndexLabels
     * @param [in] inputs input TiledIndexLabels
     * @returns a LabeledTensro object created with the input arguments
     */
    template<class... Ts>
    LabeledTensor<T> operator()(Ts... inputs) const {
        // return LabeledTensor<T>{*this, IndexLabelVec{inputs...}};
        return {};
    }

    /**
     * @brief Memory allocation method for the Tensor object
     *
     */
    void allocate() { impl_->allocate(); }

    /**
     * @brief Memory deallocation method for the Tensor object
     *
     */
    void deallocate() { impl_->deallocate(); }

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

    // Static methods for allocate/deallocate
    /**
     * @brief Static memory allocation method for a set of Tensors
     *
     * @tparam Args variadic template for set of Tensor objects
     * @param [in] exec input ExecutionContext object to be used for allocation
     * @param [in] rest set of Tensor objects to be allocated
     */
    template<typename... Args>
    static void allocate(const ExecutionContext& exec, Args... rest) {}

    /**
     * @brief Static memory deallocation method for a set of Tensors
     *
     * @tparam Args variadic template for set of Tensor objects
     * @param [in] rest set of Tensor objects to be deallocated
     */
    template<typename... Args>
    static void deallocate(Args... rest) {}

private:
    std::shared_ptr<TensorImpl> impl_;
};

} // namespace tamm

#endif // TENSOR_IMPL_HPP_
