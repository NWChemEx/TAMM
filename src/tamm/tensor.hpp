#ifndef TAMM_TENSOR_HPP_
#define TAMM_TENSOR_HPP_

#include "tamm/tensor_impl.hpp"

namespace tamm {

template<typename T>
class LabeledTensor;

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

    /**
     * @brief Construct a scalar Tensor with 0-modes
     *
     */
    Tensor() : impl_{std::make_shared<TensorImpl>()} {}

    /**
     * @brief Construct a new Tensor object from a vector of TiledIndexSpace
     * objects
     *
     * @param [in] tis_vec a vector of TiledIndexSpace objects for each mode
     */
    Tensor(std::vector<TiledIndexSpace> tis_vec) :
      impl_{std::make_shared<TensorImpl>(tis_vec)} {}

    /**
     * @brief Construct a new Tensor object from a vector of TiledIndexLabel
     * objects
     *
     * @param [in] til_vec a vector of TiledIndexLabel objects which will be
     * used to extract TiledIndexSpace for Tensor construction
     */
    Tensor(std::vector<TiledIndexLabel> til_vec) :
      impl_{std::make_shared<TensorImpl>(til_vec)} {}

#if 1
    // SpinTensor Constructors

    Tensor(TiledIndexSpaceVec t_spaces, SpinMask spin_mask) :
      impl_{std::make_shared<TensorImpl>(t_spaces, spin_mask)} {}
    Tensor(IndexLabelVec t_lbls, SpinMask spin_mask) :
      impl_{std::make_shared<TensorImpl>(t_lbls, spin_mask)} {}

    Tensor(TiledIndexSpaceVec t_spaces, std::vector<size_t> spin_sizes) :
      impl_{std::make_shared<TensorImpl>(t_spaces, spin_sizes)} {}
    Tensor(IndexLabelVec t_lbls, std::vector<size_t> spin_sizes) :
      impl_{std::make_shared<TensorImpl>(t_lbls, spin_sizes)} {}

#endif
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
     * @brief Operator overload for constructing a LabeledTensor object with
     * input TiledIndexLabel objects (main construct for Tensor operations)
     *
     * @tparam Ts variadic template for set of input TiledIndexLabels
     * @param [in] input TiledIndexLabels
     * @returns a LabeledTensor object created with the input arguments
     */

    template<class... Args>
    LabeledTensor<T> operator()(Args&&... rest) const {
        return LabeledTensor<T>{*this, std::forward<Args>(rest)...};
    }

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

    /**
     * @brief Constructs a LabeledLoopNest object  for Tensor object
     *
     * @returns a LabelLoopNest for the Tensor
     */
    LabelLoopNest loop_nest() const { return impl_->loop_nest(); }

    T trace() const { return impl_->template trace<T>(); }

    std::vector<T> diagonal() const { return impl_->template diagonal<T>(); }

    /**
     * @brief Get the size of a block
     *
     * @param [in] blockid The id of the block
     * @return size_t The size of the block
     */
    size_t block_size(const IndexVector& blockid) const {
        return impl_->block_size(blockid);
    }

    /**
     * @brief Get dimensions of a block
     *
     * @param [in] blockid The id of the block
     * @return std::vector<size_t>  The vector of dimensions
     */
    std::vector<size_t> block_dims(const IndexVector& blockid) const {
        return impl_->block_dims(blockid);
    }

    /**
     * @brief Get offsets of a block
     *
     * @param [in] blockid The id of the block
     * @returns std::vector<size_t> The vector of offsets
     */
    std::vector<size_t> block_offsets(const IndexVector& blockid) const {
        return impl_->block_offsets(blockid);
    }

    /**
     * @brief Get index spaces of a vector
     *
     * @return const std::vector<TiledIndexSpace>&
     */
    const std::vector<TiledIndexSpace>& tiled_index_spaces() const {
        return impl_->tiled_index_spaces();
    }

    /**
     * @brief Return dependency map of the tensor's index spaces
     *
     * @return const std::map<size_t,std::vector<size_t>>& The dependence map
     * that maps indices of index spaces to a vector of indices that each space
     * depends on.
     */
    const std::map<size_t, std::vector<size_t>>& dep_map() const {
        return impl_->dep_map();
    }

    /**
     * @brief Memory allocation method for the Tensor object
     *
     */
    void allocate(const ExecutionContext* ec) { impl_->allocate<T>(ec); }

    /**
     * @brief Memory deallocation method for the Tensor object
     *
     */
    void deallocate() { impl_->deallocate(); }

    // Static methods for allocate/deallocate
    /**
     * @brief Static memory allocation method for a set of Tensors
     *
     * @tparam Args variadic template for set of Tensor objects
     * @param [in] ec input ExecutionContext object to be used for
     * allocation
     * @param [in] rest set of Tensor objects to be allocated
     */
    template<typename... Args>
    static void allocate(const ExecutionContext* ec, Tensor<T>& tensor,
                         Args&... rest) {
        // tensor.impl_->template allocate<T>(ec);
        // allocate(ec, rest...);
        alloc(ec, tensor, rest...);
    }

    /**
     * @brief Static memory deallocation method for a set of Tensors
     *
     * @tparam Args variadic template for set of Tensor objects
     * @param [in] rest set of Tensor objects to be deallocated
     */
    template<typename... Args>
    static void deallocate(Tensor<T>& tensor, Args&... rest) {
        // tensor.impl_->deallocate();
        // deallocate(rest...);
        dealloc(tensor, rest...);
    }

    /**
     * @brief Get the number of modes of a Tensor
     *
     * @returns number of modes of a Tensor
     */
    size_t num_modes() const { return impl_->num_modes(); }

    bool is_non_zero(const IndexVector& blockid) const {
        return impl_->is_non_zero(blockid);
    }

private:
    std::shared_ptr<TensorImpl> impl_;

    // Private allocate and de-allocate functions

    static void alloc(const ExecutionContext* ec) {}
    static void dealloc() {}
    /**
     * @brief Static memory allocation method for a set of Tensors
     *
     * @tparam Args variadic template for set of Tensor objects
     * @param [in] ec input ExecutionContext object to be used for
     * allocation
     * @param [in] rest set of Tensor objects to be allocated
     */
    template<typename... Args>
    static void alloc(const ExecutionContext* ec, Tensor<T>& tensor,
                      Args&... rest) {
        tensor.impl_->template allocate<T>(ec);
        alloc(ec, rest...);
    }

    /**
     * @brief Static memory deallocation method for a set of Tensors
     *
     * @tparam Args variadic template for set of Tensor objects
     * @param [in] rest set of Tensor objects to be deallocated
     */
    template<typename... Args>
    static void dealloc(Tensor<T>& tensor, Args&... rest) {
        tensor.impl_->deallocate();
        dealloc(rest...);
    }
};

} // namespace tamm

#endif // TENSOR_HPP_
