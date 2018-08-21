#ifndef TAMM_SPIN_TENSOR_HPP_
#define TAMM_SPIN_TENSOR_HPP_

#include "tamm/tensor_impl.hpp"

namespace tamm {

template<typename T>
class LabeledTensor;

template <typename T>
class SpinTensor {
public:
    // Ctors
    SpinTensor() = default;

    template<typename... Params>
    SpinTensor(Params&&... params) :
      impl_{std::make_shared<SpinTensorImpl>(std::forward<Params>(params)...)} {
    }

    // Copy Ctor and Assignment Operator
    SpinTensor(const SpinTensor&) = default;
    SpinTensor& operator=(const SpinTensor&) = default;

    // Dtor
    ~SpinTensor() = default;


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
     * @return const TiledIndexSpaceVec&
     */
    const TiledIndexSpaceVec& tiled_index_spaces() const {
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
    static void allocate(const ExecutionContext* ec, SpinTensor<T>& tensor,
                         Args&... rest) {
        alloc(ec, tensor, rest...);
    }

    /**
     * @brief Static memory deallocation method for a set of Tensors
     *
     * @tparam Args variadic template for set of Tensor objects
     * @param [in] rest set of Tensor objects to be deallocated
     */
    template<typename... Args>
    static void deallocate(SpinTensor<T>& tensor, Args&... rest) {
        dealloc(tensor, rest...);
    }

    /**
     * @brief Get the number of modes of a Tensor
     * 
     * @returns number of modes of a Tensor
     */
    size_t num_modes() const { return impl_->num_modes(); }

private:
    std::shared_ptr<SpinTensorImpl> impl_;

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
    static void alloc(const ExecutionContext* ec, SpinTensor<T>& tensor,
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
    static void dealloc(SpinTensor<T>& tensor, Args&... rest) {
        tensor.impl_->deallocate();
        dealloc(rest...);
    }
}; // class SpinTensor

} // namespace tamm

#endif // TAMM_SPIN_TENSOR_HPP_