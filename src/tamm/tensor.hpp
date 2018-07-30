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
    using element_type = T;

    Tensor() : 
      impl_{std::make_shared<TensorImpl>()} {}


    /**
     * @brief Construct a new Tensor object from a set of TiledIndexSpace
     * objects as modes of the Tensor
     *
     * @param [in] tis set of TiledIndexSpace objects for each mode
     */
    // Tensor(std::initializer_list<TiledIndexSpace> tis) :
    //   impl_{std::make_shared<TensorImpl>(tis)} {}

    /**
     * @brief Construct a new Tensor object from a set of TiledIndexLabel
     * objects that are used to extract TiledIndexSpace information as
     * modes of the Tensor
     *
     * @param [in] tis set of TiledIndexLabel objects for each mode
     */
    // Tensor(const std::initializer_list<TiledIndexLabel>& lbls) :
    //   impl_{std::make_shared<TensorImpl>(lbls)} {}


    /**
     * @brief Construct a new Tensor object from a list of 
     * TiledIndexSpace/TiledIndexLabel objects as modes of the Tensor
     *
     * @param [in] tis set of TiledIndexSpace/TiledIndexLabel objects for each mode
     */
    template<class... Ts>
    Tensor(Ts... rest) :
      impl_{std::make_shared<TensorImpl>(rest...)} {}

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
     * @param [in] input TiledIndexLabels
     * @returns a LabeledTensor object created with the input arguments
     */

    template<class... Args>
    LabeledTensor<T> operator()(Args&&... rest) const {
        return LabeledTensor<T>{*this, std::forward<Args>(rest)...};
    }

    // template<class... Args>
    //  LabeledTensor<T> operator()(Args&&... rest) const; 
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

    LabelLoopNest loop_nest() const {
        return impl_->loop_nest();
    }

    size_t block_size(const IndexVector& blockid) const {
        return impl_->block_size(blockid);
    }

    std::vector<size_t> block_dims(const IndexVector& blockid) const {
        return impl_->block_dims(blockid);
    }

    const std::vector<TiledIndexSpace>& tiled_index_spaces() const {
        return impl_->tiled_index_spaces();
    }

    const std::map<size_t,std::vector<size_t>>& dep_map() const {
        return impl_->dep_map();
    }

    /**
     * @brief Memory allocation method for the Tensor object
     *
     */
    void alloc(const ExecutionContext* ec) { impl_->allocate<T>(ec); }

    /**
     * @brief Memory deallocation method for the Tensor object
     *
     */
    void dealloc() { impl_->deallocate(); }

    // Static methods for allocate/deallocate

    static void allocate(const ExecutionContext* ec) {}
    static void deallocate() {}
    /**
     * @brief Static memory allocation method for a set of Tensors
     *
     * @tparam Args variadic template for set of Tensor objects
     * @param [in] exec input ExecutionContext object to be used for
     * allocation
     * @param [in] rest set of Tensor objects to be allocated
     */
    template<typename... Args>
    static void allocate(const ExecutionContext* ec, Tensor<T>& tensor, Args& ... rest) {
       tensor.impl_->template allocate<T>(ec);
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

#endif // TENSOR_HPP_
