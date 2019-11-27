#ifndef TAMM_TENSOR_HPP_
#define TAMM_TENSOR_HPP_

#include "tamm/tensor_impl.hpp"

namespace tamm {

/**
 * @ingroup tensors
 * @brief Templated Tensor class designed using PIMPL (pointer to
 * implementation) idiom. All of the implementation (except the static
 * methods) are done in TensorImpl<T> class
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
    Tensor() : impl_{std::make_shared<TensorImpl<T>>()} {}

    /**
     * @brief Construct a new Tensor object from a vector of TiledIndexSpace
     * objects
     *
     * @param [in] tis_vec a vector of TiledIndexSpace objects for each mode
     */
    Tensor(std::vector<TiledIndexSpace> tis_vec) {
      // bool is_sparse = false;

      // for(const auto& tis : tis_vec) {
      //   is_sparse = tis.is_dependent();
      //   if(is_sparse)
      //     break;
      // }

      // if(!is_sparse && tis_vec.size() == 2) {
      //   impl_ = std::make_shared<DenseTensorImpl<T>>(tis_vec,ProcGrid{});
      // }
      // else {
         impl_ = std::make_shared<TensorImpl<T>>(tis_vec);
      //}
    }
    
    Tensor(ProcGrid pg, std::vector<TiledIndexSpace> tis_vec) {
      EXPECTS(tis_vec.size() == 2); 
      impl_ = std::make_shared<DenseTensorImpl<T>>(tis_vec, pg, true);
    }

    /**
     * @brief Construct a new Tensor object from a vector of TiledIndexLabel
     * objects
     *
     * @param [in] til_vec a vector of TiledIndexLabel objects which will be
     * used to extract TiledIndexSpace for Tensor construction
     */

    Tensor(std::vector<TiledIndexLabel> til_vec) {
      // bool is_sparse = false;
      // for(const auto& til : til_vec) {
      //   is_sparse = til.is_dependent();
      //   if(is_sparse)
      //     break;
      // }

      // if(!is_sparse && til_vec.size() == 2) {
      //   impl_ = std::make_shared<DenseTensorImpl<T>>(til_vec,ProcGrid{});
      // }
      // else {
         impl_ = std::make_shared<TensorImpl<T>>(til_vec);
      //}
    }

    Tensor(ProcGrid pg, std::vector<TiledIndexLabel> til_vec) {
      EXPECTS(til_vec.size() == 2); 
      impl_ = std::make_shared<DenseTensorImpl<T>>(til_vec, pg, true);
    }


    // SpinTensor Constructors
    /**
     * @brief Construct a new Tensor object with Spin attributes
     *
     * @param [in] t_spaces vector of TiledIndexSpace objects for each mode
     * @param [in] spin_mask spin mask for each mode
     */
    Tensor(TiledIndexSpaceVec t_spaces, SpinMask spin_mask) :
      impl_{std::make_shared<TensorImpl<T>>(t_spaces, spin_mask)} {}

    /**
     * @brief Construct a new Tensor object with Spin attributes
     *
     * @param [in] t_spaces initializer list of TiledIndexSpace objects for each
     * mode
     * @param [in] spin_mask spin mask for each mode
     */
    Tensor(std::initializer_list<TiledIndexSpace> t_spaces,
           SpinMask spin_mask) :
      impl_{std::make_shared<TensorImpl<T>>(t_spaces, spin_mask)} {}

    /**
     * @brief Construct a new Tensor object with Spin attributes
     *
     * @param [in] t_lbls vector of TiledIndexLabel objects for each mode
     * @param [in] spin_mask spin mask for each mode
     */
    Tensor(IndexLabelVec t_lbls, SpinMask spin_mask) :
      impl_{std::make_shared<TensorImpl<T>>(t_lbls, spin_mask)} {}

    /**
     * @brief Construct a new Tensor object with Spin attributes
     *
     * @param [in] t_spaces vector of TiledIndexSpace objects for each mode
     * @param [in] spin_sizes vector of sizes for each spin attribute
     */
    Tensor(TiledIndexSpaceVec t_spaces, std::vector<size_t> spin_sizes) :
      impl_{std::make_shared<TensorImpl<T>>(t_spaces, spin_sizes)} {}

    /**
     * @brief Construct a new Tensor object with Spin attributes
     *
     * @param [in] t_spaces initializer list of TiledIndexSpace objects for each
     * mode
     * @param [in] spin_sizes vector of sizes for each spin attribute
     */
    Tensor(std::initializer_list<TiledIndexSpace> t_spaces,
           std::vector<size_t> spin_sizes) :
      impl_{std::make_shared<TensorImpl<T>>(t_spaces, spin_sizes)} {}

    /**
     * @brief Construct a new Tensor object with Spin attributes
     *
     * @param [in] t_spaces vector of TiledIndexLabel objects for each mode
     * @param [in] spin_sizes vector of sizes for each spin attribute
     */
    Tensor(IndexLabelVec t_lbls, std::vector<size_t> spin_sizes) :
      impl_{std::make_shared<TensorImpl<T>>(t_lbls, spin_sizes)} {}

    // LambdaTensor Constructors
    /**
     * @brief Signature description for Lambda functor
     *
     */
    using Func = std::function<void(const IndexVector&, span<T>)>;

    /**
     * @brief Construct a new Tensor object with a Lambda function
     *
     * @param [in] t_spaces vector of TiledIndexSpace objects for each mode
     * @param [in] lambda Lambda function for Tensor construction
     * @warning Tensor objects constructed using Lambda function is a read-only
     * Tensor. It can only be on right-hand side of an equation and @param
     * lambda will be called as a get access on the Tensor.
     */
    Tensor(TiledIndexSpaceVec t_spaces, Func lambda) :
      impl_{std::make_shared<LambdaTensorImpl<T>>(t_spaces, lambda)} {}

    Tensor(IndexLabelVec t_labels, Func lambda) :
      impl_{std::make_shared<LambdaTensorImpl<T>>(t_labels, lambda)} {}

    /**
     * @brief Construct a new Tensor object with a Lambda function
     *
     * @param [in] t_spaces initializer list  of TiledIndexSpace objects for
     * each mode
     * @param [in] lambda Lambda function for Tensor construction
     * @warning Tensor objects constructed using Lambda function is a read-only
     * Tensor. It can only be on right-hand side of an equation and @param
     * lambda will be called as a get access on the Tensor.
     */
    Tensor(std::initializer_list<TiledIndexSpace> t_spaces, Func lambda) :
      impl_{std::make_shared<LambdaTensorImpl<T>>(t_spaces, lambda)} {}

    Tensor(std::initializer_list<TiledIndexLabel> t_labels, Func lambda) :
      impl_{std::make_shared<LambdaTensorImpl<T>>(t_labels, lambda)} {}

    /**
     * @brief Construct a new Tensor object from a set of TiledIndexSpace
     * objects as modes of the Tensor
     *
     * @param [in] tis set of TiledIndexSpace objects for each mode
     */
    Tensor(std::initializer_list<TiledIndexSpace> tis) {
      // bool is_sparse = false;

      // for(const auto& t : tis) {
      //   is_sparse = t.is_dependent();
      //   if(is_sparse)
      //     break;
      // }

      // if(!is_sparse && tis.size() == 2) {
      //   impl_ = std::make_shared<DenseTensorImpl<T>>(tis,ProcGrid{});
      // }
      // else {
         impl_ = std::make_shared<TensorImpl<T>>(tis);
     //}
    }
    
    Tensor(ProcGrid pg, std::initializer_list<TiledIndexSpace> tis) {
      EXPECTS(tis.size() == 2); 
      impl_ = std::make_shared<DenseTensorImpl<T>>(tis, pg, true);
    }

    /**
     * @brief Construct a new Tensor object from a set of TiledIndexLabel
     * objects that are used to extract TiledIndexSpace information as
     * modes of the Tensor
     *
     * @param [in] tis set of TiledIndexLabel objects for each mode
     */
    Tensor(const std::initializer_list<TiledIndexLabel>& lbls) {
      // bool is_sparse = false;

      // for(const auto& tlbl : lbls) {
      //   is_sparse = tlbl.is_dependent();
      //   if(is_sparse)
      //     break;
      // }

      // if(!is_sparse && lbls.size() == 2) {
      //   impl_ = std::make_shared<DenseTensorImpl<T>>(lbls, ProcGrid{});
      // }
      // else {
         impl_ = std::make_shared<TensorImpl<T>>(lbls);
      //}
    }

    Tensor(ProcGrid pg, const std::initializer_list<TiledIndexLabel>& lbls) {
      EXPECTS(lbls.size() == 2); 
      impl_ = std::make_shared<DenseTensorImpl<T>>(lbls, pg, true);
    }

    /**
     * @brief Constructs a new Tensor object recursively with a set of
     * TiledIndexSpace objects followed by a lambda expression
     *
     * @tparam Ts variadic template for the input arguments
     * @param [in] tis TiledIndexSpace object for the corresponding mode of
     * the Tensor object
     * @param [in] rest remaining parts of the input arguments
     */
    template<class... Ts>
    Tensor(const TiledIndexSpace& tis, Ts... rest) :
      impl_{std::make_shared<TensorImpl<T>>(tis, rest...)} {}

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
     * Access the underying global array
     * @return Handle to underlying global array
     */
    int ga_handle() {
        return impl_->ga_handle();
    }

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
     * @brief nonblocking Get method for Tensor values
     *
     * @param [in] idx_vec set of indices to get data from
     * @param [in] buff_span a memory span to write the fetched data
     */
    void nb_get(IndexVector idx_vec, span<T> buff_span, DataCommunicationHandlePtr data_comm_handle) const {
        impl_->nb_get(idx_vec, buff_span, data_comm_handle);
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
     * @brief nonblocking Put method for Tensor values
     *
     * @param [in] idx_vec set of indices to put data to
     * @param [in] buff_span a memory span for the data to be put
     */
    void nb_put(IndexVector idx_vec, span<T> buff_span, DataCommunicationHandlePtr data_comm_handle) {
        impl_->nb_put(idx_vec, buff_span, data_comm_handle);
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
     * @brief nonblocking Add method for Tensor values
     *
     * @param [in] idx_vec set of indices to put data to
     * @param [in] buff_span a memory span for the data to be put
     */
    void nb_add(IndexVector idx_vec, span<T> buff_span, DataCommunicationHandlePtr data_comm_handle) {
        impl_->nb_add(idx_vec, buff_span, data_comm_handle);
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
    void allocate(ExecutionContext* ec) { impl_->allocate(ec); }

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
    static void allocate(ExecutionContext* ec, Tensor<T>& tensor,
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
    static void deallocate(Tensor<T>& tensor, Args&... rest) {
        dealloc(tensor, rest...);
    }

    /**
     * @brief Get the number of modes of a Tensor
     *
     * @returns number of modes of a Tensor
     */
    size_t num_modes() const { return impl_->num_modes(); }

    /**
     * @brief Checks if the block is non-zero (calculated using Spin attributes)
     *
     * @param [in] blockid identifier for the block
     * @returns true if the block is non-zero
     */
    bool is_non_zero(const IndexVector& blockid) const {
        return impl_->is_non_zero(blockid);
    }

    TensorBase* base_ptr() const {
      return static_cast<TensorBase*>(impl_.get());
    }

    ExecutionContext* execution_context() const {
      return impl_->execution_context();
    }

    bool has_spin() const {
      return impl_->has_spin();
    }

    bool has_spatial() const {
      return impl_->has_spatial();
    }

    template <typename U>
    friend bool operator==(const Tensor<U>& lhs,
                           const Tensor<U>& rhs);

private:
    std::shared_ptr<TensorImpl<T>>
      impl_; /**< Shared pointer to the implementation object */

    // Private allocate and de-allocate functions
    /**
     * @brief Static allocation method (used internally only)
     *
     * @param [in] ec ExecutionContext for the allocation
     */
    static void alloc(const ExecutionContext* ec) {}

    /**
     * @brief Static deallocation method (used internally only)
     *
     */
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
    static void alloc(ExecutionContext* ec, Tensor<T>& tensor, Args&... rest) {
        tensor.impl_->allocate(ec);
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

template <typename T>
bool operator==(const Tensor<T>& lhs,
                const Tensor<T>& rhs){
  EXPECTS_STR(lhs.execution_context() != nullptr && 
              rhs.execution_context() != nullptr,
              "Tensors have to be allocated for comparison.");
  EXPECTS_STR(lhs.num_modes() == rhs.num_modes(), 
              "Tensors should have the same number of modes for comparison.");
  for (size_t i = 0; i < lhs.num_modes(); i++) {
    auto lhs_tis = lhs.tiled_index_spaces()[i];
    auto rhs_tis = rhs.tiled_index_spaces()[i];

    EXPECTS_STR(lhs_tis == rhs_tis,
                "Each mode on tensors should be the same for comparison.");
  }

  return (hash_tensor(lhs.execution_context(), lhs) == hash_tensor(rhs.execution_context(), rhs));
}

// This class inherits from pair ranther than using an alias because deduction
// guides are not supported for aliases in C++17 (see
// https://stackoverflow.com/questions/41008092/class-template-argument-deduction-not-working-with-alias-template)
template<typename T>
class IndexedTensor : public std::pair<Tensor<T>, IndexVector> 
{
    public:
    using std::pair<Tensor<T>, IndexVector>::pair;
    auto& tensor() { return this->first; }
    auto& tensor() const { return this->first; }
    auto& indexVector() { return this->second; }
    auto& indexVector() const { return this->second; }
    void put(span<T> span) { tensor().put(indexVector(), span); }
    void add(span<T> span) { tensor().add(indexVector(), span); }
};

template<typename T>
IndexedTensor(Tensor<T>, IndexVector) -> IndexedTensor<T>;

} // namespace tamm

#endif // TENSOR_HPP_
