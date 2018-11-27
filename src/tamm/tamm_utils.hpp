#ifndef TAMM_TAMM_UTILS_HPP_
#define TAMM_TAMM_UTILS_HPP_

#include <vector>

namespace tamm {
/**
 * @brief Overload of << operator for printing Tensor blocks
 *
 * @tparam T template type for Tensor element type
 * @param [in] os output stream
 * @param [in] vec vector to be printed
 * @returns the reference to input output stream with vector elements printed
 * out
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T>& vec) {
    os << "[";
    for(auto& x : vec) os << x << ",";
    os << "]\n";
    return os;
}

/**
 * @brief Prints a Tensor object
 *
 * @tparam T template type for Tensor element type
 * @param [in] tensor input Tensor object
 */
template<typename T>
void print_tensor(const Tensor<T>& tensor) {
    auto lt = tensor();
    for(auto it : tensor.loop_nest()) {
        auto blockid   = internal::translate_blockid(it, lt);
        TAMM_SIZE size = tensor.block_size(blockid);
        std::vector<T> buf(size);
        tensor.get(blockid, buf);
        std::cout << "block" << blockid;

        for(TAMM_SIZE i = 0; i < size; i++) {
            if(buf[i] > 0.0000000000001 || buf[i] < -0.0000000000001)
                std::cout << buf[i] << " ";
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Get the scalar value from the Tensor
 *
 * @tparam T template type for Tensor element type
 * @param [in] tensor input Tensor object
 * @returns a scalar value in type T
 *
 * @warning This function only works with scalar (zero dimensional) Tensor
 * objects
 */
template<typename T>
T get_scalar(Tensor<T>& tensor) {
    T scalar;
    EXPECTS(tensor.num_modes() == 0);
    tensor.get({}, {&scalar, 1});
    return scalar;
}

/**
 * @brief Update input LabeledTensor object with a lambda function
 *
 * @tparam T template type for Tensor element type
 * @tparam Func template type for the lambda function
 * @param [in] labeled_tensor tensor slice to be updated
 * @param [in] lambda function for updating the tensor
 */
template<typename T, typename Func>
void update_tensor(LabeledTensor<T> labeled_tensor, Func lambda) {
    LabelLoopNest loop_nest{labeled_tensor.labels()};

    for(const auto& itval : loop_nest) {
        const IndexVector blockid =
          internal::translate_blockid(itval, labeled_tensor);
        size_t size = labeled_tensor.tensor().block_size(blockid);
        std::vector<T> buf(size);

        labeled_tensor.tensor().get(blockid, buf);
        lambda(blockid, buf);
        labeled_tensor.tensor().put(blockid, buf);
    }
}

/**
 * @brief Update input LabeledTensor object with a lambda function
 *
 * @tparam T template type for Tensor element type
 * @tparam Func template type for the lambda function
 * @param [in] labeled_tensor tensor slice to be updated
 * @param [in] lambda function for updating the tensor
 */
template<typename T, typename Func>
void update_tensor_general(LabeledTensor<T> labeled_tensor, Func lambda) {
    LabelLoopNest loop_nest{labeled_tensor.labels()};

    for(const auto& itval : loop_nest) {
        const IndexVector blockid =
          internal::translate_blockid(itval, labeled_tensor);
        size_t size = labeled_tensor.tensor().block_size(blockid);
        std::vector<T> buf(size);

        labeled_tensor.tensor().get(blockid, buf);
        lambda(labeled_tensor.tensor(), blockid, buf);
        labeled_tensor.tensor().put(blockid, buf);
    }
}

/**
 * @brief Construct an ExecutionContext object
 *
 * @returns an Execution context
 * @todo there is possible memory leak as distribution will not be unallocated
 * when Execution context is destructed
 */
ExecutionContext make_execution_context() {
    ProcGroup pg{GA_MPI_Comm()};
    auto* pMM             = MemoryManagerLocal::create_coll(pg);
    Distribution_NW* dist = new Distribution_NW();
    return ExecutionContext(pg, dist, pMM);
}

} // namespace tamm

#endif // TAMM_TAMM_UTILS_HPP_
