#ifndef TAMM_DAG_IMPL_HPP_
#define TAMM_DAG_IMPL_HPP_

#include "tamm/ops.hpp"
#include "tamm/types.hpp"

namespace tamm {

namespace internal {

/**
 * @brief Interface base class for DAG-based task construction including
 *        virtual functions for accessing the operations list and constructing
 *        hash of the corresponding DAG unit.
 *
 */
class DAGBase {
public:
    /**
     * @brief A method for retrieving the list of operations in the DAG unit
     *
     * @returns an OpList with corresponding operations
     */
    virtual OpList ops() = 0;

    /**
     * @brief A method for constructing the hash of the corresponding DAG unit
     *
     * @todo Return type should be settled according to the decision on the
     *        serialization framework and format
     *
     * @returns a HashData for the DAG unit
     */
    virtual HashData hash() = 0;

    /**
     * @brief Destroy the DAGBase interface
     *
     */
    virtual ~DAGBase() {}
};

/**
 * @brief  Implementation class for DAG based execution unit
 *
 * @tparam Func a template for the lambda function to be used
 *              for constructing the DAG unit. Only requirement
 *              for the lambda function is to return an OpList
 *              object
 * @tparam Args variadic template for the argument list for the
 *              related lambda function
 */
template<typename Func, typename... Args>
class DAGImpl : public DAGBase {
public:
    // Ctors
    DAGImpl() = default;

    /**
     * @brief Construct a new DAGImpl object from a lambda function
     *        and a set of arguments for the corresponding lambda
     *        function
     *
     * @param [in] func a lambda function returning an OpList
     * @param [in] args set of arguments for the corresponding
     *                  lambda function
     */
    DAGImpl(Func func, Args... args) :
      func_{func},
      args_{std::forward_as_tuple(args...)} {}

    // Copy/Move Ctors and Assignment Operators
    DAGImpl(DAGImpl&&)      = default;
    DAGImpl(const DAGImpl&) = default;
    DAGImpl& operator=(DAGImpl&&) = default;
    DAGImpl& operator=(const DAGImpl&) = default;

    /**
     * @brief operator () overload for assigning input
     *        arguments for the lambda function
     *
     * @param [in] args set of arguments for the lambda function
     * @returns a DAGImpl object with updated arguments
     */
    DAGImpl& operator()(Args... args) {
        args_ = std::forward_as_tuple(args...);
        return *this;
    }

    /**
     * @brief Convenience method for binding arguments
     *        to the arguments in the lambda function
     *
     * @param [in] args set of arguments for the lambda function
     * @returns a DAGImpl object with updated arguments
     */
    DAGImpl& bind(Args... args) {
        args_ = std::forward_as_tuple(args...);
        return *this;
    }

    /**
     * @brief Method for retrieving the operation list from
     *        the lambda function assigned to DAG unit
     *
     * @returns an OpList object returned by the lambda function
     */
    OpList ops() { return apply(func_, args_); }

    /**
     * @brief Method for constructing a hash of the DAG unit.
     *
     * @todo implement according to the decision on serialization
     *
     * @returns a HashData for the hash of DAG unit
     */
    HashData hash() {
        // FIXME
        HashData ret{0};
        return ret;
    }

protected:
    Func func_;                /**< lambda function assigned to the DAG unit */
    std::tuple<Args...> args_; /**< set of arguments bind to the DAG unit */
};                             // DAGImpl

} // namespace internal

using internal::DAGImpl;

/**
 * @brief Constructs a DAGImpl object with provided lambda function
 *        and set of arguments for this function
 *
 * @tparam Func a template for the lambda function
 * @tparam Args a variadic template for the arguments
 * @param [in] func lambda function to be assigned to the DAG unit
 * @param [in] args set of arguments for the lambda function
 * @returns
 */
template<typename Func, typename... Args>
static DAGImpl<Func, Args...> make_dag(Func func, Args... args) {
    return {func, args...};
}

} // namespace tamm

#endif // TAMM_DAG_IMPL_HPP_