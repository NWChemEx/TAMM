#ifndef TAMM_CHOLESKY_HPP_
#define TAMM_CHOLESKY_HPP_

#include "tamm/labeled_tensor.hpp"
//#include "tamm/tensor.hpp" //uncomment when suffices to include only this
#include <memory> //for shared_ptr

namespace tamm {
namespace internal {

///Forward declaration of the class that implements Cholesky
template<typename T> class CholeskyPIMPL;

} // namespace internal


/**
 * @brief Class defining the public API for Cholesky factorization.
 *
 * Cholesky factorization of a positive-definite matrix @f$\mathbf{M}@f$
 * attempts to find a lower-triangular matrix @f$\mathbf{L}@f$ such that:
 * @f$\mathbf{M} = \mathbf{L}\mathbf{L}^\dagger@f$.  Note that since
 * @f$\mathbf{L}^\dagger@f$ is an upper-triangular matrix, @f$\mathbf{U}@f$ this
 * means that @f$\mathbf{M}@f$ can also be written as:
 * @f$\mathbf{U}^\dagger\mathbf{U}@f$.
 *
 * @tparam T the type of the scalars stored within the tensors.  Should be one
 *         of the supported types.
 */
template<typename T>
class LLT {
public:
    ///The type of a tensor returned by this class
    using tensor_type = Tensor<T>;

    ///The type of a read-only tensor returned by this class
    using const_reference = const tensor_type&;

    ///Makes the PIMPL instance
    LLT();

    ///Frees the memory associated with the PIMPL
    ~LLT();

    ///Deleted until PIMPL supports these operations
    ///@{
    LLT(const LLT&) = delete;
    LLT(LLT&& ) = delete;
    LLT& operator=(const LLT&) = delete;
    LLT& operator=(LLT&&) = delete;
    ///@}

    /**
     * @brief Initializes the PIMPL and factors the matrix in one step
     * @param M The matrix to factor.
     */
    LLT(const_reference M) : LLT() {
        compute(M);
    }

    /**
     * @brief Actually runs Cholesky.
     *
     * This function does not need to be called again if
     * `LLT::LLT(const_reference)` was used to construct the class (unless you
     * want to reuse the class
     *
     * @param M the matrix to factorize.
     */
     LLT<T>& compute(const_reference M);

    /**
     * @brief Returns the @f$\mathbf{L}@f$.
     * @return The lower-triangular factor of @f\mathbf{M}@f$
     */
    const_reference matrix_L() const;

    /**
     * @brief Returns @f$\mathbf{L}^\dagger@f$.
     * @return The transpose of the lower-triangular factor of @f\mathbf{M}@f$
     */
    const_reference matrix_U() const;
private:

    ///The instance that actually does the computing.
    std::shared_ptr<internal::CholeskyPIMPL<T>> pimpl_;
};

///Forward declare the available specializations
extern template class LLT<double>;
extern template class LLT<float>;


} //End namespace

#endif
