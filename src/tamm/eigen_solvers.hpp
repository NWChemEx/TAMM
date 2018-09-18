#ifndef TAMM_EIGEN_SOLVERS_HPP_
#define TAMM_EIGEN_SOVLERS_HPP_

#include "tamm/labeled_tensor.hpp"
//#include "tamm/tensor.hpp" //uncomment when suffices to include only this
#include <memory> //for shared_ptr

namespace tamm {
namespace internal {

///Forward declare the class that will actually implement the eigen solver
template<typename T>  class EigenSolverPIMPL;

} //end namespace internal


/**
 * @brief Class for solving a generalized eigenvalue problem.
 *
 * Solves the generalized eigenvalue problem:
 *
 * @f[
 * \mathbf{A}\vector{v} = \lambda\mathbf{B}\vector{v}
 * @f]
 *
 * The API of this class is modeled after Eigen's
 * GeneralizedSelfAdjointEigenSolver class, but with a few notable differences.
 * First the eigenvalues and eigenvectors are returned as `tamm::Tensor`
 * instances.  Second many of the finer-grained features of Eigen's class are
 * not available (*e.g.*, passing of options to compute is not supported). And
 * finally we make no guarantees about the internal workings of this class at
 * this time (other than it will produce the correct answer).  Specifically,
 * this last point implies that this class may be more inefficient than Eigen's
 * variant as cacheing of intermediates may not occur.
 *
 * @tparam T The type of the scalars within the tensors.  Should be one of the
 *          types TAMM recognizes.
 *
 */
template<typename T>
class GeneralizedSelfAdjointEigenSolver {
private:
    ///The type of this object (b/c it's really long to type)
    using my_type = GeneralizedSelfAdjointEigenSolver<T>;
public:
    ///The type of the object holding the eigenvalues/vectors
    using tensor_type = Tensor<T>;

    ///Read-only version of a held tensor
    using const_reference = const tensor_type&;

    ///Makes the PIMPL
    GeneralizedSelfAdjointEigenSolver();

    ///Frees up all memory associated with the PIMPL
    ~GeneralizedSelfAdjointEigenSolver();

    /**
     * @brief Copy/Move ctors and assignment operators.
     *
     * These are deleted until the internal PIMPLs support them.
     */
    ///@{
    GeneralizedSelfAdjointEigenSolver(const my_type&) = delete;
    GeneralizedSelfAdjointEigenSolver(my_type&& ) = delete;
    my_type& operator=(const my_type&) = delete;
    my_type& operator=(my_type&&) = delete;
    ///@}

    /**
     * @brief Makes the PIMPL and diagonalizes the matrix in one step.
     *
     * This ctor is a convenience ctor for creating a new
     * GeneralizedSelfAdjointEigenSolver instance and then using it to
     * diagonalize a matrix.
     *
     * @param A The matrix to diagonalize
     * @param B The metric matrix for A with respect to the chosen basis set.
     */
    GeneralizedSelfAdjointEigenSolver(const tensor_type& A,
                                      const tensor_type& B) :
        GeneralizedSelfAdjointEigenSolver() {
        compute(A, B);
    }

    /**
     * @brief Uses the current solver to solve a generalized eigenvalue problem.
     *
     * @param A The matrix to diagonalize.  Must be self-adjoint.
     * @param B The metric matrix for A with respect to the chosen basis set
     * @return The current instance
     */
    my_type& compute(const tensor_type& A, const tensor_type& B);

    ///The eigenvalues from the most recent call to compute
    const_reference eigenvalues() const;

    ///The eigenvectors from the most recent call to compute
    const_reference eigenvectors() const;
private:

    ///The object actually responsible for implementing this class
    std::shared_ptr<internal::EigenSolverPIMPL<T>> pimpl_;
};

//Need to declare all possible specializations
extern template class GeneralizedSelfAdjointEigenSolver<double>;
extern template class GeneralizedSelfAdjointEigenSolver<float>;

} //End namespace tamm

#endif
