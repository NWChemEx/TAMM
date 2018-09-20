#include "tamm/eigen_solvers.hpp"
#undef I //Gets declared in complex.h and interferes with Eigen's templating
#include "tamm/eigen_utils.hpp"

namespace tamm {

template<typename T>
using GSAES = GeneralizedSelfAdjointEigenSolver<T>;

namespace internal {

template<typename T>
class EigenSolverPIMPL {
public:
    ///The type of a tensor returned by an eigensolver
    using tensor_type = typename GSAES<T>::tensor_type;

    ///The type of a reference to a constant tensor
    using const_tensor = typename GSAES<T>::const_tensor;

    ///The type of the new space
    using space_type = typename GSAES<T>::space_type;

    ///The type of a read-only version of the new space
    using const_space = typename GSAES<T>::const_space;

    ///The type of a unique_ptr to an instance of this class
    using unique_pimpl = std::unique_ptr<EigenSolverPIMPL<T>>;

    EigenSolverPIMPL() = default;
    virtual ~EigenSolverPIMPL() = default;

    //TODO: Enable when TAMM tensors can be copied/moved
    EigenSolverPIMPL(const EigenSolverPIMPL&) = delete;
    EigenSolverPIMPL(EigenSolverPIMPL&&) = delete;
    EigenSolverPIMPL<T>& operator=(const EigenSolverPIMPL<T>&) = delete;
    EigenSolverPIMPL<T>& operator=(EigenSolverPIMPL<T>&&) = delete;

    ///Public API for performing the computation (same doc as main class)
    void compute(const_tensor A,
                 const_tensor B,
                 const_space evec_space) { compute_(A, B, evec_space); }

    ///Public API for getting the values
    const_tensor eigenvalues() const { return eigenvalues_(); }

    ///Public API for getting the vectors
    const_tensor eigenvectors() const { return eigenvectors_(); }

private:
    ///These functions to be implemented by the derived class
    ///@{
    virtual void compute_(const_tensor A,
                          const_tensor B,
                          const_space evec_space) = 0;
    virtual const_tensor eigenvalues_() const = 0;
    virtual const_tensor eigenvectors_() const = 0;
    ///@}
};

template<typename T>
class EigenGeneralizedSelfAdjointEigenSolverPIMPL :
        public EigenSolverPIMPL<T> {
public:
    ///Pull typedefs in from base class
    ///@{
    using tensor_type = typename EigenSolverPIMPL<T>::tensor_type;
    using const_tensor = typename EigenSolverPIMPL<T>::const_tensor;
    using const_space  = typename EigenSolverPIMPL<T>::const_space;
    using unique_pimpl  = typename EigenSolverPIMPL<T>::unique_pimpl;
    ///@}

private:
    using eigen_matrix =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    ///The instance actually doing the work for us
    Eigen::GeneralizedSelfAdjointEigenSolver<eigen_matrix> solver_;

    tensor_type evals_;
    tensor_type evecs_;

    void compute_(const_tensor A,
                  const_tensor B,
                  const_space new_space) override {
        using eigen_map = Eigen::Map<eigen_matrix>;
        call_eigen_matrix_fxn(A, [&, this](eigen_map map_A){
            call_eigen_matrix_fxn(B, [&, this](eigen_map map_B){
                solver_.compute(map_A, map_B);
            });
        });

        //solver_.compute(mapped_A, mapped_B);

        const auto& eigen_lambda = solver_.eigenvalues();
        const auto& eigen_nu     = solver_.eigenvectors();
        eigen_matrix neigen_nu = eigen_nu.colwise().normalized();


        //TODO: get from input tensors
        ProcGroup pg{GA_MPI_Comm()};
        auto mgr = MemoryManagerGA::create_coll(pg);
        Distribution_NW distribution;
        ExecutionContext ec{pg,&distribution,mgr};
        auto old_space = A.tiled_index_spaces()[0];
        tensor_type lambda{new_space};
        tensor_type nu{old_space, new_space};
        tensor_type::allocate(&ec, lambda);
        tensor_type::allocate(&ec, nu);


        eigen_matrix_to_tamm<1>(eigen_lambda, lambda);
        eigen_matrix_to_tamm<2>(neigen_nu, nu);
        evals_ = lambda;
        evecs_ = nu;
    }

    const_tensor eigenvalues_() const {return evals_;}
    const_tensor eigenvectors_() const {return evecs_;}
};

} //namespace internal

template<typename T>
GeneralizedSelfAdjointEigenSolver<T>::GeneralizedSelfAdjointEigenSolver():
    pimpl_(std::make_unique<
            internal::EigenGeneralizedSelfAdjointEigenSolverPIMPL<T>>()
    ) {}

template<typename T>
GeneralizedSelfAdjointEigenSolver<T>::~GeneralizedSelfAdjointEigenSolver() =
        default;

template<typename T>
GSAES<T>& GeneralizedSelfAdjointEigenSolver<T>::compute(const_tensor A,
        const_tensor B, const_space new_space) {
    pimpl_->compute(A, B, new_space);
    return *this;
}

template<typename T>
typename GSAES<T>::const_tensor
GeneralizedSelfAdjointEigenSolver<T>::eigenvalues() const {
    return pimpl_->eigenvalues();
}

template<typename T>
typename GSAES<T>::const_tensor
GeneralizedSelfAdjointEigenSolver<T>::eigenvectors() const {
    return pimpl_->eigenvectors();
}

//Need to instantiate each declared template specialization
template class GeneralizedSelfAdjointEigenSolver<double>;
template class GeneralizedSelfAdjointEigenSolver<float>;

} //namespace tamm
