#include "tamm/cholesky.hpp"
#undef I //Gets declared in complex.h and interferes with Eigen
#include "tamm/eigen_utils.hpp"

namespace tamm {
namespace internal {

///Defines public API for the PIMPL
template<typename T>
class CholeskyPIMPL {
public:
    ///The type of tensors managed by the PIMPL
    using tensor_type = typename LLT<T>::tensor_type;

    ///Type of read-only variants of those tensors
    using const_reference = typename LLT<T>::const_reference;


    CholeskyPIMPL() = default;
    virtual ~CholeskyPIMPL() = default;

    ///Deleted until tensors can be copied/moved
    ///@{
    CholeskyPIMPL(const CholeskyPIMPL&) = delete;
    CholeskyPIMPL(CholeskyPIMPL&&) = delete;
    CholeskyPIMPL<T>& operator=(const CholeskyPIMPL&) = delete;
    CholeskyPIMPL<T>& operator=(CholeskyPIMPL&&) = delete;
    ///@}

    ///Public API for actually computing the decomposition
    void compute(const_reference A) { compute_(A); }

    ///Public API for retrieving the L matrix
    const_reference matrix_L() const { return matrix_L_(); }

    ///Public API for retrieving L^T=U
    const_reference matrix_U() const { return matrix_U_(); }

private:

    ///To be implemented by the derived class in order to make the class work
    ///@{
    virtual void compute_(const_reference A) = 0;
    virtual const_reference matrix_L_() const = 0;
    virtual const_reference matrix_U_() const = 0;
    ///@}

};

///PIMPL that defers to Eigen
template<typename T>
class EigenCholeskyPIMPL : public CholeskyPIMPL<T> {
public:
    ///Forward type-defs from base class
    ///@{
    using tensor_type = typename CholeskyPIMPL<T>::tensor_type;
    using const_reference = typename CholeskyPIMPL<T>::const_reference;
    ///@}
private:
    ///The type of the matrix returned by Eigen's LLt
    using matrix_type =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    ///Eigen instance we'll use (upper b/c we have row-major)
    Eigen::LLT<matrix_type, Eigen::Upper> llt_;

    ///The lower-triangular part of the matrix
    tensor_type l_;

    ///The upper-triangular part of the matrix (l transposed)
    tensor_type u_;

    void compute_(const_reference A) override {
        tensor_type copy_A(A);

        auto eigen_A = tamm_to_eigen_tensor<T, 2>(copy_A);

        using eigen_map = Eigen::Map<matrix_type>;
        const auto nbf = eigen_A.dimensions()[0];

        eigen_map mapped_A(eigen_A.data(), nbf, nbf);

        llt_.compute(mapped_A);

        const auto& eigen_u = llt_.matrixU();
        const auto& eigen_l = llt_.matrixL();

        using eigen_tensor = Eigen::Tensor<T, 2, Eigen::RowMajor>;
        eigen_tensor tensor_l(std::array<long int, 2>{nbf, nbf});
        eigen_tensor tensor_u(std::array<long int, 2>{nbf, nbf});

        for(long int i = 0; i < nbf; ++i)
            for (long int j = 0; j < nbf; ++j) {
                tensor_l(i, j) = j > i ? 0.0 : eigen_l(i, j);
                tensor_u(i, j) = i > j ? 0.0 : eigen_u(i, j);
            }

        auto tis = A.tiled_index_spaces()[0];

        //TODO: get from input tensors
        ProcGroup pg{GA_MPI_Comm()};
        auto mgr = MemoryManagerGA::create_coll(pg);
        Distribution_NW distribution;
        ExecutionContext ec{pg,&distribution,mgr};
        tensor_type l{tis, tis};
        tensor_type u{tis, tis};

        tensor_type::allocate(&ec, l);
        tensor_type::allocate(&ec, u);

        eigen_to_tamm_tensor(l, tensor_l);
        eigen_to_tamm_tensor(u, tensor_u);

        l_ = l;
        u_ = u;
    }

    const_reference matrix_L_() const override { return l_; }
    const_reference matrix_U_() const override { return u_; }
};

} //namespace internal

template<typename T>
LLT<T>::LLT() : pimpl_(std::make_shared<internal::EigenCholeskyPIMPL<T>>()){}

template<typename T>
LLT<T>::~LLT() = default;

template<typename T>
LLT<T>& LLT<T>::compute(typename LLT<T>::const_reference M) {
    pimpl_->compute(M);
    return *this;
}

template<typename T>
typename LLT<T>::const_reference LLT<T>::matrix_L() const {
    return pimpl_->matrix_L();
}

template<typename T>
typename LLT<T>::const_reference LLT<T>::matrix_U() const {
    return pimpl_->matrix_U();
}

template class LLT<double>;
template class LLT<float>;

} //End namespace tamm
