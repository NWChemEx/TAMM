#include "tamm_blas.hpp"

#include "ga/ga_linalg.h"

template<typename T, typename T1, typename T2, typename T3>
void tamm::kernels::cpu::blas(int m, int n, int k, const T alpha, const T2* A, int lda, const T3* B,
                              int ldb, const T beta, T1* C, int ldc) {
  blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, k, alpha, A, lda,
             B, ldb, beta, C, ldc);
}

// Explicit template instantiations
template void tamm::kernels::cpu::blas(int m, int n, int k, const double alpha, const double* A,
                                       int lda, const double* B, int ldb, const double beta,
                                       double* C, int ldc);
template void tamm::kernels::cpu::blas(int m, int n, int k, const std::complex<double> alpha,
                                       const std::complex<double>* A, int lda,
                                       const std::complex<double>* B, int ldb,
                                       const std::complex<double> beta, std::complex<double>* C,
                                       int ldc);
template void tamm::kernels::cpu::blas(int m, int n, int k, const float alpha, const float* A,
                                       int lda, const float* B, int ldb, const float beta, float* C,
                                       int ldc);
template void tamm::kernels::cpu::blas(int m, int n, int k, const std::complex<float> alpha,
                                       const std::complex<float>* A, int lda,
                                       const std::complex<float>* B, int ldb,
                                       const std::complex<float> beta, std::complex<float>* C,
                                       int ldc);
