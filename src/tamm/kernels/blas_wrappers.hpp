#ifndef TAMM_INTERNAL_BLAS_WRAPPERS_H_
#define TAMM_INTERNAL_BLAS_WRAPPERS_H_

#include "ga/ga_linalg.h"

namespace tamm::blas_wrappers {

template <typename T>
void gemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
          const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
          T alpha, const T* A, const int lda, const T* B, const int ldb, T beta,
          T* C, const int ldc);

template <>
inline void gemm<double>(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
                         const CBLAS_TRANSPOSE TransB, const int M, const int N,
                         const int K, double alpha, const double* A,
                         const int lda, const double* B, const int ldb,
                         double beta, double* C, const int ldc) {
  cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}

template <>
inline void gemm<float>(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, float alpha, const float* A, const int lda,
                        const float* B, const int ldb, float beta, float* C,
                        const int ldc) {
  cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}

template <>
inline void gemm<std::complex<float>>(
    const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    std::complex<float> alpha, const std::complex<float>* A, const int lda,
    const std::complex<float>* B, const int ldb, std::complex<float> beta,
    std::complex<float>* C, const int ldc) {
  cblas_cgemm(Order, TransA, TransB, M, N, K, (const float*)&alpha,
              (const float*)A, lda, (const float*)B, ldb, (const float*)&beta,
              (float*)C, ldc);
}

template <>
inline void gemm<std::complex<double>>(
    const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    std::complex<double> alpha, const std::complex<double>* A, const int lda,
    const std::complex<double>* B, const int ldb, std::complex<double> beta,
    std::complex<double>* C, const int ldc) {
  cblas_zgemm(Order, TransA, TransB, M, N, K, (const double*)&alpha,
              (const double*)A, lda, (const double*)B, ldb,
              (const double*)&beta, (double*)C, ldc);
}
}  // namespace tamm::blas_wrappers

#endif  // TAMM_INTERNAL_BLAS_WRAPPERS_H_
