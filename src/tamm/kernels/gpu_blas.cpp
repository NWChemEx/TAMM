#include "tamm/utils.hpp"
#include "tamm_blas.hpp"

#if defined(USE_CUDA)
#include <cublas_v2.h>
#elif defined(USE_HIP)
#include <rocblas/rocblas.h>
#elif defined(USE_DPCPP)
#include <oneapi/mkl/blas.hpp>
#if defined(USE_PORT_BLAS)
#include <portblas.hpp>
#endif // USE_PORT_BLAS
#endif // USE_DPCPP

#if defined(USE_DPCPP)
#define ONEMKLBLAS_CHECK(FUNC)                                                         \
  do {                                                                                 \
    try {                                                                              \
      (FUNC);                                                                          \
    } catch(oneapi::mkl::exception const& ex) {                                        \
      std::ostringstream msg;                                                          \
      msg << "oneMKL Error: " << ex.what() << ", at " << __FILE__ << " : " << __LINE__ \
          << std::endl;                                                                \
      throw std::runtime_error(msg.str());                                             \
    }                                                                                  \
  } while(0)
#endif // USE_DPCPP

template<typename T>
void tamm::kernels::gpu::axpy(const int64_t n, const T* src, const int incx, T*& dst,
                              const int incy, gpuStream_t& handle) {
  T alpha = 1.0;
#if defined(USE_DPCPP)
  ONEMKLBLAS_CHECK(
    oneapi::mkl::blas::column_major::axpy(handle.first, n, alpha, src, incx, dst, incy));
#elif defined(USE_CUDA)
  CUBLAS_CHECK(cublasDaxpy(handle.second, n, &alpha, src, incx, dst, incy));
#elif defined(USE_HIP)
  ROCBLAS_CHECK(rocblas_daxpy(handle.second, n, &alpha, src, incx, dst, incy));
#endif
}

template<typename T, typename T1, typename T2, typename T3>
void tamm::kernels::gpu::gemm(int n, int m, int k, const T alpha, const T3* B, int ldb, const T2* A,
                              int lda, const T beta, T1* C, int ldc, gpuStream_t& handle) {
#if defined(USE_DPCPP)

#ifdef USE_PORT_BLAS
  blas::SB_Handle sb_handle(handle.first);
  blas::internal::_gemm(sb_handle, 'n', 'n', n, m, k, alpha, const_cast<T3*>(B), ldb,
                        const_cast<T2*>(A), lda, beta, C, ldc, {});
  handle.first.wait();
#else
  auto gemm_event = oneapi::mkl::blas::column_major::gemm(handle.first, oneapi::mkl::transpose::N,
                                                          oneapi::mkl::transpose::N, n, m, k, alpha,
                                                          B, ldb, A, lda, beta, C, ldc);
  gemm_event.wait();
#endif // USE_PORT_BLAS

#elif defined(USE_CUDA)
  if constexpr(tamm::internal::is_complex_v<T1> && tamm::internal::is_complex_v<T2> &&
               tamm::internal::is_complex_v<T3>) {
    CUBLAS_CHECK(cublasZgemm(handle.second, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                             (cuDoubleComplex*) &alpha, (cuDoubleComplex*) B, ldb,
                             (cuDoubleComplex*) A, lda, (cuDoubleComplex*) &beta,
                             (cuDoubleComplex*) C, ldc));
  }
  else {
    CUBLAS_CHECK(cublasDgemm(handle.second, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, ldb, A,
                             lda, &beta, C, ldc));
  }
#elif defined(USE_HIP)
  if constexpr(internal::is_complex_v<T1> && internal::is_complex_v<T2> &&
               internal::is_complex_v<T3>) {
    ROCBLAS_CHECK(rocblas_zgemm(handle.second, rocblas_operation_none, rocblas_operation_none, n, m,
                                k, (rocblas_double_complex*) &alpha, (rocblas_double_complex*) B,
                                ldb, (rocblas_double_complex*) A, lda,
                                (rocblas_double_complex*) &beta, (rocblas_double_complex*) C, ldc));
  }
  else {
    ROCBLAS_CHECK(rocblas_dgemm(handle.second, rocblas_operation_none, rocblas_operation_none, n, m,
                                k, &alpha, B, ldb, A, lda, &beta, C, ldc));
  }
#endif
}

template void tamm::kernels::gpu::axpy(const int64_t n, const double* src, const int incx,
                                       double*& dst, const int incy, gpuStream_t& thandle);

template void tamm::kernels::gpu::gemm(int n, int m, int k, const double alpha, const double* B,
                                       int ldb, const double* A, int lda, const double beta,
                                       double* C, int ldc, gpuStream_t& handle);
#if !defined(USE_PORT_BLAS)
template void tamm::kernels::gpu::gemm(int n, int m, int k, const std::complex<double> alpha,
                                       const std::complex<double>* B, int ldb,
                                       const std::complex<double>* A, int lda,
                                       const std::complex<double> beta, std::complex<double>* C,
                                       int ldc, gpuStream_t& handle);
#endif // USE_PORT_BLAS
