#include "tamm_blas.hpp"

#if defined(USE_CUDA)
#include <cublas_v2.h>
#elif defined(USE_HIP)
#include <rocblas.h>
#elif defined(USE_DPCPP)
#include <oneapi/mkl/blas.hpp>
#if defined(USE_SYCL_BLAS)
#include <sycl_blas.h>
#endif // USE_SYCL_BLAS
#endif // USE_DPCPP

#if defined(USE_HIP)
#define ROCBLAS_CHECK(FUNC)                                                                      \
  do {                                                                                           \
    rocblas_status err_ = (FUNC);                                                                \
    if(err_ != rocblas_status_success) {                                                         \
      std::ostringstream msg;                                                                    \
      msg << "ROCBLAS Error: " << rocblas_status_to_string(err_) << ", at " << __FILE__ << " : " \
          << __LINE__ << std::endl;                                                              \
      throw std::runtime_error(msg.str());                                                       \
    }                                                                                            \
  } while(0)
#endif // USE_HIP

#if defined(USE_CUDA)
#define CUBLAS_CHECK(FUNC)                                                                   \
  do {                                                                                       \
    cublasStatus_t err_ = (FUNC);                                                            \
    if(err_ != CUBLAS_STATUS_SUCCESS) {                                                      \
      std::ostringstream msg;                                                                \
      msg << "CUBLAS Error: " << cublasGetStatusString(err_) << ", at " << __FILE__ << " : " \
          << __LINE__ << std::endl;                                                          \
      throw std::runtime_error(msg.str());                                                   \
    }                                                                                        \
  } while(0)
#endif // USE_CUDA

#if defined(USE_DPCPP)
#define ONEMKLBLAS_CHECK(FUNC)                                                         \
  do {                                                                                 \
    try {                                                                              \
      (FUNC)                                                                           \
    } catch(oneapi::mkl::exception const& ex) {                                        \
      std::ostringstream msg;                                                          \
      msg << "oneMKL Error: " << ex.what() << ", at " << __FILE__ << " : " << __LINE__ \
          << std::endl;                                                                \
      throw std::runtime_error(msg.str());                                             \
    }                                                                                  \
  } while(0)
#endif // USE_DPCPP

template<typename T, typename T1, typename T2, typename T3>
void tamm::kernels::gpu::blas(int m, int n, int k, const T alpha, const T2* A, int lda, const T3* B,
                              int ldb, const T beta, T1* C, int ldc, gpuStream_t& handle) {
#if defined(USE_DPCPP)

#ifdef USE_SYCL_BLAS
  blas::SB_Handle sb_handle(handle.first);
  blas::_gemm(sb_handle, 'n', 'n', n, m, k, alpha, const_cast<T3*>(B), ldb, const_cast<T2*>(A), lda,
              beta, C, ldc, {});
#else
  try {
    auto dgemm = oneapi::mkl::blas::column_major::gemm(handle.first, oneapi::mkl::transpose::N,
                                                       oneapi::mkl::transpose::N, n, m, k, alpha, B,
                                                       ldb, A, lda, beta, C, ldc);
    dgemm.wait();
  } catch(oneapi::mkl::exception const& ex) {
    std::stringstream msg;
    msg << "oneMKL Exception at " << __FILE__ << " : " << __LINE__ << std::endl;
    throw(std::runtime_error(ex.what()));
  }
#endif // USE_SYCL_BLAS

#elif defined(USE_CUDA)
  if constexpr(internal::is_complex_v<T1> && internal::is_complex_v<T2> &&
               internal::is_complex_v<T3>) {
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

// Explicit template instantiations (for SYCL-BLAS APIs, complex-types are not supported yet)
extern template blas::SB_Handle::event_t
blas::_gemm(blas::SB_Handle& sb_handle, char _TransA, char _TransB, int _M, int _N, int _K,
            double _alpha, double* a_, int _lda, double* b_, int _ldb, double _beta, double* _C,
            int _ldc, const blas::SB_Handle::event_t& _dependencies = {});
extern template blas::SB_Handle::event_t
blas::_gemm(blas::SB_Handle& sb_handle, char _TransA, char _TransB, int _M, int _N, int _K,
            float _alpha, float* a_, int _lda, float* b_, int _ldb, float _beta, float* _C,
            int _ldc, const blas::SB_Handle::event_t& _dependencies = {});
// extern template blas::SB_Handle::event_t blas::_gemm(blas::SB_Handle& sb_handle, char _TransA,
// char _TransB, int _M, int _N, int _K, std::complex<double> _alpha, std::complex<double> * a_, int
// _lda, std::complex<double> * b_, int _ldb, std::complex<double> _beta, std::complex<double>* _C,
// int _ldc, const blas::SB_Handle::event_t& _dependencies = {}); template blas::SB_Handle::event_t
// blas::_gemm(blas::SB_Handle& sb_handle, char _TransA, char _TransB, int _M, int _N, int _K,
// std::complex<float> _alpha, std::complex<float> * a_, int _lda, std::complex<float> * b_, int
// _ldb, std::complex<float> _beta, std::complex<float>* _C, int _ldc, const
// blas::SB_Handle::event_t& _dependencies = {});

template void tamm::kernels::gpu::blas(int m, int n, int k, const double alpha, const double* A,
                                       int lda, const double* B, int ldb, const double beta,
                                       double* C, int ldc, gpuStream_t& handle);
template void tamm::kernels::gpu::blas(int m, int n, int k, const std::complex<double> alpha,
                                       const std::complex<double>* A, int lda,
                                       const std::complex<double>* B, int ldb,
                                       const std::complex<double> beta, std::complex<double>* C,
                                       int ldc, gpuStream_t& handle);
template void tamm::kernels::gpu::blas(int m, int n, int k, const float alpha, const float* A,
                                       int lda, const float* B, int ldb, const float beta, float* C,
                                       int ldc, gpuStream_t& handle);
template void tamm::kernels::gpu::blas(int m, int n, int k, const std::complex<float> alpha,
                                       const std::complex<float>* A, int lda,
                                       const std::complex<float>* B, int ldb,
                                       const std::complex<float> beta, std::complex<float>* C,
                                       int ldc, gpuStream_t& handle);
