#pragma once

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#include <tamm/gpu_streams.hpp>
#endif

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#include <lapack.hh> // lapack::Job, blas::real_type
#endif

namespace tamm::kernels {

namespace cpu {
template<typename T, typename T1, typename T2, typename T3>
void gemm(int m, int n, int k, const T alpha, const T2* A, int lda, const T3* B, int ldb,
          const T beta, T1* C, int ldc);
} // namespace cpu

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
namespace gpu {
template<typename T>
void axpy(const int64_t n, const T* src, const int incx, T*& dst, const int incy,
          gpuStream_t& gpuhandle);

template<typename T, typename T1, typename T2, typename T3>
void gemm(int n, int m, int k, const T alpha, const T3* B, int ldb, const T2* A, int lda,
          const T beta, T1* C, int ldc, gpuStream_t& gpuhandle);
} // namespace gpu
#endif

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
namespace gpu {
// GPU (cuSolver/rocSolver/oneMKL) counterpart of lapack::gesvd, used by tamm::svd. A/U/VT are
// host pointers (column-major, same layout lapack::gesvd expects); device staging is handled
// internally. On CUDA/HIP this requires m >= n (cusolverDn/rocsolver <t>gesvd's native
// constraint); tamm::svd falls back to the LAPACK/CPU path otherwise.
template<typename T>
void gesvd(lapack::Job jobu, lapack::Job jobvt, int64_t m, int64_t n, T* A, int64_t lda,
           blas::real_type<T>* S, T* U, int64_t ldu, T* VT, int64_t ldvt);
} // namespace gpu
#endif

} // namespace tamm::kernels
