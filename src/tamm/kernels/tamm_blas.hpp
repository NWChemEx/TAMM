#pragma once

#include <cstddef>

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

// Out-of-place N-dimensional axis-permuting transpose (in-house permute
// kernel).
//   out        : destination buffer (device), sized as the permuted tensor
//   in         : source buffer (device); must not alias out
//   ndim       : tensor rank (0 .. 8)
//   outDims    : extents of the output (permuted) tensor, length ndim
//   perm       : output-axis -> source-axis map, length ndim
//   scale      : every output element is multiplied by scale
//   accumulate : false -> out = scale*in, true -> out += scale*in
// Enqueued on handle.first (cuda/hip stream, in-order SYCL queue); async,
// no internal synchronization.
template<typename T>
void permute(T* out, const T* in, int ndim, const size_t* outDims, const int* perm, T scale,
             bool accumulate, gpuStream_t& handle);

// Column-major C = alpha op(A) op(B) + beta C on device pointers, with the same argument order
// and meaning as blas::gemm(blas::Layout::ColMajor, ...). Runs on gpuhandle's stream.
// Note: Added for the generalized eigensolver API that needs to call gemms on 2D mats with
// transpose.
// TODO: Address uniform gemm() API usage
template<typename T>
void gemm(blas::Op transa, blas::Op transb, int m, int n, int k, const T alpha, const T* A, int lda,
          const T* B, int ldb, const T beta, T* C, int ldc, gpuStream_t& gpuhandle);

// GPU (cuSolver/rocSolver/oneMKL) counterpart of lapack::gesvd, used by tamm::svd. A/U/VT are
// host pointers (column-major, same layout lapack::gesvd expects); device staging is handled
// internally. On CUDA/HIP this requires m >= n (cusolverDn/rocsolver <t>gesvd's native
// constraint); tamm::svd falls back to the LAPACK/CPU path otherwise.
template<typename T>
void gesvd(lapack::Job jobu, lapack::Job jobvt, int64_t m, int64_t n, T* A, int64_t lda,
           blas::real_type<T>* S, T* U, int64_t ldu, T* VT, int64_t ldvt);

// GPU (cuSolver/rocSolver/oneMKL) counterparts of lapack::geqrf and lapack::ungqr, used by
// tamm::qr. A/tau are host pointers (same layout and host-in/host-out contract as their LAPACK
// counterparts); device staging is handled internally, independently in each call. No shape
// restriction (unlike gesvd): both routines support any m, n.
//
// geqrf: on exit, A's upper (trapezoidal) part holds R; its strictly-lower part holds the
// Householder reflector vectors, and tau (size min(m,n)) their scalar factors.
// ungqr: forms the first n columns of Q (k <= n <= m, k = number of reflectors from geqrf) from
// that A/tau, overwriting A's first n columns with Q. The caller is responsible for giving A
// enough columns (and reading out R beforehand): see tamm::qr.
template<typename T>
void geqrf(int64_t m, int64_t n, T* A, int64_t lda, T* tau);

template<typename T>
void ungqr(int64_t m, int64_t n, int64_t k, T* A, int64_t lda, const T* tau);

} // namespace gpu
#endif

} // namespace tamm::kernels
