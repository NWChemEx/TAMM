#pragma once

// Internal GPU LAPACK kernels used by TAMM's linalg routines. This header is not installed: callers
// outside TAMM go through the public linalg API (e.g. the local tamm::eigensolve in
// tamm_linalg.hpp) instead of calling vendor kernels directly.

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)

#include <cstdint>

namespace tamm::kernels::gpu {

// GPU counterpart of lapack::syevd(Job::Vec, Uplo::Lower, n, A, lda, W): cusolverDnXsyevd (CUDA),
// rocsolver_dsyevd (HIP) or oneapi::mkl::lapack::syevd (DPCPP). A (column-major, n x n) and W
// are host pointers; A is staged to the device, overwritten there with the eigenvectors and copied
// back, mirroring lapack::syevd's host-in/host-out contract. Terminates on a backend error.
template<typename T>
void syevd(int64_t n, T* A, int64_t lda, T* W);

// GPU counterpart of the host-side steps of the local tamm::generalized_eigensolve: with the
// same column-major view of the row-major N x N F, N x M X and N x M C, computes F' = X^T F X,
// its eigendecomposition and C = X C' on the device (vendor gemm + syevd). F, X, C and W are
// host pointers. On return C and W (size M) hold the results and F holds exactly what the host
// path leaves there: C' (M x M, column-major) in its first M*M entries, the rest unchanged.
template<typename T>
void generalized_eigensolve(int64_t N, int64_t M, T* F, const T* X, T* C, T* W);

} // namespace tamm::kernels::gpu

#endif
