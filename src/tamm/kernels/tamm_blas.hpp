#pragma once

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#include <tamm/gpu_streams.hpp>
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

} // namespace tamm::kernels
