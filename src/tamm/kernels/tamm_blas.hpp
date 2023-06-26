#pragma once

#include "tamm/gpu_streams.hpp"

namespace tamm::kernels {

namespace cpu {
template<typename T, typename T1, typename T2, typename T3>
void blas(int m, int n, int k, const T alpha, const T2* A, int lda, const T3* B, int ldb,
          const T beta, T1* C, int ldc);
} // namespace cpu

namespace gpu {
template<typename T, typename T1, typename T2, typename T3>
void blas(int m, int n, int k, const T alpha, const T2* A, int lda, const T3* B, int ldb,
          const T beta, T1* C, int ldc, gpuStream_t& blashandle);
} // namespace gpu

} // namespace tamm::kernels
