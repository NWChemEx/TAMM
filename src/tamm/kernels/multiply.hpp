#pragma once

#include "tamm/errors.hpp"
#include "tamm/kernels/assign.hpp"
#include "tamm/types.hpp"

#include <algorithm>
#include <complex>
#include <numeric>
#include <vector>

#if defined(USE_GA_AT)
#include "ga_linalg.h"
#else
#include "ga/ga_linalg.h"
#endif

#if defined(USE_BLIS)
// disable BLAS prototypes within BLIS.
#define BLIS_DISABLE_BLAS_DEFS
#include "blis/blis.h"
#endif

#if defined(USE_TALSH)
#include "tamm/talsh_tamm.hpp"
using tensor_handle = talsh_tens_t;

#undef C0
#undef C4
#undef C5
#undef C6
#undef C7
#undef C8
#undef C9
#undef C10

#ifndef USE_HIP
#include "tamm/cuda_memory_allocator.hpp"
#endif

#else

#include "tamm/gpu_stream_memory_pool.hpp"

#if defined(USE_CUDA)

#define CUBLAS_CHECK(err)                                                              \
  do {                                                                                 \
    cublasStatus_t err_ = (err);                                                       \
    if(err_ != CUBLAS_STATUS_SUCCESS) {                                                \
      std::printf("CUBLAS Exception code: %d at %s : %d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cublas error");                                        \
    }                                                                                  \
  } while(0)
#endif // USE_CUDA

#if defined(USE_HIP)

#define ROCBLAS_CHECK(err)                                                            \
  do {                                                                                \
    rocblas_status err_ = (err);                                                      \
    if(err_ != rocblas_status_success) {                                              \
      std::printf("ROCBLAS Exception code: %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("rocblas error");                                      \
    }                                                                                 \
  } while(0)
#endif // USE_HIP

#if defined(USE_DPCPP)
#include "oneapi/mkl.hpp"
#endif // USE_DPCPP

#endif // USE_TALSH

namespace tamm {

namespace kernels {

template<typename T1, typename T2, typename T3>
void copy_data_to_gpu(ExecutionHW hw, const std::vector<T2>& ainter_buf, T2** ainter_buf_dev,
                      const std::vector<T3>& binter_buf, T3** binter_buf_dev,
                      const std::vector<T1>& cinter_buf, T1** cinter_buf_dev) {
  if(hw != ExecutionHW::GPU) return;

  auto& gpuPool   = tamm::GPUPool::getInstance();
  auto  memPool   = gpuPool.get_memory_pool();
  auto& devStream = gpuPool.getStream();

  *ainter_buf_dev = static_cast<T2*>(memPool->allocate(ainter_buf.size() * sizeof(T2), devStream));
  *binter_buf_dev = static_cast<T3*>(memPool->allocate(binter_buf.size() * sizeof(T3), devStream));
  *cinter_buf_dev = static_cast<T1*>(memPool->allocate(cinter_buf.size() * sizeof(T1), devStream));

  // host-->device copy
#if defined(USE_DPCPP) && !defined(USE_TALSH)
  devStream.memcpy(*ainter_buf_dev, ainter_buf.data(), ainter_buf.size() * sizeof(T2)).wait();
  devStream.memcpy(*binter_buf_dev, binter_buf.data(), binter_buf.size() * sizeof(T3)).wait();
  devStream.memset(*cinter_buf_dev, 0, cinter_buf.size() * sizeof(T1)).wait();
#elif defined(USE_CUDA) && !defined(USE_TALSH)
  cudaMemcpy(*ainter_buf_dev, ainter_buf.data(), ainter_buf.size() * sizeof(T2),
             cudaMemcpyHostToDevice);
  cudaMemcpy(*binter_buf_dev, binter_buf.data(), binter_buf.size() * sizeof(T3),
             cudaMemcpyHostToDevice);
  cudaMemset(*cinter_buf_dev, 0, cinter_buf.size() * sizeof(T1));
#elif defined(USE_HIP) && !defined(USE_TALSH)
  hipMemcpy(*ainter_buf_dev, ainter_buf.data(), ainter_buf.size() * sizeof(T2),
            hipMemcpyHostToDevice);
  hipMemcpy(*binter_buf_dev, binter_buf.data(), binter_buf.size() * sizeof(T3),
            hipMemcpyHostToDevice);
  hipMemset(*cinter_buf_dev, 0, cinter_buf.size() * sizeof(T1));
#endif
}

template<typename T, typename T1, typename T2, typename T3>
void gemm_wrapper(ExecutionHW hw, int AR, int BR, int B, int M, int N, int K, T alpha, T beta,
                  std::vector<T2>& ainter_buf, T2* ainter_buf_dev, std::vector<T3>& binter_buf,
                  T3* binter_buf_dev, std::vector<T1>& cinter_buf, T1* cinter_buf_dev) {
#if defined(USE_DPCPP) && !defined(USE_TALSH)
  auto& gpuPool   = tamm::GPUPool::getInstance();
  auto& devStream = gpuPool.getStream();
#elif defined(USE_CUDA) && !defined(USE_TALSH)
  auto& gpuPool = tamm::GPUPool::getInstance();
  auto& handle  = gpuPool.getBlasHandle();
#elif defined(USE_HIP) && !defined(USE_TALSH)
  auto& gpuPool = tamm::GPUPool::getInstance();
  auto& handle  = gpuPool.getBlasHandle();
#endif

  auto transA     = blas::Op::NoTrans;
  auto transB     = blas::Op::NoTrans;
  int  ainter_ld  = K;
  int  binter_ld  = N;
  int  cinter_ld  = N;
  int  cbatch_ld  = M * N;
  int  abatch_ld  = M * K;
  int  bbatch_ld  = K * N;
  int  areduce_ld = B * abatch_ld;
  int  breduce_ld = B * bbatch_ld;

  for(size_t ari = 0; ari < AR; ari++) {
    for(size_t bri = 0; bri < BR; bri++) {
      for(size_t i = 0; i < B; i++) {
#if defined(USE_DPCPP)
        if(hw == ExecutionHW::GPU) {
          oneapi::mkl::blas::column_major::gemm(
            devStream, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, N, M, K, alpha,
            binter_buf_dev + bri * breduce_ld + i * bbatch_ld, binter_ld,
            ainter_buf_dev + ari * areduce_ld + i * abatch_ld, ainter_ld, beta,
            cinter_buf_dev + i * cbatch_ld, cinter_ld)
            .wait();
        }
        else {
          blas::gemm(blas::Layout::RowMajor, transA, transB, M, N, K, alpha,
                     ainter_buf.data() + ari * areduce_ld + i * abatch_ld, ainter_ld,
                     binter_buf.data() + bri * breduce_ld + i * bbatch_ld, binter_ld, beta,
                     cinter_buf.data() + i * cbatch_ld, cinter_ld);
        }
#elif defined(USE_CUDA) && !defined(USE_TALSH)
        if(hw == ExecutionHW::GPU) {
          if constexpr(internal::is_complex_v<T1> && internal::is_complex_v<T2> &&
                       internal::is_complex_v<T3>) {
            cuDoubleComplex calpha = make_cuDoubleComplex(alpha.real(), 0);
            cuDoubleComplex cbeta  = make_cuDoubleComplex(beta.real(), 0);
            CUBLAS_CHECK(cublasZgemm(
              handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, (cuDoubleComplex*) &alpha,
              (cuDoubleComplex*) binter_buf_dev + bri * breduce_ld + i * bbatch_ld, binter_ld,
              (cuDoubleComplex*) ainter_buf_dev + ari * areduce_ld + i * abatch_ld, ainter_ld,
              (cuDoubleComplex*) &beta, (cuDoubleComplex*) cinter_buf_dev + i * cbatch_ld,
              cinter_ld));
          }
          else {
            CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                      binter_buf_dev + bri * breduce_ld + i * bbatch_ld, binter_ld,
                                      ainter_buf_dev + ari * areduce_ld + i * abatch_ld, ainter_ld,
                                      &beta, cinter_buf_dev + i * cbatch_ld, cinter_ld));
          }
        }
        else {
          blas::gemm(blas::Layout::RowMajor, transA, transB, M, N, K, alpha,
                      ainter_buf.data() + ari * areduce_ld + i * abatch_ld, ainter_ld,
                      binter_buf.data() + bri * breduce_ld + i * bbatch_ld, binter_ld, beta,
                      cinter_buf.data() + i * cbatch_ld, cinter_ld);
        }
#elif defined(USE_HIP) && !defined(USE_TALSH)
        if(hw == ExecutionHW::GPU) {
          if constexpr(internal::is_complex_v<T1> && internal::is_complex_v<T2> &&
                       internal::is_complex_v<T3>) {
            ROCBLAS_CHECK(rocblas_zgemm(
              handle, rocblas_operation_none, rocblas_operation_none, N, M, K,
              (rocblas_double_complex*) &alpha,
              (rocblas_double_complex*) binter_buf_dev + bri * breduce_ld + i * bbatch_ld,
              binter_ld,
              (rocblas_double_complex*) ainter_buf_dev + ari * areduce_ld + i * abatch_ld,
              ainter_ld, (rocblas_double_complex*) &beta,
              (rocblas_double_complex*) cinter_buf_dev + i * cbatch_ld, cinter_ld));
          }
          else {
            ROCBLAS_CHECK(
              rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none, N, M, K, &alpha,
                             binter_buf_dev + bri * breduce_ld + i * bbatch_ld, binter_ld,
                             ainter_buf_dev + ari * areduce_ld + i * abatch_ld, ainter_ld, &beta,
                             cinter_buf_dev + i * cbatch_ld, cinter_ld));
          }
        }
        else {
          blas::gemm(blas::Layout::RowMajor, transA, transB, M, N, K, alpha,
                      ainter_buf.data() + ari * areduce_ld + i * abatch_ld, ainter_ld,
                      binter_buf.data() + bri * breduce_ld + i * bbatch_ld, binter_ld, beta,
                      cinter_buf.data() + i * cbatch_ld, cinter_ld);
        }
#else
        blas::gemm(blas::Layout::RowMajor, transA, transB, M, N, K, alpha,
                   ainter_buf.data() + ari * areduce_ld + i * abatch_ld, ainter_ld,
                   binter_buf.data() + bri * breduce_ld + i * bbatch_ld, binter_ld, beta,
                   cinter_buf.data() + i * cbatch_ld, cinter_ld);
#endif
      }
    }
  }
}

template<typename T1>
void copy_result_to_host(ExecutionHW hw, std::vector<T1>& cinter_buf, T1* cinter_buf_dev) {
  if(hw != ExecutionHW::GPU) return;

// device-->host copy
#if defined(USE_DPCPP) && !defined(USE_TALSH)
  auto& gpuPool   = tamm::GPUPool::getInstance();
  auto& devStream = gpuPool.getStream();
  devStream.memcpy(cinter_buf.data(), cinter_buf_dev, cinter_buf.size() * sizeof(T1)).wait();
#elif defined(USE_CUDA) && !defined(USE_TALSH)
  cudaMemcpy(cinter_buf.data(), cinter_buf_dev, cinter_buf.size() * sizeof(T1),
             cudaMemcpyDeviceToHost);
#elif defined(USE_HIP) && !defined(USE_TALSH)
  hipMemcpy(cinter_buf.data(), cinter_buf_dev, cinter_buf.size() * sizeof(T1),
            hipMemcpyDeviceToHost);
#endif
}

template<typename T1, typename T2, typename T3>
void free_device_buffers(ExecutionHW hw, std::size_t ainter_size, std::size_t binter_size,
                         std::size_t cinter_size, T2* ainter_buf_dev, T3* binter_buf_dev,
                         T1* cinter_buf_dev) {
  if(hw != ExecutionHW::GPU) return;

  auto& gpuPool   = tamm::GPUPool::getInstance();
  auto  memPool   = gpuPool.get_memory_pool();
  auto& devStream = gpuPool.getStream();

  memPool->deallocate(static_cast<void*>(ainter_buf_dev), ainter_size * sizeof(T2), devStream);
  memPool->deallocate(static_cast<void*>(binter_buf_dev), binter_size * sizeof(T3), devStream);
  memPool->deallocate(static_cast<void*>(cinter_buf_dev), cinter_size * sizeof(T1), devStream);
}

template<typename T, typename T1, typename T2, typename T3>
void block_multiply(bool& isgpuOp,
#if defined(USE_TALSH)
                    TALSH& gpu_mult, talsh_task_t& talsh_task, tensor_handle& th_c,
                    tensor_handle& th_a, tensor_handle& th_b, int copy_ctrl, int talsh_dev_id,
#endif
                    T alpha, const T2* abuf, const SizeVec& adims, const IntLabelVec& alabels,
                    const T3* bbuf, const SizeVec& bdims, const IntLabelVec& blabels, T beta,
                    T1* cbuf, const SizeVec& cdims, const IntLabelVec& clabels,
                    ExecutionHW hw = ExecutionHW::CPU, bool has_gpu = false,
                    bool is_assign = true) {

  const Size asize = std::accumulate(adims.begin(), adims.end(), Size{1}, std::multiplies<Size>());
  const Size bsize = std::accumulate(bdims.begin(), bdims.end(), Size{1}, std::multiplies<Size>());
  const Size csize = std::accumulate(cdims.begin(), cdims.end(), Size{1}, std::multiplies<Size>());

  EXPECTS(abuf != nullptr && bbuf != nullptr && cbuf != nullptr);

  IntLabelVec asorted_labels{alabels}, bsorted_labels{blabels}, csorted_labels{clabels};
  std::sort(asorted_labels.begin(), asorted_labels.end());
  std::sort(bsorted_labels.begin(), bsorted_labels.end());
  std::sort(csorted_labels.begin(), csorted_labels.end());

  std::vector<IntLabel> inner_labels, aouter_labels, bouter_labels, batch_labels, areduce_labels,
    breduce_labels;
  std::vector<Size> inner_dims, aouter_dims, bouter_dims, batch_dims, areduce_dims, breduce_dims;

  int B = 1, M = 1, N = 1, K = 1, AR = 1, BR = 1;
  for(size_t i = 0; i < cdims.size(); i++) {
    const auto& lbl     = clabels[i];
    bool        is_in_a = std::binary_search(asorted_labels.begin(), asorted_labels.end(), lbl);
    bool        is_in_b = std::binary_search(bsorted_labels.begin(), bsorted_labels.end(), lbl);
    if(is_in_a && is_in_b) {
      batch_labels.push_back(lbl);
      batch_dims.push_back(cdims[i]);
      B *= static_cast<int>(cdims[i].value());
    }
    else if(is_in_a) {
      aouter_labels.push_back(lbl);
      aouter_dims.push_back(cdims[i]);
      M *= static_cast<int>(cdims[i].value());
    }
    else if(is_in_b) {
      bouter_labels.push_back(lbl);
      bouter_dims.push_back(cdims[i]);
      N *= static_cast<int>(cdims[i].value());
    }
    else {
      // UNREACHABLE();
    }
  }

  for(size_t i = 0; i < adims.size(); i++) {
    const auto& lbl     = alabels[i];
    bool        is_in_b = std::binary_search(bsorted_labels.begin(), bsorted_labels.end(), lbl);
    bool        is_in_c = std::binary_search(csorted_labels.begin(), csorted_labels.end(), lbl);
    if(is_in_b && is_in_c) {
      // no-op -- already added in batch_labels
    }
    else if(is_in_b) {
      inner_labels.push_back(lbl);
      inner_dims.push_back(adims[i]);
      K *= static_cast<int>(adims[i].value());
    }
    else if(is_in_c) {
      // no-op -- already added to aouter
    }
    else {
      AR *= adims[i].value();
      areduce_dims.push_back(adims[i]);
      areduce_labels.push_back(lbl);
    }
  }

  for(size_t i = 0; i < bdims.size(); i++) {
    const auto& lbl     = blabels[i];
    bool        is_in_a = std::binary_search(asorted_labels.begin(), asorted_labels.end(), lbl);
    bool        is_in_c = std::binary_search(csorted_labels.begin(), csorted_labels.end(), lbl);
    if(is_in_a && is_in_c) {
      // no-op -- already added in batch_labels
    }
    else if(is_in_a) {
      // no-op -- already in inner_labels
    }
    else if(is_in_c) {
      // no-op -- already added to bouter
    }
    else {
      BR *= bdims[i].value();
      breduce_dims.push_back(bdims[i]);
      breduce_labels.push_back(lbl);
    }
  }

  std::vector<IntLabel> ainter_labels{areduce_labels};
  ainter_labels.insert(ainter_labels.end(), batch_labels.begin(), batch_labels.end());
  ainter_labels.insert(ainter_labels.end(), aouter_labels.begin(), aouter_labels.end());
  ainter_labels.insert(ainter_labels.end(), inner_labels.begin(), inner_labels.end());

  std::vector<IntLabel> binter_labels{breduce_labels};
  binter_labels.insert(binter_labels.end(), batch_labels.begin(), batch_labels.end());
  binter_labels.insert(binter_labels.end(), inner_labels.begin(), inner_labels.end());
  binter_labels.insert(binter_labels.end(), bouter_labels.begin(), bouter_labels.end());

  std::vector<IntLabel> cinter_labels{batch_labels};
  cinter_labels.insert(cinter_labels.end(), aouter_labels.begin(), aouter_labels.end());
  cinter_labels.insert(cinter_labels.end(), bouter_labels.begin(), bouter_labels.end());

  SizeVec ainter_dims{areduce_dims};
  ainter_dims.insert(ainter_dims.end(), batch_dims.begin(), batch_dims.end());
  ainter_dims.insert(ainter_dims.end(), aouter_dims.begin(), aouter_dims.end());
  ainter_dims.insert(ainter_dims.end(), inner_dims.begin(), inner_dims.end());

  SizeVec binter_dims{breduce_dims};
  binter_dims.insert(binter_dims.end(), batch_dims.begin(), batch_dims.end());
  binter_dims.insert(binter_dims.end(), inner_dims.begin(), inner_dims.end());
  binter_dims.insert(binter_dims.end(), bouter_dims.begin(), bouter_dims.end());

  SizeVec cinter_dims{batch_dims};
  cinter_dims.insert(cinter_dims.end(), aouter_dims.begin(), aouter_dims.end());
  cinter_dims.insert(cinter_dims.end(), bouter_dims.begin(), bouter_dims.end());

  auto transA     = blas::Op::NoTrans;
  auto transB     = blas::Op::NoTrans;
  int  ainter_ld  = K;
  int  binter_ld  = N;
  int  cinter_ld  = N;
  int  cbatch_ld  = M * N;
  int  abatch_ld  = M * K;
  int  bbatch_ld  = K * N;
  int  areduce_ld = B * abatch_ld;
  int  breduce_ld = B * bbatch_ld;

  auto bmult_cpu_lambda = [&]() {
    std::vector<T2> ainter_buf(static_cast<size_t>(asize.value()));
    std::vector<T3> binter_buf(static_cast<size_t>(bsize.value()));
    std::vector<T1> cinter_buf(static_cast<size_t>(csize.value()));
    assign<T2>(ainter_buf.data(), ainter_dims, ainter_labels, T2{1}, abuf, adims, alabels, true);
    assign<T3>(binter_buf.data(), binter_dims, binter_labels, T3{1}, bbuf, bdims, blabels, true);

    T2* ainter_buf_dev{nullptr};
    T3* binter_buf_dev{nullptr};
    T1* cinter_buf_dev{nullptr};

    // dgemm
    if constexpr(std::is_same_v<T1, T2> && std::is_same_v<T1, T3>) {
      copy_data_to_gpu(hw, ainter_buf, &ainter_buf_dev, binter_buf, &binter_buf_dev, cinter_buf,
                       &cinter_buf_dev);

      gemm_wrapper(hw, AR, BR, B, M, N, K, alpha, beta, ainter_buf, ainter_buf_dev, binter_buf,
                   binter_buf_dev, cinter_buf, cinter_buf_dev);

      copy_result_to_host(hw, cinter_buf, cinter_buf_dev);
      free_device_buffers(hw, ainter_buf.size(), binter_buf.size(), cinter_buf.size(),
                          ainter_buf_dev, binter_buf_dev, cinter_buf_dev);
    }
#if defined(USE_BLIS)
    else {
      // TODO: actually check if one of T2, T3 is real, T1 is complex
      if constexpr(std::is_same_v<T1, T2>) {
        // T2 (matrix A) is complex, T3 (B) is real
        if constexpr(internal::is_complex_v<T1>) {
          // copy B to complex buffer
          std::vector<T1> bbuf_complex(bsize.value());
          if constexpr(std::is_same_v<T3, double>)
            bli_dcopyv(BLIS_NO_CONJUGATE, bsize.value(), binter_buf.data(), 1,
                       reinterpret_cast<T3*>(bbuf_complex.data()), 2);
          else if constexpr(std::is_same_v<T3, float>)
            bli_scopyv(BLIS_NO_CONJUGATE, bsize.value(), binter_buf.data(), 1,
                       reinterpret_cast<T3*>(bbuf_complex.data()), 2);

          T1* bbuf_complex_dev{nullptr};
          copy_data_to_gpu(hw, ainter_buf, &ainter_buf_dev, bbuf_complex, &bbuf_complex_dev,
                           cinter_buf, &cinter_buf_dev);

          gemm_wrapper(hw, AR, BR, B, M, N, K, alpha, beta, ainter_buf, ainter_buf_dev,
                       bbuf_complex, bbuf_complex_dev, cinter_buf, cinter_buf_dev);

          copy_result_to_host(hw, cinter_buf, cinter_buf_dev);
          free_device_buffers(hw, ainter_buf.size(), bbuf_complex.size(), cinter_buf.size(),
                              ainter_buf_dev, bbuf_complex_dev, cinter_buf_dev);

        } // is_complex<T1>
        else {
          // T1,T2 (C,A) are real, T3 (B) is complex
          std::vector<T1> bbuf_real(bsize.value());
          if constexpr(std::is_same_v<T1, double>)
            bli_dcopyv(BLIS_NO_CONJUGATE, bsize.value(), reinterpret_cast<T1*>(binter_buf.data()),
                       2, bbuf_real.data(), 1);
          else if constexpr(std::is_same_v<T1, float>)
            bli_scopyv(BLIS_NO_CONJUGATE, bsize.value(), reinterpret_cast<T1*>(binter_buf.data()),
                       2, bbuf_real.data(), 1);

          T1* bbuf_real_dev{nullptr};
          copy_data_to_gpu(hw, ainter_buf, &ainter_buf_dev, bbuf_real, &bbuf_real_dev, cinter_buf,
                           &cinter_buf_dev);

          gemm_wrapper(hw, AR, BR, B, M, N, K, alpha, beta, ainter_buf, ainter_buf_dev, bbuf_real,
                       bbuf_real_dev, cinter_buf, cinter_buf_dev);

          copy_result_to_host(hw, cinter_buf, cinter_buf_dev);
          free_device_buffers(hw, ainter_buf.size(), bbuf_real.size(), cinter_buf.size(),
                              ainter_buf_dev, bbuf_real_dev, cinter_buf_dev);
        } // is_real<T1>

      } // is_same_v<T1,T2>
      else if constexpr(std::is_same_v<T1, T3>) {
        // T3 (matrix B) is complex, T2 (A) is real
        if constexpr(internal::is_complex_v<T1>) {
          std::vector<T1> abuf_complex(asize.value());
          if constexpr(std::is_same_v<T2, double>)
            bli_dcopyv(BLIS_NO_CONJUGATE, asize.value(), ainter_buf.data(), 1,
                       reinterpret_cast<T2*>(abuf_complex.data()), 2);
          else if constexpr(std::is_same_v<T2, float>)
            bli_scopyv(BLIS_NO_CONJUGATE, asize.value(), ainter_buf.data(), 1,
                       reinterpret_cast<T2*>(abuf_complex.data()), 2);

          T1* abuf_complex_dev{nullptr};
          copy_data_to_gpu(hw, abuf_complex, &abuf_complex_dev, binter_buf, &binter_buf_dev,
                           cinter_buf, &cinter_buf_dev);

          gemm_wrapper(hw, AR, BR, B, M, N, K, alpha, beta, abuf_complex, abuf_complex_dev,
                       binter_buf, binter_buf_dev, cinter_buf, cinter_buf_dev);

          copy_result_to_host(hw, cinter_buf, cinter_buf_dev);
          free_device_buffers(hw, abuf_complex.size(), binter_buf.size(), cinter_buf.size(),
                              abuf_complex_dev, binter_buf_dev, cinter_buf_dev);
        }
        else {
          // T1,T3 (C,B) are real, T2 (A) is complex
          std::vector<T1> abuf_real(asize.value());
          if constexpr(std::is_same_v<T1, double>)
            bli_dcopyv(BLIS_NO_CONJUGATE, asize.value(), reinterpret_cast<T1*>(ainter_buf.data()),
                       2, abuf_real.data(), 1);
          else if constexpr(std::is_same_v<T1, float>)
            bli_scopyv(BLIS_NO_CONJUGATE, asize.value(), reinterpret_cast<T1*>(ainter_buf.data()),
                       2, abuf_real.data(), 1);

          T1* abuf_real_dev{nullptr};
          copy_data_to_gpu(hw, abuf_real, &abuf_real_dev, binter_buf, &binter_buf_dev, cinter_buf,
                           &cinter_buf_dev);

          gemm_wrapper(hw, AR, BR, B, M, N, K, alpha, beta, abuf_real, abuf_real_dev, binter_buf,
                       binter_buf_dev, cinter_buf, cinter_buf_dev);

          copy_result_to_host(hw, cinter_buf, cinter_buf_dev);
          free_device_buffers(hw, abuf_real.size(), binter_buf.size(), cinter_buf.size(),
                              abuf_real_dev, binter_buf_dev, cinter_buf_dev);
        }

      } // is_same_v<T1,T3>

      else if constexpr(internal::is_complex_v<T1> && std::is_same_v<T2, T3>) {
        // T1 is complex, T2, T3 are real
        std::vector<T1> abuf_complex(asize.value());
        T2*             abuf_comp_ptr = reinterpret_cast<T2*>(abuf_complex.data());
        std::vector<T1> bbuf_complex(bsize.value());
        T2*             bbuf_comp_ptr = reinterpret_cast<T2*>(bbuf_complex.data());

        if constexpr(std::is_same_v<T2, double>) {
          bli_dcopyv(BLIS_NO_CONJUGATE, asize.value(), ainter_buf.data(), 1, abuf_comp_ptr, 2);
          bli_dcopyv(BLIS_NO_CONJUGATE, bsize.value(), binter_buf.data(), 1, bbuf_comp_ptr, 2);
        }
        else if constexpr(std::is_same_v<T2, float>) {
          bli_scopyv(BLIS_NO_CONJUGATE, asize.value(), ainter_buf.data(), 1, abuf_comp_ptr, 2);
          bli_scopyv(BLIS_NO_CONJUGATE, bsize.value(), binter_buf.data(), 1, bbuf_comp_ptr, 2);
        }

        T1* abuf_complex_dev{nullptr};
        T2* bbuf_complex_dev{nullptr};
        copy_data_to_gpu(hw, abuf_complex, &abuf_complex_dev, bbuf_complex, &bbuf_complex_dev,
                         cinter_buf, &cinter_buf_dev);

        gemm_wrapper(hw, AR, BR, B, M, N, K, alpha, beta, abuf_complex, abuf_complex_dev,
                     bbuf_complex, bbuf_complex_dev, cinter_buf, cinter_buf_dev);

        copy_result_to_host(hw, cinter_buf, cinter_buf_dev);
        free_device_buffers(hw, abuf_complex.size(), bbuf_complex.size(), cinter_buf.size(),
                            abuf_complex_dev, bbuf_complex_dev, cinter_buf_dev);
      }

      else
        NOT_IMPLEMENTED();
    }
#endif

    assign<T1>(cbuf, cdims, clabels, T{1}, cinter_buf.data(), cinter_dims, cinter_labels,
               is_assign);
  };
#ifndef USE_TALSH
  bmult_cpu_lambda();

#else

  auto talsh_op_string = internal::talsh_mult_op_string(clabels, alabels, blabels);

  auto aid_size = adims.size();
  auto bid_size = bdims.size();
  auto cid_size = cdims.size();
  int  tal_adims[aid_size];
  int  tal_bdims[bid_size];
  int  tal_cdims[cid_size];

  std::vector<int> taid;
  std::vector<int> tbid;
  std::vector<int> tcid;
  std::transform(std::begin(adims), std::end(adims), std::back_inserter(taid),
                 [](tamm::Size i) -> int { return i.value(); });
  std::transform(std::begin(bdims), std::end(bdims), std::back_inserter(tbid),
                 [](tamm::Size i) -> int { return i.value(); });
  std::transform(std::begin(cdims), std::end(cdims), std::back_inserter(tcid),
                 [](tamm::Size i) -> int { return i.value(); });

  std::reverse(taid.begin(), taid.end());
  std::reverse(tbid.begin(), tbid.end());
  std::reverse(tcid.begin(), tcid.end());

  std::copy(taid.begin(), taid.end(), tal_adims);
  std::copy(tbid.begin(), tbid.end(), tal_bdims);
  std::copy(tcid.begin(), tcid.end(), tal_cdims);

  bool hadamard = false;
  for(auto x: cinter_labels) {
    auto r1 = std::find(std::begin(ainter_labels), std::end(ainter_labels), x) !=
              std::end(ainter_labels);
    auto r2 = std::find(std::begin(binter_labels), std::end(binter_labels), x) !=
              std::end(binter_labels);
    if(r1 && r2) hadamard = true;
  }

  bool reduction_op = false;
  for(auto x: ainter_labels) {
    auto r1 = std::find(std::begin(cinter_labels), std::end(cinter_labels), x) ==
              std::end(cinter_labels);
    auto r2 = std::find(std::begin(binter_labels), std::end(binter_labels), x) ==
              std::end(binter_labels);
    if(r1 && r2) reduction_op = true;
  }
  for(auto x: binter_labels) {
    auto r1 = std::find(std::begin(cinter_labels), std::end(cinter_labels), x) ==
              std::end(cinter_labels);
    auto r2 = std::find(std::begin(ainter_labels), std::end(ainter_labels), x) ==
              std::end(ainter_labels);
    if(r1 && r2) reduction_op = true;
  }

  if(hadamard || reduction_op || hw == ExecutionHW::CPU || !has_gpu) { bmult_cpu_lambda(); }

  else {
    isgpuOp = true;
    // std::cout << "not hadamard\n";
    // std::cout << talsh_op_string << std::endl;
    // std::cout << aid_size << ":" << bid_size << ":" << cid_size << std::endl;

    // adata, bdata, cdata will have to be created
    // using pinned memory else where for now using
    // regular memory
    // double *adata = host_pinned_memory(abatch_ld*sizeof(double));
    // double *bdata = host_pinned_memory(bbatch_ld*sizeof(double));
    // double *cdata = host_pinned_memory(cbatch_ld*sizeof(double));

    // TALSH gpu_mult{ngpu};
    T2* abufp = const_cast<T2*>(abuf);
    T3* bbufp = const_cast<T3*>(bbuf);

    if constexpr(std::is_same_v<T1, T2> && std::is_same_v<T1, T3>) {
      th_a = gpu_mult.host_block(adims.size(), tal_adims, abufp);
      th_b = gpu_mult.host_block(bdims.size(), tal_bdims, bbufp);
      if(copy_ctrl == COPY_TTT) th_c = gpu_mult.host_block(cdims.size(), tal_cdims, cbuf);

      gpu_mult.mult_block(talsh_task, talsh_dev_id, th_c, th_a, th_b, talsh_op_string, alpha,
                          copy_ctrl, is_assign);

      // talshTensorDestruct(&th_a);
      // talshTensorDestruct(&th_b);
      // talshTensorDestruct(&th_c);
    }
#if defined(USE_BLIS)
    else {
      // TODO: actually check if one of T2, T3 is real, T1 is complex
      if constexpr(std::is_same_v<T1, T2>) {
        // T2 (matrix A) is complex, T3 (B) is real
        if constexpr(internal::is_complex_v<T1>) {
          // copy B to complex buffer
          std::vector<T1> bbuf_complex(bsize.value());
          T3*             bbuf_comp_ptr = reinterpret_cast<T3*>(&bbuf_complex[0]);
          if constexpr(std::is_same_v<T3, double>)
            bli_dcopyv(BLIS_NO_CONJUGATE, bsize.value(), bbufp, 1, bbuf_comp_ptr, 2);
          else if constexpr(std::is_same_v<T3, float>)
            bli_scopyv(BLIS_NO_CONJUGATE, bsize.value(), bbufp, 1, bbuf_comp_ptr, 2);

          th_a = gpu_mult.host_block(adims.size(), tal_adims, abufp);
          th_b = gpu_mult.host_block(bdims.size(), tal_bdims, bbuf_complex.data());
          if(copy_ctrl == COPY_TTT) th_c = gpu_mult.host_block(cdims.size(), tal_cdims, cbuf);

          gpu_mult.mult_block(talsh_task, talsh_dev_id, th_c, th_a, th_b, talsh_op_string, alpha,
                              copy_ctrl, is_assign);

          // talshTensorDestruct(&th_a);
          // talshTensorDestruct(&th_b);
          // talshTensorDestruct(&th_c);

        } // is_complex<T1>
        else {
          // T1,T2 (C,A) are real, T3 (B) is complex
          std::vector<T1> bbuf_real(bsize.value());
          T1*             bbuf_comp_ptr = reinterpret_cast<T1*>(bbufp);
          if constexpr(std::is_same_v<T1, double>)
            bli_dcopyv(BLIS_NO_CONJUGATE, bsize.value(), bbuf_comp_ptr, 2, &bbuf_real[0], 1);
          else if constexpr(std::is_same_v<T1, float>)
            bli_scopyv(BLIS_NO_CONJUGATE, bsize.value(), bbuf_comp_ptr, 2, &bbuf_real[0], 1);

          th_a = gpu_mult.host_block(adims.size(), tal_adims, abufp);
          th_b = gpu_mult.host_block(bdims.size(), tal_bdims, bbuf_real.data());
          if(copy_ctrl == COPY_TTT) th_c = gpu_mult.host_block(cdims.size(), tal_cdims, cbuf);

          gpu_mult.mult_block(talsh_task, talsh_dev_id, th_c, th_a, th_b, talsh_op_string, alpha,
                              copy_ctrl, is_assign);

          // talshTensorDestruct(&th_a);
          // talshTensorDestruct(&th_b);
          // talshTensorDestruct(&th_c);

        } // is_real<T1>

      } // is_same_v<T1,T2>
      else if constexpr(std::is_same_v<T1, T3>) {
        // T3 (matrix B) is complex, T2 (A) is real
        if constexpr(internal::is_complex_v<T1>) {
          std::vector<T1> abuf_complex(asize.value());
          T2*             abuf_comp_ptr = reinterpret_cast<T2*>(&abuf_complex[0]);
          if constexpr(std::is_same_v<T2, double>)
            bli_dcopyv(BLIS_NO_CONJUGATE, asize.value(), abufp, 1, abuf_comp_ptr, 2);
          else if constexpr(std::is_same_v<T2, float>)
            bli_scopyv(BLIS_NO_CONJUGATE, asize.value(), abufp, 1, abuf_comp_ptr, 2);

          th_a = gpu_mult.host_block(adims.size(), tal_adims, abuf_complex.data());
          th_b = gpu_mult.host_block(bdims.size(), tal_bdims, bbufp);
          if(copy_ctrl == COPY_TTT) th_c = gpu_mult.host_block(cdims.size(), tal_cdims, cbuf);

          gpu_mult.mult_block(talsh_task, talsh_dev_id, th_c, th_a, th_b, talsh_op_string, alpha,
                              copy_ctrl, is_assign);

          // talshTensorDestruct(&th_a);
          // talshTensorDestruct(&th_b);
          // talshTensorDestruct(&th_c);
        }
        else {
          // T1,T3 (C,B) are real, T2 (A) is complex
          std::vector<T1> abuf_real(asize.value());
          T1*             abuf_comp_ptr = reinterpret_cast<T1*>(abufp);
          if constexpr(std::is_same_v<T1, double>)
            bli_dcopyv(BLIS_NO_CONJUGATE, asize.value(), abuf_comp_ptr, 2, &abuf_real[0], 1);
          else if constexpr(std::is_same_v<T1, float>)
            bli_scopyv(BLIS_NO_CONJUGATE, asize.value(), abuf_comp_ptr, 2, &abuf_real[0], 1);

          th_a = gpu_mult.host_block(adims.size(), tal_adims, abuf_real.data());
          th_b = gpu_mult.host_block(bdims.size(), tal_bdims, bbufp);
          if(copy_ctrl == COPY_TTT) th_c = gpu_mult.host_block(cdims.size(), tal_cdims, cbuf);

          gpu_mult.mult_block(talsh_task, talsh_dev_id, th_c, th_a, th_b, talsh_op_string, alpha,
                              copy_ctrl, is_assign);

          // talshTensorDestruct(&th_a);
          // talshTensorDestruct(&th_b);
          // talshTensorDestruct(&th_c);
        }

      } // is_same_v<T1,T3>

      else if constexpr(internal::is_complex_v<T1> && std::is_same_v<T2, T3>) {
        // T1 is complex, T2, T3 are real
        std::vector<T1> abuf_complex(asize.value());
        T2*             abuf_comp_ptr = reinterpret_cast<T2*>(&abuf_complex[0]);
        std::vector<T1> bbuf_complex(bsize.value());
        T2*             bbuf_comp_ptr = reinterpret_cast<T2*>(&bbuf_complex[0]);

        if constexpr(std::is_same_v<T2, double>) {
          bli_dcopyv(BLIS_NO_CONJUGATE, asize.value(), abufp, 1, abuf_comp_ptr, 2);
          bli_dcopyv(BLIS_NO_CONJUGATE, bsize.value(), bbufp, 1, bbuf_comp_ptr, 2);
        }
        else if constexpr(std::is_same_v<T2, float>) {
          bli_scopyv(BLIS_NO_CONJUGATE, asize.value(), abufp, 1, abuf_comp_ptr, 2);
          bli_scopyv(BLIS_NO_CONJUGATE, bsize.value(), bbufp, 1, bbuf_comp_ptr, 2);
        }

        th_a = gpu_mult.host_block(adims.size(), tal_adims, abuf_complex.data());
        th_b = gpu_mult.host_block(bdims.size(), tal_bdims, bbuf_complex.data());
        if(copy_ctrl == COPY_TTT) th_c = gpu_mult.host_block(cdims.size(), tal_cdims, cbuf);

        gpu_mult.mult_block(talsh_task, talsh_dev_id, th_c, th_a, th_b, talsh_op_string, alpha,
                            copy_ctrl, is_assign);

        // talshTensorDestruct(&th_a);
        // talshTensorDestruct(&th_b);
        // talshTensorDestruct(&th_c);
      }
      else
        NOT_IMPLEMENTED();
    }
#endif

    // Create tensor objects
    // tensor_handle th_a = gpu_mult.host_block(adims.size(),
    //     tal_adims, abufp);
    // tensor_handle th_b = gpu_mult.host_block(bdims.size(),
    //     tal_bdims, bbufp);
    // tensor_handle th_c = gpu_mult.host_block(cdims.size(),
    //     tal_cdims, cbuf);

    // gpu_mult.mult_block(talsh_task,talsh_dev_id, th_c, th_a, th_b, talsh_op_string,
    //     alpha, copy_ctrl, is_assign);

    // talshTensorDestruct(&th_a);
    // talshTensorDestruct(&th_b);
    // talshTensorDestruct(&th_c);
    // free_host_pinned_memory(adata);
    // free_host_pinned_memory(bdata);
    // free_host_pinned_memory(cdata);
  }
#endif
} // block_multiply()

} // namespace kernels

} // namespace tamm
