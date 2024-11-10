#pragma once

#include "tamm/errors.hpp"
#include "tamm/kernels/assign.hpp"
#include "tamm/types.hpp"

#include <complex>
#include <cstring> // for std::memset
#include <numeric>
#include <vector>

#include "tamm/op_profiler.hpp"
#include "tamm/rmm_memory_pool.hpp"
#include "tamm/utils.hpp"
#include "tamm_blas.hpp"

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#include "librett/librett.h"
#else
namespace tamm {
using gpuStream_t = int; // not used
}
#endif

namespace tamm {

namespace kernels {

template<typename T2, typename T3>
void copy_data_to_gpu(ExecutionHW hw, gpuStream_t& thandle, const T2* ainter_buf, size_t asize,
                      T2* ainter_buf_dev, const T3* binter_buf, size_t bsize, T3* binter_buf_dev) {
  if(hw == ExecutionHW::CPU) return;

  auto&      oprof = tamm::OpProfiler::instance();
  TimerGuard tg_copy{&oprof.multOpCopyTime};
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  gpuMemcpyAsync<T2>(ainter_buf_dev, ainter_buf, asize, gpuMemcpyHostToDevice, thandle);
  gpuMemcpyAsync<T3>(binter_buf_dev, binter_buf, bsize, gpuMemcpyHostToDevice, thandle);
#endif
}

template<typename T, typename T1, typename T2, typename T3>
void gemm_wrapper(ExecutionHW hw, gpuStream_t& thandle, int AR, int BR, int B, int M, int N, int K,
                  T alpha, T beta, const T2* ainter_buf, const T2* ainter_buf_dev,
                  const T3* binter_buf, const T3* binter_buf_dev, T1*& cinter_buf,
                  T1*& cinter_buf_dev) {
  int ainter_ld  = K;
  int binter_ld  = N;
  int cinter_ld  = N;
  int cbatch_ld  = M * N;
  int abatch_ld  = M * K;
  int bbatch_ld  = K * N;
  int areduce_ld = B * abatch_ld;
  int breduce_ld = B * bbatch_ld;

  for(size_t ari = 0; ari < AR; ari++) {
    for(size_t bri = 0; bri < BR; bri++) {
      for(size_t i = 0; i < B; i++) {
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
        if(hw == ExecutionHW::GPU) {
          gpu::gemm(N, M, K, alpha, binter_buf_dev + bri * breduce_ld + i * bbatch_ld, binter_ld,
                    ainter_buf_dev + ari * areduce_ld + i * abatch_ld, ainter_ld, beta,
                    cinter_buf_dev + i * cbatch_ld, cinter_ld, thandle);
          continue;
        }
#endif
        cpu::gemm(M, N, K, alpha, ainter_buf + ari * areduce_ld + i * abatch_ld, ainter_ld,
                  binter_buf + bri * breduce_ld + i * bbatch_ld, binter_ld, beta,
                  cinter_buf + i * cbatch_ld, cinter_ld);

      } // for-i
    }   // for-bri
  }     // for-ari
}

template<typename T>
void allocate_host_buffers(ExecutionHW hw, T*& host_buf, size_t buf_size) {
  if(hw == ExecutionHW::GPU) return;
  auto& memPool = RMMMemoryManager::getInstance().getHostMemoryPool();
  host_buf      = static_cast<T*>(memPool.allocate(buf_size * sizeof(T)));
}

template<typename T>
void free_host_buffers(ExecutionHW hw, T*& host_buf, std::size_t buf_size) {
  if(hw == ExecutionHW::GPU) return;
  auto& memPool = RMMMemoryManager::getInstance().getHostMemoryPool();
  memPool.deallocate(host_buf, buf_size * sizeof(T));
}

template<typename T>
void allocate_device_buffers(ExecutionHW hw, T*& dev_buf, size_t buf_size) {
  if(hw == ExecutionHW::CPU) return;
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  auto& memPool = RMMMemoryManager::getInstance().getDeviceMemoryPool();
  dev_buf       = static_cast<T*>(memPool.allocate(buf_size * sizeof(T)));
#endif
}

template<typename T>
void free_device_buffers(ExecutionHW hw, T*& dev_buf, std::size_t buf_size) {
  if(hw == ExecutionHW::CPU) return;
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  auto& memPool = RMMMemoryManager::getInstance().getDeviceMemoryPool();
  memPool.deallocate(static_cast<void*>(dev_buf), buf_size * sizeof(T));
#endif
}

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
template<typename T>
void assign_gpu(gpuStream_t& thandle, T*& dst, const SizeVec& ddims, const IntLabelVec& dlabels,
                T scale, const T* src, const SizeVec& sdims, const IntLabelVec& slabels,
                bool is_assign) {
  const int ndim = sdims.size();

  const Size ssize = std::accumulate(sdims.begin(), sdims.end(), Size{1}, std::multiplies<Size>());
  if(ndim <= 1 || ssize.value() == 1) {
    // device-->device copy
    gpuMemcpyAsync<T>(dst, src, ssize.value(), gpuMemcpyDeviceToDevice, thandle);
  }

  std::vector<int> r_sdims;
  std::transform(std::begin(sdims), std::end(sdims), std::back_inserter(r_sdims),
                 [](tamm::Size i) -> int { return i.value(); });

  tamm::IntLabelVec r_dlabels = dlabels;
  tamm::IntLabelVec r_slabels = slabels;

  // if(is_assign)
  std::reverse(r_sdims.begin(), r_sdims.end());
  std::reverse(r_slabels.begin(), r_slabels.end());
  std::reverse(r_dlabels.begin(), r_dlabels.end());

  int perm[ndim];
  int size[ndim];
  // T beta         = is_assign ? 0 : 1;

  for(size_t i = 0; i < r_sdims.size(); i++) { size[i] = r_sdims[i]; }
  for(size_t i = 0; i < r_dlabels.size(); i++) {
    auto it = std::find(r_slabels.begin(), r_slabels.end(), r_dlabels[i]);
    EXPECTS(it != r_slabels.end());
    perm[i] = it - r_slabels.begin();
  }

  // create plan
  librettHandle plan;
#if defined(USE_DPCPP)
  sycl::queue* ptrQueue = &(thandle.first);
  librettPlan(&plan, ndim, size, perm, sizeof(T), ptrQueue);
#else
  librettPlan(&plan, ndim, size, perm, sizeof(T), thandle.first);
#endif

  // ABB: following casts were required since librett API only accepts void* as args
  librettExecute(plan, reinterpret_cast<void*>(const_cast<T*>(src)), reinterpret_cast<void*>(dst));
  librettDestroy(plan);
}
#endif

template<typename T2, typename T3>
bool transpose_inputs(ExecutionHW hw, gpuStream_t& thandle, T2* ainter_buf,
                      const SizeVec& ainter_dims, const IntLabelVec& ainter_labels, const T2* abuf,
                      size_t asize, const SizeVec& adims, const IntLabelVec& alabels,
                      T3* binter_buf, const SizeVec& binter_dims, const IntLabelVec& binter_labels,
                      const T3* bbuf, size_t bsize, const SizeVec& bdims,
                      const IntLabelVec& blabels, T2*& ainter_buf_dev, T3*& binter_buf_dev) {
  bool gpu_trans = false;

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  if(hw == ExecutionHW::GPU) {
    gpu_trans = true;

    T2* ainter_buf_dev_in{nullptr};
    T3* binter_buf_dev_in{nullptr};
    allocate_device_buffers(hw, ainter_buf_dev_in, asize);
    allocate_device_buffers(hw, binter_buf_dev_in, bsize);

    copy_data_to_gpu(hw, thandle, abuf, asize, ainter_buf_dev_in, bbuf, bsize, binter_buf_dev_in);

    assign_gpu<T2>(thandle, ainter_buf_dev, ainter_dims, ainter_labels, T2{1}, ainter_buf_dev_in,
                   adims, alabels, true);
    assign_gpu<T3>(thandle, binter_buf_dev, binter_dims, binter_labels, T3{1}, binter_buf_dev_in,
                   bdims, blabels, true);

    free_device_buffers(hw, ainter_buf_dev_in, asize);
    free_device_buffers(hw, binter_buf_dev_in, bsize);

    return gpu_trans;
  }
#endif

  assign<T2>(ainter_buf, ainter_dims, ainter_labels, T2{1}, abuf, adims, alabels, true);
  assign<T3>(binter_buf, binter_dims, binter_labels, T3{1}, bbuf, bdims, blabels, true);
  return gpu_trans;
}

template<typename T1>
void transpose_output(ExecutionHW hw, gpuStream_t& thandle, bool gpu_trans, T1* cinter_buf,
                      const SizeVec& cinter_dims, const IntLabelVec& cinter_labels, T1* cbuf,
                      const SizeVec& cdims, const IntLabelVec& clabels, T1*& cinter_buf_dev,
                      T1*& cinter_tmp_buf_dev, bool is_assign) {
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  if(hw == ExecutionHW::GPU) {
    assign_gpu<T1>(thandle, cinter_buf_dev, cdims, clabels, T1{1}, cinter_tmp_buf_dev, cinter_dims,
                   cinter_labels, is_assign);
    return;
  }
#endif

  assign<T1>(cbuf, cdims, clabels, T1{1}, cinter_buf, cinter_dims, cinter_labels, is_assign);
}

template<typename T, typename T1, typename T2, typename T3>
void block_multiply(
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  T2*& th_a, T3*& th_b,
#endif
  gpuStream_t& thandle, T alpha, const T2* abuf, const SizeVec& adims, const IntLabelVec& alabels,
  const T3* bbuf, const SizeVec& bdims, const IntLabelVec& blabels, T beta, T1* cbuf,
  const SizeVec& cdims, const IntLabelVec& clabels, ExecutionHW hw, bool is_assign,
  T1*& cinter_buf_dev, T1*& cinter_tmp_buf_dev) {

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

  // int ainter_ld  = K;
  // int binter_ld  = N;
  // int cinter_ld  = N;
  // int cbatch_ld  = M * N;
  // int abatch_ld  = M * K;
  // int bbatch_ld  = K * N;
  // int areduce_ld = B * abatch_ld;
  // int breduce_ld = B * bbatch_ld;

  bool gpu_trans = false;

  T1* cinter_buf{nullptr};
  allocate_host_buffers(hw, cinter_buf, static_cast<size_t>(csize.value()));
  if(hw == ExecutionHW::CPU) {
    // if(csize.value() != 1)
    std::memset(static_cast<void*>(cinter_buf), 0, static_cast<size_t>(csize.value() * sizeof(T1)));
  }

  T2* ainter_buf_dev{nullptr};
  T3* binter_buf_dev{nullptr};
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  ainter_buf_dev = th_a;
  binter_buf_dev = th_b;
#endif

  // dgemm
  if constexpr(std::is_same_v<T1, T2> && std::is_same_v<T1, T3>) { // R=RxR, C=CxC
    T2* ainter_buf{nullptr};
    T3* binter_buf{nullptr};
    allocate_host_buffers(hw, ainter_buf, asize.value());
    allocate_host_buffers(hw, binter_buf, bsize.value());

    gpu_trans = transpose_inputs(hw, thandle, ainter_buf, ainter_dims, ainter_labels, abuf,
                                 asize.value(), adims, alabels, binter_buf, binter_dims,
                                 binter_labels, bbuf, bsize.value(), bdims, blabels, ainter_buf_dev,
                                 binter_buf_dev);

    if(!gpu_trans)
      copy_data_to_gpu(hw, thandle, ainter_buf, asize.value(), ainter_buf_dev, binter_buf,
                       bsize.value(), binter_buf_dev);

    gemm_wrapper(hw, thandle, AR, BR, B, M, N, K, alpha, beta, ainter_buf, ainter_buf_dev,
                 binter_buf, binter_buf_dev, cinter_buf, cinter_tmp_buf_dev);

    transpose_output(hw, thandle, gpu_trans, cinter_buf, cinter_dims, cinter_labels, cbuf, cdims,
                     clabels, cinter_buf_dev, cinter_tmp_buf_dev, is_assign);

    free_host_buffers(hw, ainter_buf, asize.value());
    free_host_buffers(hw, binter_buf, bsize.value());
  }
  else {
    T2* abufp = const_cast<T2*>(abuf);
    T3* bbufp = const_cast<T3*>(bbuf);
    // TODO: actually check if one of T2, T3 is real, T1 is complex
    if constexpr(std::is_same_v<T1, T2>) {
      T2* ainter_buf{nullptr};
      T1* binter_buf{nullptr};
      allocate_host_buffers(hw, ainter_buf, asize.value());
      allocate_host_buffers(hw, binter_buf, bsize.value());

      // T2 (matrix A) is complex, T3 (B) is real, C=CxR
      if constexpr(internal::is_complex_v<T1>) {
        // copy B to complex buffer
        T1* bbuf_complex{nullptr};
        allocate_host_buffers(ExecutionHW::CPU, bbuf_complex, bsize.value());
        std::copy(bbufp, bbufp + bsize.value(), bbuf_complex);

        T1* bbuf_complex_dev{nullptr};
        allocate_device_buffers(hw, bbuf_complex_dev, bsize.value());

        gpu_trans = transpose_inputs(hw, thandle, ainter_buf, ainter_dims, ainter_labels, abuf,
                                     asize.value(), adims, alabels, binter_buf, binter_dims,
                                     binter_labels, bbuf_complex, bsize.value(), bdims, blabels,
                                     ainter_buf_dev, bbuf_complex_dev);

        if(!gpu_trans) {
          bbuf_complex = binter_buf;
          copy_data_to_gpu(hw, thandle, ainter_buf, asize.value(), ainter_buf_dev, bbuf_complex,
                           bsize.value(), bbuf_complex_dev);
        }

        gemm_wrapper(hw, thandle, AR, BR, B, M, N, K, alpha, beta, ainter_buf, ainter_buf_dev,
                     bbuf_complex, bbuf_complex_dev, cinter_buf, cinter_tmp_buf_dev);
        transpose_output(hw, thandle, gpu_trans, cinter_buf, cinter_dims, cinter_labels, cbuf,
                         cdims, clabels, cinter_buf_dev, cinter_tmp_buf_dev, is_assign);

        free_device_buffers(hw, bbuf_complex_dev, bsize.value());
        if(gpu_trans) free_host_buffers(ExecutionHW::CPU, bbuf_complex, bsize.value());
      } // is_complex<T1>
      else {
        // T1,T2 (C,A) are real, T3 (B) is complex, R=RxC
        T1* bbuf_real{nullptr};
        allocate_host_buffers(ExecutionHW::CPU, bbuf_real, bsize.value());
        std::transform(bbufp, bbufp + bsize.value(), bbuf_real,
                       [](const T3& val) { return val.real(); });

        T1* bbuf_real_dev{nullptr};
        allocate_device_buffers(hw, bbuf_real_dev, bsize.value());

        gpu_trans = transpose_inputs(hw, thandle, ainter_buf, ainter_dims, ainter_labels, abuf,
                                     asize.value(), adims, alabels, binter_buf, binter_dims,
                                     binter_labels, bbuf_real, bsize.value(), bdims, blabels,
                                     ainter_buf_dev, bbuf_real_dev);

        if(!gpu_trans) {
          bbuf_real = binter_buf;
          copy_data_to_gpu(hw, thandle, ainter_buf, asize.value(), ainter_buf_dev, bbuf_real,
                           bsize.value(), bbuf_real_dev);
        }

        gemm_wrapper(hw, thandle, AR, BR, B, M, N, K, alpha, beta, ainter_buf, ainter_buf_dev,
                     bbuf_real, bbuf_real_dev, cinter_buf, cinter_tmp_buf_dev);
        transpose_output(hw, thandle, gpu_trans, cinter_buf, cinter_dims, cinter_labels, cbuf,
                         cdims, clabels, cinter_buf_dev, cinter_tmp_buf_dev, is_assign);

        free_device_buffers(hw, bbuf_real_dev, bsize.value());
        if(gpu_trans) free_host_buffers(ExecutionHW::CPU, bbuf_real, bsize.value());
      } // is_real<T1>

      free_host_buffers(hw, ainter_buf, asize.value());
      free_host_buffers(hw, binter_buf, bsize.value());
    } // is_same_v<T1,T2>
    else if constexpr(std::is_same_v<T1, T3>) {
      T1* ainter_buf{nullptr};
      T3* binter_buf{nullptr};
      allocate_host_buffers(hw, ainter_buf, asize.value());
      allocate_host_buffers(hw, binter_buf, bsize.value());

      // T3 (matrix B) is complex, T2 (A) is real, C=RxC
      if constexpr(internal::is_complex_v<T1>) {
        T1* abuf_complex{nullptr};
        allocate_host_buffers(ExecutionHW::CPU, abuf_complex, asize.value());
        std::copy(abufp, abufp + asize.value(), abuf_complex);

        T1* abuf_complex_dev{nullptr};
        allocate_device_buffers(hw, abuf_complex_dev, asize.value());

        gpu_trans = transpose_inputs(hw, thandle, ainter_buf, ainter_dims, ainter_labels,
                                     abuf_complex, asize.value(), adims, alabels, binter_buf,
                                     binter_dims, binter_labels, bbuf, bsize.value(), bdims,
                                     blabels, abuf_complex_dev, binter_buf_dev);

        if(!gpu_trans) {
          abuf_complex = ainter_buf;
          copy_data_to_gpu(hw, thandle, abuf_complex, asize.value(), abuf_complex_dev, binter_buf,
                           bsize.value(), binter_buf_dev);
        }

        gemm_wrapper(hw, thandle, AR, BR, B, M, N, K, alpha, beta, abuf_complex, abuf_complex_dev,
                     binter_buf, binter_buf_dev, cinter_buf, cinter_tmp_buf_dev);

        transpose_output(hw, thandle, gpu_trans, cinter_buf, cinter_dims, cinter_labels, cbuf,
                         cdims, clabels, cinter_buf_dev, cinter_tmp_buf_dev, is_assign);

        free_device_buffers(hw, abuf_complex_dev, asize.value());
        if(gpu_trans) free_host_buffers(ExecutionHW::CPU, abuf_complex, asize.value());
      }
      else {
        // T1,T3 (C,B) are real, T2 (A) is complex, //R=CxR
        T1* abuf_real{nullptr};
        allocate_host_buffers(ExecutionHW::CPU, abuf_real, asize.value());
        std::transform(abufp, abufp + asize.value(), abuf_real,
                       [](const T2& val) { return val.real(); });

        T1* abuf_real_dev{nullptr};
        allocate_device_buffers(hw, abuf_real_dev, asize.value());

        gpu_trans = transpose_inputs(hw, thandle, ainter_buf, ainter_dims, ainter_labels, abuf_real,
                                     asize.value(), adims, alabels, binter_buf, binter_dims,
                                     binter_labels, bbuf, bsize.value(), bdims, blabels,
                                     abuf_real_dev, binter_buf_dev);

        if(!gpu_trans) {
          abuf_real = ainter_buf;
          copy_data_to_gpu(hw, thandle, abuf_real, asize.value(), abuf_real_dev, binter_buf,
                           bsize.value(), binter_buf_dev);
        }

        gemm_wrapper(hw, thandle, AR, BR, B, M, N, K, alpha, beta, abuf_real, abuf_real_dev,
                     binter_buf, binter_buf_dev, cinter_buf, cinter_tmp_buf_dev);
        transpose_output(hw, thandle, gpu_trans, cinter_buf, cinter_dims, cinter_labels, cbuf,
                         cdims, clabels, cinter_buf_dev, cinter_tmp_buf_dev, is_assign);

        free_device_buffers(hw, abuf_real_dev, asize.value());
        if(gpu_trans) free_host_buffers(ExecutionHW::CPU, abuf_real, asize.value());
      }

      free_host_buffers(hw, ainter_buf, asize.value());
      free_host_buffers(hw, binter_buf, bsize.value());
    } // is_same_v<T1,T3>

    else if constexpr(internal::is_complex_v<T1> && std::is_same_v<T2, T3>) { // C=RxR
      T2* ainter_buf{nullptr};
      T2* binter_buf{nullptr};
      T2* cinter_buf_real{nullptr};
      allocate_host_buffers(hw, ainter_buf, asize.value());
      allocate_host_buffers(hw, binter_buf, bsize.value());
      allocate_host_buffers(hw, cinter_buf_real, csize.value());
#if !defined(USE_CUDA) && !defined(USE_HIP) && !defined(USE_DPCPP)
      std::memset(static_cast<void*>(cinter_buf_real), 0, csize.value() * sizeof(T2));
#endif

      T2* cbuf_tmp_real_dev{nullptr};
      allocate_device_buffers(hw, cbuf_tmp_real_dev, csize.value());
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
      gpuMemsetAsync(reinterpret_cast<void*&>(cbuf_tmp_real_dev), csize.value() * sizeof(T2),
                     thandle);
#endif

      gpu_trans = transpose_inputs(hw, thandle, ainter_buf, ainter_dims, ainter_labels, abuf,
                                   asize.value(), adims, alabels, binter_buf, binter_dims,
                                   binter_labels, bbuf, bsize.value(), bdims, blabels,
                                   ainter_buf_dev, binter_buf_dev);

      if(!gpu_trans) {
        copy_data_to_gpu(hw, thandle, ainter_buf, asize.value(), ainter_buf_dev, binter_buf,
                         bsize.value(), binter_buf_dev);
      }

      gemm_wrapper(hw, thandle, AR, BR, B, M, N, K, alpha.real(), beta.real(), ainter_buf,
                   ainter_buf_dev, binter_buf, binter_buf_dev, cinter_buf_real, cbuf_tmp_real_dev);

      if(hw == ExecutionHW::GPU) {
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
        gpu::axpy(csize.value(), cbuf_tmp_real_dev, 1, reinterpret_cast<T2*&>(cinter_tmp_buf_dev),
                  2, thandle);
#endif
      }
      else { std::copy(cinter_buf_real, cinter_buf_real + csize.value(), cinter_buf); }

      transpose_output(hw, thandle, gpu_trans, cinter_buf, cinter_dims, cinter_labels, cbuf, cdims,
                       clabels, cinter_buf_dev, reinterpret_cast<T1*&>(cinter_tmp_buf_dev),
                       is_assign);

      free_device_buffers(hw, cbuf_tmp_real_dev, csize.value());
      free_host_buffers(hw, ainter_buf, asize.value());
      free_host_buffers(hw, binter_buf, bsize.value());
      free_host_buffers(hw, cinter_buf_real, csize.value());
    }

    else NOT_IMPLEMENTED();
  }

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  th_a = ainter_buf_dev;
  th_b = binter_buf_dev;
#endif

  if(is_assign && hw != ExecutionHW::GPU) // not using bufacc code path
    assign<T1>(cbuf, cdims, clabels, T{1}, cinter_buf, cinter_dims, cinter_labels, is_assign);

  free_host_buffers(hw, cinter_buf, csize.value());

} // block_multiply()

} // namespace kernels

} // namespace tamm
