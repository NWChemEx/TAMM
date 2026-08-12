#pragma once

#include "tamm/errors.hpp"
#include "tamm/kernels/assign.hpp"
#include "tamm/types.hpp"

#include <complex>
#include <cstdint>
#include <cstring> // for std::memset
#include <numeric>
#include <span>
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
  // Leading dimensions stay `int`: they are passed straight to the BLAS lda/ldb/ldc
  // parameters, which are `int` in the cuBLAS/rocBLAS/MKL interfaces used here.
  int const ainter_ld = K;
  int const binter_ld = N;
  int const cinter_ld = N;

  // Batch/reduction strides are computed in int64_t. Previously these were `int`, so the
  // products below were evaluated in int arithmetic and could overflow before any
  // promotion took place. The offset *accumulation* was already 64-bit (the loop counters
  // were size_t, which promoted the int strides), so the failure is in the strides
  // themselves:
  //
  //   B=256, K=N=4096  ->  breduce_ld = B*K*N = 2^32, which wraps to exactly 0 in int.
  //
  // With BR > 1 every reduction iteration then re-reads batch 0 instead of advancing,
  // producing silently wrong results rather than a crash. M*K overflows the same way once
  // a single stride exceeds INT_MAX. Both are signed overflow, i.e. UB.
  //
  // Note the loop counters are int64_t here too, so the offsets stay exact without relying
  // on the old size_t-promotion accident.
  std::int64_t const cbatch_ld  = static_cast<std::int64_t>(M) * N;
  std::int64_t const abatch_ld  = static_cast<std::int64_t>(M) * K;
  std::int64_t const bbatch_ld  = static_cast<std::int64_t>(K) * N;
  std::int64_t const areduce_ld = static_cast<std::int64_t>(B) * abatch_ld;
  std::int64_t const breduce_ld = static_cast<std::int64_t>(B) * bbatch_ld;

  for(std::int64_t ari = 0; ari < AR; ari++) {
    for(std::int64_t bri = 0; bri < BR; bri++) {
      for(std::int64_t i = 0; i < B; i++) {
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
    } // for-bri
  } // for-ari
}

// Buffer helpers return and consume std::span rather than a raw pointer plus a separately
// recomputed size. The span carries its own length, so a free can no longer disagree with
// its allocation -- the failure mode that silently orphans or overlaps pool blocks.
//
// An empty span is the "not allocated on this hardware path" state, replacing the previous
// convention of leaving a raw pointer null. Freeing an empty span is a no-op, so the
// conditional-free asymmetries that leaked here before are now impossible.

template<typename T>
[[nodiscard]] std::span<T> allocate_host_buffer(ExecutionHW hw, size_t buf_size) {
  if(hw == ExecutionHW::GPU) return {};
  auto& memPool = RMMMemoryManager::getInstance().getHostMemoryPool();
  return memPool.template allocate_span<T>(buf_size);
}

template<typename T>
void free_host_buffer(ExecutionHW hw, std::span<T> host_buf) {
  if(hw == ExecutionHW::GPU) return;
  auto& memPool = RMMMemoryManager::getInstance().getHostMemoryPool();
  memPool.deallocate(host_buf);
}

template<typename T>
[[nodiscard]] std::span<T> allocate_device_buffer([[maybe_unused]] ExecutionHW hw,
                                                  [[maybe_unused]] size_t      buf_size) {
  if(hw == ExecutionHW::CPU) return {};
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  auto& memPool = RMMMemoryManager::getInstance().getDeviceMemoryPool();
  return memPool.template allocate_span<T>(buf_size);
#else
  return {};
#endif
}

template<typename T>
void free_device_buffer([[maybe_unused]] ExecutionHW hw, [[maybe_unused]] std::span<T> dev_buf) {
  if(hw == ExecutionHW::CPU) return;
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  auto& memPool = RMMMemoryManager::getInstance().getDeviceMemoryPool();
  memPool.deallocate(dev_buf);
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

    std::span<T2> ainter_dev_in = allocate_device_buffer<T2>(hw, asize);
    std::span<T3> binter_dev_in = allocate_device_buffer<T3>(hw, bsize);

    copy_data_to_gpu(hw, thandle, abuf, asize, ainter_dev_in.data(), bbuf, bsize,
                     binter_dev_in.data());

    assign_gpu<T2>(thandle, ainter_buf_dev, ainter_dims, ainter_labels, T2{1},
                   ainter_dev_in.data(), adims, alabels, true);
    assign_gpu<T3>(thandle, binter_buf_dev, binter_dims, binter_labels, T3{1},
                   binter_dev_in.data(), bdims, blabels, true);

    // The H2D copies above and librettExecute() inside assign_gpu() are enqueued on
    // `thandle` and return immediately on CUDA/HIP, but returning these staging buffers to
    // the pool makes their addresses instantly re-allocatable (the pool's deallocate is
    // host-side bookkeeping, not a stream-ordered free). Without this sync the next
    // allocation can alias memory that librett is still reading.
    gpuStreamSynchronize(thandle);

    free_device_buffer(hw, ainter_dev_in);
    free_device_buffer(hw, binter_dev_in);

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

  std::span<T1> cinter_span =
    allocate_host_buffer<T1>(hw, static_cast<size_t>(csize.value()));
  T1* cinter_buf = cinter_span.data();
  if(hw == ExecutionHW::CPU) {
    // if(csize.value() != 1)
    std::memset(static_cast<void*>(cinter_span.data()), 0, cinter_span.size_bytes());
  }

  T2* ainter_buf_dev{nullptr};
  T3* binter_buf_dev{nullptr};
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  ainter_buf_dev = th_a;
  binter_buf_dev = th_b;
#endif

  // dgemm
  if constexpr(std::is_same_v<T1, T2> && std::is_same_v<T1, T3>) { // R=RxR, C=CxC
    std::span<T2> ainter_span = allocate_host_buffer<T2>(hw, asize.value());
    std::span<T3> binter_span = allocate_host_buffer<T3>(hw, bsize.value());
    T2*           ainter_buf  = ainter_span.data();
    T3*           binter_buf  = binter_span.data();

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

    free_host_buffer(hw, ainter_span);
    free_host_buffer(hw, binter_span);
  }
  else {
    T2* abufp = const_cast<T2*>(abuf);
    T3* bbufp = const_cast<T3*>(bbuf);
    // TODO: actually check if one of T2, T3 is real, T1 is complex
    if constexpr(std::is_same_v<T1, T2>) {
      std::span<T2> ainter_span = allocate_host_buffer<T2>(hw, asize.value());
      std::span<T1> binter_span = allocate_host_buffer<T1>(hw, bsize.value());
      T2*           ainter_buf  = ainter_span.data();
      T1*           binter_buf  = binter_span.data();

      // T2 (matrix A) is complex, T3 (B) is real, C=CxR
      if constexpr(internal::is_complex_v<T1>) {
        // copy B to complex buffer.
        // `*_owned` holds the pool allocation and is never reassigned, so it can always be
        // returned to the pool with the size it was allocated with. `bbuf_complex` is only a
        // view, which may be re-pointed at `binter_buf` on the CPU path.
        std::span<T1> bbuf_complex_span =
          allocate_host_buffer<T1>(ExecutionHW::CPU, bsize.value());
        std::copy(bbufp, bbufp + bsize.value(), bbuf_complex_span.data());
        T1* bbuf_complex = bbuf_complex_span.data();

        std::span<T1> bbuf_complex_dev_span = allocate_device_buffer<T1>(hw, bsize.value());
        T1*           bbuf_complex_dev      = bbuf_complex_dev_span.data();

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

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
        // gemm_wrapper() and transpose_output() enqueue stream-async work on CUDA/HIP;
        // sync before returning this device block to the pool, which makes the address
        // immediately re-allocatable.
        if(hw == ExecutionHW::GPU) { gpuStreamSynchronize(thandle); }
#endif

        free_device_buffer(hw, bbuf_complex_dev_span);
        free_host_buffer(ExecutionHW::CPU, bbuf_complex_span);
      } // is_complex<T1>
      else {
        // T1,T2 (C,A) are real, T3 (B) is complex, R=RxC
        std::span<T1> bbuf_real_span = allocate_host_buffer<T1>(ExecutionHW::CPU, bsize.value());
        std::transform(bbufp, bbufp + bsize.value(), bbuf_real_span.data(),
                       [](const T3& val) { return val.real(); });
        T1* bbuf_real = bbuf_real_span.data();

        std::span<T1> bbuf_real_dev_span = allocate_device_buffer<T1>(hw, bsize.value());
        T1*           bbuf_real_dev      = bbuf_real_dev_span.data();

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

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
        // gemm_wrapper() and transpose_output() enqueue stream-async work on CUDA/HIP;
        // sync before returning this device block to the pool, which makes the address
        // immediately re-allocatable.
        if(hw == ExecutionHW::GPU) { gpuStreamSynchronize(thandle); }
#endif

        free_device_buffer(hw, bbuf_real_dev_span);
        free_host_buffer(ExecutionHW::CPU, bbuf_real_span);
      } // is_real<T1>

      free_host_buffer(hw, ainter_span);
      free_host_buffer(hw, binter_span);
    } // is_same_v<T1,T2>
    else if constexpr(std::is_same_v<T1, T3>) {
      std::span<T1> ainter_span = allocate_host_buffer<T1>(hw, asize.value());
      std::span<T3> binter_span = allocate_host_buffer<T3>(hw, bsize.value());
      T1*           ainter_buf  = ainter_span.data();
      T3*           binter_buf  = binter_span.data();

      // T3 (matrix B) is complex, T2 (A) is real, C=RxC
      if constexpr(internal::is_complex_v<T1>) {
        // `*_owned` holds the pool allocation and is never reassigned; `abuf_complex` is a view
        // that may be re-pointed at `ainter_buf` on the CPU path.
        std::span<T1> abuf_complex_span =
          allocate_host_buffer<T1>(ExecutionHW::CPU, asize.value());
        std::copy(abufp, abufp + asize.value(), abuf_complex_span.data());
        T1* abuf_complex = abuf_complex_span.data();

        std::span<T1> abuf_complex_dev_span = allocate_device_buffer<T1>(hw, asize.value());
        T1*           abuf_complex_dev      = abuf_complex_dev_span.data();

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

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
        // gemm_wrapper() and transpose_output() enqueue stream-async work on CUDA/HIP;
        // sync before returning this device block to the pool, which makes the address
        // immediately re-allocatable.
        if(hw == ExecutionHW::GPU) { gpuStreamSynchronize(thandle); }
#endif

        free_device_buffer(hw, abuf_complex_dev_span);
        free_host_buffer(ExecutionHW::CPU, abuf_complex_span);
      }
      else {
        // T1,T3 (C,B) are real, T2 (A) is complex, //R=CxR
        std::span<T1> abuf_real_span = allocate_host_buffer<T1>(ExecutionHW::CPU, asize.value());
        std::transform(abufp, abufp + asize.value(), abuf_real_span.data(),
                       [](const T2& val) { return val.real(); });
        T1* abuf_real = abuf_real_span.data();

        std::span<T1> abuf_real_dev_span = allocate_device_buffer<T1>(hw, asize.value());
        T1*           abuf_real_dev      = abuf_real_dev_span.data();

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

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
        // gemm_wrapper() and transpose_output() enqueue stream-async work on CUDA/HIP;
        // sync before returning this device block to the pool, which makes the address
        // immediately re-allocatable.
        if(hw == ExecutionHW::GPU) { gpuStreamSynchronize(thandle); }
#endif

        free_device_buffer(hw, abuf_real_dev_span);
        free_host_buffer(ExecutionHW::CPU, abuf_real_span);
      }

      free_host_buffer(hw, ainter_span);
      free_host_buffer(hw, binter_span);
    } // is_same_v<T1,T3>

    else if constexpr(internal::is_complex_v<T1> && std::is_same_v<T2, T3>) { // C=RxR
      std::span<T2> ainter_span      = allocate_host_buffer<T2>(hw, asize.value());
      std::span<T2> binter_span      = allocate_host_buffer<T2>(hw, bsize.value());
      std::span<T2> cinter_real_span = allocate_host_buffer<T2>(hw, csize.value());
      T2*           ainter_buf       = ainter_span.data();
      T2*           binter_buf       = binter_span.data();
      T2*           cinter_buf_real  = cinter_real_span.data();
#if !defined(USE_CUDA) && !defined(USE_HIP) && !defined(USE_DPCPP)
      std::memset(static_cast<void*>(cinter_real_span.data()), 0, cinter_real_span.size_bytes());
#endif

      std::span<T2> cbuf_tmp_real_dev_span = allocate_device_buffer<T2>(hw, csize.value());
      T2*           cbuf_tmp_real_dev      = cbuf_tmp_real_dev_span.data();
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
      gpuMemsetAsync(cbuf_tmp_real_dev, csize.value() * sizeof(T2),
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

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
      // gpu::axpy and the librett transpose in transpose_output are stream-async on
      // CUDA/HIP; sync before returning this device block to the pool, which would
      // otherwise make it immediately re-allocatable while that work is still reading it.
      if(hw == ExecutionHW::GPU) { gpuStreamSynchronize(thandle); }
#endif

      free_device_buffer(hw, cbuf_tmp_real_dev_span);
      free_host_buffer(hw, ainter_span);
      free_host_buffer(hw, binter_span);
      free_host_buffer(hw, cinter_real_span);
    }

    else NOT_IMPLEMENTED();
  }

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  th_a = ainter_buf_dev;
  th_b = binter_buf_dev;

  // Establish the postcondition that every device buffer passed in (th_a/th_b, the
  // cinter_*_dev buffers) and every device buffer allocated internally is free of
  // in-flight work by the time this function returns.
  //
  // On CUDA/HIP the cuBLAS/rocBLAS gemm above is enqueued on `thandle` and returns
  // immediately, whereas the memory pool's deallocate() is pure host-side bookkeeping that
  // makes a block instantly re-allocatable -- it does not defer the reuse behind a stream
  // event the way upstream RMM's stream-ordered free lists do. Callers that free a device
  // buffer right after this call (e.g. the reduction loop in MultOp::execute_bufacc, and
  // the equivalent loops in exachem's cd_ccsd_{cs,os}_ann.cpp) would otherwise hand the
  // same address to the next iteration's allocation while the gemm is still reading it.
  //
  // DPC++ already synchronises inside gpu::gemm (gemm_event.wait()), so this is a no-op
  // there; the cost on CUDA/HIP is the sync the callers' correctness already assumed.
  if(hw == ExecutionHW::GPU) { gpuStreamSynchronize(thandle); }
#endif

  if(is_assign && hw != ExecutionHW::GPU) // not using bufacc code path
    assign<T1>(cbuf, cdims, clabels, T{1}, cinter_buf, cinter_dims, cinter_labels, is_assign);

  free_host_buffer(hw, cinter_span);

} // block_multiply()

} // namespace kernels

} // namespace tamm
