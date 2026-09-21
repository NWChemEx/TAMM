#include "gpu_reorder.hpp"

#include "tamm_blas.hpp"

#include <algorithm>
#include <complex>

namespace tamm::kernels::gpu {

// Decompose the host-side scale into raw {real, imag} doubles for the kernel.
template<typename T>
inline void reorder_split_scale(T scale, double& re, double& im) {
  if constexpr(reorder_is_complex_v<T>) {
    re = static_cast<double>(scale.real());
    im = static_cast<double>(scale.imag());
  }
  else {
    re = static_cast<double>(scale);
    im = 0.0;
  }
}

template<typename T>
inline bool reorder_scale_is_one(T scale) {
  if constexpr(reorder_is_complex_v<T>) { return scale.real() == 1 && scale.imag() == 0; }
  else { return scale == T{1}; }
}

#if defined(USE_CUDA) || defined(USE_HIP)
// One thread per output element, grid-stride loop: any total works with a
// bounded grid, and the kernel never touches the caller's stream ordering
// (no internal sync, no plan, no host round-trip).
template<typename T, typename Idx>
__global__ void
#if defined(USE_CUDA) || defined(USE_HIP)
__launch_bounds__(256)
#endif
reorder_kernel(T* out, const T* in, ReorderMeta meta, size_t total, double scale_re, double scale_im,
               bool accumulate)
{
  const size_t tid0   = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for(size_t t = tid0; t < total; t += stride) {
    const Idx tid = static_cast<Idx>(t);
    const size_t src = reorder_src_index<Idx>(tid, meta);
    const T      y   = reorder_scaled<T>(in[src], scale_re, scale_im);
    out[tid]         = accumulate ? reorder_add<T>(out[tid], y) : y;
  }
}
#endif // USE_CUDA || USE_HIP

template<typename T>
void transpose_reorder(T* out, const T* in, int ndim, const size_t* outDims, const int* perm,
                       T scale, bool accumulate, gpuStream_t& handle) {
  EXPECTS(ndim >= 0 && ndim <= reorder_maxrank);
  EXPECTS(out != in); // out-of-place only; in-place permutes need a temp buffer
  if(ndim == 0) {
    // Rank-0: single element, still honors scale/accumulate.
    double scale_re, scale_im;
    reorder_split_scale(scale, scale_re, scale_im);
#if defined(USE_CUDA) || defined(USE_HIP)
    // Route through the kernel for stream ordering (async, like every
    // other path through this function).
    if(reorder_scale_is_one(scale) && !accumulate) {
      gpuMemcpyAsync<T>(out, in, 1, gpuMemcpyDeviceToDevice, handle);
    }
    else {
      ReorderMeta meta{};
      meta.ndim = 0;
      reorder_kernel<T, uint32_t>
        <<<1, 1, 0, handle.first>>>(out, in, meta, 1, scale_re, scale_im, accumulate);
    }
#elif defined(USE_DPCPP)
    if(reorder_scale_is_one(scale) && !accumulate) {
      gpuMemcpyAsync<T>(out, in, 1, gpuMemcpyDeviceToDevice, handle);
    }
    else {
      handle.first.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
        const T y = reorder_scaled<T>(in[0], scale_re, scale_im);
        out[0]    = accumulate ? reorder_add<T>(out[0], y) : y;
      });
    }
#endif
    return;
  }
  EXPECTS(outDims != nullptr && perm != nullptr);

  const ReorderMeta meta  = reorder_build_meta(ndim, outDims, perm);
  const size_t    total = reorder_total(ndim, outDims);
  if(total == 0) return;

  double scale_re, scale_im;
  reorder_split_scale(scale, scale_re, scale_im);
  if constexpr(!reorder_is_complex_v<T>) { EXPECTS(scale_im == 0.0); }

  // Identity permutation with scale==1 and overwrite: pure copy, no kernel.
  if(!accumulate && reorder_scale_is_one(scale) && reorder_is_identity(meta)) {
    gpuMemcpyAsync<T>(out, in, total, gpuMemcpyDeviceToDevice, handle);
    return;
  }

#if defined(USE_CUDA) || defined(USE_HIP)
  // Bounded grid + stride loop (CUDA grid dims cap at 2^31-1, so a 1:1
  // mapping is impossible for huge totals).
  constexpr size_t block     = 256;
  constexpr size_t maxBlocks = 1 << 20; // stride loop covers the rest
  const size_t nblocks = std::min<size_t>((total + block - 1) / block, maxBlocks);
  if(reorder_meta_fits32(meta, total)) {
    reorder_kernel<T, uint32_t><<<static_cast<unsigned>(nblocks), static_cast<unsigned>(block), 0,
                                          handle.first>>>(out, in, meta, total, scale_re, scale_im,
                                                          accumulate);
  }
  else {
    reorder_kernel<T, uint64_t><<<static_cast<unsigned>(nblocks), static_cast<unsigned>(block), 0,
                                          handle.first>>>(out, in, meta, total, scale_re, scale_im,
                                                          accumulate);
  }
#elif defined(USE_DPCPP)
  // Flat one-item-per-element range: no grid sizing, no stride loop. (A
  // literal single_task would serialize the whole transpose onto one
  // work-item.)
  if(reorder_meta_fits32(meta, total)) {
    handle.first.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
      const uint32_t tid = static_cast<uint32_t>(idx[0]);
      const size_t   src = reorder_src_index<uint32_t>(tid, meta);
      const T        y   = reorder_scaled<T>(in[src], scale_re, scale_im);
      out[tid]           = accumulate ? reorder_add<T>(out[tid], y) : y;
    });
  }
  else {
    handle.first.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
      const size_t t   = idx[0];
      const size_t src = reorder_src_index<uint64_t>(t, meta);
      const T      y   = reorder_scaled<T>(in[src], scale_re, scale_im);
      out[t]           = accumulate ? reorder_add<T>(out[t], y) : y;
    });
  }
  // No wait: the caller's in-order queue preserves ordering, and
  // block_multiply() synchronizes before pool buffers are recycled.
#endif
}

template void transpose_reorder(double* out, const double* in, int ndim, const size_t* outDims,
                                const int* perm, double scale, bool accumulate,
                                gpuStream_t& handle);
template void transpose_reorder(float* out, const float* in, int ndim, const size_t* outDims,
                                const int* perm, float scale, bool accumulate,
                                gpuStream_t& handle);
template void transpose_reorder(std::complex<double>* out, const std::complex<double>* in, int ndim,
                                const size_t* outDims, const int* perm, std::complex<double> scale,
                                bool accumulate, gpuStream_t& handle);
template void transpose_reorder(std::complex<float>* out, const std::complex<float>* in, int ndim,
                                const size_t* outDims, const int* perm, std::complex<float> scale,
                                bool accumulate, gpuStream_t& handle);

} // namespace tamm::kernels::gpu
