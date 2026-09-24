#include "gpu_permute.hpp"

#include "tamm_blas.hpp"

#include <algorithm>
#include <complex>

namespace tamm::kernels::gpu {

// Decompose the host-side scale into raw {real, imag} doubles for the kernel.
template<typename T>
inline void permute_split_scale(T scale, double& re, double& im) {
  if constexpr(permute_is_complex_v<T>) {
    re = static_cast<double>(scale.real());
    im = static_cast<double>(scale.imag());
  }
  else {
    re = static_cast<double>(scale);
    im = 0.0;
  }
}

template<typename T>
inline bool permute_scale_is_one(T scale) {
  if constexpr(permute_is_complex_v<T>) { return scale.real() == 1 && scale.imag() == 0; }
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
permute_kernel(T* out, const T* in, PermuteMeta meta, size_t total, double scale_re, double scale_im,
               bool accumulate)
{
  const size_t tid0   = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for(size_t t = tid0; t < total; t += stride) {
    const Idx tid = static_cast<Idx>(t);
    const size_t src = permute_src_index<Idx>(tid, meta);
    const T      y   = permute_scaled<T>(in[src], scale_re, scale_im);
    out[tid]         = accumulate ? permute_add<T>(out[tid], y) : y;
  }
}
#endif // USE_CUDA || USE_HIP

template<typename T>
void permute(T* out, const T* in, int ndim, const size_t* outDims, const int* perm,
                       T scale, bool accumulate, gpuStream_t& handle) {
  EXPECTS(ndim >= 0 && ndim <= permute_maxrank);
  EXPECTS(out != in); // out-of-place only; in-place permutes need a temp buffer
  if(ndim == 0) {
    // Rank-0: single element, still honors scale/accumulate.
    if(permute_scale_is_one(scale) && !accumulate) {
      gpuMemcpyAsync<T>(out, in, 1, gpuMemcpyDeviceToDevice, handle);
      return;
    }
    double scale_re, scale_im;
    permute_split_scale(scale, scale_re, scale_im);
#if defined(USE_CUDA) || defined(USE_HIP)
    // Route through the kernel for stream ordering (async, like every
    // other path through this function).
    PermuteMeta meta{};
    meta.ndim = 0;
    permute_kernel<T, uint32_t>
      <<<1, 1, 0, handle.first>>>(out, in, meta, 1, scale_re, scale_im, accumulate);
#elif defined(USE_DPCPP)
    handle.first.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
      const T y = permute_scaled<T>(in[0], scale_re, scale_im);
      out[0]    = accumulate ? permute_add<T>(out[0], y) : y;
    });
#endif
    return;
  }
  EXPECTS(outDims != nullptr && perm != nullptr);

  const PermuteMeta meta  = permute_build_meta(ndim, outDims, perm);
  const size_t    total = permute_total(ndim, outDims);
  if(total == 0) return;

  double scale_re, scale_im;
  permute_split_scale(scale, scale_re, scale_im);
  if constexpr(!permute_is_complex_v<T>) { EXPECTS(scale_im == 0.0); }

  // Identity permutation with scale==1 and overwrite: pure copy, no kernel.
  if(!accumulate && permute_scale_is_one(scale) && permute_is_identity(meta)) {
    gpuMemcpyAsync<T>(out, in, total, gpuMemcpyDeviceToDevice, handle);
    return;
  }

#if defined(USE_CUDA) || defined(USE_HIP)
  // Bounded grid + stride loop (CUDA grid dims cap at 2^31-1, so a 1:1
  // mapping is impossible for huge totals).
  constexpr size_t block     = 256;
  constexpr size_t maxBlocks = 1 << 20; // stride loop covers the rest
  const size_t nblocks = std::min<size_t>((total + block - 1) / block, maxBlocks);
  if(permute_meta_fits32(meta, total)) {
    permute_kernel<T, uint32_t><<<static_cast<unsigned>(nblocks), static_cast<unsigned>(block), 0,
                                          handle.first>>>(out, in, meta, total, scale_re, scale_im,
                                                          accumulate);
  }
  else {
    permute_kernel<T, uint64_t><<<static_cast<unsigned>(nblocks), static_cast<unsigned>(block), 0,
                                          handle.first>>>(out, in, meta, total, scale_re, scale_im,
                                                          accumulate);
  }
#elif defined(USE_DPCPP)
  if(permute_meta_fits32(meta, total)) {
    handle.first.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
      const uint32_t tid = static_cast<uint32_t>(idx[0]);
      const size_t   src = permute_src_index<uint32_t>(tid, meta);
      const T        y   = permute_scaled<T>(in[src], scale_re, scale_im);
      out[tid]           = accumulate ? permute_add<T>(out[tid], y) : y;
    });
  }
  else {
    handle.first.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
      const size_t t   = idx[0];
      const size_t src = permute_src_index<uint64_t>(t, meta);
      const T      y   = permute_scaled<T>(in[src], scale_re, scale_im);
      out[t]           = accumulate ? permute_add<T>(out[t], y) : y;
    });
  }
  // No wait: the caller's in-order queue preserves ordering, and
  // block_multiply() synchronizes before pool buffers are recycled.
#endif
}

template void permute(double* out, const double* in, int ndim, const size_t* outDims,
                                const int* perm, double scale, bool accumulate,
                                gpuStream_t& handle);
template void permute(float* out, const float* in, int ndim, const size_t* outDims,
                                const int* perm, float scale, bool accumulate,
                                gpuStream_t& handle);
template void permute(std::complex<double>* out, const std::complex<double>* in, int ndim,
                                const size_t* outDims, const int* perm, std::complex<double> scale,
                                bool accumulate, gpuStream_t& handle);
template void permute(std::complex<float>* out, const std::complex<float>* in, int ndim,
                                const size_t* outDims, const int* perm, std::complex<float> scale,
                                bool accumulate, gpuStream_t& handle);

} // namespace tamm::kernels::gpu
