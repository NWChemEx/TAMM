#include "tamm/utils.hpp"
#include "tamm_blas.hpp"

#include <complex>

#if defined(USE_CUDA)
#include <cuda_runtime.h>
#elif defined(USE_HIP)
#include <hip/hip_runtime.h>
#endif
// For USE_DPCPP, SYCL is pulled in transitively via tamm_blas.hpp ->
// gpu_streams.hpp -> sycl_device.hpp.

namespace tamm::kernels::gpu {

// Maximum tensor rank supported by the reorder kernel. Must be >= tamm::maxrank.
static constexpr int TAMM_REORDER_MAXRANK = 8;

// POD payload passed by value into the device kernel. Holds the per-axis
// metadata needed to map an output linear index to the corresponding source
// linear index for an arbitrary axis permutation.
template<int MAXRANK>
struct ReorderMeta {
  int ndim;
  int outDims[MAXRANK];    // extents of the (permuted) output tensor
  int outStrides[MAXRANK]; // strides of the output tensor
  int inStrides[MAXRANK];  // strides of the source tensor
  int perm[MAXRANK];       // output-axis -> source-axis map
};

#if defined(USE_CUDA) || defined(USE_HIP)
// Element type used inside the device kernel. std::complex<T>'s copy assignment
// is not annotated __device__, so we transport values as a trivially-copyable
// POD of the same size/alignment (a plain byte copy of the element), exactly as
// the previous librett path did (it treated elements as opaque bytes).
template<typename T>
struct DevElem {
  alignas(T) unsigned char bytes[sizeof(T)];
};

template<typename T>
__global__ void reorder_kernel(DevElem<T>* out, const DevElem<T>* in,
                               ReorderMeta<TAMM_REORDER_MAXRANK> meta, size_t total) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if(tid >= total) return;

  const int ndim = meta.ndim;

  // Decode output linear index -> per-axis output index -> source linear index.
  // Column-major (first-axis-fastest) layout, matching librett's convention.
  size_t iIdx = 0;
  for(int k = 0; k < ndim; ++k) {
    int oidx = static_cast<int>((tid / meta.outStrides[k]) % meta.outDims[k]);
    // output axis k maps to source axis perm[k]
    iIdx += static_cast<size_t>(oidx) * meta.inStrides[meta.perm[k]];
  }
  out[tid] = in[iIdx]; // trivially-copyable POD assignment (device-safe)
}
#endif

template<typename T>
void transpose_reorder(T* out, const T* in, int ndim, const int* outDims, const int* perm,
                       gpuStream_t& handle) {
  EXPECTS(ndim >= 0 && ndim <= TAMM_REORDER_MAXRANK);
  if(ndim == 0) return;

  // Source extents: inDims[perm[k]] == outDims[k].
  ReorderMeta<TAMM_REORDER_MAXRANK> meta{};
  meta.ndim = ndim;

  int inDims[TAMM_REORDER_MAXRANK] = {0};
  for(int k = 0; k < ndim; ++k) {
    meta.outDims[k] = outDims[k];
    meta.perm[k]    = perm[k];
    inDims[perm[k]] = outDims[k];
  }

  // Column-major (first-axis-fastest) contiguous strides for both tensors.
  // This exactly matches librett's tensor-conversion convention (verified
  // against librett's TensorTester reference), so this kernel is a drop-in
  // replacement producing byte-identical output for any permutation/rank.
  meta.outStrides[0] = 1;
  meta.inStrides[0]  = 1;
  for(int k = 1; k < ndim; ++k) {
    meta.outStrides[k] = meta.outStrides[k - 1] * outDims[k - 1];
    meta.inStrides[k]  = meta.inStrides[k - 1] * inDims[k - 1];
  }

  size_t total = 1;
  for(int k = 0; k < ndim; ++k) total *= static_cast<size_t>(outDims[k]);
  if(total == 0) return;

#if defined(USE_CUDA) || defined(USE_HIP)
  constexpr unsigned block = 256;
  unsigned           grid  = static_cast<unsigned>((total + block - 1) / block);
  reorder_kernel<T><<<grid, block, 0, handle.first>>>(reinterpret_cast<DevElem<T>*>(out),
                                                      reinterpret_cast<const DevElem<T>*>(in), meta,
                                                      total);
#elif defined(USE_DPCPP)
  // Transport elements as trivially-copyable POD bytes (matches the CUDA/HIP
  // path and the previous librett behavior of treating elements as opaque).
  struct DevElem {
    alignas(T) unsigned char bytes[sizeof(T)];
  };
  auto*       out_pod = reinterpret_cast<DevElem*>(out);
  const auto* in_pod  = reinterpret_cast<const DevElem*>(in);

  constexpr size_t  block = 256;
  size_t            grid  = (total + block - 1) / block;
  sycl::nd_range<1> ndr{sycl::range<1>(grid * block), sycl::range<1>(block)};
  handle.first.parallel_for(ndr, [=](sycl::nd_item<1> item) {
    size_t tid = item.get_global_id(0);
    if(tid >= total) return;

    const int ndim_ = meta.ndim;
    size_t    iIdx  = 0;
    // Column-major (first-axis-fastest) layout, matching librett's convention.
    for(int k = 0; k < ndim_; ++k) {
      int oidx = static_cast<int>((tid / meta.outStrides[k]) % meta.outDims[k]);
      iIdx += static_cast<size_t>(oidx) * meta.inStrides[meta.perm[k]];
    }
    out_pod[tid] = in_pod[iIdx];
  });
#endif
}

template void transpose_reorder(double* out, const double* in, int ndim, const int* outDims,
                                const int* perm, gpuStream_t& handle);
template void transpose_reorder(float* out, const float* in, int ndim, const int* outDims,
                                const int* perm, gpuStream_t& handle);
template void transpose_reorder(std::complex<double>* out, const std::complex<double>* in, int ndim,
                                const int* outDims, const int* perm, gpuStream_t& handle);
template void transpose_reorder(std::complex<float>* out, const std::complex<float>* in, int ndim,
                                const int* outDims, const int* perm, gpuStream_t& handle);

} // namespace tamm::kernels::gpu
