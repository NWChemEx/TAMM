#pragma once

// Host/device shared logic for TAMM's in-house GPU tensor transpose (reorder).
//
// This header holds everything about the transpose that is backend-agnostic:
//   - the permutation metadata layout,
//   - the output-linear-index -> input-linear-index decode (column-major,
//     first-axis-fastest, matching the convention the old librett path used),
//   - the scale/accumulate elementwise apply,
//   - host-side helpers that build the metadata from TAMM's (dims, labels).
//
// The per-backend kernels in gpu_reorder.cpp are thin launch wrappers around
// these helpers, so all three backends (CUDA/HIP/SYCL) provably implement the
// same math. The CPU unit test (tests/tamm/Test_TransposeReorder.cpp)
// exercises these helpers directly, without needing a GPU.
//
// Layout note: TAMM callers pass dims/labels in reverse (Fortran) order
// (see build_reorder_spec); the kernel itself always works in column-major
// (first-axis-fastest) order. Reverse + column-major is exactly equivalent to
// the natural row-major (last-axis-fastest) transpose, which is what the CPU
// reference (kernels::assign / HPTT path) computes.

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "tamm/errors.hpp"
#include "tamm/types.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define TAMM_REORDER_HD __host__ __device__
#else
#define TAMM_REORDER_HD
#endif

namespace tamm::kernels::gpu {

// Maximum tensor rank supported by the reorder kernel.
inline constexpr int reorder_maxrank = 8;
static_assert(reorder_maxrank >= static_cast<int>(tamm::maxrank),
              "reorder kernel must support at least tamm::maxrank");

template<typename T>
struct reorder_is_complex: std::false_type {};
template<typename T>
struct reorder_is_complex<std::complex<T>>: std::true_type {};
template<typename T>
inline constexpr bool reorder_is_complex_v = reorder_is_complex<T>::value;

// POD metadata passed by value into the device kernel. Maps an output linear
// index to the corresponding source linear index for an arbitrary axis
// permutation. All extents/strides are 64-bit: TAMM blocks can exceed 2^31
// elements (cf. the B*K*N = 2^32 overflow previously fixed in gemm_wrapper).
struct ReorderMeta {
  int    ndim = 0;
  size_t outDims[reorder_maxrank]    = {};
  size_t outStrides[reorder_maxrank] = {};
  size_t inStrides[reorder_maxrank]  = {};
  int    perm[reorder_maxrank]       = {}; // output-axis -> source-axis map
};
// Passed by value into CUDA/HIP kernels (4 KB parameter limit) and captured
// by value into SYCL kernels (must stay trivially copyable).
static_assert(sizeof(ReorderMeta) <= 1024, "ReorderMeta must stay small");
static_assert(std::is_trivially_copyable_v<ReorderMeta>,
              "ReorderMeta must be trivially copyable for device kernels");

// Decode: output linear index -> source linear index (column-major).
// Templated on the integer type so launches with < 2^32 elements can use
// 32-bit division/modulo (much faster on GPUs); the caller guarantees every
// value used below fits in Idx (see meta_fits32).
template<typename Idx>
TAMM_REORDER_HD inline size_t reorder_src_index(Idx tid, const ReorderMeta& meta) {
  size_t src = 0;
  for(int k = 0; k < meta.ndim; ++k) {
    const Idx oidx =
      static_cast<Idx>((tid / static_cast<Idx>(meta.outStrides[k])) %
                       static_cast<Idx>(meta.outDims[k]));
    src += static_cast<size_t>(oidx) * meta.inStrides[meta.perm[k]];
  }
  return src;
}

// Elementwise math, split in two so device code never default-constructs a
// T (std::complex's default ctor is not reliably usable in CUDA/HIP/SYCL
// device code) and the overwrite path never reads the output buffer.
// Complex arithmetic runs on the raw {real, imag} words so no std::complex
// device calls are needed; the complex<T> <-> T[2] pun is blessed by the
// standard.
template<typename T>
TAMM_REORDER_HD inline T reorder_scaled(T v, double scale_re, double scale_im) {
  if constexpr(reorder_is_complex_v<T>) {
    using R    = typename T::value_type;
    const R* w = reinterpret_cast<const R*>(&v);
    T        out;
    R*       o = reinterpret_cast<R*>(&out);
    o[0] = static_cast<R>(scale_re * w[0] - scale_im * w[1]);
    o[1] = static_cast<R>(scale_re * w[1] + scale_im * w[0]);
    return out;
  }
  else {
    return static_cast<T>(scale_re * static_cast<double>(v));
  }
}

template<typename T>
TAMM_REORDER_HD inline T reorder_add(T a, T b) {
  if constexpr(reorder_is_complex_v<T>) {
    using R    = typename T::value_type;
    const R* wa = reinterpret_cast<const R*>(&a);
    const R* wb = reinterpret_cast<const R*>(&b);
    T        out;
    R*       o = reinterpret_cast<R*>(&out);
    o[0]       = wa[0] + wb[0];
    o[1]       = wa[1] + wb[1];
    return out;
  }
  else {
    return a + b;
  }
}

// Per-element store idiom (use at every call site):
//   const T y = reorder_scaled<T>(in[src], scale_re, scale_im);
//   out[tid]  = accumulate ? reorder_add<T>(out[tid], y) : y;
// so the overwrite path never reads the output buffer and no code path
// default-constructs a T on the device.

// True when every extent/stride/total used by the decode fits in 32 bits.
inline bool reorder_meta_fits32(const ReorderMeta& meta, size_t total) {
  constexpr auto max32 = static_cast<size_t>(std::numeric_limits<uint32_t>::max());
  if(total > max32) return false;
  for(int k = 0; k < meta.ndim; ++k) {
    if(meta.outDims[k] > max32 || meta.outStrides[k] > max32 || meta.inStrides[k] > max32) {
      return false;
    }
  }
  return true;
}

inline bool reorder_is_identity(const ReorderMeta& meta) {
  for(int k = 0; k < meta.ndim; ++k) {
    if(meta.perm[k] != k) return false;
  }
  return true;
}

// Host-side: build the metadata from the (already reversed) output extents
// and output-axis -> source-axis permutation. Source extents follow from
// inDims[perm[k]] == outDims[k]; both tensors are contiguous column-major.
inline ReorderMeta reorder_build_meta(int ndim, const size_t* outDims, const int* perm) {
  EXPECTS(ndim >= 0 && ndim <= reorder_maxrank);
  ReorderMeta meta{};
  meta.ndim = ndim;
  if(ndim == 0) return meta;

  size_t inDims[reorder_maxrank] = {};
  for(int k = 0; k < ndim; ++k) {
    EXPECTS(perm[k] >= 0 && perm[k] < ndim);
    meta.outDims[k] = outDims[k];
    meta.perm[k]    = perm[k];
    inDims[perm[k]] = outDims[k];
  }
  meta.outStrides[0] = 1;
  meta.inStrides[0]  = 1;
  for(int k = 1; k < ndim; ++k) {
    meta.outStrides[k] = meta.outStrides[k - 1] * outDims[k - 1];
    meta.inStrides[k]  = meta.inStrides[k - 1] * inDims[k - 1];
  }
  return meta;
}

inline size_t reorder_total(int ndim, const size_t* outDims) {
  size_t total = 1;
  for(int k = 0; k < ndim; ++k) total *= outDims[k];
  return total;
}

// Host-side: translate TAMM's (src dims, src labels, dst labels) into the
// reversed (Fortran-order) output extents + permutation the column-major
// kernel consumes. Reversing both dim orders converts between TAMM's natural
// row-major transpose and the kernel's column-major transpose.
//
// Allocation-free: everything lives in fixed-size stack arrays (rank <=
// reorder_maxrank), so this can sit in the hottest transpose path without
// touching the heap. (An earlier version used std::vectors here: 4 heap
// allocations per transpose call.)
struct ReorderSpec {
  int    ndim = 0;
  size_t outDims[reorder_maxrank] = {};
  int    perm[reorder_maxrank]    = {}; // reversed output-axis -> source-axis map
};

inline void build_reorder_spec(const SizeVec& sdims, const IntLabelVec& slabels,
                               const IntLabelVec& dlabels, ReorderSpec& spec) {
  const size_t ndim = sdims.size();
  EXPECTS(dlabels.size() == ndim && slabels.size() == ndim);
  EXPECTS(ndim <= static_cast<size_t>(reorder_maxrank));

  // Reverse into Fortran order (librett used the same convention).
  size_t   r_size[reorder_maxrank];
  IntLabel r_slabels[reorder_maxrank];
  IntLabel r_dlabels[reorder_maxrank];
  for(size_t i = 0; i < ndim; ++i) {
    r_size[i]    = sdims[ndim - 1 - i].value();
    r_slabels[i] = slabels[ndim - 1 - i];
    r_dlabels[i] = dlabels[ndim - 1 - i];
  }

  spec.ndim = static_cast<int>(ndim);
  for(size_t i = 0; i < ndim; ++i) {
    int j = 0;
    while(j < static_cast<int>(ndim) && r_slabels[static_cast<size_t>(j)] != r_dlabels[i]) ++j;
    EXPECTS(j < static_cast<int>(ndim));
    spec.perm[i] = j;
  }
  for(size_t i = 0; i < ndim; ++i) {
    spec.outDims[i] = r_size[static_cast<size_t>(spec.perm[i])];
  }
}

} // namespace tamm::kernels::gpu
