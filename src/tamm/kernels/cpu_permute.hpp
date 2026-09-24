#pragma once

// Host-only CPU tensor permute, the HPTT replacement.
//
// Row-major (TAMM-native) counterpart of the GPU kernel in gpu_permute.hpp.
// Reuses its PermuteMeta storage, elementwise scale/add helpers and identity
// check, so both backends implement the same math; only the metadata build
// differs (row-major strides, natural axis order).
//
//   out = beta * out + alpha * permute(in)
//
// beta==0 never reads out; beta==1 accumulates. Single-threaded and
// allocation-free except on the aliased in-place path.

#include "tamm/kernels/gpu_permute.hpp"

#include <complex>
#include <cstring>
#include <type_traits>
#include <vector>

namespace tamm::kernels::cpu {

// Split a host-side alpha/beta into raw {real, imag} doubles.
template<typename S>
inline void permute_split_scale_host(S scale, double& re, double& im) {
  if constexpr(gpu::permute_is_complex_v<S>) {
    re = static_cast<double>(scale.real());
    im = static_cast<double>(scale.imag());
  }
  else {
    re = static_cast<double>(scale);
    im = 0.0;
  }
}

template<typename S>
inline bool permute_scale_is_zero_host(S scale) {
  if constexpr(gpu::permute_is_complex_v<S>) { return scale.real() == 0 && scale.imag() == 0; }
  else { return scale == S{0}; }
}

template<typename S>
inline bool permute_scale_is_one_host(S scale) {
  if constexpr(gpu::permute_is_complex_v<S>) { return scale.real() == 1 && scale.imag() == 0; }
  else { return scale == S{1}; }
}

// Build row-major metadata from natural output extents + dst->src perm.
// Source extents follow from inDims[perm[k]] == outDims[k].
inline gpu::PermuteMeta permute_build_meta_rowmajor(int ndim, const size_t* outDims,
                                                    const int* perm) {
  EXPECTS(ndim >= 0 && ndim <= gpu::permute_maxrank);
  gpu::PermuteMeta meta{};
  meta.ndim = ndim;
  if(ndim == 0) return meta;

  size_t inDims[gpu::permute_maxrank] = {};
  for(int k = 0; k < ndim; ++k) {
    EXPECTS(perm[k] >= 0 && perm[k] < ndim);
    meta.outDims[k] = outDims[k];
    meta.perm[k]    = perm[k];
    inDims[perm[k]] = outDims[k];
  }
  meta.outStrides[ndim - 1] = 1;
  meta.inStrides[ndim - 1]  = 1;
  for(int k = ndim - 2; k >= 0; --k) {
    meta.outStrides[k] = meta.outStrides[k + 1] * outDims[k + 1];
    meta.inStrides[k]  = meta.inStrides[k + 1] * inDims[k + 1];
  }
  return meta;
}

template<typename T, typename B>
inline void permute_impl(T* out, const T* in, const gpu::PermuteMeta& meta,
                                       size_t total, double alpha_re, double alpha_im, B beta) {
  const bool beta_is_zero = permute_scale_is_zero_host(beta);
  const bool beta_is_one  = permute_scale_is_one_host(beta);
  double     beta_re = 0.0, beta_im = 0.0;
  if(!beta_is_zero && !beta_is_one) permute_split_scale_host(beta, beta_re, beta_im);

  // Odometer over the output coordinates: no division/modulo per element,
  // just ndim multiply-adds plus an amortized single increment.
  const int ndim = meta.ndim;
  size_t    coord[gpu::permute_maxrank] = {};
  for(size_t t = 0; t < total; ++t) {
    size_t src = 0;
    for(int k = 0; k < ndim; ++k) src += coord[k] * meta.inStrides[meta.perm[k]];
    const T y = gpu::permute_scaled<T>(in[src], alpha_re, alpha_im);
    if(beta_is_zero) { out[t] = y; }
    else if(beta_is_one) { out[t] = gpu::permute_add<T>(out[t], y); }
    else { out[t] = gpu::permute_add<T>(gpu::permute_scaled<T>(out[t], beta_re, beta_im), y); }
    for(int k = ndim - 1; k >= 0; --k) {
      if(++coord[k] < meta.outDims[k]) break;
      coord[k] = 0;
    }
  }
}

// out = beta * out + alpha * permute(in); outDims/perm in natural order.
template<typename T, typename A, typename B>
inline void permute(T* out, const T* in, int ndim, const size_t* outDims,
                                  const int* perm, A alpha, B beta) {
  EXPECTS(ndim >= 0 && ndim <= gpu::permute_maxrank);
  if(ndim == 0) {
    double alpha_re, alpha_im, beta_re, beta_im;
    permute_split_scale_host(alpha, alpha_re, alpha_im);
    permute_split_scale_host(beta, beta_re, beta_im);
    const T y = gpu::permute_scaled<T>(in[0], alpha_re, alpha_im);
    if(permute_scale_is_zero_host(beta)) { out[0] = y; }
    else { out[0] = gpu::permute_add<T>(gpu::permute_scaled<T>(out[0], beta_re, beta_im), y); }
    return;
  }
  EXPECTS(outDims != nullptr && perm != nullptr && in != nullptr && out != nullptr);

  const gpu::PermuteMeta meta  = permute_build_meta_rowmajor(ndim, outDims, perm);
  const size_t           total = gpu::permute_total(ndim, outDims);
  if(total == 0) return;

  double alpha_re, alpha_im;
  permute_split_scale_host(alpha, alpha_re, alpha_im);

  // Identity + alpha==1 + beta==0: pure copy.
  if(permute_scale_is_zero_host(beta) && permute_scale_is_one_host(alpha) &&
     gpu::permute_is_identity(meta)) {
    std::memcpy(out, in, total * sizeof(T));
    return;
  }

  // Aliased non-identity permute goes via a temp buffer.
  if(out == in && !gpu::permute_is_identity(meta)) {
    std::vector<T> tmp(in, in + total);
    permute_impl(out, tmp.data(), meta, total, alpha_re, alpha_im, beta);
    return;
  }

  permute_impl(out, in, meta, total, alpha_re, alpha_im, beta);
}

} // namespace tamm::kernels::cpu
