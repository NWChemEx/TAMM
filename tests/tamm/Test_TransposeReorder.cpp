// Unit tests for the in-house GPU tensor transpose (reorder).
//
// Exercises the backend-agnostic logic in tamm/kernels/gpu_reorder.hpp
// (spec building, metadata, index decode, scale/accumulate apply) directly on
// the host, so no GPU is needed. Every test compares the exact helpers the
// CUDA/HIP/SYCL kernels call against an independent row-major reference for
// all permutations of ranks 0..5 (plus rank-6/8 spot checks), several shapes
// (including unit extents), and double/complex<double> values with a range of
// scale/accumulate settings.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <tamm/kernels/gpu_reorder.hpp>

#include <algorithm>
#include <complex>
#include <numeric>
#include <vector>

using namespace tamm;
using namespace tamm::kernels::gpu;

namespace {

// Row-major (last-axis-fastest) strides.
std::vector<size_t> row_strides(const std::vector<size_t>& dims) {
  const size_t n = dims.size();
  std::vector<size_t> st(n, 1);
  for(size_t k = n; k-- > 1;) st[k - 1] = st[k] * dims[k];
  return st;
}

// Independent reference: dst[d] = (acc ? dst[d] + : 0) + scale*src[s] with a
// plain row-major transpose driven by the natural (non-reversed) permutation.
template<typename T>
void reference_transpose(const std::vector<T>& src, std::vector<T>& dst,
                         const std::vector<size_t>& sdims, const std::vector<int>& slabels,
                         const std::vector<size_t>& ddims, const std::vector<int>& dlabels, T scale,
                         bool accum) {
  const size_t ndim = sdims.size();
  const auto sst = row_strides(sdims);
  const auto dst_ = row_strides(ddims);
  // dest-axis -> src-axis in natural order
  std::vector<int> perm(ndim);
  for(size_t i = 0; i < ndim; ++i) {
    perm[i] = static_cast<int>(std::find(slabels.begin(), slabels.end(), dlabels[i]) -
                               slabels.begin());
  }
  size_t total = 1;
  for(auto d: ddims) total *= d;
  REQUIRE(dst.size() == total);
  std::vector<size_t> coord(ndim);
  for(size_t lin = 0; lin < total; ++lin) {
    size_t rem = lin;
    for(size_t k = ndim; k-- > 0;) {
      coord[k] = (ndim == 0) ? 0 : rem % ddims[k];
      if(ndim > 0) rem /= ddims[k];
    }
    size_t s = 0;
    for(size_t k = 0; k < ndim; ++k) s += coord[k] * sst[static_cast<size_t>(perm[k])];
    dst[lin] = accum ? dst[lin] + scale * src[s] : scale * src[s];
  }
}

// Replica of the device loop in gpu_reorder.cpp, but on the host, using the
// shipped helpers (build_reorder_spec / reorder_build_meta /
// reorder_src_index / reorder_scaled / reorder_add). This is the exact math
// each backend kernel executes; only the thread loop itself is replicated.
template<typename T, typename Idx>
void device_math_replica(const std::vector<T>& src, std::vector<T>& dst, const SizeVec& sdims,
                         const IntLabelVec& slabels, const IntLabelVec& dlabels, T scale,
                         bool accum) {
  const int ndim = static_cast<int>(sdims.size());
  ReorderSpec spec{};
  build_reorder_spec(sdims, slabels, dlabels, spec);
  const ReorderMeta meta = reorder_build_meta(ndim, spec.outDims, spec.perm);
  const size_t total = reorder_total(ndim, spec.outDims);
  REQUIRE(dst.size() == total);
  double sre, sim;
  if constexpr(reorder_is_complex_v<T>) {
    sre = static_cast<double>(scale.real());
    sim = static_cast<double>(scale.imag());
  }
  else {
    sre = static_cast<double>(scale);
    sim = 0.0;
  }
  for(size_t t = 0; t < total; ++t) {
    const Idx tid = static_cast<Idx>(t);
    const size_t s = reorder_src_index<Idx>(tid, meta);
    const T      y = reorder_scaled<T>(src[s], sre, sim);
    dst[t]         = accum ? reorder_add<T>(dst[t], y) : y;
  }
}

template<typename T>
T make_val(size_t i) {
  if constexpr(reorder_is_complex_v<T>) {
    return T(static_cast<typename T::value_type>(0.25 * (i + 1)),
             static_cast<typename T::value_type>(-0.125 * (i + 3)));
  }
  else { return static_cast<T>(0.25 * (i + 1)); }
}

template<typename T>
void check_case(const std::vector<size_t>& sdims, const std::vector<int>& slabels,
                const std::vector<int>& dlabels, T scale, bool accum) {
  const size_t ndim = sdims.size();
  SizeVec sv;
  for(auto d: sdims) sv.emplace_back(d);
  IntLabelVec sl(slabels.begin(), slabels.end()), dl(dlabels.begin(), dlabels.end());

  std::vector<size_t> ddims(ndim);
  for(size_t i = 0; i < ndim; ++i) {
    const int j =
      static_cast<int>(std::find(slabels.begin(), slabels.end(), dlabels[i]) - slabels.begin());
    ddims[i] = sdims[static_cast<size_t>(j)];
  }
  size_t total = 1;
  for(auto d: ddims) total *= d;

  std::vector<T> src(total), ref(total), got32(total), got64(total), init(total);
  for(size_t i = 0; i < total; ++i) {
    src[i]   = make_val<T>(i + 1);
    init[i]  = make_val<T>(1000 + i);
    ref[i]   = init[i];
    got32[i] = init[i];
    got64[i] = init[i];
  }
  reference_transpose(src, ref, sdims, slabels, ddims, dlabels, scale, accum);
  device_math_replica<T, uint32_t>(src, got32, sv, sl, dl, scale, accum);
  device_math_replica<T, uint64_t>(src, got64, sv, sl, dl, scale, accum);
  for(size_t i = 0; i < total; ++i) {
    CHECK(got32[i] == ref[i]);
    CHECK(got64[i] == ref[i]);
  }
  // The 64-bit total path must agree with the 32-bit path elementwise.
  for(size_t i = 0; i < total; ++i) { CHECK(got64[i] == got32[i]); }
}

template<typename T>
void check_all_perms(const std::vector<size_t>& sdims, T scale, bool accum) {
  const size_t ndim = sdims.size();
  std::vector<int> labels(ndim);
  std::iota(labels.begin(), labels.end(), 0);
  if(ndim == 0) {
    check_case<T>({}, {}, {}, scale, accum);
    return;
  }
  std::vector<int> perm = labels;
  do { check_case<T>(sdims, labels, perm, scale, accum); } while(
    std::next_permutation(perm.begin(), perm.end()));
}

} // namespace

TEST_CASE("reorder rank 0..2, all perms, scales, assign/accumulate") {
  using C = std::complex<double>;
  for(bool accum: {false, true}) {
    for(double s: {1.0, 2.5, 0.0, -1.0}) {
      check_all_perms<double>({}, static_cast<double>(s), accum);
      check_all_perms<double>({1}, static_cast<double>(s), accum);
      check_all_perms<double>({5}, static_cast<double>(s), accum);
      check_all_perms<double>({1, 1}, static_cast<double>(s), accum);
      check_all_perms<double>({2, 3}, static_cast<double>(s), accum);
      check_all_perms<double>({4, 1}, static_cast<double>(s), accum);
      check_all_perms<C>({2, 3}, C(s, 0.5 * s), accum);
    }
  }
}

TEST_CASE("reorder rank 3, all perms") {
  using C = std::complex<double>;
  for(bool accum: {false, true}) {
    check_all_perms<double>({2, 3, 4}, 1.0, accum);
    check_all_perms<double>({1, 5, 2}, 2.0, accum);
    check_all_perms<double>({3, 1, 1}, -1.5, accum);
    check_all_perms<C>({2, 3, 4}, C(1.0, -1.0), accum);
    check_all_perms<C>({1, 1, 6}, C(0.0, 2.0), accum);
  }
}

TEST_CASE("reorder rank 4, all perms") {
  using C = std::complex<double>;
  for(bool accum: {false, true}) {
    check_all_perms<double>({2, 3, 2, 4}, 1.0, accum);
    check_all_perms<double>({4, 1, 3, 2}, 0.5, accum);
    check_all_perms<C>({2, 2, 3, 2}, C(2.0, 1.0), accum);
  }
}

TEST_CASE("reorder rank 5, all perms") {
  for(bool accum: {false, true}) {
    check_all_perms<double>({2, 1, 3, 2, 2}, 1.0, accum);
    check_all_perms<std::complex<double>>({2, 1, 2, 2, 3}, std::complex<double>(1.0, 1.0),
                                           accum);
  }
}

TEST_CASE("reorder rank 6 and 8 spot checks") {
  // 720 perms at rank 6: check identity, reversal, and a rotation plus a few
  // random perms rather than all of them.
  const std::vector<size_t> d6{2, 3, 1, 4, 2, 2};
  const std::vector<int>    id6{0, 1, 2, 3, 4, 5};
  check_case<double>(d6, id6, {5, 4, 3, 2, 1, 0}, 1.0, false);
  check_case<double>(d6, id6, {5, 4, 3, 2, 1, 0}, 3.0, true);
  check_case<double>(d6, id6, id6, 1.0, false);
  check_case<double>(d6, id6, {1, 2, 3, 4, 5, 0}, -2.0, true);
  check_case<double>(d6, id6, {2, 0, 4, 1, 5, 3}, 1.0, false);
  check_case<std::complex<double>>(d6, id6, {5, 4, 3, 2, 1, 0}, std::complex<double>(0.5, -0.5),
                                   true);
  const std::vector<size_t> d8{2, 1, 2, 3, 1, 2, 2, 2};
  const std::vector<int>    id8{0, 1, 2, 3, 4, 5, 6, 7};
  check_case<double>(d8, id8, {7, 6, 5, 4, 3, 2, 1, 0}, 1.0, false);
  check_case<double>(d8, id8, id8, 1.0, false);
  check_case<double>(d8, id8, {3, 7, 1, 5, 0, 6, 2, 4}, 2.0, true);
}

TEST_CASE("reorder metadata invariants") {
  // Column-major strides + perm round-trip on a known case.
  const size_t outDims[3] = {4, 2, 3};
  const int    perm[3]    = {2, 0, 1};
  const ReorderMeta meta = reorder_build_meta(3, outDims, perm);
  CHECK(meta.outStrides[0] == 1);
  CHECK(meta.outStrides[1] == 4);
  CHECK(meta.outStrides[2] == 8);
  // inDims = {2, 3, 4} -> inStrides {1, 2, 6}
  CHECK(meta.inStrides[0] == 1);
  CHECK(meta.inStrides[1] == 2);
  CHECK(meta.inStrides[2] == 6);
  CHECK(reorder_total(3, outDims) == 24);
  CHECK_FALSE(reorder_is_identity(meta));
  const int    idp[3]    = {0, 1, 2};
  const ReorderMeta idm = reorder_build_meta(3, outDims, idp);
  CHECK(reorder_is_identity(idm));
  CHECK(reorder_meta_fits32(meta, 24));
  CHECK_FALSE(reorder_meta_fits32(meta, size_t{1} << 33));
}
