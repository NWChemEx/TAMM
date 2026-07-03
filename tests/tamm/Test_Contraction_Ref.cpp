// Per-element, Eigen-referenced verification of TAMM tensor contractions.
//
// Unlike Test_Ops (uniform fills + single-scalar checks) and Test_Mult_Ops (one
// hardcoded Frobenius norm at N==50), this test fills operands with *non-uniform*
// index-dependent data and compares the TAMM contraction result element-by-element
// against a golden reference computed with Eigen::Tensor::contract / .shuffle().
//
// This directly guards the block-multiply plans (GemmPlan / GeneralMultPlan /
// FlatBlockMultPlan in block_mult_plan.hpp) and the multop execute path: a bug
// that permutes/mis-strides *within* a block but preserves the block sum (which a
// uniform fill or a norm check would miss) is caught here.
//
// Self-contained: uses only tamm_to_eigen_tensor<T,ndim>() from eigen_utils.hpp
// plus Eigen directly — no dependency on Test_Eigen.cpp's local helpers.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include "tamm/eigen_includes.hpp"
#include "tamm/eigen_utils.hpp"
#include "tamm/tamm.hpp"

#include <complex>

using namespace tamm;
using complex_double = std::complex<double>;

static size_t g_is_size   = 6;
static Tile   g_tile_size = 3;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic, non-uniform, per-element fill using global (untiled) indices.
/// Values are distinct across positions so permutation/stride errors change the
/// result (unlike a uniform fill).
template<typename T, typename Gen>
void fill_nonuniform(ExecutionContext& ec, Tensor<T> tensor, Gen&& gen) {
  auto lambda = [&](const IndexVector& blockid) {
    const TAMM_SIZE size = tensor.block_size(blockid);
    std::vector<T>  buf(size);
    const auto      bdims = tensor.block_dims(blockid);
    const auto      boff  = tensor.block_offsets(blockid);
    const size_t    rank  = bdims.size();

    // Row-major iteration over the block, tracking global indices.
    std::vector<size_t> idx(rank, 0);
    for(TAMM_SIZE c = 0; c < size; c++) {
      // decode flat c -> multi-index within block (row-major)
      size_t rem = c;
      for(size_t d = 0; d < rank; d++) {
        size_t stride = 1;
        for(size_t e = d + 1; e < rank; e++) stride *= bdims[e];
        idx[d] = boff[d] + (rem / stride);
        rem %= stride;
      }
      buf[c] = gen(idx);
    }
    tensor.put(blockid, buf);
  };
  block_for(ec, tensor(), lambda);
}

template<typename T>
double abs_of(const T& v) {
  if constexpr(tamm::internal::is_complex_v<T>) return std::abs(v);
  else return std::abs(static_cast<double>(v));
}

/// Compare an Eigen tensor (reference) to a TAMM tensor element-by-element.
template<typename T, int NDIM>
void expect_equal_to_eigen(const Tensor<T>&                              tamm_t,
                           const Eigen::Tensor<T, NDIM, Eigen::RowMajor>& ref,
                           double tol = 1e-9) {
  auto got = tamm_to_eigen_tensor<T, NDIM>(tamm_t);
  // dimensions must match
  for(int d = 0; d < NDIM; d++) { REQUIRE(got.dimension(d) == ref.dimension(d)); }
  const Eigen::Index n = ref.size();
  const T*           gp = got.data();
  const T*           rp = ref.data();
  double             max_err = 0.0;
  for(Eigen::Index i = 0; i < n; i++) { max_err = std::max(max_err, abs_of<T>(gp[i] - rp[i])); }
  CHECK(max_err <= tol);
}

// ---------------------------------------------------------------------------
// 2-D:  C(i,j) = alpha * A(i,k) * B(k,j)   (matrix multiply, plus a transpose)
// ---------------------------------------------------------------------------
template<typename T>
void test_mm_2d(ExecutionContext& ec, size_t N, Tile ts) {
  TiledIndexSpace tis{IndexSpace{range(N)}, ts};
  auto [i, j, k] = tis.labels<3>("all");

  Tensor<T> A{i, k}, B{k, j}, C{i, j}, Ct{j, i};
  Tensor<T>::allocate(&ec, A, B, C, Ct);
  Scheduler sch{ec};
  sch(A() = T{0})(B() = T{0})(C() = T{0})(Ct() = T{0}).execute();

  // Non-uniform fills: A[i,k]=1+i*0.5-k*0.25 ; B[k,j]=2-k*0.3+j*0.1
  fill_nonuniform<T>(ec, A, [](const std::vector<size_t>& x) {
    return T{1} + T(0.5) * T(double(x[0])) - T(0.25) * T(double(x[1]));
  });
  fill_nonuniform<T>(ec, B, [](const std::vector<size_t>& x) {
    return T{2} - T(0.3) * T(double(x[0])) + T(0.1) * T(double(x[1]));
  });
  ec.pg().barrier();

  const T alpha{1.5};

  // TAMM: C(i,j) = alpha*A(i,k)*B(k,j) ; Ct(j,i) = alpha*A(i,k)*B(k,j)
  sch(C(i, j) = alpha * A(i, k) * B(k, j)).execute();
  sch(Ct(j, i) = alpha * A(i, k) * B(k, j)).execute();
  ec.pg().barrier();

  // Eigen reference
  auto ea = tamm_to_eigen_tensor<T, 2>(A);
  auto eb = tamm_to_eigen_tensor<T, 2>(B);
  Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> dims_ik_kj = {Eigen::IndexPair<Eigen::Index>(1, 0)};
  Eigen::Tensor<T, 2, Eigen::RowMajor> eab = ea.contract(eb, dims_ik_kj);
  Eigen::Tensor<T, 2, Eigen::RowMajor> ec_ref = eab * eab.constant(alpha);
  Eigen::Tensor<T, 2, Eigen::RowMajor> ect_ref =
    ec_ref.shuffle(Eigen::array<int, 2>{1, 0});

  expect_equal_to_eigen<T, 2>(C, ec_ref);
  expect_equal_to_eigen<T, 2>(Ct, ect_ref);

  Tensor<T>::deallocate(A, B, C, Ct);
}

// ---------------------------------------------------------------------------
// 2-D accumulate (+=) semantics on non-uniform data.
// ---------------------------------------------------------------------------
template<typename T>
void test_mm_2d_accumulate(ExecutionContext& ec, size_t N, Tile ts) {
  TiledIndexSpace tis{IndexSpace{range(N)}, ts};
  auto [i, j, k] = tis.labels<3>("all");

  Tensor<T> A{i, k}, B{k, j}, C{i, j};
  Tensor<T>::allocate(&ec, A, B, C);
  Scheduler sch{ec};
  sch(A() = T{0})(B() = T{0}).execute();
  fill_nonuniform<T>(ec, A, [](const std::vector<size_t>& x) {
    return T{1} + T(0.5) * T(double(x[0])) - T(0.25) * T(double(x[1]));
  });
  fill_nonuniform<T>(ec, B, [](const std::vector<size_t>& x) {
    return T{2} - T(0.3) * T(double(x[0])) + T(0.1) * T(double(x[1]));
  });
  // C starts non-zero so += must add onto it.
  sch(C() = T{3}).execute();
  ec.pg().barrier();

  sch(C(i, j) += A(i, k) * B(k, j)).execute();
  ec.pg().barrier();

  auto ea = tamm_to_eigen_tensor<T, 2>(A);
  auto eb = tamm_to_eigen_tensor<T, 2>(B);
  Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> d = {Eigen::IndexPair<Eigen::Index>(1, 0)};
  Eigen::Tensor<T, 2, Eigen::RowMajor> ref = ea.contract(eb, d);
  ref = ref + ref.constant(T{3});
  expect_equal_to_eigen<T, 2>(C, ref);

  Tensor<T>::deallocate(A, B, C);
}

// ---------------------------------------------------------------------------
// 4-D:  C(i,j,k,l) = A(i,j,m,o) * B(m,o,k,l)  (two contracted indices)
// plus permuted output C(j,i,k,l).
// ---------------------------------------------------------------------------
template<typename T>
void test_mm_4d(ExecutionContext& ec, size_t N, Tile ts) {
  TiledIndexSpace tis{IndexSpace{range(N)}, ts};
  auto [i, j, k, l, m, o] = tis.labels<6>("all");

  Tensor<T> A{i, j, m, o}, B{m, o, k, l}, C{i, j, k, l}, Cp{j, i, k, l};
  Tensor<T>::allocate(&ec, A, B, C, Cp);
  Scheduler sch{ec};
  sch(A() = T{0})(B() = T{0})(C() = T{0})(Cp() = T{0}).execute();

  fill_nonuniform<T>(ec, A, [](const std::vector<size_t>& x) {
    return T{1} + T(0.1) * T(double(x[0])) - T(0.05) * T(double(x[1])) +
           T(0.2) * T(double(x[2])) - T(0.15) * T(double(x[3]));
  });
  fill_nonuniform<T>(ec, B, [](const std::vector<size_t>& x) {
    return T{2} - T(0.07) * T(double(x[0])) + T(0.11) * T(double(x[1])) -
           T(0.03) * T(double(x[2])) + T(0.09) * T(double(x[3]));
  });
  ec.pg().barrier();

  sch(C(i, j, k, l) = A(i, j, m, o) * B(m, o, k, l)).execute();
  sch(Cp(j, i, k, l) = A(i, j, m, o) * B(m, o, k, l)).execute();
  ec.pg().barrier();

  auto ea = tamm_to_eigen_tensor<T, 4>(A);
  auto eb = tamm_to_eigen_tensor<T, 4>(B);
  // contract A(i,j,m,o) with B(m,o,k,l) over (m,o): A dims 2,3 with B dims 0,1
  Eigen::array<Eigen::IndexPair<Eigen::Index>, 2> d = {Eigen::IndexPair<Eigen::Index>(2, 0),
                                                       Eigen::IndexPair<Eigen::Index>(3, 1)};
  Eigen::Tensor<T, 4, Eigen::RowMajor> ref = ea.contract(eb, d); // (i,j,k,l)
  Eigen::Tensor<T, 4, Eigen::RowMajor> refp =
    ref.shuffle(Eigen::array<int, 4>{1, 0, 2, 3}); // (j,i,k,l)

  expect_equal_to_eigen<T, 4>(C, ref);
  expect_equal_to_eigen<T, 4>(Cp, refp);

  Tensor<T>::deallocate(A, B, C, Cp);
}

// ---------------------------------------------------------------------------
// TEST_CASEs — run across a couple of tile sizes to exercise blocking.
// ---------------------------------------------------------------------------
TEST_CASE("Contraction ref: 2D matrix multiply (double)") {
  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  for(Tile ts: {Tile(1), g_tile_size, Tile(g_is_size)}) {
    Tile t = std::min<Tile>(ts, Tile(g_is_size));
    test_mm_2d<double>(ec, g_is_size, t);
    test_mm_2d_accumulate<double>(ec, g_is_size, t);
  }
}

TEST_CASE("Contraction ref: 2D matrix multiply (complex<double>)") {
  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  for(Tile ts: {Tile(1), g_tile_size}) {
    Tile t = std::min<Tile>(ts, Tile(g_is_size));
    test_mm_2d<complex_double>(ec, g_is_size, t);
  }
}

TEST_CASE("Contraction ref: 4D two-index contraction (double)") {
  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  for(Tile ts: {Tile(1), g_tile_size}) {
    Tile t = std::min<Tile>(ts, Tile(g_is_size));
    test_mm_4d<double>(ec, g_is_size, t);
  }
}

TEST_CASE("Contraction ref: 4D two-index contraction (complex<double>)") {
  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  test_mm_4d<complex_double>(ec, g_is_size, g_tile_size);
}

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  // Optional CLI overrides: <index_space_size> <tile_size> (kept small for CI).
  if(argc > 1) g_is_size = static_cast<size_t>(atoi(argv[1]));
  if(argc > 2) g_tile_size = static_cast<Tile>(atoi(argv[2]));
  if(g_is_size < g_tile_size) g_tile_size = static_cast<Tile>(g_is_size);

  doctest::Context context(argc, argv);
  int              res = context.run();

  tamm::finalize();
  return res;
}
