// Scheduler dependency-ordering correctness (scheduler.hpp levelization).
//
// Previously uncovered.  Builds a chain with a read-after-write hazard inside a
// single scheduler batch — E depends on C which is produced earlier in the same
// batch — and requires the final result to equal a reference computed step by
// step.  If the scheduler mis-orders dependent ops (or a levelization refactor
// regresses), E would be computed from a stale/zero C and the check fails.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include "tamm/tamm.hpp"

#include <vector>

using namespace tamm;

static size_t g_is_size   = 6;
static Tile   g_tile_size = 3;

template<typename T>
static void fill_nonuniform_2d(ExecutionContext& ec, Tensor<T> t, double base, double s0,
                               double s1) {
  auto lam = [&](const IndexVector& blockid) {
    const TAMM_SIZE size = t.block_size(blockid);
    std::vector<T>  buf(size);
    auto            bdims = t.block_dims(blockid);
    auto            boff  = t.block_offsets(blockid);
    TAMM_SIZE       c     = 0;
    for(size_t r = boff[0]; r < boff[0] + bdims[0]; r++)
      for(size_t s = boff[1]; s < boff[1] + bdims[1]; s++, c++)
        buf[c] = T(base + s0 * double(r) + s1 * double(s));
    t.put(blockid, buf);
  };
  block_for(ec, t(), lam);
}

template<typename T>
static std::vector<std::vector<T>> to_dense(Tensor<T> t, size_t N) {
  std::vector<std::vector<T>> m(N, std::vector<T>(N, T{0}));
  for(const auto& blockid: t.loop_nest()) {
    const TAMM_SIZE size = t.block_size(blockid);
    std::vector<T>  buf(size);
    t.get(blockid, buf);
    auto      bdims = t.block_dims(blockid);
    auto      boff  = t.block_offsets(blockid);
    TAMM_SIZE c     = 0;
    for(size_t r = boff[0]; r < boff[0] + bdims[0]; r++)
      for(size_t s = boff[1]; s < boff[1] + bdims[1]; s++, c++) m[r][s] = buf[c];
  }
  return m;
}

template<typename T>
static std::vector<std::vector<T>> matmul(const std::vector<std::vector<T>>& a,
                                          const std::vector<std::vector<T>>& b, size_t N) {
  std::vector<std::vector<T>> c(N, std::vector<T>(N, T{0}));
  for(size_t i = 0; i < N; i++)
    for(size_t j = 0; j < N; j++)
      for(size_t k = 0; k < N; k++) c[i][j] += a[i][k] * b[k][j];
  return c;
}

TEST_CASE("Scheduler orders dependent contractions within one batch") {
  using T = double;
  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  const size_t    N = g_is_size;
  TiledIndexSpace tis{IndexSpace{range(N)}, g_tile_size};
  auto [i, j, k] = tis.labels<3>("all");

  Tensor<T> A{i, k}, B{k, j}, C{i, j}, D{i, k}, E{i, j};
  Tensor<T>::allocate(&ec, A, B, C, D, E);
  Scheduler sch{ec};
  sch(A() = T{0})(B() = T{0})(C() = T{0})(D() = T{0})(E() = T{0}).execute();

  fill_nonuniform_2d<T>(ec, A, 1.0, 0.5, -0.25);
  fill_nonuniform_2d<T>(ec, B, 2.0, -0.3, 0.1);
  fill_nonuniform_2d<T>(ec, D, 0.7, 0.2, 0.15);
  ec.pg().barrier();

  // Reference: sequential C = A*B, then E = C*D.
  auto refA = to_dense<T>(A, N);
  auto refB = to_dense<T>(B, N);
  auto refD = to_dense<T>(D, N);
  auto refC = matmul<T>(refA, refB, N);
  auto refE = matmul<T>(refC, refD, N); // E(i,j) = C(i,k)*D(k,j)

  // Single scheduler batch with a RAW hazard: E reads C produced above.
  sch(C(i, j) = A(i, k) * B(k, j))(E(i, j) = C(i, k) * D(k, j)).execute();
  ec.pg().barrier();

  auto gotC = to_dense<T>(C, N);
  auto gotE = to_dense<T>(E, N);
  for(size_t r = 0; r < N; r++)
    for(size_t s = 0; s < N; s++) {
      CHECK(std::abs(gotC[r][s] - refC[r][s]) < 1e-9);
      CHECK(std::abs(gotE[r][s] - refE[r][s]) < 1e-9);
    }

  // WAW/WAR: reuse C as an output again after it was read; must still be ordered.
  sch(C(i, j) = A(i, k) * D(k, j)).execute();
  ec.pg().barrier();
  auto refC2 = matmul<T>(refA, refD, N);
  auto gotC2 = to_dense<T>(C, N);
  for(size_t r = 0; r < N; r++)
    for(size_t s = 0; s < N; s++) CHECK(std::abs(gotC2[r][s] - refC2[r][s]) < 1e-9);

  Tensor<T>::deallocate(A, B, C, D, E);
}

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);
  if(argc > 1) g_is_size = static_cast<size_t>(atoi(argv[1]));
  if(argc > 2) g_tile_size = static_cast<Tile>(atoi(argv[2]));
  if(g_is_size < g_tile_size) g_tile_size = static_cast<Tile>(g_is_size);
  doctest::Context context(argc, argv);
  int              res = context.run();
  tamm::finalize();
  return res;
}
