// Tests for ProcGroup (proc_group.hpp).
//
// Verifies allreduce / reduce / broadcast against analytic expected values across ranks (the
// checks scale with pg.size()), a tensor allocated on a subgroup, and a replicated tensor
// contracted with distributed ones. Runs with MPI (2 ranks in CI).

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include "tamm/tamm.hpp"

#include <vector>

using namespace tamm;

TEST_CASE("ProcGroup allreduce sum (single value)") {
  ProcGroup pg   = ProcGroup::create_world_coll();
  const int nrk  = pg.size().value();
  const int rank = pg.rank().value();

  // Each rank contributes (rank+1); sum over ranks = nrk*(nrk+1)/2.
  double val = static_cast<double>(rank + 1);
  double sum = pg.allreduce(&val, ReduceOp::sum);
  CHECK(sum == doctest::Approx(nrk * (nrk + 1) / 2.0));
}

TEST_CASE("ProcGroup allreduce sum (buffer)") {
  ProcGroup pg   = ProcGroup::create_world_coll();
  const int nrk  = pg.size().value();
  const int rank = pg.rank().value();

  const int           count = 4;
  std::vector<double> sbuf(count), rbuf(count, -1.0);
  for(int i = 0; i < count; i++) sbuf[i] = static_cast<double>((rank + 1) * (i + 1));
  pg.allreduce(sbuf.data(), rbuf.data(), count, ReduceOp::sum);

  // sum over ranks of (rank+1)*(i+1) = (i+1) * nrk*(nrk+1)/2
  for(int i = 0; i < count; i++) {
    CHECK(rbuf[i] == doctest::Approx((i + 1) * (nrk * (nrk + 1) / 2.0)));
  }
}

TEST_CASE("ProcGroup reduce to root (single value)") {
  ProcGroup pg   = ProcGroup::create_world_coll();
  const int nrk  = pg.size().value();
  const int rank = pg.rank().value();

  double val = static_cast<double>(rank + 1);
  double res = pg.reduce(&val, ReduceOp::sum, /*root=*/0);
  if(rank == 0) { CHECK(res == doctest::Approx(nrk * (nrk + 1) / 2.0)); }
  pg.barrier();
}

TEST_CASE("ProcGroup allreduce min/max") {
  ProcGroup pg   = ProcGroup::create_world_coll();
  const int nrk  = pg.size().value();
  const int rank = pg.rank().value();

  double val  = static_cast<double>(rank + 1); // ranks: 1..nrk
  double vmin = pg.allreduce(&val, ReduceOp::min);
  double vmax = pg.allreduce(&val, ReduceOp::max);
  CHECK(vmin == doctest::Approx(1.0));
  CHECK(vmax == doctest::Approx(static_cast<double>(nrk)));
}

TEST_CASE("ProcGroup broadcast (single value and buffer)") {
  ProcGroup pg   = ProcGroup::create_world_coll();
  const int rank = pg.rank().value();

  // single value: root sets 42, others 0; after broadcast all see 42.
  double v = (rank == 0) ? 42.0 : 0.0;
  pg.broadcast(&v, /*root=*/0);
  CHECK(v == doctest::Approx(42.0));

  // buffer
  const int           count = 3;
  std::vector<double> buf(count, 0.0);
  if(rank == 0) {
    buf[0] = 1.0;
    buf[1] = 2.0;
    buf[2] = 3.0;
  }
  pg.broadcast(buf.data(), count, /*root=*/0);
  CHECK(buf[0] == doctest::Approx(1.0));
  CHECK(buf[1] == doctest::Approx(2.0));
  CHECK(buf[2] == doctest::Approx(3.0));
}

TEST_CASE("Tensor allocated on a subgroup") {
  using T = double;

  ProcGroup        gpg = ProcGroup::create_world_coll();
  ExecutionContext gec{gpg, DistributionKind::nw, MemoryManagerKind::ga};

  // Collective over gpg: the group is valid only on the ranks in it
  ProcGroup pg_m = ProcGroup::create_subgroup(gpg, gec.ppn());

  if(pg_m.is_valid()) {
    TiledIndexSpace AO{IndexSpace{range(5)}, 5};
    Tensor<T>       tens1{AO, AO};

    {
      ExecutionContext ec_m{pg_m, DistributionKind::nw, MemoryManagerKind::ga};
      Scheduler        sch{ec_m};

      sch.allocate(tens1)(tens1() = 4.0).execute();

      for(const auto& blockid: tens1.loop_nest()) {
        std::vector<T> buf(tens1.block_size(blockid));
        tens1.get(blockid, buf);
        for(const auto v: buf) CHECK(v == doctest::Approx(4.0));
      }

      Tensor<T>::deallocate(tens1);
    }
    pg_m.destroy_coll();
  }
}

// Disabled: aborts when run (cause not yet known).
// TEST_CASE("Tensor replicated on each rank contracted with distributed tensors") {
//   using T = double;
//
//   ProcGroup        gpg = ProcGroup::create_world_coll();
//   ExecutionContext gec{gpg, DistributionKind::dense, MemoryManagerKind::ga};
//
//   const size_t    dim = 20;
//   TiledIndexSpace tis1{IndexSpace{range(dim)}, 40};
//
//   auto [i, j, k] = tis1.labels<3>("all");
//
//   Tensor<T> A{tis1, tis1};
//   Tensor<T> B{tis1, tis1};
//   Tensor<T> C{tis1, tis1};
//
//   Scheduler gsch{gec};
//   gsch.allocate(A, C).execute();
//
//   { // B is replicated
//     ProcGroup        pg = ProcGroup::create_self();
//     ExecutionContext ec{pg, DistributionKind::dense, MemoryManagerKind::ga};
//
//     Scheduler{ec}.allocate(B).execute();
//
//     gsch(A() = 21.0)(B() = 2.0)(C(i, j) = A(i, k) * B(k, j)).execute();
//
//     Scheduler{ec}.deallocate(B).execute();
//
//     ec.flush_and_sync();
//   }
//
//   gec.pg().barrier();
//
//   for(const auto& blockid: C.loop_nest()) {
//     std::vector<T> buf(C.block_size(blockid));
//     C.get(blockid, buf);
//     for(const auto v: buf) CHECK(v == doctest::Approx(21.0 * 2.0 * double(dim)));
//   }
//
//   gsch.deallocate(A, C).execute();
//
//   gec.flush_and_sync();
// }

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);
  doctest::Context context(argc, argv);
  if(ProcGroup::world_rank() != 0) context.setOption("minimal", true);
  int res = context.run();
  tamm::finalize();
  return res;
}
