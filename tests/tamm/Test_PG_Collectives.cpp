// Correctness tests for ProcGroup collectives (proc_group.hpp).
//
// Previously uncovered (Test_PG is disabled and print-only).  Verifies
// allreduce / reduce / broadcast against analytic expected values across ranks.
// Runs with MPI (2 ranks in CI); the analytic checks scale with pg.size().

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

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);
  doctest::Context context(argc, argv);
  int              res = context.run();
  tamm::finalize();
  return res;
}
