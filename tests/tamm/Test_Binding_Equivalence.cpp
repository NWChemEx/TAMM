// Core C++ TAMM half of the binding-equivalence check.
//
// Runs a fixed set of tensor operations through core C++ TAMM and writes the
// results to disk under the `cpp_tamm_equiv` prefix.
//
// Part of a 3-file check proving the pytamm (Python) bindings produce
// numerically identical results to core C++ TAMM:
//   1. Test_Binding_Equivalence.cpp    -> writes `cpp_tamm_equiv` (this file)
//   2. Test_Binding_Equivalence.py     -> writes `py_tamm_equiv`  (same ops, Python)
//   3. Compare_Binding_Equivalence.py  -> reads both, asserts they match
// Run 1 and 2 (order-independent), then 3.

#include "tamm/op_executor.hpp"
#include "tamm/opmin.hpp"
#include "tamm/tamm.hpp"

#include <complex>
#include <exception>
#include <iostream>
#include <string>

using namespace tamm;

namespace {

std::string arg_value(int argc, char** argv, const std::string& key, std::string def) {
  for(int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if(a == key && i + 1 < argc) return argv[i + 1];
    const std::string p = key + "=";
    if(a.rfind(p, 0) == 0) return a.substr(p.size());
  }
  return def;
}

int bi(const IndexVector& b, size_t i) { return static_cast<int>(b[i]); }

bool nz_a(const IndexVector& b) { return ((bi(b, 0) + 2 * bi(b, 1)) % 3) != 1; }

bool nz_b(const IndexVector& b) { return ((2 * bi(b, 0) + bi(b, 1)) % 3) != 2; }

bool nz_d(const IndexVector& b) {
  const int x = bi(b, 0);
  const int y = bi(b, 1);
  return x == y || ((x + y) % 2 == 0);
}

bool nz_c(const IndexVector& b) {
  const int x = bi(b, 0);
  const int y = bi(b, 1);
  return x <= y || ((x + y) % 2 == 0);
}

TensorInfo make_info2(const TiledIndexSpace& x, NonZeroCheck check) {
  return TensorInfo{TiledIndexSpaceVec{x, x}, check};
}

double dval(int which, const IndexVector& b, size_t p) {
  const long long x = bi(b, 0);
  const long long y = bi(b, 1);
  const long long q = static_cast<long long>(p);

  long long n = 0;
  if(which == 0) n = 16 * x + 4 * y + q + 1;
  if(which == 1) n = 8 * x - 6 * y + q + 3;
  if(which == 2) n = 5 * x + 7 * y + 2 * q + 1;

  return static_cast<double>(n) / 8.0;
}

std::complex<double> zval(int which, const IndexVector& b, size_t p) {
  const long long x = bi(b, 0);
  const long long y = bi(b, 1);
  const long long q = static_cast<long long>(p);

  long long re = 0;
  long long im = 0;

  if(which == 0) {
    re = 6 * x + 2 * y + q + 1;
    im = 3 * x - 5 * y + q + 2;
  }
  else {
    re = -4 * x + 7 * y + q + 1;
    im = 5 * x + y - q - 1;
  }

  return {static_cast<double>(re) / 8.0, static_cast<double>(im) / 8.0};
}

void fill_d(Tensor<double>& t, int which) {
  fill_sparse_tensor<double>(t, [which](const IndexVector& bid, span<double> buf) {
    for(size_t p = 0; p < buf.size(); ++p) buf[p] = dval(which, bid, p);
  });
}

void fill_z(Tensor<std::complex<double>>& t, int which) {
  fill_sparse_tensor<std::complex<double>>(
    t, [which](const IndexVector& bid, span<std::complex<double>> buf) {
      for(size_t p = 0; p < buf.size(); ++p) buf[p] = zval(which, bid, p);
    });
}

void run_scheduler_double(ExecutionContext& ec, const TiledIndexLabel& i, const TiledIndexLabel& j,
                          const TiledIndexLabel& k, Tensor<double>& A, Tensor<double>& B,
                          Tensor<double>& D, Tensor<double>& C) {
  Scheduler{ec}(C(i, j) = 0.0)(C(i, j) += 1.5 * A(i, k) * B(k, j))(C(i, j) -= 0.25 * D(i, j))
    .execute(ExecutionHW::CPU, false);
}

void run_scheduler_complex(ExecutionContext& ec, const TiledIndexLabel& i, const TiledIndexLabel& j,
                           const TiledIndexLabel& k, Tensor<std::complex<double>>& ZA,
                           Tensor<std::complex<double>>& ZB, Tensor<std::complex<double>>& ZC) {
  const std::complex<double> alpha{0.5, -0.25};
  const std::complex<double> beta{-0.125, 0.25};

  Scheduler{ec}(ZC(i, j) = std::complex<double>{0.0, 0.0})(ZC(i, j) += alpha * ZA(i, k) * ZB(k, j))(
    ZC(i, j) += beta * ZA(i, j))
    .execute(ExecutionHW::CPU, false);
}

void run_opmin_double(ExecutionContext& ec, const TiledIndexLabel& i, const TiledIndexLabel& j,
                      const TiledIndexLabel& k, Tensor<double>& A, Tensor<double>& B,
                      Tensor<double>& D, Tensor<double>& R) {
  Scheduler{ec}(R(i, j) = 0.0).execute(ExecutionHW::CPU, false);

  auto rhs = (1.25 * tamm::new_ops::LTOp{A(i, k)} * tamm::new_ops::LTOp{B(k, j)}) +
             (-0.5 * tamm::new_ops::LTOp{D(i, j)});

  R(i, j).update(rhs);

  SymbolTable symbol_table;
  TAMM_REGISTER_SYMBOLS(symbol_table, A, B, D, R);
  TAMM_REGISTER_SYMBOLS(symbol_table, i, j, k);

  Scheduler  sch{ec};
  OpExecutor op_exec{sch, symbol_table};

  op_exec.opmin_execute(R);
}

template<typename T>
void safe_deallocate(Tensor<T>& t) {
  if(t.is_allocated()) t.deallocate();
}

} // namespace

int main(int argc, char** argv) {
  tamm::initialize(argc, argv, false);

  int rc = 0;

  try {
    const std::string prefix = arg_value(argc, argv, "--prefix", "cpp_tamm_equiv");

    ProcGroup        pg = ProcGroup::create_world_coll();
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

    IndexSpace      is{range(6)};
    TiledIndexSpace x{is, 2};

    auto [i, j, k] = x.labels<3>("all");

    const auto info_a = make_info2(x, NonZeroCheck{nz_a});
    const auto info_b = make_info2(x, NonZeroCheck{nz_b});
    const auto info_d = make_info2(x, NonZeroCheck{nz_d});
    const auto info_c = make_info2(x, NonZeroCheck{nz_c});

    Tensor<double> A{TiledIndexSpaceVec{x, x}, info_a};
    Tensor<double> B{TiledIndexSpaceVec{x, x}, info_b};
    Tensor<double> D{TiledIndexSpaceVec{x, x}, info_d};
    Tensor<double> C{TiledIndexSpaceVec{x, x}, info_c};
    Tensor<double> R{TiledIndexSpaceVec{x, x}, info_c};

    Tensor<std::complex<double>> ZA{TiledIndexSpaceVec{x, x}, info_a};
    Tensor<std::complex<double>> ZB{TiledIndexSpaceVec{x, x}, info_b};
    Tensor<std::complex<double>> ZC{TiledIndexSpaceVec{x, x}, info_c};

    A.allocate(&ec);
    B.allocate(&ec);
    D.allocate(&ec);
    C.allocate(&ec);
    R.allocate(&ec);
    ZA.allocate(&ec);
    ZB.allocate(&ec);
    ZC.allocate(&ec);

    fill_d(A, 0);
    fill_d(B, 1);
    fill_d(D, 2);
    fill_z(ZA, 0);
    fill_z(ZB, 1);

    run_scheduler_double(ec, i, j, k, A, B, D, C);
    run_opmin_double(ec, i, j, k, A, B, D, R);
    run_scheduler_complex(ec, i, j, k, ZA, ZB, ZC);

    ec.flush_and_sync();

    write_to_disk<double>(C, prefix + "_sched_d", true, false, 0);
    write_to_disk<double>(R, prefix + "_opmin_d", true, false, 0);
    write_to_disk<std::complex<double>>(ZC, prefix + "_sched_z", true, false, 0);

    ec.flush_and_sync();
    pg.barrier();

    if(pg.rank() == 0) {
      std::cout << "Wrote " << prefix << "_sched_d, " << prefix << "_opmin_d, " << prefix
                << "_sched_z\n";
    }

    safe_deallocate(ZC);
    safe_deallocate(ZB);
    safe_deallocate(ZA);
    safe_deallocate(R);
    safe_deallocate(C);
    safe_deallocate(D);
    safe_deallocate(B);
    safe_deallocate(A);
  } catch(const std::exception& ex) {
    std::cerr << "C++ TAMM equivalence test failed: " << ex.what() << "\n";
    rc = 1;
  }

  tamm::finalize(true);
  return rc;
}