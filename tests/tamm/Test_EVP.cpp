#include <tamm/eigen_utils.hpp>
#include <tamm/tamm.hpp>
#include <tamm/tamm_config.hpp>
#include <tamm/tamm_git.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <random>

using namespace tamm;

using RMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Wall-clock time (s) of a single (already collective, where applicable) solver call.
template<typename F>
static double timed_s(F&& f) {
  const auto t0 = std::chrono::high_resolution_clock::now();
  std::forward<F>(f)();
  return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
}

// Random symmetric N x N matrix (same on every rank: fixed seed).
static RMatrix random_symmetric(int64_t N, unsigned seed) {
  std::mt19937                           gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  RMatrix                                R(N, N);
  for(int64_t i = 0; i < N; i++)
    for(int64_t j = 0; j < N; j++) R(i, j) = dist(gen);
  return 0.5 * (R + R.transpose());
}

// Checks the local tamm::eigensolve on a random symmetric N x N matrix: residual,
// orthonormality, ascending eigenvalues and the row = eigenvector layout, for both
// ExecutionHW::CPU and ec.exhw() (the GPU solver in GPU builds), and that the two
// paths give the same eigenvalues.
void test_local_eigensolve(tamm::ExecutionContext& ec, int64_t N) {
  if(ec.pg().rank() != 0) return; // rank-local solve

  // Untimed warm-up so the timed GPU solve does not include one-time GPU/solver initialization.
  if(ec.exhw() == ExecutionHW::GPU) {
    RMatrix             W = random_symmetric(64, 1);
    std::vector<double> w;
    tamm::eigensolve(64, W.data(), w, ExecutionHW::GPU);
  }

  std::mt19937                           gen(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  RMatrix                                R(N, N);
  for(int64_t i = 0; i < N; i++)
    for(int64_t j = 0; j < N; j++) R(i, j) = dist(gen);
  const RMatrix S = 0.5 * (R + R.transpose());

  auto check = [&](tamm::ExecutionHW hw, const std::string& label) {
    RMatrix             V = S;
    std::vector<double> eps;
    const double        elapsed_s = timed_s([&]() { tamm::eigensolve(N, V.data(), eps, hw); });

    EXPECTS(static_cast<int64_t>(eps.size()) == N);
    EXPECTS(std::is_sorted(eps.begin(), eps.end()));

    // Row i of V (row-major) is eigenvector i, so the columns of V^T are the eigenvectors.
    const RMatrix         Vt   = V.transpose();
    const Eigen::VectorXd lam  = Eigen::Map<const Eigen::VectorXd>(eps.data(), N);
    const double          res  = (S * Vt - Vt * lam.asDiagonal()).norm() / S.norm();
    const double          orth = (V * V.transpose() - RMatrix::Identity(N, N)).norm();

    std::cout << "eigensolve (local) [" << label << "] N=" << N
              << " : ||SV-VL||/||S||=" << std::scientific << std::setprecision(3) << res
              << " ||VV^T-I||=" << orth << " time=" << std::defaultfloat << std::setprecision(3)
              << elapsed_s << " s" << std::endl;
    EXPECTS(res < 1e-10);
    EXPECTS(orth < 1e-10);
    return eps;
  };

  const std::vector<double> eps_cpu = check(tamm::ExecutionHW::CPU, "CPU");
  const std::vector<double> eps_hw =
    check(ec.exhw(), ec.exhw() == tamm::ExecutionHW::GPU ? "GPU" : "CPU (exhw)");

  double max_diff = 0.0, max_abs = 1.0;
  for(int64_t i = 0; i < N; i++) {
    max_diff = std::max(max_diff, std::abs(eps_cpu[i] - eps_hw[i]));
    max_abs  = std::max(max_abs, std::abs(eps_cpu[i]));
  }
  std::cout << "eigensolve (local) CPU vs exhw: max |eps diff| = " << std::scientific
            << std::setprecision(3) << max_diff << std::endl;
  EXPECTS(max_diff < 1e-10 * max_abs);
}

// Checks the local generalized_eigensolve with a random rectangular basis X (N x M), for both
// ExecutionHW::CPU and ec.exhw(): C^T S C = diag(eps) (C = X C' with orthonormal C') and the two
// paths give the same eigenvalues.
void test_local_generalized(ExecutionContext& ec, int64_t N, int64_t M) {
  if(ec.pg().rank() != 0) return; // rank-local solve

  const RMatrix                          S = random_symmetric(N, 5);
  std::mt19937                           gen(13);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  RMatrix                                X(N, M);
  for(int64_t i = 0; i < N; i++)
    for(int64_t j = 0; j < M; j++) X(i, j) = dist(gen);

  auto check = [&](ExecutionHW hw, const std::string& label) {
    RMatrix             F = S, C(N, M);
    std::vector<double> eps;
    const double        elapsed_s =
      timed_s([&]() { generalized_eigensolve(N, M, F.data(), X.data(), C.data(), eps, hw); });

    EXPECTS(static_cast<int64_t>(eps.size()) == M);
    EXPECTS(std::is_sorted(eps.begin(), eps.end()));
    const RMatrix L   = RMatrix(Eigen::Map<const Eigen::VectorXd>(eps.data(), M).asDiagonal());
    const double  res = (C.transpose() * S * C - L).norm() / S.norm();
    std::cout << "generalized_eigensolve (local) [" << label << "] N=" << N << " M=" << M
              << " : ||C^TSC-L||/||S||=" << std::scientific << std::setprecision(3) << res
              << " time=" << std::defaultfloat << std::setprecision(3) << elapsed_s << " s"
              << std::endl;
    EXPECTS(res < 1e-8);
    return eps;
  };

  const std::vector<double> eps_cpu = check(ExecutionHW::CPU, "CPU");
  const std::vector<double> eps_hw =
    check(ec.exhw(), ec.exhw() == ExecutionHW::GPU ? "GPU" : "CPU (exhw)");

  double max_diff = 0.0, max_abs = 1.0;
  for(int64_t i = 0; i < M; i++) {
    max_diff = std::max(max_diff, std::abs(eps_cpu[i] - eps_hw[i]));
    max_abs  = std::max(max_abs, std::abs(eps_cpu[i]));
  }
  std::cout << "generalized_eigensolve (local) CPU vs exhw: max |eps diff| = " << std::scientific
            << std::setprecision(3) << max_diff << std::endl;
  EXPECTS(max_diff < 1e-10 * max_abs);
}

// Checks the distributed eigensolve in every build (solved on the ScaLAPACK grid, or on rank 0
// otherwise) for hw = CPU and ec.exhw(): residual, orthonormality and eigenvalues vs the local
// reference. No grid exists beforehand: with ScaLAPACK the first call creates the process group's
// grid, which is released at the end; without ScaLAPACK the group only has the empty grid.
void test_tensor_eigensolve(ExecutionContext& ec, int64_t N) {
  const bool root = ec.pg().rank() == 0;
  EXPECTS(!find_scalapack_grid(ec).participates());

  const RMatrix   S = random_symmetric(N, 3);
  TiledIndexSpace tN{IndexSpace{range(N)}, 17};
  Tensor<double>  A{tN, tN}, V{tN, tN};
  Tensor<double>::allocate(&ec, A, V);
  if(root) {
    RMatrix Sc = S;
    eigen_to_tamm_tensor(A, Sc);
  }
  ec.pg().barrier();

  std::vector<double> eps_ref;
  RMatrix             Vref = S;
  eigensolve(N, Vref.data(), eps_ref);

  auto check = [&](ExecutionHW hw, const std::string& label) {
    std::vector<double> eps;
    const double        elapsed_s = timed_s([&]() { eigensolve(ec, A, V, eps, hw); });
    if(root) {
      const RMatrix         Vm   = tamm_to_eigen_matrix(V);
      const RMatrix         Vt   = Vm.transpose();
      const Eigen::VectorXd lam  = Eigen::Map<const Eigen::VectorXd>(eps.data(), N);
      const double          res  = (S * Vt - Vt * lam.asDiagonal()).norm() / S.norm();
      const double          orth = (Vm * Vm.transpose() - RMatrix::Identity(N, N)).norm();
      double                dmax = 0.0;
      for(int64_t i = 0; i < N; i++) dmax = std::max(dmax, std::abs(eps[i] - eps_ref[i]));
      std::cout << "eigensolve (tensor) [" << label << "] N=" << N
                << " : ||SV-VL||/||S||=" << std::scientific << std::setprecision(3) << res
                << " ||VV^T-I||=" << orth << " max|eps-eps_local|=" << dmax
                << " time=" << std::defaultfloat << std::setprecision(3) << elapsed_s << " s"
                << std::endl;
      EXPECTS(std::is_sorted(eps.begin(), eps.end()));
      EXPECTS(res < 1e-10 && orth < 1e-10 && dmax < 1e-10);
    }
  };
  check(ExecutionHW::CPU, "CPU");
  check(ec.exhw(), ec.exhw() == ExecutionHW::GPU ? "GPU" : "CPU (exhw)");

#if defined(USE_SCALAPACK)
  // The first eigensolve created the grid; it uses the group's first ranks, so rank 0 is in it.
  if(root) EXPECTS(find_scalapack_grid(ec).participates());
#else
  EXPECTS(!find_scalapack_grid(ec).participates());
#endif
  Tensor<double>::deallocate(A, V);
  release_scalapack_grid(ec);
  EXPECTS(!find_scalapack_grid(ec).participates());
}

#if !defined(USE_SCALAPACK)
// Without ScaLAPACK every process group only has the empty grid: scalapack_grid and
// find_scalapack_grid return it, and release_scalapack_grid does nothing.
void test_empty_grid(ExecutionContext& ec) {
  const ScalapackGridHints hints{.N = 100, .max_nranks = 1};
  const ScalapackGrid&     grid = scalapack_grid(ec, hints);
  EXPECTS(!grid.participates());
  EXPECTS(!find_scalapack_grid(ec).participates());
  release_scalapack_grid(ec);
  release_scalapack_grid(ec);
  EXPECTS(!find_scalapack_grid(ec).participates());
  if(ec.pg().rank() == 0) std::cout << "empty grid (build without ScaLAPACK): ok" << std::endl;
}
#endif

#if defined(USE_SCALAPACK)

// Checks the distributed API on the process group's ScaLAPACK grid: eigensolve (into regular and
// allocate_dense V), generalized eigensolve with a rectangular orthogonalizer,
// to_block_cyclic/from_block_cyclic/from_block_cyclic_dense, release + re-creation, and max_nranks.
void test_distributed(ExecutionContext& ec, int64_t N, int64_t M) {
  const bool root = ec.pg().rank() == 0;

  // An explicit 2x2 grid (needs >= 4 ranks) with a small block size, so matrices span ranks.
  const ScalapackGridHints hints{.N = N, .npr = 2, .npc = 2, .nb = 16};
  const ScalapackGrid&     grid = scalapack_grid(ec, hints);
  // Explicit hints that differ from the existing grid: reused, with a one-time warning.
  scalapack_grid(ec, {.N = N, .npr = 1, .npc = 1});
  EXPECTS(&find_scalapack_grid(ec) == &grid);

  const RMatrix   S = random_symmetric(N, 7);
  TiledIndexSpace tN{IndexSpace{range(N)}, 17};
  TiledIndexSpace tM{IndexSpace{range(M)}, 17};
  Tensor<double>  A{tN, tN}, V{tN, tN};
  Tensor<double>::allocate(&ec, A, V);
  if(root) {
    RMatrix Sc = S;
    eigen_to_tamm_tensor(A, Sc);
  }
  ec.pg().barrier();

  // Reference: local eigensolve of S on the host.
  std::vector<double> eps_ref;
  RMatrix             Vref = S;
  eigensolve(N, Vref.data(), eps_ref);

  // 1) eigensolve with hw = CPU and ec.exhw() (ELPA on the GPU in CUDA+ELPA builds): residual,
  // orthonormality and eigenvalues vs the local reference (rank 0), and CPU vs exhw.
  const std::string exhw_label       = ec.exhw() == ExecutionHW::GPU ? "GPU" : "CPU (exhw)";
  auto              check_eigensolve = [&](ExecutionHW hw, const std::string& label) {
    std::vector<double> eps;
    const double        elapsed_s = timed_s([&]() { eigensolve(ec, A, V, eps, hw); });
    if(root) {
      const RMatrix         Vm  = tamm_to_eigen_matrix(V);
      const RMatrix         Vt  = Vm.transpose();
      const Eigen::VectorXd lam = Eigen::Map<const Eigen::VectorXd>(eps.data(), N);
      const double          res = (S * Vt - Vt * lam.asDiagonal()).norm() / S.norm();
      const double          orth = (Vm * Vm.transpose() - RMatrix::Identity(N, N)).norm();
      double                dmax = 0.0;
      for(int64_t i = 0; i < N; i++) dmax = std::max(dmax, std::abs(eps[i] - eps_ref[i]));
      std::cout << "eigensolve (distributed) [" << label << "] N=" << N
                << " : ||SV-VL||/||S||=" << std::scientific << std::setprecision(3) << res
                << " ||VV^T-I||=" << orth << " max|eps-eps_local|=" << dmax
                << " time=" << std::defaultfloat << std::setprecision(3) << elapsed_s << " s"
                << std::endl;
      EXPECTS(std::is_sorted(eps.begin(), eps.end()));
      EXPECTS(res < 1e-10 && orth < 1e-10 && dmax < 1e-10);
    }
    return eps;
  };
  const std::vector<double> eps_cpu = check_eigensolve(ExecutionHW::CPU, "CPU");
  const std::vector<double> eps_hw  = check_eigensolve(ec.exhw(), exhw_label);
  if(root) {
    double dmax = 0.0;
    for(int64_t i = 0; i < N; i++) dmax = std::max(dmax, std::abs(eps_cpu[i] - eps_hw[i]));
    std::cout << "eigensolve (distributed) CPU vs exhw: max |eps diff| = " << std::scientific
              << std::setprecision(3) << dmax << std::endl;
    EXPECTS(dmax < 1e-10);
  }

  // 1b) V from allocate_dense (a dense-kind tensor on the grid's ranks only), read back with
  // from_dense_tensor on the grid's ranks.
  {
    Tensor<double> Vd = grid.allocate_dense<double>(grid.index_space(N), grid.index_space(N));
    Tensor<double> Vg{tN, tN};
    Tensor<double>::allocate(&ec, Vg);
    std::vector<double> eps_d;
    eigensolve(ec, A, Vd, eps_d);
    if(grid.participates()) {
      tamm::from_dense_tensor(Vd, Vg);
      Tensor<double>::deallocate(Vd);
    }
    ec.pg().barrier();
    if(root) {
      const RMatrix         Vt  = tamm_to_eigen_matrix(Vg).transpose();
      const Eigen::VectorXd lam = Eigen::Map<const Eigen::VectorXd>(eps_d.data(), N);
      const double          res = (S * Vt - Vt * lam.asDiagonal()).norm() / S.norm();
      std::cout << "eigensolve (distributed, allocate_dense V) N=" << N
                << " : ||SV-VL||/||S||=" << std::scientific << std::setprecision(3) << res
                << std::endl;
      EXPECTS(res < 1e-10);
    }
    Tensor<double>::deallocate(Vg);
  }

  // 2) generalized_eigensolve with a random rectangular basis X (N x M) vs the host overload.
  std::mt19937                           gen(11);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  RMatrix                                Xm(N, M);
  for(int64_t i = 0; i < N; i++)
    for(int64_t j = 0; j < M; j++) Xm(i, j) = dist(gen);

  Tensor<double> X{tN, tM}, Cg{tN, tM};
  Tensor<double>::allocate(&ec, X, Cg);
  if(root) eigen_to_tamm_tensor(X, Xm);
  ec.pg().barrier();

  Tensor<double> X_bc = grid.allocate<double>(grid.index_space(N), grid.index_space(M));
  Tensor<double> C_bc = grid.allocate<double>(grid.index_space(N), grid.index_space(M));
  grid.to_block_cyclic(X, X_bc);
  ec.pg().barrier();

  // The checked solve uses ec.exhw(); a second solve with hw = CPU is compared on eigenvalues.
  // Each timed call ends with a barrier, since generalized_eigensolve has none of its own.
  std::vector<double> eps_p, eps_pc;
  const double        elapsed_p_s = timed_s([&]() {
    generalized_eigensolve(ec, A, X_bc, C_bc, eps_p, ec.exhw());
    ec.pg().barrier();
  });
  grid.from_block_cyclic(C_bc, Cg);
  ec.pg().barrier();
  double elapsed_pc_s = 0.0;
  {
    Tensor<double> C_bc_cpu = grid.allocate<double>(grid.index_space(N), grid.index_space(M));
    elapsed_pc_s            = timed_s([&]() {
      generalized_eigensolve(ec, A, X_bc, C_bc_cpu, eps_pc, ExecutionHW::CPU);
      ec.pg().barrier();
    });
    if(grid.participates()) Tensor<double>::deallocate(C_bc_cpu);
  }
  ec.pg().barrier();

  // 3) to_block_cyclic/from_block_cyclic round trip of X, and from_block_cyclic_dense of C.
  Tensor<double> Xg{tN, tM};
  Tensor<double>::allocate(&ec, Xg);
  grid.from_block_cyclic(X_bc, Xg);
  Tensor<double> C_dense = grid.from_block_cyclic_dense(C_bc);
  ec.pg().barrier();

  if(root) {
    RMatrix             Fh = S, Ch(N, M);
    std::vector<double> eps_h;
    generalized_eigensolve(N, M, Fh.data(), Xm.data(), Ch.data(), eps_h);

    // C^T S C = diag(eps) for C = X C' with orthonormal C' (independent of eigenvector signs).
    const RMatrix         Cm  = tamm_to_eigen_matrix(Cg);
    const Eigen::VectorXd lam = Eigen::Map<const Eigen::VectorXd>(eps_p.data(), M);
    const double res  = (Cm.transpose() * S * Cm - RMatrix(lam.asDiagonal())).norm() / S.norm();
    const double resh = (Ch.transpose() * S * Ch -
                         RMatrix(Eigen::Map<const Eigen::VectorXd>(eps_h.data(), M).asDiagonal()))
                          .norm() /
                        S.norm();
    double dmax = 0.0;
    for(int64_t i = 0; i < M; i++) dmax = std::max(dmax, std::abs(eps_p[i] - eps_h[i]));
    const double xdiff = (tamm_to_eigen_matrix(Xg) - Xm).norm();
    double       dhw   = 0.0;
    for(int64_t i = 0; i < M; i++) dhw = std::max(dhw, std::abs(eps_p[i] - eps_pc[i]));
    std::cout << "generalized_eigensolve [" << exhw_label << "] N=" << N << " M=" << M
              << " : ||C^TSC-L||/||S|| distributed=" << std::scientific << std::setprecision(3)
              << res << " host=" << resh << " max|eps_dist-eps_host|=" << dmax
              << " ||from_block_cyclic(to_block_cyclic(X))-X||=" << xdiff
              << " time=" << std::defaultfloat << std::setprecision(3) << elapsed_p_s << " s"
              << std::endl;
    std::cout << "generalized_eigensolve CPU vs exhw: max |eps diff| = " << std::scientific
              << std::setprecision(3) << dhw << std::defaultfloat << " time[" << exhw_label
              << "]=" << elapsed_p_s << " s time[CPU]=" << elapsed_pc_s << " s" << std::endl;
    EXPECTS(std::is_sorted(eps_p.begin(), eps_p.end()));
    EXPECTS(res < 1e-8 && resh < 1e-8 && dmax < 1e-10 && xdiff == 0.0 && dhw < 1e-10);
  }

  // from_block_cyclic_dense gives a TAMM-dense tensor usable with tensor_block
  if(grid.participates()) {
    Tensor<double> C_first = tensor_block(C_dense, {0, 0}, {N, 3});
    Tensor<double>::deallocate(C_first, C_dense);
  }

  // 4) caller-owned grid tensors are freed before the grid is released; then re-create it.
  if(grid.participates()) Tensor<double>::deallocate(X_bc, C_bc);
  Tensor<double>::deallocate(A, V, X, Cg, Xg);
  release_scalapack_grid(ec);
  EXPECTS(!find_scalapack_grid(ec).participates());

  const ScalapackGridHints auto_hints{.N = N}; // automatic sizing
  const ScalapackGrid&     grid2 = scalapack_grid(ec, auto_hints);
  if(root)
    std::cout << "released and re-created the grid (participates on rank 0: "
              << grid2.participates() << ")" << std::endl;
  release_scalapack_grid(ec);

  // 5) max_nranks caps the grid: a large N asks for many ranks, max_nranks = 1 gives a 1x1 grid on
  // the group's first rank only.
  const ScalapackGridHints capped{.N = 10000, .max_nranks = 1};
  const ScalapackGrid&     grid3 = scalapack_grid(ec, capped);
  EXPECTS(grid3.participates() == root);
  if(root) std::cout << "max_nranks = 1: 1x1 grid on rank 0 only" << std::endl;
  release_scalapack_grid(ec);
}
// The sub-group recipe: a grid on a sub-group (the first half of the ranks), used for a generalized
// eigensolve and released automatically by the sub-group's destroy_coll(). The release runs the
// live-allocation check, so a clean destroy_coll() also shows the grid tensors were freed first.
void test_subgroup_grid(ExecutionContext& ec, int64_t N, int64_t M) {
  const bool root   = ec.pg().rank() == 0;
  const int  nranks = std::max(1, static_cast<int>(ec.pg().size().value() / 2));
  EXPECTS(!find_scalapack_grid(ec).participates());

  const RMatrix                          S = random_symmetric(N, 21);
  std::mt19937                           gen(23);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  RMatrix                                Xm(N, M);
  for(int64_t i = 0; i < N; i++)
    for(int64_t j = 0; j < M; j++) Xm(i, j) = dist(gen);

  ProcGroup  sub_pg    = ProcGroup::create_subgroup(ec.pg(), nranks);
  const bool in_subgrp = sub_pg.is_valid();
  if(in_subgrp) {
    ExecutionContext sub_ec{sub_pg, DistributionKind::nw, MemoryManagerKind::ga};
    const bool       sub_root = sub_pg.rank() == 0;

    TiledIndexSpace tN{IndexSpace{range(N)}, 17};
    TiledIndexSpace tM{IndexSpace{range(M)}, 17};
    Tensor<double>  F{tN, tN}, X{tN, tM}, C{tN, tM};
    Tensor<double>::allocate(&sub_ec, F, X, C);
    if(sub_root) {
      RMatrix Sc = S, Xc = Xm;
      eigen_to_tamm_tensor(F, Sc);
      eigen_to_tamm_tensor(X, Xc);
    }
    sub_pg.barrier();

    const ScalapackGridHints   hints{.N = N, .nb = 128};
    const tamm::ScalapackGrid& grid = tamm::scalapack_grid(sub_ec, hints);
    Tensor<double> X_bc = grid.allocate<double>(grid.index_space(N), grid.index_space(M));
    Tensor<double> C_bc = grid.allocate<double>(grid.index_space(N), grid.index_space(M));
    grid.to_block_cyclic(X, X_bc);
    sub_pg.barrier();

    std::vector<double> eps;
    const double        elapsed_s = timed_s([&]() {
      tamm::generalized_eigensolve(sub_ec, F, X_bc, C_bc, eps, sub_ec.exhw());
      sub_pg.barrier();
    });
    grid.from_block_cyclic(C_bc, C);
    sub_pg.barrier();

    if(sub_root) {
      RMatrix             Fh = S, Ch(N, M);
      std::vector<double> eps_h;
      generalized_eigensolve(N, M, Fh.data(), Xm.data(), Ch.data(), eps_h);
      const RMatrix Cm   = tamm_to_eigen_matrix(C);
      const RMatrix L    = RMatrix(Eigen::Map<const Eigen::VectorXd>(eps.data(), M).asDiagonal());
      const double  res  = (Cm.transpose() * S * Cm - L).norm() / S.norm();
      double        dmax = 0.0;
      for(int64_t i = 0; i < M; i++) dmax = std::max(dmax, std::abs(eps[i] - eps_h[i]));
      std::cout << "sub-group grid (" << nranks << " of " << ec.pg().size().value()
                << " ranks): generalized_eigensolve N=" << N << " M=" << M
                << " : ||C^TSC-L||/||S||=" << std::scientific << std::setprecision(3) << res
                << " max|eps-eps_local|=" << dmax << " time=" << std::defaultfloat
                << std::setprecision(3) << elapsed_s << " s" << std::endl;
      EXPECTS(res < 1e-8 && dmax < 1e-10);
    }

    if(grid.participates()) Tensor<double>::deallocate(X_bc, C_bc);
    Tensor<double>::deallocate(F, X, C);
    sub_ec.flush_and_sync();
    sub_pg.destroy_coll(); // also releases the grid
    if(sub_root) std::cout << "sub-group destroyed; its grid was released" << std::endl;
  }
  ec.pg().barrier();
  EXPECTS(!find_scalapack_grid(ec).participates());
  if(root) EXPECTS(in_subgrp); // the sub-group is the group's first ranks
}
#endif

int main(int argc, char* argv[]) {
  // Let small test matrices reach the GPU solver in GPU builds (unless the caller set it already).
  setenv("TAMM_GPU_EIGENSOLVE_MIN_N", "0", 0);

  tamm::initialize(argc, argv);

  // usage: Test_EVP [N [M [N_local]]]  -- distributed N x N solve with an N x M basis (ScaLAPACK
  // builds; >= 4 usable ranks for the 2x2 grid), and a local N_local x N_local solve.
  [[maybe_unused]] int64_t N = 100, M = 80;
  int64_t                  N_local = 200;
  if(argc >= 2) N = std::atoi(argv[1]);
  if(argc >= 3) M = std::atoi(argv[2]);
  if(argc >= 4) N_local = std::atoi(argv[3]);

  {
    ProcGroup        pg = ProcGroup::create_world_coll();
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

    if(ec.print()) {
      std::cout << tamm_git_info() << std::endl;

      std::cout << std::endl;
      ec.print_execution_environment();
      std::cout << std::endl << tamm_build_config();
      std::cout << std::endl;
      std::cout << "N, M, N_local = " << N << ", " << M << ", " << N_local << std::endl
                << std::endl;
    }

    test_local_eigensolve(ec, N_local);
    ec.pg().barrier();
    test_local_generalized(ec, N_local, (3 * N_local) / 4);
    ec.pg().barrier();
    test_tensor_eigensolve(ec, N);
    ec.pg().barrier();
#if defined(USE_SCALAPACK)
    test_distributed(ec, N, M);
    ec.pg().barrier();
    test_subgroup_grid(ec, N, M);
#else
    test_empty_grid(ec);
#endif
    ec.pg().barrier();
  }

  tamm::finalize();
  return 0;
}
