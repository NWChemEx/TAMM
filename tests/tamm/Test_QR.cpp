#include <tamm/eigen_utils.hpp>
#include <tamm/tamm.hpp>
#include <tamm/tamm_config.hpp>
#include <tamm/tamm_git.hpp>

#include <iomanip>

using namespace tamm;

// Reconstruction test for tamm::qr.
// Builds a random distributed A (M x N), computes A = Q R, and checks
// || A - Q R ||_F / ||A||_F is small along with orthonormality of Q and the upper-triangularity
// of R.
template<typename T>
void test_qr(ExecutionContext& ec, size_t M, size_t N, Tile tilesize, const std::string& label) {
  // exhw=GPU when built with USE_CUDA/HIP/DPCPP, else CPU
  Scheduler   sch{ec};
  ExecutionHW exhw = ec.exhw();

  TiledIndexSpace MR{IndexSpace{range(M)}, tilesize};
  TiledIndexSpace NC{IndexSpace{range(N)}, tilesize};

  Tensor<T> A{MR, NC};
  sch.allocate(A).execute();
  random_ip(A);

  QROptions opts;
  opts.full_matrices                    = true;
  auto t1                               = std::chrono::high_resolution_clock::now();
  auto [Q, R]                           = qr(ec, A, opts, exhw);
  auto                          t2      = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = t2 - t1;
  if(ec.print()) {
    const auto old_flags = std::cout.flags();
    const auto old_prec  = std::cout.precision();
    std::cout << "QR elapsed time: " << std::fixed << std::setprecision(2) << elapsed.count()
              << " seconds" << std::endl;
    std::cout.flags(old_flags);
    std::cout.precision(old_prec);
  }

  // Verify from the ACTUAL returned factor shapes, so this works for both full_matrices
  // (Q: M x M, R: M x N) and reduced (Q: M x K, R: K x N), and for M<N. The reconstruction
  // Qe*Re is M x N in every case. Q's columns are orthonormal in every case.
  using EM              = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  EM                 Ae = tamm_to_eigen_matrix(A); // M x N
  EM                 Qe = tamm_to_eigen_matrix(Q); // M x qc
  EM                 Re = tamm_to_eigen_matrix(R); // rr x N
  const Eigen::Index qc = Qe.cols();

  // adjoint() (conjugate transpose) rather than transpose(): Q is unitary for complex T,
  // orthogonal for real T, and adjoint() reduces to transpose() in the real case.
  const double rel = (Qe * Re - Ae).norm() / (Ae.norm() > 0 ? Ae.norm() : 1.0);
  const double qtq = (Qe.adjoint() * Qe - EM::Identity(qc, qc)).norm();
  // R's strictly-lower part should be exactly zero (upper triangular / upper trapezoidal).
  const double lower = Re.template triangularView<Eigen::StrictlyLower>().toDenseMatrix().norm();

  const EM     A_reconstructed = Qe * Re;
  const double err             = (Ae - A_reconstructed).norm();

  if(ec.print()) {
    std::cout << "QR [" << label << "] M=" << M << " N=" << N << " tile=" << tilesize
              << " : ||A-QR||/||A||=" << rel << " ||Q^HQ-I||=" << qtq
              << " ||strictly-lower(R)||=" << lower << std::endl
              << "Q Dims=" << Qe.rows() << "x" << Qe.cols() << "\nR Dims=" << Re.rows() << "x"
              << Re.cols() << std::endl
              << "Reconstruction error: " << std::scientific << std::setprecision(3) << err
              << std::endl;
  }

  EXPECTS(rel < 1e-8);
  EXPECTS(qtq < 1e-8);
  EXPECTS(lower < 1e-8);

  sch.deallocate(A, Q, R).execute();
}

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  size_t M = 100, N = 20;
  Tile   tile = 50;
  if(argc >= 3) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
  }
  if(argc >= 4) tile = std::atoi(argv[3]);

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  if(ec.print()) {
    std::cout << tamm_git_info() << std::endl;

    std::cout << std::endl;
    ec.print_execution_environment();
    std::cout << std::endl << tamm_build_config();
    std::cout << std::endl;
  }

  test_qr<double>(ec, M, N, tile, "double");
  test_qr<std::complex<double>>(ec, M, N, tile, "complex<double>");

  tamm::finalize();
  return 0;
}
