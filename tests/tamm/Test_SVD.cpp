#include <tamm/eigen_utils.hpp>
#include <tamm/tamm.hpp>
#include <tamm/tamm_git.hpp>

#include <iomanip>

using namespace tamm;

// Reconstruction test for tamm::svd.
// Builds a random distributed A (M x N, M>=N), computes A = U diag(S) Vh, and checks
// || A - U diag(S) Vh ||_F / ||A||_F is small along with orthonormality of U and Vh.
template<typename T>
void test_svd(ExecutionContext& ec, size_t M, size_t N, Tile tilesize, const std::string& label) {
  // exhw=GPU when built with USE_CUDA/HIP/DPCPP, else CPU
  Scheduler   sch{ec};
  ExecutionHW exhw = ec.exhw();

  // EXPECTS(M >= N);

  TiledIndexSpace MR{IndexSpace{range(M)}, tilesize};
  TiledIndexSpace NC{IndexSpace{range(N)}, tilesize};

  Tensor<T> A{MR, NC};
  sch.allocate(A).execute();
  random_ip(A);

  SVDOptions opts;
  opts.full_matrices                    = true;
  auto t1                               = std::chrono::high_resolution_clock::now();
  auto [U, S, Vh]                       = svd(ec, A, opts, exhw);
  auto                          t2      = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = t2 - t1;
  if(ec.print()) {
    std::cout << "SVD elapsed time: " << std::fixed << std::setprecision(2) << elapsed.count()
              << " seconds" << std::endl;
  }

  // Verify from the ACTUAL returned factor shapes, so this works for both full_matrices
  // (U: M x M, Vh: N x N) and reduced (U: M x K, Vh: K x N), and for M<N. Sigma is uc x vr with
  // the min(uc, vr) singular values on the diagonal; the reconstruction Ue*Sigma*Ve is then M x N
  // in every case. U's columns and Vh's rows are orthonormal in every case.
  using EM              = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  EM                 Ae = tamm_to_eigen_matrix(A);  // M x N
  EM                 Ue = tamm_to_eigen_matrix(U);  // M x uc
  EM                 Ve = tamm_to_eigen_matrix(Vh); // vr x N
  const Eigen::Index uc = Ue.cols();
  const Eigen::Index vr = Ve.rows();
  EM                 Sd = EM::Zero(uc, vr);
  const Eigen::Index Kd = std::min(uc, vr);
  for(Eigen::Index k = 0; k < Kd; k++) Sd(k, k) = S[k];

  // adjoint() (conjugate transpose) rather than transpose(): U/Vh are unitary for complex T,
  // orthogonal for real T, and adjoint() reduces to transpose() in the real case.
  const double rel = (Ue * Sd * Ve - Ae).norm() / (Ae.norm() > 0 ? Ae.norm() : 1.0);
  const double utu = (Ue.adjoint() * Ue - EM::Identity(uc, uc)).norm();
  const double vvt = (Ve * Ve.adjoint() - EM::Identity(vr, vr)).norm();

  const EM     A_reconstructed = Ue * Sd * Ve;
  const double err             = (Ae - A_reconstructed).norm();

  if(ec.print()) {
    std::cout << "SVD [" << label << "] M=" << M << " N=" << N << " tile=" << tilesize
              << " : ||A-USVh||/||A||=" << rel << " ||U^HU-I||=" << utu << " ||VhVh^H-I||=" << vvt
              << std::endl
              << "U Dims=" << Ue.rows() << "x" << Ue.cols() << "\nVt Dims=" << Ve.rows() << "x"
              << Ve.cols() << std::endl
              << "Reconstruction error: " << std::scientific << std::setprecision(3) << err
              << std::endl;
  }

  EXPECTS(rel < 1e-8);
  EXPECTS(utu < 1e-8);
  EXPECTS(vvt < 1e-8);

  sch.deallocate(A, U, Vh).execute();
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
    auto current_time   = std::chrono::system_clock::now();
    auto current_time_t = std::chrono::system_clock::to_time_t(current_time);
    auto cur_local_time = localtime(&current_time_t);
    std::cout << std::endl << "date: " << std::put_time(cur_local_time, "%c") << std::endl;

    std::cout << "nnodes: " << ec.nnodes() << ", ";
    std::cout << "nproc_per_node: " << ec.ppn() << ", ";
    std::cout << "nproc_total: " << ec.nnodes() * ec.ppn() << ", ";
    if(ec.has_gpu()) {
      std::cout << "ngpus_per_node: " << ec.gpn() << ", ";
      std::cout << "ngpus_total: " << ec.nnodes() * ec.gpn() << std::endl;
    }
    std::cout << std::endl;
    ec.print_mem_info();
    std::cout << std::endl;
  }

  test_svd<double>(ec, M, N, tile, "double");
  test_svd<std::complex<double>>(ec, M, N, tile, "complex<double>");

  tamm::finalize();
  return 0;
}
