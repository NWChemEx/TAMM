#include "tamm/eigen_utils.hpp"
#include "tamm/scalapack_grid_impl.hpp"
#include "tamm/tamm.hpp"

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#include "tamm/kernels/gpu_lapack_internal.hpp"
#endif

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace tamm {

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
namespace detail {

// Parse a GPU size-threshold env var (a non-negative integer). Reported once on world rank 0 when
// set, so a run's output shows that the default was overridden.
static int64_t parse_gpu_min_n(const char* name, int64_t default_value) {
  const char* raw = std::getenv(name);
  if(raw == nullptr) { return default_value; }

  char*     end = nullptr;
  long long val = std::strtoll(raw, &end, 10);
  if(end == raw || *end != '\0' || val < 0) {
    std::ostringstream os;
    os << "[TAMM ERROR] " << name << " must be a non-negative integer; got \"" << raw << "\".\n"
       << __FILE__ << ":L" << __LINE__;
    tamm_terminate(os.str());
  }
  if(ProcGroup::world_rank().value() == 0)
    std::cout << "[TAMM] " << name << " = " << val << " (default " << default_value << ")"
              << std::endl;
  return static_cast<int64_t>(val);
}

// TAMM_GPU_EIGENSOLVE_MIN_N = 1000 (default): smallest N for which
// the local eigensolve(..., ExecutionHW::GPU) runs on the GPU. Read on first use.
static int64_t gpu_eigensolve_min_n() {
  static const int64_t min_n = parse_gpu_min_n("TAMM_GPU_EIGENSOLVE_MIN_N", 1000);
  return min_n;
}

} // namespace detail
#endif

template<typename T>
void eigensolve(int64_t N, T* A, std::vector<blas::real_type<T>>& eps,
                [[maybe_unused]] ExecutionHW hw) {
  eps.resize(N);

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  if(hw == ExecutionHW::GPU && N >= detail::gpu_eigensolve_min_n()) {
    kernels::gpu::syevd(N, A, N, eps.data());
    return;
  }
#endif

  const int64_t info = lapack::syevd(lapack::Job::Vec, lapack::Uplo::Lower, N, A, N, eps.data());
  if(info != 0)
    tamm_terminate("[TAMM ERROR] LAPACK syevd failed with info = " + std::to_string(info));
}

template void eigensolve<double>(int64_t N, double* A, std::vector<double>& eps, ExecutionHW hw);

template<typename T>
void eigensolve(ExecutionContext& ec, const Tensor<T>& A, Tensor<T>& V,
                std::vector<blas::real_type<T>>& eps, ExecutionHW hw) {
  const auto tis = A.tiled_index_spaces();
  EXPECTS(tis.size() == 2);
  const int64_t N = static_cast<int64_t>(tis[0].index_space().num_indices());
  EXPECTS(static_cast<int64_t>(tis[1].index_space().num_indices()) == N);

  eps.resize(N);

#if defined(USE_SCALAPACK)
  ScalapackGrid::Impl& g = scalapack_grid(ec, ScalapackGridHints{N}).impl();
  if(g.participates()) {
    TiledIndexSpace tN_bc{IndexSpace{range(N)}, static_cast<Tile>(g.mb)};
    Tensor<T>       A_BC{tN_bc, tN_bc};
    Tensor<T>       V_BC{tN_bc, tN_bc};
    A_BC.set_block_cyclic({g.npr, g.npc});
    V_BC.set_block_cyclic({g.npr, g.npc});
    Tensor<T>::allocate(&g.ec, A_BC, V_BC);

    tamm::to_block_cyclic_tensor(A, A_BC);
    g.pg.barrier();

    detail::distributed_eigensolve(g, N, A_BC.access_local_buf(), V_BC.access_local_buf(), eps, hw);
    g.pg.barrier();

    tamm::from_block_cyclic_tensor(V_BC, V);
    Tensor<T>::deallocate(A_BC, V_BC);
  }
#else
  if(ec.pg().rank() == 0) {
    auto Vm = tamm_to_eigen_matrix(A);
    eigensolve(N, Vm.data(), eps, hw);
    eigen_to_tamm_tensor(V, Vm);
  }
#endif
  ec.pg().barrier();
}

#if defined(USE_SCALAPACK)
template<typename T>
void generalized_eigensolve(ExecutionContext& ec, const Tensor<T>& F, const Tensor<T>& X,
                            Tensor<T>& C, std::vector<blas::real_type<T>>& eps, ExecutionHW hw) {
  const ScalapackGrid& grid = find_scalapack_grid(ec);
  if(!grid.participates()) return;
  ScalapackGrid::Impl& g = grid.impl();

  const blacspp::Grid& blacs = *g.blacs_grid;
  if(blacs.ipr() < 0 || blacs.ipc() < 0) return;

  const auto    tis_F = F.tiled_index_spaces();
  const auto    tis_X = X.tiled_index_spaces();
  const int64_t N     = static_cast<int64_t>(tis_F[0].index_space().num_indices());
  const int64_t M     = static_cast<int64_t>(tis_X[1].index_space().num_indices());
  const int64_t mb    = g.mb;

  // TAMM's block-cyclic buffers are row-major, so ScaLAPACK (column-major) sees an N x M tensor
  // as its M x N transpose: X is addressed through an M x N descriptor, and C = X C' is written
  // as (X C')^T with the same descriptor.
  scalapackpp::BlockCyclicMatrix<T> Fp_sca(blacs, M, M, mb, mb), Cp_sca(blacs, M, M, mb, mb),
    TMP_sca(blacs, N, M, mb, mb);
  const auto desc_F = g.descriptor(N, N);
  const auto desc_X = g.descriptor(M, N);

  Tensor<T>& F_BC = g.scratch(N);
  tamm::to_block_cyclic_tensor(F, F_BC);
  g.pg.barrier();

  const T* F_lptr = F_BC.access_local_buf();
  const T* X_lptr = X.access_local_buf();
  T*       C_lptr = C.access_local_buf();

  // TMP = F * X  (F * X**T in ScaLAPACK's view of the row-major X)
  scalapackpp::pgemm(scalapackpp::Op::NoTrans, scalapackpp::Op::Trans, TMP_sca.m(), TMP_sca.n(),
                     desc_F[3], 1., F_lptr, 1, 1, desc_F, X_lptr, 1, 1, desc_X, 0., TMP_sca.data(),
                     1, 1, TMP_sca.desc());

  // Fp = X^T * TMP  (X * TMP in ScaLAPACK's view)
  scalapackpp::pgemm(scalapackpp::Op::NoTrans, scalapackpp::Op::NoTrans, Fp_sca.m(), Fp_sca.n(),
                     desc_X[3], 1., X_lptr, 1, 1, desc_X, TMP_sca.data(), 1, 1, TMP_sca.desc(), 0.,
                     Fp_sca.data(), 1, 1, Fp_sca.desc());

  detail::distributed_eigensolve(g, M, Fp_sca.data(), Cp_sca.data(), eps, hw);

  // C = X * Cp  ->  C**T = Cp**T * X  (row-major C)
  scalapackpp::pgemm(scalapackpp::Op::Trans, scalapackpp::Op::NoTrans, desc_X[2], desc_X[3],
                     Cp_sca.m(), 1., Cp_sca.data(), 1, 1, Cp_sca.desc(), X_lptr, 1, 1, desc_X, 0.,
                     C_lptr, 1, 1, desc_X);
}
#endif

template<typename T>
void generalized_eigensolve(int64_t N, int64_t M, T* F, const T* X, T* C,
                            std::vector<blas::real_type<T>>& eps, ExecutionHW hw) {
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  // With the GPU requested, the gemms and the inner eigensolve all run on the device. The inner
  // eigensolve is M x M, so M is compared with the GPU threshold, as in the host path below.
  if(hw == ExecutionHW::GPU && M >= detail::gpu_eigensolve_min_n()) {
    eps.resize(M);
    kernels::gpu::generalized_eigensolve(N, M, F, X, C, eps.data());
    return;
  }
#endif

  // Column-major BLAS on row-major buffers: the row-major N x N F, N x M X and N x M C are seen
  // as F, X^T (M x N) and C^T (M x N).
  // C (as N x M column-major scratch) = F * X
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans, N, M, N, 1., F, N, X, M,
             0., C, N);
  // F (as M x M) = X^T * F * X
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, M, M, N, 1., X, M, C, N,
             0., F, M);
  eigensolve(M, F, eps, hw);
  // C (row-major N x M) = X * C'
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, M, N, M, 1., F, M, X, M,
             0., C, M);
}

template void eigensolve<double>(ExecutionContext& ec, const Tensor<double>& A, Tensor<double>& V,
                                 std::vector<double>& eps, ExecutionHW hw);
#if defined(USE_SCALAPACK)
template void generalized_eigensolve<double>(ExecutionContext& ec, const Tensor<double>& F,
                                             const Tensor<double>& X, Tensor<double>& C,
                                             std::vector<double>& eps, ExecutionHW hw);
#endif
template void generalized_eigensolve<double>(int64_t N, int64_t M, double* F, const double* X,
                                             double* C, std::vector<double>& eps, ExecutionHW hw);

} // namespace tamm
