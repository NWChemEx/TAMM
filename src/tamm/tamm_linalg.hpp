#pragma once

#include "eigen_includes.hpp"

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#include "tamm/kernels/tamm_blas.hpp"
#endif

namespace tamm {

/**
 * @brief Rank-local standard eigensolve A v_i = eps_i v_i of an N x N symmetric matrix.
 *
 * Solves the standard symmetric eigenvalue problem for all N eigenpairs of a matrix held entirely
 * by the calling rank. Rank-local: not collective and no communication; each calling rank solves
 * its own matrix independently. Terminates on a backend error.
 *
 * @param N          Size of A.
 * @param[in,out] A  Symmetric N x N matrix, host buffer with leading dimension N. Overwritten with
 *                   the orthonormal eigenvectors: row i = eigenvector i when viewed row-major (as
 *                   TAMM and Eigen's RowMajor matrices are), column i when viewed column-major.
 * @param[out] eps   Eigenvalues in ascending order, resized to N.
 * @param hw         Where the solve runs:
 *                     - ExecutionHW::CPU (default): LAPACK syevd on the host.
 *                     - ExecutionHW::GPU: cuSolver/rocSolver/oneMKL syevd on the calling rank's
 *                       GPU in GPU builds when N is at least the GPU threshold (1000, or the
 *                       environment variable TAMM_GPU_EIGENSOLVE_MIN_N); LAPACK otherwise.
 *
 * Instantiated for T = double.
 */
template<typename T>
void eigensolve(int64_t N, T* A, std::vector<blas::real_type<T>>& eps,
                ExecutionHW hw = ExecutionHW::CPU);

/**
 * @brief Standard eigensolve A v_i = eps_i v_i of a symmetric N x N tensor.
 *
 * In builds with ScaLAPACK the solve is distributed (ELPA when built with it) on the process
 * group's ScaLAPACK grid, which is created if it does not exist yet (see scalapack_grid());
 * otherwise it runs on rank 0 of ec. Collective over ec.
 *
 * @param ec       Execution context; every rank of its process group must make the call.
 * @param A        Symmetric N x N regular (not block-cyclic) tensor; not modified.
 * @param[out] V   Caller-allocated N x N tensor (regular or dense-kind), same shape as A; row i is
 *                 eigenvector i (orthonormal). With ScaLAPACK only the grid's ranks write V, so
 *                 V may live on the grid's ranks only.
 * @param[out] eps Eigenvalues in ascending order, resized to N; valid on the grid's ranks in
 *                 builds with ScaLAPACK or ELPA, on rank 0 otherwise.
 * @param hw       With ScaLAPACK: ELPA on the GPU in CUDA builds when hw == GPU, the CPU
 *                 otherwise (ScaLAPACK is CPU-only). Without ScaLAPACK: passed to the local
 *                 eigensolve on rank 0.
 *
 * Instantiated for T = double.
 */
template<typename T>
void eigensolve(ExecutionContext& ec, const Tensor<T>& A, Tensor<T>& V,
                std::vector<blas::real_type<T>>& eps, ExecutionHW hw = ExecutionHW::CPU);

#if defined(USE_SCALAPACK)
/**
 * @brief Generalized eigensolve F C = S C eps, given the canonical orthogonalizer X of S.
 *
 * Solves the generalized eigenvalue problem by transforming it into a standard eigenvalue problem
 * using canonical orthogonalization, followed by back-transformation of the resulting eigenvectors
 * to the original non-orthogonal basis: (X^T F X) C' = C' eps, C = X C'.
 *
 * This overload is distributed and is declared only in builds with ScaLAPACK or ELPA (there is no
 * rank-0 fallback as for the distributed eigensolve; other builds use the local overload). X and C
 * are block-cyclic tensors on the ScaLAPACK grid of ec's process group (e.g. from
 * ScalapackGrid::allocate with index_space(N) x index_space(M)); the grid must exist. A
 * block-cyclic copy of F is cached on the grid and reused across calls. Collective over the grid's
 * ranks; a no-op elsewhere.
 *
 * @param ec       Execution context whose process group owns the ScaLAPACK grid.
 * @param F        Symmetric N x N matrix (regular tensor).
 * @param X        N x M canonical orthogonalizer of S (X^T S X = I, M <= N), built once by the
 *                 caller, e.g. from an eigensolve of S; M < N when near-linear dependencies are
 *                 removed.
 * @param[out] C   N x M eigenvectors in the original basis (C^T S C = I); column j is the
 *                 eigenvector for eps[j].
 * @param[out] eps Eigenvalues in ascending order, resized to M; valid on the grid's ranks.
 * @param hw       Hardware for the inner eigensolve, as for the distributed eigensolve.
 *
 * Instantiated for T = double.
 */
template<typename T>
void generalized_eigensolve(ExecutionContext& ec, const Tensor<T>& F, const Tensor<T>& X,
                            Tensor<T>& C, std::vector<blas::real_type<T>>& eps,
                            ExecutionHW hw = ExecutionHW::CPU);
#endif

/**
 * @brief Rank-local generalized eigensolve on host buffers held by the calling rank.
 *
 * The same problem as the distributed overload, rank-local in the same way as the local eigensolve.
 *
 * @param N          Size of F.
 * @param M          Number of columns of X and C (M <= N).
 * @param[in,out] F  Symmetric N x N matrix, row-major; overwritten (used as workspace).
 * @param X          N x M canonical orthogonalizer of S, row-major.
 * @param[out] C     N x M eigenvectors in the original basis, row-major; receives X C'.
 * @param[out] eps   Eigenvalues in ascending order, resized to M.
 * @param hw         Where the solve runs: with ExecutionHW::GPU in GPU builds and M at least the
 *                   GPU threshold (as for the local eigensolve), the gemms and the inner
 *                   eigensolve all run on the GPU; otherwise on the host.
 *
 * Instantiated for T = double.
 */
template<typename T>
void generalized_eigensolve(int64_t N, int64_t M, T* F, const T* X, T* C,
                            std::vector<blas::real_type<T>>& eps,
                            ExecutionHW                      hw = ExecutionHW::CPU);

/**
 * @brief Options for tamm::svd
 */
struct SVDOptions {
  bool full_matrices = true;
  bool compute_uv    = true;
  // bool   hermitian     = false;
};

/**
 * @brief Singular Value Decomposition of a 2D tensor
 *
 * Computes A = U * diag(S) * Vh by gathering A onto rank 0 as a dense Eigen matrix and
 * solving it there. execute_on selects the solver:
 *   - ExecutionHW::CPU (default) : LAPACK's gesvd on the host, any M/N.
 *   - ExecutionHW::GPU           : cuSolver's/rocSolver's/oneMKL's gesvd on rank 0's GPU
 *                                  (tamm::kernels::gpu::gesvd), only available when built with
 *                                  USE_CUDA/USE_HIP/USE_DPCPP (else ignored, LAPACK/CPU is
 *                                  used). tamm::svd requires M >= N for the GPU path uniformly
 *                                  across backends (cusolverDn/rocsolver <t>gesvd's native
 *                                  constraint; not a hard requirement of oneMKL, but applied
 *                                  the same way there for consistency) and transparently falls
 *                                  back to LAPACK/CPU when M < N.
 *
 * Options are:
 *   - full_matrices=true  : U is M x M, Vh is N x N.
 *   - full_matrices=false : U is M x K, Vh is K x N, K = min(M, N) (reduced).
 *   - compute_uv=false    : only S is computed; U and Vh are returned empty.
 * S holds the singular values in non-increasing order, as LAPACK returns them.
 *
 * @return tuple (U, S, Vh); S is a std::vector of the (real) singular values. S is always
 * real-valued (via blas::real_type<T>) even when T is complex, matching LAPACK's gesvd.
 */
template<typename T>
std::tuple<Tensor<T>, std::vector<blas::real_type<T>>, Tensor<T>>
svd(ExecutionContext& ec, Tensor<T> A, SVDOptions opts = {},
    ExecutionHW execute_on = ExecutionHW::CPU) {
  // LAPACK expects column-major; hold the gathered matrix column-major so its buffer is
  // directly consumable with leading dimension = number of rows.
  using CMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using RMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using real_t  = blas::real_type<T>;

  const auto rank = ec.pg().rank().value();
  const auto tis  = A.tiled_index_spaces();
  EXPECTS(tis.size() == 2);
  const TiledIndexSpace MR = tis[0]; // row space
  const TiledIndexSpace NC = tis[1]; // column space
  const int64_t         M  = static_cast<int64_t>(MR.max_num_indices());
  const int64_t         N  = static_cast<int64_t>(NC.max_num_indices());
  const int64_t         K  = std::min(M, N);

  // Gather A onto every rank. tamm_to_eigen_matrix returns row-major; assigning into a
  // column-major matrix transposes the storage order (not the logical matrix).
  CMatrix             Am;
  std::vector<real_t> sigma(static_cast<size_t>(K), real_t{0});

  if(rank == 0) { Am = tamm_to_eigen_matrix(A); }

  if(!opts.compute_uv) {
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
    if(execute_on == ExecutionHW::GPU && M >= N) {
      tamm::kernels::gpu::gesvd<T>(lapack::Job::NoVec, lapack::Job::NoVec, M, N, Am.data(), M,
                                   sigma.data(), nullptr, 1, nullptr, 1);
    }
    else {
      lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec, M, N, Am.data(), M, sigma.data(),
                    nullptr, 1, nullptr, 1);
    }
#else
    lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec, M, N, Am.data(), M, sigma.data(), nullptr,
                  1, nullptr, 1);
#endif
    ec.pg().broadcast(sigma.data(), sigma.size(), 0);
    return {Tensor<T>{}, sigma, Tensor<T>{}};
  }

  const bool        full  = opts.full_matrices;
  const int64_t     ucols = full ? M : K; // U is M x ucols
  const int64_t     vrows = full ? N : K; // Vh is vrows x N
  const lapack::Job job   = full ? lapack::Job::AllVec : lapack::Job::SomeVec;

  // Output index spaces: U is M x ucols, Vh is vrows x N.
  const TiledIndexSpace KS{IndexSpace{range(static_cast<size_t>(K))}, static_cast<Tile>(K)};
  const TiledIndexSpace UC = full ? MR : KS;
  const TiledIndexSpace VR = full ? NC : KS;

  Tensor<T> U{MR, UC};
  Tensor<T> Vh{VR, NC};
  Scheduler{ec}.allocate(U, Vh).execute();

  if(rank == 0) {
    CMatrix Um(M, ucols);  // ldu  = M
    CMatrix VTm(vrows, N); // ldvt = vrows
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
    // GPU path requires m >= n (see svd()'s doc comment); fall back to LAPACK/CPU otherwise
    if(execute_on == ExecutionHW::GPU && M >= N) {
      tamm::kernels::gpu::gesvd(job, job, M, N, Am.data(), M, sigma.data(), Um.data(), M,
                                VTm.data(), vrows);
    }
    else {
      lapack::gesvd(job, job, M, N, Am.data(), M, sigma.data(), Um.data(), M, VTm.data(), vrows);
    }
#else
    lapack::gesvd(job, job, M, N, Am.data(), M, sigma.data(), Um.data(), M, VTm.data(), vrows);
#endif
    Am.resize(0, 0);

    // Convert the column-major LAPACK factors back to row-major for eigen_to_tamm_tensor.
    {
      RMatrix Ur = Um;
      eigen_to_tamm_tensor(U, Ur);
    }

    RMatrix Vhr = VTm;
    eigen_to_tamm_tensor(Vh, Vhr);
  }
  ec.pg().broadcast(sigma.data(), sigma.size(), 0);

  return {U, sigma, Vh};
}

} // namespace tamm
