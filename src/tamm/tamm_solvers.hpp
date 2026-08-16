#pragma once

#include "eigen_includes.hpp"

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#include "tamm/kernels/tamm_blas.hpp"
#endif

namespace tamm {

/*-----------------SOLVERS-----------------*/
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
