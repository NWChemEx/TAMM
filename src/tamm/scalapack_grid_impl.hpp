#pragma once

// Internal to TAMM's sources (not installed): the state behind a ScalapackGrid.

#include "tamm/scalapack_grid.hpp"

#include <functional>
#include <vector>

#if defined(USE_SCALAPACK)
#include <blacspp/grid.hpp>
#include <scalapackpp/block_cyclic_matrix.hpp>
#include <scalapackpp/eigenvalue_problem/sevp.hpp>
#include <scalapackpp/pblas/gemm.hpp>
#endif

namespace tamm {

struct ScalapackGrid::Impl {
#if defined(USE_SCALAPACK)
  ScalapackGridHints hints; // as requested at creation (for the mismatch warning)
  bool               warned{false};
  int64_t            npr{1}, npc{1}, nranks{1};
  int64_t            mb{1}; // effective block size, valid on every rank of the parent group
  ProcGroup          pg;    // the grid's sub-group (valid on participating ranks only)
  ExecutionContext   ec;    // dense GA context on pg
  std::unique_ptr<blacspp::Grid>                  blacs_grid;
  std::unique_ptr<scalapackpp::BlockCyclicDist2D> dist;

  // Tensors handed to the caller (for the live-allocation check at release).
  std::vector<std::function<bool()>> caller_tensors;
  // generalized_eigensolve's cached block-cyclic copy of F (N x N), owned by the grid.
  Tensor<double> f_scratch;
  int64_t        f_scratch_n{0};

  bool participates() const { return pg.is_valid(); }

  // ScaLAPACK descriptor of an M x N matrix laid out per dist (leading dimension = local rows).
  scalapackpp::scalapack_desc descriptor(int64_t M, int64_t N) const {
    auto [m_loc, n_loc] = dist->get_local_dims(M, N);
    return dist->descinit_noerror(M, N, m_loc);
  }

  // The cached N x N block-cyclic scratch, (re)allocated on the grid when N changes.
  Tensor<double>& scratch(int64_t N);

  // Free the grid (collective over its ranks). Terminates if caller tensors are still allocated,
  // unless check_live is false (then it warns and leaves the grid to GA/MPI shutdown).
  void release(bool check_live = true);
#endif
};

#if defined(USE_SCALAPACK)
namespace detail {
// Eigensolve of an N x N symmetric matrix whose local buffers are distributed per g.dist
// (ELPA when built with TAMM_USE_ELPA, ScaLAPACK otherwise). A is overwritten; V receives the
// eigenvectors (row i = eigenvector i in the row-major view); eps (resized to N) the eigenvalues
// in ascending order on every grid rank. No-op on ranks outside the grid. hw == GPU runs ELPA on
// the GPU in CUDA builds; every other case (ScaLAPACK, ELPA without CUDA, hw != GPU) runs on the
// CPU.
template<typename T>
void distributed_eigensolve(ScalapackGrid::Impl& g, int64_t N, T* A_local, T* V_local,
                            std::vector<T>& eps, ExecutionHW hw);
} // namespace detail
#endif

} // namespace tamm
