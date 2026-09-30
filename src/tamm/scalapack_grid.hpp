#pragma once

#include "tamm/execution_context.hpp"
#include "tamm/tensor.hpp"
#include "tamm/tiled_index_space.hpp"

#include <cstdint>
#include <memory>

namespace tamm {

/**
 * @brief Sizing requests for a process group's ScaLAPACK grid. A value of 0 means "choose
 * automatically".
 */
struct ScalapackGridHints {
  int64_t N{0};   ///< matrix size the grid is sized for (ranks ~ 4% of N, block size <= N/npr)
  int     npr{0}; ///< requested process rows (with npc; rounded down to a square grid)
  int     npc{0}; ///< requested process columns
  int     nb{0};  ///< requested block size (default 256; reduced for small N)
  int     max_nranks{0}; ///< cap on the number of grid ranks (default: size of the process group)
};

/**
 * @brief The ScaLAPACK/BLACS process grid of a process group: a sub-group of the group's first
 * ranks laid out as a square grid, with a block-cyclic distribution.
 *
 * A process group has at most one grid. It is created by scalapack_grid() (or on first use by
 * tamm::eigensolve), found with find_scalapack_grid(), and released by release_scalapack_grid()
 * or automatically when its process group is destroyed (ProcGroup::destroy_coll()) or at
 * tamm::finalize(). In builds without ScaLAPACK the grid is always empty.
 *
 * Tensors obtained from allocate(), allocate_dense() and from_block_cyclic_dense() are owned by the
 * caller, who deallocates them (on participating ranks) before the grid is released.
 *
 * Grids are managed by TAMM and handed out read-only (const ScalapackGrid&). To change a grid,
 * release it and create a new one.
 */
class ScalapackGrid {
public:
  ScalapackGrid();
  ~ScalapackGrid();
  ScalapackGrid(const ScalapackGrid&)            = delete;
  ScalapackGrid& operator=(const ScalapackGrid&) = delete;

  /// Whether the calling rank is part of the grid (false for an empty grid).
  bool participates() const;

#if defined(USE_SCALAPACK)
  /// range(N) tiled by the grid's block size; valid on every rank of the process group.
  TiledIndexSpace index_space(int64_t N) const;

  /// A block-cyclic tensor on the grid, allocated on participating ranks.
  template<typename T>
  Tensor<T> allocate(const TiledIndexSpace& rows, const TiledIndexSpace& cols) const;

  /// A dense-kind tensor on the grid's ranks, allocated on participating ranks.
  template<typename T>
  Tensor<T> allocate_dense(const TiledIndexSpace& rows, const TiledIndexSpace& cols) const;

  /// Copy a regular tensor into a block-cyclic grid tensor. No-op on non-participating ranks.
  template<typename T>
  void to_block_cyclic(const Tensor<T>& src, Tensor<T>& bc) const;

  /// Copy a block-cyclic grid tensor into a regular tensor. No-op on non-participating ranks.
  template<typename T>
  void from_block_cyclic(const Tensor<T>& bc, Tensor<T>& dst) const;

  /// A new dense-kind copy of a block-cyclic grid tensor (empty on non-participating ranks).
  template<typename T>
  Tensor<T> from_block_cyclic_dense(const Tensor<T>& bc) const;
#endif

  struct Impl;                          ///< Defined in TAMM's sources only.
  Impl& impl() const { return *impl_; } ///< For TAMM's solvers; not part of the public API.

private:
  std::unique_ptr<Impl> impl_;
  friend struct ScalapackGridFactory;
};

/**
 * @brief The ScaLAPACK grid of ec's process group, created (collectively over the group) if it
 * does not exist yet. An existing grid is reused; explicit hints (npr, npc, nb, max_nranks)
 * that differ from the ones it was created with produce a one-time warning.
 */
const ScalapackGrid& scalapack_grid(ExecutionContext& ec, const ScalapackGridHints& hints = {});

/// The ScaLAPACK grid of ec's process group, or an empty grid if it has none. Never creates one.
const ScalapackGrid& find_scalapack_grid(ExecutionContext& ec);

/// Release the ScaLAPACK grid of ec's process group, if any. Collective over the group.
void release_scalapack_grid(ExecutionContext& ec);

namespace detail {
/// Release every remaining grid (called by tamm::finalize()).
void release_all_scalapack_grids();
} // namespace detail

} // namespace tamm
