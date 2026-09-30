#include "tamm/scalapack_grid_impl.hpp"
#include "tamm/tamm.hpp"

#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

#if defined(TAMM_USE_ELPA)
#include <elpa/elpa.h>
#endif

namespace tamm {

ScalapackGrid::ScalapackGrid()  = default;
ScalapackGrid::~ScalapackGrid() = default;

bool ScalapackGrid::participates() const {
#if defined(USE_SCALAPACK)
  return impl_ && impl_->participates();
#else
  return false;
#endif
}

namespace {

// A grid attached to its parent process group; released by ProcGroup::destroy_coll().
struct GridAttachment: ProcGroupAttachment {
  ScalapackGrid grid;
  void          release() override {
#if defined(USE_SCALAPACK)
    if(grid.participates()) grid.impl().release();
#endif
  }
};

const ScalapackGrid& empty_grid() {
  static const ScalapackGrid empty;
  return empty;
}

std::shared_ptr<GridAttachment> attached_grid(ExecutionContext& ec) {
  return std::dynamic_pointer_cast<GridAttachment>(ec.pg().attachment());
}

#if defined(USE_SCALAPACK)
// Grids created so far, released at tamm::finalize() if still alive. Grids are created
// collectively in the same order on every rank, so releasing in this order is collective-safe.
std::vector<std::weak_ptr<GridAttachment>>& grid_registry() {
  static std::vector<std::weak_ptr<GridAttachment>> registry;
  return registry;
}
#endif

} // namespace

struct ScalapackGridFactory {
  static std::shared_ptr<GridAttachment> create(ExecutionContext&         ec,
                                                const ScalapackGridHints& hints);
};

#if defined(USE_SCALAPACK)

std::shared_ptr<GridAttachment> ScalapackGridFactory::create(ExecutionContext&         ec,
                                                             const ScalapackGridHints& hints) {
  auto attachment         = std::make_shared<GridAttachment>();
  attachment->grid.impl_  = std::make_unique<ScalapackGrid::Impl>();
  ScalapackGrid::Impl& g  = *attachment->grid.impl_;
  g.hints                 = hints;
  const int64_t N         = hints.N;
  const int     ppn       = ec.ppn();
  const int     pg_nranks = ec.pg().size().value();

  // Number of grid ranks: ~4% of N (or the requested npr * npc), capped, rounded down to a square.
  int sca_user_ranks = hints.npr * hints.npc;
  int sca_nranks     = std::ceil(N * (4 / 100.0));
  if(sca_user_ranks > 0) sca_nranks = sca_user_ranks;
  const int max_nranks = hints.max_nranks > 0 ? hints.max_nranks : pg_nranks;
  if(sca_nranks > max_nranks) sca_nranks = max_nranks;
  sca_nranks = std::pow(std::floor(std::sqrt(sca_nranks)), 2);
  if(sca_nranks == 0) sca_nranks = 1;

  int sca_nnodes = sca_nranks / ppn;
  if(sca_nranks % ppn > 0 || sca_nnodes == 0) sca_nnodes++;

  g.nranks = sca_nranks;
  g.npr    = std::sqrt(sca_nranks);
  g.npc    = g.npr;

  g.pg = ProcGroup::create_subgroup(ec.pg(), sca_nranks);

  // Block size: requested (default 256), reduced to a power of 2 <= N / npr for small N.
  // Computed on every rank of the parent group so index_space() agrees everywhere.
  const int64_t mb_requested = hints.nb > 0 ? hints.nb : 256;
  g.mb                       = mb_requested;
  const bool reset_mb        = g.mb > N / g.npr;
  if(reset_mb) {
    g.mb = N / g.npr;
    if(g.mb < 1) g.mb = 1;
    g.mb = static_cast<int64_t>(std::pow(2, static_cast<int>(std::log2(g.mb))));
  }

  if(!g.pg.is_valid()) return attachment; // ranks outside the grid

  g.ec = ExecutionContext{g.pg, DistributionKind::dense, MemoryManagerKind::ga};

  if(g.pg.rank() == 0) {
    std::cout << "scalapack_nnodes = " << sca_nnodes << std::endl;
    std::cout << "scalapack_nranks = " << sca_nranks << std::endl;
    std::cout << "scalapack_np_row = " << g.npr << std::endl;
    std::cout << "scalapack_np_col = " << g.npc << std::endl;
    std::cout << "scalapack_nb     = " << mb_requested << std::endl;
    if(reset_mb)
      std::cout << "WARNING: Resetting scalapack block size (scalapack_nb) to: " << g.mb
                << std::endl;
  }

  std::vector<int64_t> grid_ranks(sca_nranks);
  std::iota(grid_ranks.begin(), grid_ranks.end(), 0);
  g.blacs_grid =
    std::make_unique<blacspp::Grid>(g.pg.comm(), g.npr, g.npc, grid_ranks.data(), g.npr);
  g.dist = std::make_unique<scalapackpp::BlockCyclicDist2D>(*g.blacs_grid, g.mb, g.mb, 0, 0);

  return attachment;
}

Tensor<double>& ScalapackGrid::Impl::scratch(int64_t N) {
  if(f_scratch_n != N) {
    if(f_scratch.is_allocated()) Tensor<double>::deallocate(f_scratch);
    TiledIndexSpace tN{IndexSpace{range(N)}, static_cast<Tile>(mb)};
    f_scratch = Tensor<double>{tN, tN};
    f_scratch.set_block_cyclic({npr, npc});
    Tensor<double>::allocate(&ec, f_scratch);
    f_scratch_n = N;
  }
  return f_scratch;
}

void ScalapackGrid::Impl::release(bool check_live) {
  if(!participates()) return;

  int64_t live = 0;
  for(auto& is_allocated: caller_tensors) live += is_allocated() ? 1 : 0;
  if(live > 0) {
    const std::string msg = "[TAMM ERROR] " + std::to_string(live) +
                            " tensor(s) allocated on the ScaLAPACK grid are still allocated; "
                            "deallocate them before releasing the grid or destroying its "
                            "process group.";
    if(check_live) tamm_terminate(msg);
    if(pg.rank() == 0) std::cout << msg << " Leaving the grid to GA/MPI shutdown." << std::endl;
    return;
  }

  if(f_scratch.is_allocated()) Tensor<double>::deallocate(f_scratch);
  dist.reset();
  blacs_grid.reset();
  ec.flush_and_sync();
  pg.destroy_coll();
}

TiledIndexSpace ScalapackGrid::index_space(int64_t N) const {
  EXPECTS(impl_ != nullptr);
  return TiledIndexSpace{IndexSpace{range(N)}, static_cast<Tile>(impl_->mb)};
}

template<typename T>
Tensor<T> ScalapackGrid::allocate(const TiledIndexSpace& rows, const TiledIndexSpace& cols) const {
  EXPECTS(impl_ != nullptr);
  Tensor<T> t{rows, cols};
  t.set_block_cyclic({impl_->npr, impl_->npc});
  if(participates()) {
    Tensor<T>::allocate(&impl_->ec, t);
    impl_->caller_tensors.push_back([t]() { return t.is_allocated(); });
  }
  return t;
}

template<typename T>
Tensor<T> ScalapackGrid::allocate_dense(const TiledIndexSpace& rows,
                                        const TiledIndexSpace& cols) const {
  EXPECTS(impl_ != nullptr);
  Tensor<T> t{rows, cols};
  t.set_dense();
  if(participates()) {
    Tensor<T>::allocate(&impl_->ec, t);
    impl_->caller_tensors.push_back([t]() { return t.is_allocated(); });
  }
  return t;
}

template<typename T>
void ScalapackGrid::to_block_cyclic(const Tensor<T>& src, Tensor<T>& bc) const {
  if(!participates()) return;
  tamm::to_block_cyclic_tensor(src, bc);
}

template<typename T>
void ScalapackGrid::from_block_cyclic(const Tensor<T>& bc, Tensor<T>& dst) const {
  if(!participates()) return;
  tamm::from_block_cyclic_tensor(bc, dst);
}

template<typename T>
Tensor<T> ScalapackGrid::from_block_cyclic_dense(const Tensor<T>& bc) const {
  if(!participates()) return Tensor<T>{};
  Tensor<T> t = tamm::from_block_cyclic_tensor(bc);
  impl_->caller_tensors.push_back([t]() { return t.is_allocated(); });
  return t;
}

template Tensor<double> ScalapackGrid::allocate<double>(const TiledIndexSpace&,
                                                        const TiledIndexSpace&) const;
template Tensor<double> ScalapackGrid::allocate_dense<double>(const TiledIndexSpace&,
                                                              const TiledIndexSpace&) const;
template void ScalapackGrid::to_block_cyclic<double>(const Tensor<double>&, Tensor<double>&) const;
template void ScalapackGrid::from_block_cyclic<double>(const Tensor<double>&,
                                                       Tensor<double>&) const;
template Tensor<double> ScalapackGrid::from_block_cyclic_dense<double>(const Tensor<double>&) const;

// ---------------------------------------------------------------------------------------------
// Distributed eigensolve (internal)

#if defined(TAMM_USE_ELPA)
namespace {
// Owns one ELPA handle (and the ELPA library init) for the duration of a single solve.
class ElpaSolver {
public:
  ElpaSolver() {
    if(elpa_init(20221109) != ELPA_OK) tamm_terminate("[TAMM ERROR] ELPA API not supported");
    int error;
    handle_ = elpa_allocate(&error);
    check(error, "elpa_allocate");
  }

  ~ElpaSolver() {
    int error;
    elpa_deallocate(handle_, &error);
    check(error, "elpa_deallocate");
    elpa_uninit(&error);
    check(error, "elpa_uninit");
  }

  ElpaSolver(const ElpaSolver&)            = delete;
  ElpaSolver& operator=(const ElpaSolver&) = delete;

  void set(const char* name, int value) {
    int error;
    elpa_set(handle_, name, value, &error);
    check(error, std::string("elpa_set(") + name + ")");
  }

  void setup() { check(elpa_setup(handle_), "elpa_setup"); }

  template<typename T>
  void eigenvectors(T* A, T* eps, T* V) {
    int error;
    elpa_eigenvectors(handle_, A, eps, V, &error);
    check(error, "elpa_eigenvectors");
  }

private:
  elpa_t handle_;

  static void check(int error, const std::string& what) {
    if(error != ELPA_OK) tamm_terminate("[TAMM ERROR] ELPA " + what + " failed");
  }
};
} // namespace
#endif

template<typename T>
void detail::distributed_eigensolve(ScalapackGrid::Impl& g, int64_t N, T* A_local, T* V_local,
                                    std::vector<T>& eps, [[maybe_unused]] ExecutionHW hw) {
  if(!g.participates()) return;
  const blacspp::Grid&                  grid = *g.blacs_grid;
  const scalapackpp::BlockCyclicDist2D& dist = *g.dist;
  if(grid.ipr() < 0 || grid.ipc() < 0) return;

  eps.resize(N);
  const auto [N_loc_rows, N_loc_cols] = dist.get_local_dims(N, N);

#if defined(TAMM_USE_ELPA)
  ElpaSolver elpa;
  elpa.set("na", static_cast<int>(N));
  elpa.set("nev", static_cast<int>(N));
  elpa.set("local_nrows", static_cast<int>(N_loc_rows));
  elpa.set("local_ncols", static_cast<int>(N_loc_cols));
  elpa.set("nblk", static_cast<int>(dist.mb()));
  elpa.set("mpi_comm_parent", static_cast<int>(g.pg.comm_c2f()));
  elpa.set("process_row", static_cast<int>(grid.ipr()));
  elpa.set("process_col", static_cast<int>(grid.ipc()));
  // ELPA's GPU support is enabled only for NVIDIA GPUs for now; HIP/SYCL builds run it on the CPU.
#if defined(USE_CUDA)
  const bool use_gpu = (hw == ExecutionHW::GPU);
#else
  const bool use_gpu = false;
#endif
  if(use_gpu) elpa.set("nvidia-gpu", 1);
  elpa.setup();

  elpa.set("solver", ELPA_SOLVER_2STAGE);
  elpa.set("real_kernel", use_gpu ? ELPA_2STAGE_REAL_NVIDIA_GPU : ELPA_2STAGE_REAL_AVX2_BLOCK2);

  elpa.eigenvectors(A_local, eps.data(), V_local);
#else
  const auto    desc = dist.descinit_noerror(N, N, N_loc_rows);
  const int64_t info = scalapackpp::hereig(scalapackpp::Job::Vec, scalapackpp::Uplo::Lower, N,
                                           A_local, 1, 1, desc, eps.data(), V_local, 1, 1, desc);
  if(info != 0)
    tamm_terminate("[TAMM ERROR] ScaLAPACK eigensolve failed with info = " + std::to_string(info));
#endif
}

template void detail::distributed_eigensolve<double>(ScalapackGrid::Impl& g, int64_t N,
                                                     double* A_local, double* V_local,
                                                     std::vector<double>& eps, ExecutionHW hw);

#endif

// ---------------------------------------------------------------------------------------------
// Access

const ScalapackGrid& scalapack_grid([[maybe_unused]] ExecutionContext&         ec,
                                    [[maybe_unused]] const ScalapackGridHints& hints) {
#if defined(USE_SCALAPACK)
  if(auto existing = attached_grid(ec)) {
    ScalapackGrid::Impl& g        = existing->grid.impl();
    const auto&          old      = g.hints;
    const bool           mismatch = (hints.npr > 0 && hints.npr != old.npr) ||
                          (hints.npc > 0 && hints.npc != old.npc) ||
                          (hints.nb > 0 && hints.nb != old.nb) ||
                          (hints.max_nranks > 0 && hints.max_nranks != old.max_nranks);
    if(mismatch && !g.warned) {
      g.warned = true;
      if(ec.pg().rank() == 0)
        std::cout << "WARNING: a ScaLAPACK grid already exists for this process group; "
                     "reusing it and ignoring the new sizing hints (call "
                     "release_scalapack_grid() first to resize it)."
                  << std::endl;
    }
    return existing->grid;
  }
  auto created = ScalapackGridFactory::create(ec, hints);
  ec.pg().set_attachment(created);
  grid_registry().push_back(created);
  return created->grid;
#else
  return empty_grid();
#endif
}

const ScalapackGrid& find_scalapack_grid(ExecutionContext& ec) {
  if(auto existing = attached_grid(ec)) return existing->grid;
  return empty_grid();
}

void release_scalapack_grid(ExecutionContext& ec) {
  auto existing = attached_grid(ec);
  if(!existing) return;
  existing->release();
  ec.pg().set_attachment(nullptr);
}

void detail::release_all_scalapack_grids() {
#if defined(USE_SCALAPACK)
  for(auto& weak: grid_registry()) {
    if(auto attachment = weak.lock()) {
      if(attachment->grid.participates()) attachment->grid.impl().release(/*check_live=*/false);
    }
  }
  grid_registry().clear();
#endif
}

} // namespace tamm
