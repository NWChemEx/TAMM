Dense linear algebra
====================

TAMM provides direct operations on explicitly stored matrices (2-D tensors or raw
buffers): dense eigensolves and singular value decompositions. A backend failure terminates the run
with an error message.

The eigensolvers come in two forms:

- **Local**: takes a matrix held in the calling rank's memory, e.g.
  ``eigensolve(N, A, eps, hw)`` with ``A`` a pointer to an ``N x N`` array. The call is rank-local:
  each calling rank solves its own matrix, with no communication.
- **Distributed**: takes TAMM tensors distributed over the ranks of ``ec``, e.g.
  ``eigensolve(ec, A, V, eps, hw)``. The call is collective over ``ec``, and TAMM decides which
  ranks do the work.

Complete, compiling uses of these routines, including the ScaLAPACK grid, are in
``tests/tamm/Test_EVP.cpp`` (``tests/tamm/Test_SVD.cpp`` for ``svd``).

Backends
--------

Which libraries are available is fixed when TAMM is built; ``hw`` chooses between the CPU and the
GPU where the build offers both. The calling code is the same in every build.

.. list-table::
   :header-rows: 1

   * - Build
     - Local form (calling rank)
     - Distributed form (collective over ``ec``)
   * - CPU (default)
     - LAPACK
     - Solved on rank 0 with LAPACK
   * - GPU (``USE_CUDA`` / ``USE_HIP`` / ``USE_DPCPP``)
     - LAPACK, or cuSOLVER / rocSOLVER / oneMKL when ``hw`` requests the GPU (see
       `Standard eigenvalue problem`_)
     - Solved on rank 0 with LAPACK, or with the GPU solver when ``hw`` requests the GPU
   * - ScaLAPACK (``USE_SCALAPACK``)
     - As above for CPU or GPU builds
     - ScaLAPACK on the process group's ScaLAPACK grid. ScaLAPACK runs only on the CPU, so the
       solve stays on the CPU even when ``hw`` requests the GPU
   * - ELPA (``TAMM_USE_ELPA``, implies ScaLAPACK)
     - As above for CPU or GPU builds
     - ELPA on the process group's ScaLAPACK grid: on the GPU when ``hw`` requests it in CUDA
       builds, on the CPU otherwise (ELPA's GPU support is not enabled for HIP / SYCL builds)

Standard eigenvalue problem
---------------------------

Solves the standard symmetric eigenvalue problem

.. math::

   A\, v_i = \varepsilon_i\, v_i, \qquad i = 1, \dots, N

for a symmetric ``N x N`` matrix ``A``, returning all ``N`` eigenvalues and an orthonormal set of
eigenvectors.

Parameters:

- ``A``: the symmetric ``N x N`` matrix.
- ``hw``: where the solve runs (see `Backends`_).

Results:

- The eigenvectors, by row: in a row-major view of the output (the layout of TAMM tensors and of
  ``Eigen::Matrix<..., Eigen::RowMajor>``), row ``i`` is :math:`v_i`, the eigenvector for
  ``eps[i]``; in a column-major view it is column ``i``. They are orthonormal.
- ``eps`` (resized to ``N``): the eigenvalues in ascending order.

It comes in the two forms described at the top of this page, which differ in what they modify:

.. list-table::
   :header-rows: 1

   * - Argument
     - Local
     - Distributed
   * - ``A``
     - Input; overwritten with the eigenvectors
     - Input; unchanged
   * - ``V``
     - (none: the eigenvectors are returned in ``A``)
     - Output, allocated by the caller; previous contents are ignored
   * - ``eps``
     - Output; resized to ``N``
     - Output; resized to ``N``; valid on the grid's ranks in builds with ScaLAPACK or ELPA, on
       rank 0 otherwise

Local
~~~~~

.. code:: cpp

   template<typename T>
   void eigensolve(int64_t N, T* A, std::vector<blas::real_type<T>>& eps,
                   ExecutionHW hw = ExecutionHW::CPU);

``N`` is the size of ``A``. ``A`` is a row-major host buffer held entirely by the calling rank.

The call is rank-local: it is not collective and does no communication, and each calling rank
solves its own matrix independently. Two common patterns:

- Solve on one rank (e.g. rank 0) and broadcast the result to the ranks that need it.
- Call it on every rank with the same (replicated) matrix; each rank then computes the full
  result itself, with no communication but with every rank doing the whole solve.

``hw`` selects where the solve runs:

- ``ExecutionHW::CPU`` (default): LAPACK on the host.
- ``ExecutionHW::GPU``: cuSOLVER / rocSOLVER / oneMKL on the calling rank's GPU in GPU builds,
  when ``N`` is at least the GPU threshold (1000 by default, see ``TAMM_GPU_EIGENSOLVE_MIN_N`` in
  :doc:`runtime_parameters`); LAPACK otherwise. In builds without a GPU the request is ignored.

Passing ``ec.exhw()`` lets the execution context decide (``GPU`` in GPU builds, ``CPU`` otherwise).

.. code:: cpp

   using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

   Matrix              A = ...;              // symmetric N x N
   std::vector<double> eps;
   tamm::eigensolve(N, A.data(), eps, ec.exhw());
   // A.row(i) is the eigenvector for eps[i]

Distributed
~~~~~~~~~~~

.. code:: cpp

   template<typename T>
   void eigensolve(ExecutionContext& ec, const Tensor<T>& A, Tensor<T>& V,
                   std::vector<blas::real_type<T>>& eps, ExecutionHW hw = ExecutionHW::CPU);

``A`` is a regular (not block-cyclic) tensor. ``V`` has the same shape as ``A`` and may be regular
or dense-kind (i.e. ``set_dense()`` / ``TensorKind::dense``).

- In builds with ScaLAPACK the solve is distributed over the process group's ScaLAPACK grid
  (ELPA when available), which is created automatically if the group does not have one yet (see
  `ScaLAPACK grids`_). Only the grid's ranks write ``V``, so ``V`` may be allocated on the grid's
  ranks only. ``hw`` applies to ELPA as described in `Backends`_; ScaLAPACK runs on the CPU.
- In other builds every rank makes the call, and TAMM performs the solve on rank 0 of ``ec``,
  with LAPACK or, when ``hw`` requests the GPU, the GPU solver (as for the local form).

The call is collective over ``ec``.

.. code:: cpp

   Tensor<double> A{tN, tN}, V{tN, tN};
   Tensor<double>::allocate(&ec, A, V);
   // ... fill A ...
   std::vector<double> eps;
   tamm::eigensolve(ec, A, V, eps, ec.exhw());

Generalized eigenvalue problem
------------------------------

Solves the generalized eigenvalue problem

.. math::

   F C = S C \varepsilon

for a symmetric ``F`` and a symmetric positive definite (possibly ill-conditioned) overlap
matrix ``S``, both ``N x N``, by transforming it into a standard eigenvalue problem using canonical
orthogonalization, followed by back-transformation of the resulting eigenvectors to the original
non-orthogonal basis:

.. math::

   (X^T F X)\, C' = C' \varepsilon, \qquad C = X C'

The caller supplies the canonical orthogonalizer ``X`` rather than ``S``: :math:`X = U s^{-1/2}`,
with ``U`` the eigenvectors of ``S`` whose eigenvalues ``s`` are above a linear-dependency
threshold, so that :math:`X^T S X = I`. With ``M`` eigenvectors kept, ``X`` is ``N x M``. Leaving
out the eigenvectors with small eigenvalues (``M < N``) removes the near-linear dependencies of
``S``, and the problem is solved in the remaining ``M``-dimensional space; with ``M = N`` the
solution is exact. Because ``X`` depends only on ``S``, it can be built once and reused for many
solves with different ``F``.

Parameters:

- ``F``: the symmetric ``N x N`` matrix.
- ``X``: the canonical orthogonalizer, ``N x M`` with ``M <= N``.
- ``hw``: where the standard eigensolve runs (see `Backends`_).

Results:

- ``C`` (``N x M``, allocated by the caller): the eigenvectors in the original basis. Column ``j``
  is the eigenvector for ``eps[j]`` (unlike ``eigensolve``, whose eigenvectors are rows). They are
  orthonormal with respect to ``S``: :math:`C^T S C = I`.
- ``eps`` (resized to ``M``): the eigenvalues in ascending order.

``X`` can be built once with ``eigensolve`` of ``S``: keep the eigenvectors whose eigenvalues are at
or above the threshold (the last ``M``, since eigenvalues are in ascending order), transpose them
into columns, and scale column ``k`` by :math:`1/\sqrt{s_k}`. For the distributed form, copy the
result to the ScaLAPACK grid with ``to_block_cyclic``.

It comes in the same two forms as the standard eigensolve and uses the same backends, with the same
effect of ``hw`` (see `Backends`_). The math is the same in both forms.

.. note::

   Unlike the standard eigensolve, the distributed ``generalized_eigensolve`` exists only in builds
   with ScaLAPACK or ELPA. Its ``X`` and ``C`` are block-cyclic tensors on the ScaLAPACK grid, which
   other builds do not have, so there is no fallback to a solve on rank 0. In builds without
   ScaLAPACK or ELPA, use the local form, for example on rank 0 followed by a broadcast of the
   results.

The forms differ in what they modify:

.. list-table::
   :header-rows: 1

   * - Argument
     - Local
     - Distributed
   * - ``F``
     - Input; overwritten (used as workspace)
     - Input; unchanged
   * - ``X``
     - Input; unchanged
     - Input; unchanged
   * - ``C``
     - Output, allocated by the caller; previous contents are ignored
     - Output, allocated by the caller; previous contents are ignored
   * - ``eps``
     - Output; resized to ``M``
     - Output; resized to ``M``, valid on the grid's ranks

Local
~~~~~

.. code:: cpp

   template<typename T>
   void generalized_eigensolve(int64_t N, int64_t M, T* F, const T* X, T* C,
                               std::vector<blas::real_type<T>>& eps,
                               ExecutionHW hw = ExecutionHW::CPU);

``N`` is the size of ``F`` and ``M`` the number of columns of ``X`` and ``C`` (``M <= N``). ``F``,
``X`` and ``C`` are row-major host buffers on the calling rank. Rank-local in the same way as the
local ``eigensolve``. Available in every build, including builds with ScaLAPACK, for problems small
enough to solve on one rank.

Distributed
~~~~~~~~~~~

.. code:: cpp

   template<typename T>
   void generalized_eigensolve(ExecutionContext& ec, const Tensor<T>& F, const Tensor<T>& X,
                               Tensor<T>& C, std::vector<blas::real_type<T>>& eps,
                               ExecutionHW hw = ExecutionHW::CPU);

Declared only in builds with ScaLAPACK or ELPA (see the note above). ``F`` is a regular ``N x N``
tensor; ``X`` and ``C`` are block-cyclic tensors on the ScaLAPACK grid of ``ec``'s process group,
which must exist. The call is collective over the grid's ranks and does nothing on other ranks. A
block-cyclic copy of ``F`` is cached on the grid and reused across calls, so calling it repeatedly
with a changing ``F`` and a fixed ``X`` does not reallocate.

.. code:: cpp

   const tamm::ScalapackGrid& grid = tamm::scalapack_grid(ec, {.N = N});
   Tensor<double> X_bc = grid.allocate<double>(grid.index_space(N), grid.index_space(M));
   Tensor<double> C_bc = grid.allocate<double>(grid.index_space(N), grid.index_space(M));
   grid.to_block_cyclic(X, X_bc);            // X: regular N x M tensor

   std::vector<double> eps;
   tamm::generalized_eigensolve(ec, F, X_bc, C_bc, eps, ec.exhw());
   grid.from_block_cyclic(C_bc, C);          // C: regular N x M tensor

Singular value decomposition
----------------------------

.. code:: cpp

   struct SVDOptions {
     bool full_matrices = true;
     bool compute_uv    = true;
   };

   template<typename T>
   std::tuple<Tensor<T>, std::vector<blas::real_type<T>>, Tensor<T>>
   svd(ExecutionContext& ec, Tensor<T> A, SVDOptions opts = {},
       ExecutionHW execute_on = ExecutionHW::CPU);

Computes :math:`A = U\,\mathrm{diag}(S)\,V^H` for an ``M x N`` tensor by gathering ``A`` on rank 0.
Returns ``(U, S, Vh)``: ``U`` is ``M x M`` and ``Vh`` is ``N x N`` with ``full_matrices``, or
``M x K`` and ``K x N`` (``K = min(M, N)``) without; with ``compute_uv = false`` only ``S`` is
computed. ``S`` holds the singular values in non-increasing order on every rank. With
``execute_on = ExecutionHW::GPU`` in GPU builds the decomposition runs on rank 0's GPU when
``M >= N``, and on the CPU otherwise (the GPU path supports ``double`` and
``std::complex<double>``). The caller deallocates ``U`` and ``Vh``.

ScaLAPACK grids
---------------

In builds with ScaLAPACK, distributed solves run on a *ScaLAPACK grid*: a square BLACS process
grid made of the first ranks of a process group, with a block-cyclic distribution. A process group
has at most one grid, which TAMM owns.

Creating, finding and releasing a grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: cpp

   const ScalapackGrid& scalapack_grid(ExecutionContext& ec, const ScalapackGridHints& hints = {});
   const ScalapackGrid& find_scalapack_grid(ExecutionContext& ec);
   void                 release_scalapack_grid(ExecutionContext& ec);

- ``scalapack_grid`` returns the grid of ``ec``'s process group, creating it if the group does
  not have one. Creation is collective over the process group.
- ``find_scalapack_grid`` returns the existing grid, or an empty grid if there is none. It never
  creates one.
- ``release_scalapack_grid`` frees the grid (its sub-group and BLACS resources). It is collective
  over the process group; the group itself stays valid, and a later request creates a new grid.

A grid is also released automatically when its process group is destroyed and at
``tamm::finalize()``. ``eigensolve`` creates a grid on first use if none exists, sized for the
matrix it solves.

Release a grid explicitly when a long-lived process group (for example the world group) only needs
it for part of a run; a grid on a sub-group that is destroyed afterwards needs no explicit release.

Grids are handed out read-only (``const ScalapackGrid&``). To change a grid's sizing, release it
and create a new one.

Sizing
~~~~~~

.. code:: cpp

   struct ScalapackGridHints {
     int64_t N{0};          // matrix size the grid is sized for
     int     npr{0};        // requested process rows
     int     npc{0};        // requested process columns
     int     nb{0};         // requested block size
     int     max_nranks{0}; // cap on the number of grid ranks
   };

A value of 0 means "choose automatically":

- **Ranks**: about 4% of ``N``, or ``npr * npc`` if both are given; capped at ``max_nranks``
  (default: the size of the process group); rounded down to a square grid (at least 1x1). The grid
  uses the first ranks of the process group.
- **Block size**: ``nb`` (default 256), reduced to the largest power of 2 not exceeding
  ``N / npr`` for small ``N``.

On creation the grid prints its size (``scalapack_nnodes``, ``scalapack_nranks``,
``scalapack_np_row``, ``scalapack_np_col``, ``scalapack_nb``) and a warning if the block size was
reduced. If a grid already exists, ``scalapack_grid`` returns it; explicit hints (``npr``, ``npc``,
``nb``, ``max_nranks``) that differ from the ones it was created with produce a one-time warning.
One grid serves matrices of different sizes.

Working with block-cyclic tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: cpp

   class ScalapackGrid {
   public:
     bool participates() const;
     TiledIndexSpace index_space(int64_t N) const;
     template<typename T> Tensor<T> allocate(const TiledIndexSpace& rows, const TiledIndexSpace& cols) const;
     template<typename T> Tensor<T> allocate_dense(const TiledIndexSpace& rows, const TiledIndexSpace& cols) const;
     template<typename T> void      to_block_cyclic(const Tensor<T>& src, Tensor<T>& bc) const;
     template<typename T> void      from_block_cyclic(const Tensor<T>& bc, Tensor<T>& dst) const;
     template<typename T> Tensor<T> from_block_cyclic_dense(const Tensor<T>& bc) const;
   };

- ``participates()``: whether the calling rank is part of the grid (``false`` for an empty grid).
- ``index_space(N)``: ``range(N)`` tiled by the grid's block size; use it for the dimensions of
  block-cyclic tensors.
- ``allocate``: a block-cyclic tensor on the grid, allocated on participating ranks.
- ``allocate_dense``: a dense-kind tensor on the grid's ranks (for routines such as
  ``tensor_block`` that require one).
- ``to_block_cyclic`` / ``from_block_cyclic``: copy a regular tensor into a block-cyclic grid
  tensor, and back. Both are no-ops on non-participating ranks, so they can be called on every rank
  of the process group.
- ``from_block_cyclic_dense``: a new dense-kind copy of a block-cyclic tensor (empty on
  non-participating ranks).

Everything except ``participates()`` is declared only in builds with ScaLAPACK.

Rules
~~~~~

- Calls that create or release a grid, and the distributed solves, are collective: call them on
  every rank of the process group.
- Tensors from ``allocate``, ``allocate_dense`` and ``from_block_cyclic_dense`` belong to the
  caller. Deallocate them (on participating ranks) before the grid is released or its process
  group is destroyed; otherwise the release stops with an error naming the number of tensors still
  allocated.
- A grid is not safe to use from several threads at once.
- In builds without ScaLAPACK every process group has an empty grid: ``scalapack_grid`` and
  ``find_scalapack_grid`` return it, ``release_scalapack_grid`` does nothing, and the distributed
  ``eigensolve`` solves on rank 0.

Recipes
~~~~~~~

A grid that lives as long as a sub-group, sized once:

.. code:: cpp

   ProcGroup        sub_pg = ProcGroup::create_subgroup(ec.pg(), nranks);
   if(sub_pg.is_valid()) {
     ExecutionContext sub_ec{sub_pg, DistributionKind::nw, MemoryManagerKind::ga};

     const tamm::ScalapackGrid& grid = tamm::scalapack_grid(sub_ec, {.N = N, .nb = 128});
     Tensor<double> X_bc = grid.allocate<double>(grid.index_space(N), grid.index_space(M));
     Tensor<double> C_bc = grid.allocate<double>(grid.index_space(N), grid.index_space(M));
     // ... repeated tamm::generalized_eigensolve(sub_ec, F, X_bc, C_bc, eps, sub_ec.exhw()) ...

     if(grid.participates()) Tensor<double>::deallocate(X_bc, C_bc);
     sub_ec.flush_and_sync();
     sub_pg.destroy_coll(); // also releases the grid
   }

A grid needed only briefly on a long-lived process group:

.. code:: cpp

   tamm::scalapack_grid(ec, {.N = N, .max_nranks = nranks_cap});
   tamm::eigensolve(ec, A, V, eps);            // distributed on ec's grid
   tamm::release_scalapack_grid(ec);           // free it; ec stays usable
