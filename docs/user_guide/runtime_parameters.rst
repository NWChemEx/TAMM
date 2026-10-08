
Runtime parameters
==================

- ``TAMM_GPU_EIGENSOLVE_MIN_N (int)`` Smallest matrix size for which the local ``eigensolve`` (and
   the routines built on it) runs on the GPU when the GPU is requested, in GPU builds.
   ``[default=1000]`` - Smaller matrices are solved with LAPACK on the CPU. Set to 0 to use the GPU
   for every requested solve. When set, the value is printed once at the first GPU-eligible solve.
   See :doc:`linalg`.

- ``TAMM_IO_STRIPE_COUNT (int)`` Number of Lustre OSTs each tensor file written by
   ``write_to_disk`` is striped over.
   ``[default: one per 4 GiB of tensor data, at least 1 and at most 64]``. Ignored on filesystems
   other than Lustre. See :doc:`tensor_io`.

- ``TAMM_IO_STRIPE_SIZE (int)`` Lustre stripe size, in MiB, of each tensor file written by
   ``write_to_disk``.
   ``[default=4]``. Ignored on filesystems other than Lustre. See :doc:`tensor_io`.

- ``TAMM_IO_GIB_PER_NODE (int)`` GiB of tensor data per node used to choose how many nodes write
   or read each tensor file in ``write_to_disk`` / ``read_from_disk``.
   ``[default=5]``. See :doc:`tensor_io`.

- ``TAMM_IO_GROUPS (size|all)`` How the nodes are split into I/O groups in ``write_to_disk`` /
   ``read_from_disk``. ``size`` sizes the groups from the tensor data (see
   ``TAMM_IO_GIB_PER_NODE``); ``all`` makes all of ``ec`` handle each tensor file, one file after
   another, and ignores ``TAMM_IO_GIB_PER_NODE``. ``[default=size]``. See :doc:`tensor_io`.
