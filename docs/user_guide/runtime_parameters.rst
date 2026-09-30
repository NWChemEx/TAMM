
Runtime parameters
==================

- ``TAMM_ENABLE_SPRHBM (int)`` Enables the use of HBM memory on CPUs such as Intel SPR. 
   ``[default=0]`` - Does not use HBM memory and instead uses DDR partition. Set to 1 to enable use of HBM memory.

- ``TAMM_GPU_EIGENSOLVE_MIN_N (int)`` Smallest matrix size for which the local ``eigensolve`` (and
   the routines built on it) runs on the GPU when the GPU is requested, in GPU builds.
   ``[default=1000]`` - Smaller matrices are solved with LAPACK on the CPU. Set to 0 to use the GPU
   for every requested solve. When set, the value is printed once at the first GPU-eligible solve.
   See :doc:`linalg`.
