
Runtime parameters
==================

- ``TAMM_ENABLE_SPRHBM (int)`` Enables the use of HBM memory on CPUs such as Intel SPR. 
   ``[default=0]`` - Does not use HBM memory and instead uses DDR partition. Set to 1 to enable use of HBM memory.

- ``TAMM_RANKS_PER_GPU_POOL (int)`` Specify the number of ranks to bind per GPU. 
   ``[default=1]``. When binding multiple ranks to the GPU, please set the integer value appropriately.

