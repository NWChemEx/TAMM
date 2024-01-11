
Runtime parameters
==================

- ``TAMM_GPU_POOL`` Specifies the size of memory pool per GPU. Default is set to 80% of the free memory reported by the GPU runtime APIs. 
   Also refer to ``TAMM_RANKS_PER_GPU_POOL``. Valid values range between 1-100.

- ``TAMM_CPU_POOL`` Specifies the size of memory pool for CPU. Default is set to 100% of the size that is 
   left after reserving sufficient for Global arrays, and taking into account binding of multiple ranks per 
   numa-node as well. Valid values range between 1-100.

