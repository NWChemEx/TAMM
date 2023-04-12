#!/bin/bash

# Add the number of Nvidia GPUs per node here
num_gpus=

#Usage: mpiexec -n X --ppn Y --env OMP_NUM_THREADS=1 $PWD/sample_mpich_gpu_bind.sh $EXEC ....
export CUDA_VISIBLE_DEVICES=$((${PMI_LOCAL_RANK} % ${num_gpus}))

echo "RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} gpu= ${CUDA_VISIBLE_DEVICES}"
exec $*
