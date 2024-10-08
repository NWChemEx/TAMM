#!/bin/bash

#SBATCH -A <account>   -q regular
#SBATCH -C gpu
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH -J test
#SBATCH --gpus-per-node 4
#SBATCH -o tamm_test.%j

set -x

module load PrgEnv-gnu
module load cudatoolkit
module load cpe-cuda
module load cmake
module unload cray-libsci
module unload craype-accel-nvidia80
module load cray-python

export MPICH_GPU_SUPPORT_ENABLED=0
export CRAYPE_LINK_TYPE=dynamic

ppn=4

export OMP_NUM_THREADS=1

export MPICH_OFI_SKIP_NIC_SYMMETRY_TEST=1

export MPICH_OFI_VERBOSE=1

export MPICH_OFI_NIC_VERBOSE=1

# export MPI_OFI_NIC_POLICY=GPU

cd /$PSCRATCH/output

EXE=<tamm-exe>
INPUT=<args>

export GA_NUM_PROGRESS_RANKS_PER_NODE=1
export GA_PROGRESS_RANKS_DISTRIBUTION_PACKED=1
#--mem-bind=map_mem:0,1,2,3,0
srun -u --cpu_bind=map_cpu:0,16,32,48,64 --mem-bind=map_mem:0,1,2,3,0 --gpus-per-node 4 --ntasks-per-node $(( ppn + GA_NUM_PROGRESS_RANKS_PER_NODE )) ./perlmutter_bind.sh $EXE $INP
