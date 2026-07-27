#!/bin/bash
#SBATCH -A <projectid>
#SBATCH -J test
#SBATCH -t 0:10:00
#SBATCH --gpus-per-node 8
#SBATCH -N 2
#SBATCH -o tamm_test.%j

set -x 
date

module load cpe
module load cray-python cmake cray-hdf5-parallel
module load cce
module load cray-mpich
module load rocm
module list

export OMP_NUM_THREADS=1
export CRAYPE_LINK_TYPE=dynamic

export MPICH_GPU_SUPPORT_ENABLED=0
export SLURM_MPI_TYPE=cray_shasta
export FI_CXI_RX_MATCH_MODE=hybrid
export MPICH_SMP_SINGLE_COPY_MODE=NONE

export GA_NUM_PROGRESS_RANKS_PER_NODE=8
export GA_PROGRESS_RANKS_DISTRIBUTION_PACKED=0
export GA_PROGRESS_RANKS_DISTRIBUTION_CYCLIC=1

cd /lustre/orion/<projectid>/scratch/<userid>/output

EXE=<tamm-exe>
INP=<args>

ppn=8
NRANKS_PER_NODE=$(( ppn + GA_NUM_PROGRESS_RANKS_PER_NODE ))
NTOTRANKS=$(( $SLURM_NNODES * NRANKS_PER_NODE ))

srun -N${SLURM_NNODES} -n${NTOTRANKS} -c1 --ntasks-per-gpu=1 --gpus-per-node=8 --gpu-bind=closest -m block:cyclic $EXE  $INP
