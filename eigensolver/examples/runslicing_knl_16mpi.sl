#!/bin/bash
#SBATCH -N 16
#SBATCH -C knl
#SBATCH -q debug
#SBATCH -J slicing_mpi
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=68
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 16 -c 272 --cpu_bind=cores ./runslicing_mpi Si16H10.input >& Si16H10.out.16mpi
mv results.txt results.Si16H10
srun -n 16 -c 272 --cpu_bind=cores ./runslicing_mpi Graphene1620.input >& Graphene1620.out.16mpi
mv results.txt results.Graphene1620
srun -n 16 -c 272 --cpu_bind=cores ./runslicing_mpi Si512.input >& Si512.out.16mpi
mv results.txt results.Si512
srun -n 16 -c 272 --cpu_bind=cores ./runslicing_mpi Si2.input >& Si2.out.16mpi
mv results.txt results.Si2
srun -n 16 -c 272 --cpu_bind=cores ./runslicing_mpi SiNa.input >& SiNa.out.16mpi
mv results.txt results.SiNa.mpi
srun -n 16 -c 272 --cpu_bind=cores ./runslicing_mpi SiH4.input >& SiH4.out.16mpi
mv results.txt results.SiH4
srun -n 16 -c 272 --cpu_bind=cores ./runslicing_mpi Na5.input >& Na5.out.16mpi
mv results.txt results.Na5
srun -n 16 -c 272 --cpu_bind=cores ./runslicing_mpi benzene.input >& benzene.out.16mpi
mv results.txt results.benzene
srun -n 16 -c 272 --cpu_bind=cores ./runslicing_mpi Graphene720.input >& Graphene720.out.16mpi
mv results.txt results.Graphene720
srun -n 16 -c 272 --cpu_bind=cores ./runslicing_mpi SWNT66.input >& SWNT66.out.16mpi
mv results.txt results.SWNT66
srun -n 16 -c 272 --cpu_bind=cores ./runslicing_mpi Si216.input >& Si216.out.16mpi
mv results.txt results.Si216
