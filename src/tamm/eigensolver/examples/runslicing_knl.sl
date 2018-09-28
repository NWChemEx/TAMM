#!/bin/bash
#SBATCH -N 1
#SBATCH -C knl
#SBATCH -q debug
#SBATCH -J filter
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=68
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
#srun -n 1 -c 272 --cpu_bind=cores ./runslicing Si16H10.input >& Si16H10.out
#mv results.txt results.Si16H10
#srun -n 1 -c 272 --cpu_bind=cores ./runslicing Graphene1620.input >& Graphene1620.out
#mv results.txt results.Graphene1620
#srun -n 1 -c 272 --cpu_bind=cores ./runslicing Si512.input >& Si512.out
#mv results.txt results.Si512
srun -n 1 -c 272 --cpu_bind=cores ./runslicing Si2.input >& Si2.out
mv results.txt results.Si2
srun -n 1 -c 272 --cpu_bind=cores ./runslicing SiNa.input >& SiNa.out
mv results.txt results.SiNa
srun -n 1 -c 272 --cpu_bind=cores ./runslicing SiH4.input >& SiH4.out
mv results.txt results.SiH4
srun -n 1 -c 272 --cpu_bind=cores ./runslicing Na5.input >& Na5.out
mv results.txt results.Na5
srun -n 1 -c 272 --cpu_bind=cores ./runslicing benzene.input >& benzene.out
mv results.txt results.benzene
srun -n 1 -c 272 --cpu_bind=cores ./runslicing Graphene720.input >& Graphene720.out
mv results.txt results.Graphene720
#srun -n 1 -c 272 --cpu_bind=cores ./runslicing SWNT66.input >& SWNT66.out
#mv results.txt results.SWNT66
#srun -n 1 -c 272 --cpu_bind=cores ./runslicing Si216.input >& Si216.out
#mv results.txt results.Si216
