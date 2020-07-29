
The prerequisites needed to build:

- autotools
- cmake >= 3.13
- MPI Library
- C++17 compiler
- CUDA >= 10.1 (only if building with GPU support)

`MACOSX`: We recommend using brew to install the prerequisites:  
- `brew install gcc openmpi cmake wget autoconf automake`

Supported Compilers
--------------------
- GCC versions >= 8.x
- LLVM Clang >= 7.x (Tested on Linux Only)
- `MACOSX`: We only support brew installed `GCC`, AppleClang is not supported.


Supported Configurations
-------------------------
- The following configurations are recommended since they are tested and are known to work:
  - GCC versions >= 8.x + OpenMPI-2.x/MPICH-3.x built using corresponding gcc versions.
  - (Linux Only) LLVM Clang versions >= 7.x + OpenMPI-2.x/MPICH-3.x 


Build Instructions
=====================

untar the sources and set the following environment variables.
-------------------
```
export TAMM_SRC=$HOME/TAMM
export TAMM_INSTALL_PATH=$HOME/install  (or your preferred installation directory)
```

Step 1: Setup CMakeBuild
========================

```
cd $TAMM_SRC/CMakeBuild
mkdir build && cd build
CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH ..

make -j2
make install
```

Step 2: There are various build configurations given below. They mainly differ in the BLAS library that one wants to use.
=====================

General TAMM build using GCC (uses reference BLAS from netlib.org)
------------------------------------------------------------------
```
cd $TAMM_SRC
mkdir build && cd build

CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH -DCMAKE_PREFIX_PATH=$TAMM_INSTALL_PATH -DUSE_CUDA=ON -DCUDA_MAXREGCOUNT=128 ..

make -j3
make install
```

Building TAMM using GCC+MKL
----------------------------

### Set `INTEL_ROOT` accordingly

```
cd $TAMM_SRC
mkdir build && cd build

export INTEL_ROOT=/opt/intel/compilers_and_libraries_2019.0.117

export MKL_INC=$INTEL_ROOT/linux/mkl/include
export MKL_LIBS=$INTEL_ROOT/linux/mkl/lib/intel64

export TAMM_BLASLIBS="$MKL_LIBS/libmkl_intel_ilp64.a;$MKL_LIBS/libmkl_lapack95_ilp64.a;$MKL_LIBS/libmkl_blas95_ilp64.a;$MKL_LIBS/libmkl_intel_thread.a;$MKL_LIBS/libmkl_core.a;$INTEL_ROOT/linux/compiler/lib/intel64/libiomp5.a;-lpthread;-ldl"

CC=gcc CXX=g++ FC=gfortran cmake \
-DCBLAS_INCLUDE_DIRS=$MKL_INC \
-DLAPACKE_INCLUDE_DIRS=$MKL_INC \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH \
-DCMAKE_PREFIX_PATH=$TAMM_INSTALL_PATH \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS -DUSE_CUDA=ON -DCUDA_MAXREGCOUNT=128 ..
```

```
make -j3
make install
```

Build instructions for Summit (using GCC+ESSL)
----------------------------------------------

```
module load gcc/8.1.1
module load cmake/3.14.2
module load spectrum-mpi/10.3.0.1-20190611
module load essl/6.1.0-2
module load cuda/10.1.105
module load netlib-lapack/3.8.0
```

```
cd $TAMM_SRC
mkdir build && cd build
```

### The following paths may need to be adjusted if the modules change:

```
export ESSL_INC=/sw/summit/essl/6.1.0-2/essl/6.1/include
export TAMM_BLASLIBS="/sw/summit/essl/6.1.0-2/essl/6.1/lib64/libesslsmp.so"
export NETLIB_BLAS_LIBS="/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-8.1.1/netlib-lapack-3.8.0-moo2tlhxtaae4ij2vkhrkzcgu2pb3bmy/lib64"
```
```
CC=gcc CXX=g++ FC=gfortran cmake \
-DCBLAS_INCLUDE_DIRS=$ESSL_INC \
-DLAPACKE_INCLUDE_DIRS=$ESSL_INC \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH \
-DCMAKE_PREFIX_PATH=$TAMM_INSTALL_PATH \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS \
-DTAMM_CXX_FLAGS="-mcpu=power9" \
-DTAMM_EXTRA_LIBS="$NETLIB_BLAS_LIBS/liblapack.a" \
-DBLIS_CONFIG=power9 -DUSE_CUDA=ON -DCUDA_MAXREGCOUNT=128 -DCMAKE_BUILD_TYPE=Release ..

make -j2
make install

```


Step 3: Testing the perturbative triples code.
=====================

```
export TRIPLES_EXE=$TAMM_SRC/build/methods_stage/$TAMM_INSTALL_PATH/methods/CCSD_T_Fused_Fast

export OMP_NUM_THREADS=1
```

### General run:
```
export TRIPLES_INPUT=$TAMM_SRC/inputs/ozone.nwx

mpirun -n 2 $TRIPLES_EXE $TRIPLES_INPUT
```

### On Summit:
```
export ARMCI_DEFAULT_SHMMAX_UBOUND=65536

export PAMI_IBV_ENABLE_DCT=1
export PAMI_ENABLE_STRIPING=1
export PAMI_IBV_ADAPTER_AFFINITY=1
export PAMI_IBV_DEVICE_NAME="mlx5_0:1,mlx5_3:1"
export PAMI_IBV_DEVICE_NAME_1="mlx5_3:1,mlx5_0:1"

export GA_PROGRESS_RANKS_DISTRIBUTION_PACKED=1
export GA_NUM_PROGRESS_RANKS_PER_NODE=6

export TRIPLES_INPUT=$TAMM_SRC/inputs/ubiquitin_dgrtl.nwx

jsrun -a12 -c12 -g6 -r1 -dpacked $TRIPLES_EXE $TRIPLES_INPUT

