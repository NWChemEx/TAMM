
The prerequisites needed to build TAMM can be found [here](prerequisites.md).

TAMM uses [CMakeBuild Repository](https://github.com/NWChemEx-Project/CMakeBuild) to manage the build process and can be built as explained below.

First need to setup the [CMakeBuild Repository](https://github.com/NWChemEx-Project/CMakeBuild).


Build Instructions
=====================

```
export TAMM_SRC=$HOME/TAMM
export TAMM_INSTALL_PATH=$HOME/tamm_install
```

Step 1: Setup CMakeBuild
========================

```
git clone https://github.com/NWChemEx-Project/CMakeBuild.git
cd CMakeBuild
mkdir build && cd build
CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH ..

make install
```

Step 2: Choose Build Options
=====================

<!-- ### BLIS Options 
```
-DBLIS_CONFIG=arch
Ex: -DBLIS_CONFIG=haswell
If BLIS_CONFIG is not provided, the BLIS build will try to
auto-detect (only for x86_64 systems) the architecture.
``` -->

### CUDA Options 
```
-DUSE_CUDA=ON (OFF by Default)  
-DCUDA_MAXREGCOUNT=128 (set to 64 by Default)
```
### Optionally build with cuTensor support when USE_CUDA=ON  
```
-DUSE_CUTENSOR=ON -DCUTENSOR_INSTALL_PREFIX=/path/to/cutensor_install_prefix  
```

### GlobalArrays options. 
``` 
We only recommend building with MPI-PR (default) or OPENIB
 -DARMCI_NETWORK=OPENIB
````

### To enable DPCPP code path
``` 
-DUSE_DPCPP=ON (OFF by default, requires -DUSE_OPENMP=OFF) 
```

### CMake options for developers (optional)
```
-DUSE_GA_DEV=ON #Build GA's latest development code.

-DUSE_OPENMP=OFF (ON by default, also required to be ON when USE_CUDA=ON)
```


Step 3: Building TAMM
=====================

```
git clone https://github.com/NWChemEx-Project/TAMM.git $TAMM_SRC
cd $TAMM_SRC
# Checkout the branch you want to build
mkdir build && cd build
```

## In addition to the build options chosen in Step 2, there are various build configurations depending on the BLAS library one wants to use.

* **[Build using reference BLAS from NETLIB](install.md#build-using-reference-blas-from-netlib)**

* **[Build using Intel MKL](install.md#build-using-intel-mkl)**

* **[Build instructions for Summit using ESSL](install.md#build-instructions-for-summit-using-essl)**

* **[Build instructions for Cori](install.md#build-instructions-for-cori)**

* **[Build instructions for Theta](install.md#build-instructions-for-theta)**


## Build using reference BLAS from NETLIB

### To enable CUDA build, add `-DUSE_CUDA=ON`

```
cd $TAMM_SRC/build 
CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH ..

make -j3
make install
```

## Build using Intel MKL

### Set `INTEL_ROOT` accordingly

```
export INTEL_ROOT=/opt/intel/compilers_and_libraries_2019.0.117

export MKL_INC=$INTEL_ROOT/linux/mkl/include
export MKL_LIBS=$INTEL_ROOT/linux/mkl/lib/intel64

export TAMM_BLASLIBS="$MKL_LIBS/libmkl_intel_ilp64.a;$MKL_LIBS/libmkl_lapack95_ilp64.a;$MKL_LIBS/libmkl_blas95_ilp64.a;$MKL_LIBS/libmkl_intel_thread.a;$MKL_LIBS/libmkl_core.a;$INTEL_ROOT/linux/compiler/lib/intel64/libiomp5.a;-lpthread;-ldl"
```

### To enable CUDA build, add `-DUSE_CUDA=ON`

```
cd $TAMM_SRC/build 

CC=gcc CXX=g++ FC=gfortran cmake \
-DCBLAS_INCLUDE_DIRS=$MKL_INC \
-DLAPACKE_INCLUDE_DIRS=$MKL_INC \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS ..

make -j3
make install
```

## Build instructions for Summit using ESSL

```
module load gcc/8.1.1
module load cmake/3.17.3
module load spectrum-mpi/10.3.0.1-20190611
module load essl/6.1.0-2
module load cuda/10.1.105
module load netlib-lapack/3.8.0
```

```
The following paths may need to be adjusted if the modules change:

export ESSL_INC=/sw/summit/essl/6.1.0-2/essl/6.1/include
export TAMM_BLASLIBS="/sw/summit/essl/6.1.0-2/essl/6.1/lib64/libesslsmp.so"
export NETLIB_BLAS_LIBS="/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-8.1.1/netlib-lapack-3.8.0-moo2tlhxtaae4ij2vkhrkzcgu2pb3bmy/lib64"
```

### To enable CUDA build, add `-DUSE_CUDA=ON`

```
cd $TAMM_SRC/build

CC=gcc CXX=g++ FC=gfortran cmake \
-DCBLAS_INCLUDE_DIRS=$ESSL_INC \
-DLAPACKE_INCLUDE_DIRS=$ESSL_INC \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS \
-DTAMM_CXX_FLAGS="-mcpu=power9" \
-DTAMM_EXTRA_LIBS="$NETLIB_BLAS_LIBS/liblapack.a" \
-DBLIS_CONFIG=power9 ..

make -j3
make install
```

## Build instructions for Cori

```
module unload PrgEnv-intel/6.0.5
module load PrgEnv-gnu/6.0.5
module swap gcc/8.2.0 
module swap craype/2.5.18
module swap cray-mpich/7.7.6 
module load cmake
module load cuda/10.1.168
```

```
export CRAYPE_LINK_TYPE=dynamic
export INTEL_ROOT=/opt/intel/compilers_and_libraries_2019.3.199

export MKL_INC=$INTEL_ROOT/linux/mkl/include
export MKL_LIBS=$INTEL_ROOT/linux/mkl/lib/intel64
export TAMM_BLASLIBS="$MKL_LIBS/libmkl_intel_ilp64.a;$MKL_LIBS/libmkl_lapack95_ilp64.a;$MKL_LIBS/libmkl_blas95_ilp64.a;$MKL_LIBS/libmkl_gnu_thread.a;$MKL_LIBS/libmkl_core.a;-lgomp;-lpthread;-ldl"
```

### To enable CUDA build, add `-DUSE_CUDA=ON`

```
cd $TAMM_SRC/build

CC=cc CXX=CC FC=ftn cmake -DCBLAS_INCLUDE_DIRS=$MKL_INC \
-DLAPACKE_INCLUDE_DIRS=$MKL_INC \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS ..

make -j3
make install
```

## Build instructions for Theta

```
module unload darshan xalt perftools-base
module swap PrgEnv-intel PrgEnv-gnu
module unload darshan xalt perftools-base
module load cmake
```

```
export CRAYPE_LINK_TYPE=dynamic
export INTEL_ROOT=/theta-archive/intel/compilers_and_libraries_2019.5.281
export MKL_INC=$INTEL_ROOT/linux/mkl/include
export MKL_LIBS=$INTEL_ROOT/linux/mkl/lib/intel64
export TAMM_BLASLIBS="$MKL_LIBS/libmkl_intel_ilp64.a;$MKL_LIBS/libmkl_lapack95_ilp64.a;$MKL_LIBS/libmkl_blas95_ilp64.a;$MKL_LIBS/libmkl_intel_thread.a;$MKL_LIBS/libmkl_core.a;$INTEL_ROOT/linux/compiler/lib/intel64/libiomp5.a;-lpthread;-ldl"
```

```
cd $TAMM_SRC/build

CC=cc CXX=CC FC=ftn cmake -DCBLAS_INCLUDE_DIRS=$MKL_INC -DLAPACKE_INCLUDE_DIRS=$MKL_INC -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH -DCBLAS_LIBRARIES=$TAMM_BLASLIBS -DLAPACKE_LIBRARIES=$TAMM_BLASLIBS ..

make -j3
make install
```

Running the code
=====================
- SCF  
`export TAMM_EXE=$TAMM_SRC/build/methods_stage/$TAMM_INSTALL_PATH/methods/HartreeFock_TAMM`  

- CCSD  
`export TAMM_EXE=$TAMM_SRC/build/methods_stage/$TAMM_INSTALL_PATH/methods/CD_CCSD_CS`  
`export TAMM_EXE=$TAMM_SRC/build/methods_stage/$TAMM_INSTALL_PATH/methods/CD_CCSD_OS`

- CCSD(T)   
`export TAMM_EXE=$TAMM_SRC/build/methods_stage/$TAMM_INSTALL_PATH/methods/CCSD_T_Fused`

### General run:
```
export OMP_NUM_THREADS=1
export TAMM_INPUT=$TAMM_SRC/inputs/ozone.nwx

mpirun -n 2 $TAMM_EXE $TAMM_INPUT
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

export TAMM_INPUT=$TAMM_SRC/inputs/ubiquitin_dgrtl.nwx

jsrun -a12 -c12 -g6 -r1 -dpacked $TAMM_EXE $TAMM_INPUT
