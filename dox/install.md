
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

### To enable DPCPP code path
``` 
-DUSE_DPCPP=ON (OFF by default, requires -DUSE_OPENMP=OFF) 
```

### CMake options for developers (optional)
```
-DUSE_GA_PROFILER=ON #Enable GA's profiling feature (GCC Only).

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

* **[Building the DPCPP code path using Intel OneAPI SDK](install.md#build-dpcpp-code-path-using-intel-oneapi-sdk)**

## Build using reference BLAS from NETLIB

### To enable CUDA build, add `-DUSE_CUDA=ON`

```
cd $TAMM_SRC/build 
CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH ..

make -j3
make install
```

## Build using Intel MKL

### Set `MKLROOT` accordingly

```
export MKLROOT=/opt/intel/compilers_and_libraries_2019.0.117/linux/mkl
```

### To enable CUDA build, add `-DUSE_CUDA=ON`

```
cd $TAMM_SRC/build 

CC=gcc CXX=g++ FC=gfortran cmake -DBLAS_VENDOR=IntelMKL -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH ..

make -j3
make install
```

## Build instructions for Summit using ESSL

```
module load gcc/8.3.0
module load cmake/3.17.3
module load essl/6.1.0-2
module load cuda/10.1.105
module load netlib-lapack/3.8.0
```

```
The following paths may need to be adjusted if the modules change:

export ESSLROOT=/sw/summit/essl/6.1.0-2/essl/6.1
export NETLIB_BLAS_LIBS="/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-8.1.1/netlib-lapack-3.8.0-moo2tlhxtaae4ij2vkhrkzcgu2pb3bmy/lib64"
```

### To enable CUDA build, add `-DUSE_CUDA=ON`

```
cd $TAMM_SRC/build

CC=gcc CXX=g++ FC=gfortran cmake \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH \
-DTAMM_EXTRA_LIBS=$NETLIB_BLAS_LIBS/liblapack.a \
-DBLIS_CONFIG=power9 \
-DBLAS_VENDOR=IBMESSL ..

make -j3
make install
```

## Build instructions for Cori

```
module unload PrgEnv-intel/6.0.5
module load PrgEnv-gnu/6.0.5
module swap gcc/8.3.0 
module swap craype/2.5.18
module swap cray-mpich/7.7.6 
module load cmake
module load cuda/10.1.168
```

```
export CRAYPE_LINK_TYPE=dynamic
export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
```

### To enable CUDA build, add `-DUSE_CUDA=ON`

```
cd $TAMM_SRC/build

CC=cc CXX=CC FC=ftn cmake -DBLAS_VENDOR=IntelMKL -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH ..

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
export MKLROOT=/theta-archive/intel/compilers_and_libraries_2019.5.281/linux/mkl
```

```
cd $TAMM_SRC/build

CC=cc CXX=CC FC=ftn cmake -DBLAS_VENDOR=IntelMKL -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH ..

make -j3
make install
```
## Build DPCPP code path using Intel OneAPI SDK

- `IMPORTANT:` `OneAPI compilers currently only work with CMake versions <= 3.18`
- `MPI:` Tested using `MPICH`. Use `impi` branch of `CMakeBuild` repository if using `IntelMPI` from the OneAPI SDK.

- Set `MKLROOT` and `DPCPP_ROOT` accordingly

```
export MKLROOT=/opt/oneapi/mkl/latest
export DPCPP_ROOT=/opt/oneapi/compiler/latest/linux
```

- Set ROOT dir of the GCC installation (need gcc >= v8.3)
```
export GCCROOT=/opt/gcc8.3
```

```
cd $TAMM_SRC/build 

CC=icx CXX=dpcpp FC=ifx cmake \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH \
-DMPIEXEC_EXECUTABLE=mpiexec -DUSE_GA_AT=ON \
-DUSE_OPENMP=OFF -DBLAS_VENDOR=IntelMKL -DUSE_DPCPP=ON -DGCCROOT=$GCCROOT \
-DTAMM_CXX_FLAGS="-fno-sycl-early-optimizations -fsycl -fsycl-targets=spir64_gen-unknown-linux-sycldevice -Xsycl-target-backend '-device skl'"
```

`TAMM_CXX_FLAGS` shown above build for the Intel GEN9 GPU. Please change the `-device skl` flag as needed for other GENX devices.

```
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
export TAMM_INPUT=$TAMM_SRC/inputs/ozone.json

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

export TAMM_INPUT=$TAMM_SRC/inputs/ubiquitin_dgrtl.json

jsrun -a12 -c12 -g6 -r1 -dpacked $TAMM_EXE $TAMM_INPUT
