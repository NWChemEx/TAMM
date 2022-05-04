
The prerequisites needed to build TAMM can be found [here](prerequisites.md).

Build Instructions
=====================

```
export TAMM_SRC=$HOME/TAMM
export TAMM_INSTALL_PATH=$HOME/tamm_install
export REPO_URL=https://github.com/NWChemEx-Project
```

Choose Build Options
============================

### CUDA Options 
```
-DUSE_CUDA=ON (OFF by default)  
-DCUDA_MAXREGCOUNT=128 (64 by default)
-DGPU_ARCH=70 (GPU arch is detected automatically, only set this option if need to override)
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


Building TAMM
=====================

```
git clone $REPO_URL/TAMM.git $TAMM_SRC
cd $TAMM_SRC
# Checkout the branch you want to build
mkdir build && cd build
```

## In addition to the build options chosen, there are various build configurations depending on the BLAS library one wants to use.

* **[Default build using BLIS and NETLIB LAPACK](install.md#default-build-using-blis-and-netlib-lapack)**

* **[Default build on MACOS](install.md#default-build-on-macos)**

* **[Build using Intel MKL](install.md#build-using-intel-mkl)**

* **[Build instructions for Summit using ESSL](install.md#build-instructions-for-summit-using-essl)**

* **[Build instructions for Crusher](install.md#build-instructions-for-crusher)**

* **[Build instructions for Cori](install.md#build-instructions-for-cori)**

* **[Build instructions for Perlmutter and Polaris](install.md#build-instructions-for-perlmutter-and-polaris)**

* **[Build instructions for Theta](install.md#build-instructions-for-theta)**

* **[Building the DPCPP code path using Intel OneAPI SDK](install.md#build-dpcpp-code-path-using-intel-oneapi-sdk)**

## Default build using BLIS and NETLIB LAPACK

### To enable CUDA build, add `-DUSE_CUDA=ON`

```
cd $TAMM_SRC/build 
CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH ..

make -j3
make install
```
## Default build on MACOS

### NOTE: We only support building with GNU compilers on `MACOS`. They can be installed using brew as detailed [here](CMake_Build_Notes.md#on-mac-osx).

```
cd $TAMM_SRC/build 
CC=gcc-10 CXX=g++-10 FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH ..

make -j3
make install
```

## Build using Intel MKL

### To enable CUDA build, add `-DUSE_CUDA=ON`

```
cd $TAMM_SRC/build 

CC=gcc CXX=g++ FC=gfortran cmake -DLINALG_VENDOR=IntelMKL \
-DLINALG_PREFIX=/opt/intel/mkl \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH ..

make -j3
make install
```

## Build instructions for Summit using ESSL

```
module load gcc
module load cmake
module load essl/6.3.0
module load cuda
```


```
cd $TAMM_SRC/build

CC=gcc CXX=g++ FC=gfortran cmake \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH \
-DBLIS_CONFIG=power9 \
-DLINALG_VENDOR=IBMESSL -DUSE_CUDA=ON \
-DLINALG_PREFIX=/sw/summit/essl/6.3.0/essl/6.3 ..

make -j3
make install
```

## Build instructions for Crusher

```
module load cmake
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm
module unload cray-libsci
export CRAYPE_LINK_TYPE=dynamic
```

```
cd $TAMM_SRC/build

CC=cc CXX=CC FC=ftn cmake \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH \
-DGPU_ARCH=gfx90a \
-DUSE_OPENMP=OFF -DUSE_HIP=ON -DROCM_ROOT=$ROCM_PATH \
-DGCCROOT=/opt/cray/pe/gcc/10.3.0/snos ..

make -j3
make install
```

## Build instructions for Cori

```
export CRAYPE_LINK_TYPE=dynamic
export HDF5_USE_FILE_LOCKING="FALSE"
```

### CPU only build

```
module unload PrgEnv-intel/6.0.5
module load PrgEnv-gnu/6.0.5
module swap gcc/8.3.0 
module swap craype/2.5.18
module swap cray-mpich/7.7.6
module unload cmake
module load cmake
```

```
cd $TAMM_SRC/build

CC=cc CXX=CC FC=ftn cmake -DLINALG_VENDOR=IntelMKL \
-DLINALG_PREFIX=/opt/intel/mkl \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH -H$TAMM_SRC

make -j3
make install
```

### GPU build

```
module purge && module load cgpu cuda gcc openmpi cmake
```

```
cd $TAMM_SRC/build

CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH \
-DUSE_CUDA=ON -DGPU_ARCH=70 -H$TAMM_SRC

make -j3
make install
```

## Build instructions for Perlmutter and Polaris

```
module load PrgEnv-gnu
module load cudatoolkit
module load cpe-cuda (perlmutter only)
module load gcc/9.3.0
module load cmake
export CRAYPE_LINK_TYPE=dynamic
```

```
##ADJUST CUBLAS_PATH IF NEEDED

export CUBLAS_PATH=$CUDA_HOME/../../math_libs/11.5/lib64
export CPATH=$CPATH:$CUBLAS_PATH/include
```

```
cd $TAMM_SRC/build

cmake -DUSE_CUDA=ON -DBLIS_CONFIG=generic \
-DCMAKE_PREFIX_PATH=$CUBLAS_PATH/lib \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH ..

make -j3
make install
```

## Build instructions for Theta

```
module unload PrgEnv-intel/6.0.7
module load PrgEnv-gnu/6.0.7
module unload cmake
module load cmake
export CRAYPE_LINK_TYPE=dynamic
```

```
cd $TAMM_SRC/build

CC=cc CXX=CC FC=ftn cmake -DLINALG_VENDOR=IntelMKL \
-DLINALG_PREFIX=/opt/intel/mkl \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH ..

make -j3
make install
```

## Build DPCPP code path using Intel OneAPI SDK

- `MPI:` Only tested using `MPICH`.
- Set `DPCPP_ROOT` accordingly

```
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
-DMPIEXEC_EXECUTABLE=mpiexec -DUSE_OPENMP=OFF \
-DLINALG_VENDOR=IntelMKL -DLINALG_PREFIX=/opt/oneapi/mkl/latest \
-DUSE_DPCPP=ON -DGCCROOT=$GCCROOT \
-DTAMM_CXX_FLAGS="-fsycl-device-code-split=per_kernel"
```

```
make -j3
make install
```

Running the code
=====================
- SCF  
`export TAMM_EXE=$TAMM_INSTALL_PATH/bin/HartreeFock`  

- CCSD  
`export TAMM_EXE=$TAMM_INSTALL_PATH/bin/CD_CCSD`  

- CCSD(T)   
`export TAMM_EXE=$TAMM_INSTALL_PATH/bin/CCSD_T`

### General run:
```
export OMP_NUM_THREADS=1
export TAMM_INPUT=$TAMM_SRC/inputs/ozone.json

mpirun -n 2 $TAMM_EXE $TAMM_INPUT
```

### On Summit:
```
export PAMI_IBV_ENABLE_DCT=1
export PAMI_ENABLE_STRIPING=1
export PAMI_IBV_ADAPTER_AFFINITY=1
export PAMI_IBV_DEVICE_NAME="mlx5_0:1,mlx5_3:1"
export PAMI_IBV_DEVICE_NAME_1="mlx5_3:1,mlx5_0:1"

export GA_PROGRESS_RANKS_DISTRIBUTION_PACKED=1
export GA_NUM_PROGRESS_RANKS_PER_NODE=6

export TAMM_INPUT=$TAMM_SRC/inputs/ubiquitin_dgrtl.json

jsrun -a12 -c12 -g6 -r1 -dpacked $TAMM_EXE $TAMM_INPUT
```
