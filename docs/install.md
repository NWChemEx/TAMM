
The prerequisites needed to build this repository can be found [here](prerequisites.md).

Build Instructions
=====================

```
export REPO_ROOT_PATH=$HOME/TAMM
export REPO_INSTALL_PATH=$HOME/tamm_install
```

Choose Build Options
============================

### CUDA Options
```
-DUSE_CUDA=ON (OFF by default)  
-DGPU_ARCH=70 (GPU arch is detected automatically, only set this option if need to override)
-DCUDA_MAXREGCOUNT=64 (128 by default)
```

### To enable DPCPP code path
``` 
-DUSE_DPCPP=ON
```

### CMake options for developers (optional)
```
-DUSE_GA_PROFILER=ON #Enable GA's profiling feature (GCC Only).

-DBUILD_LIBINT=ON (OFF by default)
```


Building this repository
========================

```
git clone <repo-url> $REPO_ROOT_PATH
cd $REPO_ROOT_PATH
# Checkout the branch you want to build
mkdir build && cd build
```

## In addition to the build options chosen, there are various build configurations depending on the BLAS library one wants to use.

* **[Default build using BLIS and NETLIB LAPACK](install.md#default-build-using-blis-and-netlib-lapack)**

* **[Default build on MACOS](install.md#default-build-on-macos)**

* **[Build using Intel MKL](install.md#build-using-intel-mkl)**

* **[Build instructions for Summit using ESSL](install.md#build-instructions-for-summit-using-essl)**

* **[Build instructions for Summit using ESSL and UPC++](install.md#build-instructions-for-summit-using-essl-and-upc++)**

* **[Build instructions for Crusher](install.md#build-instructions-for-crusher)**

* **[Build instructions for Cori](install.md#build-instructions-for-cori)**

* **[Build instructions for Perlmutter and Polaris](install.md#build-instructions-for-perlmutter-and-polaris)**

* **[Build instructions for Theta](install.md#build-instructions-for-theta)**

* **[Building the DPCPP code path using Intel OneAPI SDK](install.md#build-dpcpp-code-path-using-intel-oneapi-sdk)**

## Default build using BLIS and NETLIB LAPACK

### To enable CUDA build, add `-DUSE_CUDA=ON`

```
cd $REPO_ROOT_PATH/build 
CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH ..

make -j3
make install
```
## Default build on MACOS

### NOTE: The prerequisites for `MACOS` can be installed using `brew` as detailed [here](prerequisites.md#on-mac-osx).

```
cd $REPO_ROOT_PATH/build 
CC=gcc-10 CXX=g++-10 FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH ..

make -j3
make install
```

## Build using Intel MKL

### To enable CUDA build, add `-DUSE_CUDA=ON`

```
cd $REPO_ROOT_PATH/build 

CC=gcc CXX=g++ FC=gfortran cmake -DLINALG_VENDOR=IntelMKL \
-DLINALG_PREFIX=/opt/intel/mkl \
-DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH ..

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
cd $REPO_ROOT_PATH/build

CC=gcc CXX=g++ FC=gfortran cmake \
-DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH \
-DBLIS_CONFIG=power9 \
-DLINALG_VENDOR=IBMESSL -DUSE_CUDA=ON \
-DLINALG_PREFIX=/sw/summit/essl/6.3.0/essl/6.3 ..

make -j3
make install
```

## Build instructions for Summit using ESSL and UPC++

### Note that UPC++ support is experimental.

```
module load gcc
module load cmake
module load essl/6.3.0
module load cuda
module load upcxx
```


```
cd $REPO_ROOT_PATH/build

UPCXX_CODEMODE=O3 CC=gcc CXX=upcxx FC=gfortran cmake \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH \
-DBLIS_CONFIG=power9 \
-DLINALG_VENDOR=IBMESSL \
-DLINALG_PREFIX=/sw/summit/essl/6.3.0/essl/6.3 \
-DUSE_CUDA=ON \
-DUSE_UPCXX=ON ..

UPCXX_CODEMODE=O3 make -j3
UPCXX_CODEMODE=O3 make install
```


## Build instructions for Crusher

```
module load cmake
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm
module unload cray-libsci
export CRAYPE_LINK_TYPE=dynamic
export HDF5_USE_FILE_LOCKING=FALSE
```

```
cd $REPO_ROOT_PATH/build

CC=cc CXX=CC FC=ftn cmake \
-DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH \
-DGPU_ARCH=gfx90a \
-DUSE_HIP=ON -DROCM_ROOT=$ROCM_PATH \
-DGCCROOT=/opt/cray/pe/gcc/10.3.0/snos ..

make -j3
make install
```

## Build instructions for Cori

```
export CRAYPE_LINK_TYPE=dynamic
export HDF5_USE_FILE_LOCKING=FALSE
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
cd $REPO_ROOT_PATH/build

CC=cc CXX=CC FC=ftn cmake -DLINALG_VENDOR=IntelMKL \
-DLINALG_PREFIX=/opt/intel/mkl \
-DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH -H$REPO_ROOT_PATH

make -j3
make install
```

### GPU build

```
module purge && module load cgpu cuda gcc openmpi cmake
```

```
cd $REPO_ROOT_PATH/build

CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH \
-DUSE_CUDA=ON -DGPU_ARCH=70 -H$REPO_ROOT_PATH

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
cd $REPO_ROOT_PATH/build

cmake -DUSE_CUDA=ON -DBLIS_CONFIG=generic \
-DCMAKE_PREFIX_PATH=$CUBLAS_PATH/lib \
-DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH ..

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
cd $REPO_ROOT_PATH/build

CC=cc CXX=CC FC=ftn cmake -DLINALG_VENDOR=IntelMKL \
-DLINALG_PREFIX=/opt/intel/mkl \
-DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH ..

make -j3
make install
```

## Build DPCPP code path using Intel OneAPI SDK

- `MPI:` Only tested using `MPICH`.
- Set `DPCPP_ROOT` accordingly

```
export DPCPP_ROOT=/opt/oneapi/compiler/latest/linux
```

- Set ROOT dir of the GCC installation (need gcc >= v9.1)
```
export GCC_ROOT_PATH=/opt/gcc-9.1.0
```

```
cd $REPO_ROOT_PATH/build 

CC=icx CXX=dpcpp FC=ifx cmake \
-DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH \
-DLINALG_VENDOR=IntelMKL -DLINALG_PREFIX=/opt/oneapi/mkl/latest \
-DUSE_DPCPP=ON -DGCCROOT=$GCC_ROOT_PATH \
-DTAMM_CXX_FLAGS="-fsycl-device-code-split=per_kernel"
```

```
make -j3
make install
```


