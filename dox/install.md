
The prerequisites needed to build TAMM can be found [here](prerequisites.md).

TAMM uses [CMakeBuild Repository](https://github.com/NWChemEx-Project/CMakeBuild) to manage the build process and can be built as explained below.

First need to setup the [CMakeBuild Repository](https://github.com/NWChemEx-Project/CMakeBuild).


```
export TAMM_INSTALL_PATH=/opt/NWChemEx/install

git clone https://github.com/NWChemEx-Project/CMakeBuild.git
cd CMakeBuild
mkdir build && cd build
CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH ..

make -j2
make install
```

General TAMM build using GCC
----------------------------
```
git clone https://github.com/NWChemEx-Project/TAMM.git
cd TAMM
# Checkout the branch you want to build
mkdir build && cd build

CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH ..

#BLIS Options 
-DUSE_BLIS=ON
-DBLIS_CONFIG=arch
Ex: -DBLIS_CONFIG=haswell
If BLIS_CONFIG is not provided, the BLIS build will try to
auto-detect (only for x86_64 systems) the architecture.

#CUDA Options
[-DNWX_CUDA=ON] #OFF by Default

#GlobalArrays options
[-DARMCI_NETWORK=MPI-TS] #Default is MPI-PR

#CMake options for developers (optional)
-DCMAKE_CXX_FLAGS "-fsanitize=address -fsanitize=leak -fsanitize=pointer-compare -fsanitize=pointer-subtract -fsanitize=undefined"


make -j3
make install
```

Building TAMM using GCC+MKL
----------------------------

Set `TAMM_INSTALL_PATH` and `INTEL_ROOT` accordingly

```
export TAMM_INSTALL_PATH=$HOME/NWChemEx/install
export INTEL_ROOT=/opt/intel/compilers_and_libraries_2019.0.117

export MKL_INC=$INTEL_ROOT/linux/mkl/include
export MKL_LIBS=$INTEL_ROOT/linux/mkl/lib/intel64

export TAMM_BLASLIBS="$MKL_LIBS/libmkl_intel_ilp64.a;$MKL_LIBS/libmkl_lapack95_ilp64.a;$MKL_LIBS/libmkl_blas95_ilp64.a;$MKL_LIBS/libmkl_intel_thread.a;$MKL_LIBS/libmkl_core.a;$INTEL_ROOT/linux/compiler/lib/intel64/libiomp5.a;-lpthread;-ldl"

CC=gcc CXX=g++ FC=gfortran cmake \
-DCBLAS_INCLUDE_DIRS=$MKL_INC \
-DLAPACKE_INCLUDE_DIRS=$MKL_INC \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS ..
```

To use `ScaLAPACK`, change `TAMM_BLASLIBS` and the `cmake` line as shown below:

```
export TAMM_BLASLIBS="$MKL_LIBS/libmkl_scalapack_ilp64.a;$MKL_LIBS/libmkl_intel_ilp64.a;$MKL_LIBS/libmkl_lapack95_ilp64.a;$MKL_LIBS/libmkl_blas95_ilp64.a;$MKL_LIBS/libmkl_intel_thread.a;$MKL_LIBS/libmkl_core.a;$MKL_LIBS/libmkl_blacs_openmpi_ilp64.a;$INTEL_ROOT/linux/compiler/lib/intel64/libiomp5.a;-lpthread;-ldl"

CC=gcc CXX=g++ FC=gfortran cmake \
-DCBLAS_INCLUDE_DIRS=$MKL_INC \
-DLAPACKE_INCLUDE_DIRS=$MKL_INC \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS \
-DSCALAPACK_LIBRARIES=$TAMM_BLASLIBS -DSCALAPACK=ON ..
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
```

```
The following paths may need to be adjusted if the modules change:

export TAMM_INSTALL_PATH=$HOME/NWChemEx/install
export ESSL_INC=/sw/summit/essl/6.1.0-2/essl/6.1/include
export TAMM_BLASLIBS="/sw/summit/essl/6.1.0-2/essl/6.1/lib64/libesslsmp6464.so"
```
```
CC=gcc CXX=g++ FC=gfortran cmake \
-DCBLAS_INCLUDE_DIRS=$ESSL_INC \
-DLAPACKE_INCLUDE_DIRS=$ESSL_INC \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS \
-DTAMM_CXX_FLAGS="-mcpu=power9" \
[-DUSE_BLIS=ON -DBLIS_CONFIG=power9 ] ..

To enable CUDA build, add -DNWX_CUDA=ON

```

```
For Scalapack build, the following need to be changed above:

module load netlib-scalapack

export TAMM_BLASLIBS="/sw/summit/essl/6.1.0-2/essl/6.1/lib64/libesslsmp.so;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-8.1.1/netlib-scalapack-2.0.2-re7if5fomjhxgqa5morvan7mptnkihdx/lib/libscalapack.so"

Add -DSCALAPACK=ON to the cmake line.


```

Build instructions for Cori
----------------------------

```
module unload PrgEnv-intel/6.0.5
module load PrgEnv-gnu/6.0.5
module swap gcc/8.2.0 
module swap craype/2.5.18
module swap cray-mpich/7.7.6 
module load cmake/3.14.4 
module load cuda/10.1.168

```

- `NOTE:` CMakeBuild repository should be built with the following compiler options.
  - Remove the compiler options from the cmake line or change them to:  
    CC=cc CXX=CC FC=ftn 

 
```
export CRAYPE_LINK_TYPE=dynamic

export TAMM_INSTALL_PATH=$HOME/code/NWChemEx/install
export INTEL_ROOT=/opt/intel/compilers_and_libraries_2019.3.199

export MKL_INC=$INTEL_ROOT/linux/mkl/include
export MKL_LIBS=$INTEL_ROOT/linux/mkl/lib/intel64
export TAMM_BLASLIBS="$MKL_LIBS/libmkl_intel_ilp64.a;$MKL_LIBS/libmkl_lapack95_ilp64.a;$MKL_LIBS/libmkl_blas95_ilp64.a;$MKL_LIBS/libmkl_gnu_thread.a;$MKL_LIBS/libmkl_core.a;-lgomp;-lpthread;-ldl"

CC=cc CXX=CC FC=ftn cmake -DCBLAS_INCLUDE_DIRS=$MKL_INC \
-DLAPACKE_INCLUDE_DIRS=$MKL_INC \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS ..

To enable CUDA build, add -DNWX_CUDA=ON

```

