
The prerequisites needed to build TAMM can be found [here](prerequisites.md).

TAMM uses [CMakeBuild Repository](https://github.com/NWChemEx-Project/CMakeBuild) to manage the build process and can be built as explained below.

First need to setup the [CMakeBuild Repository](https://github.com/NWChemEx-Project/CMakeBuild).


```
export TAMM_INSTALL_PATH=/opt/NWChemEx/install

git clone https://github.com/NWChemEx-Project/CMakeBuild.git
cd CMakeBuild
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH/CMakeBuild \
-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran ..

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

cmake \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH/TAMM \
-DCMAKE_PREFIX_PATH=$TAMM_INSTALL_PATH/CMakeBuild \
-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran ..

#CUDA Options
[-DTAMM_CUDA=ON] #Disabled by Default
#GlobalArrays options
[-DARMCI_NETWORK=MPI3] #Default is MPI-PR

#CMake options for developers (optional)
-DCMAKE_CXX_FLAGS "-fsanitize=address -fsanitize=leak -fsanitize=pointer-compare -fsanitize=pointer-subtract -fsanitize=undefined"
-DCMAKE_BUILD_TYPE=Debug
-DCMAKE_BUILD_TYPE=RELWITHDEBINFO

make -j3
make install
```

Building TAMM using GCC+MKL
----------------------------

Set `TAMM_INSTALL_PATH` and `INTEL_ROOT` accordingly

```
export TAMM_INSTALL_PATH=/opt/NWChemEx/install
export INTEL_ROOT=/opt/intel/compilers_and_libraries_2019.0.117

export MKL_INC=$INTEL_ROOT/linux/mkl/include
export MKL_LIBS=$INTEL_ROOT/linux/mkl/lib/intel64

export TAMM_BLASLIBS="$MKL_LIBS/libmkl_intel_ilp64.a;$MKL_LIBS/libmkl_lapack95_ilp64.a;$MKL_LIBS/libmkl_blas95_ilp64.a;$MKL_LIBS/libmkl_intel_thread.a;$MKL_LIBS/libmkl_core.a;$INTEL_ROOT/linux/compiler/lib/intel64/libiomp5.a;-lpthread;-ldl"

cmake \
-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran \
-DCBLAS_INCLUDE_DIRS=$MKL_INC \
-DLAPACKE_INCLUDE_DIRS=$MKL_INC \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH/TAMMGCCMKL \
-DCMAKE_PREFIX_PATH=$TAMM_INSTALL_PATH/CMakeBuild \
-DTAMM_CXX_FLAGS="-DMKL_ILP64 -m64" \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS ..
```

To use `ScaLAPACK`, change `TAMM_BLASLIBS` and the `cmake` line as shown below:

```
export TAMM_BLASLIBS="$MKL_LIBS/libmkl_scalapack_ilp64.a;$MKL_LIBS/libmkl_intel_ilp64.a;$MKL_LIBS/libmkl_lapack95_ilp64.a;$MKL_LIBS/libmkl_blas95_ilp64.a;$MKL_LIBS/libmkl_intel_thread.a;$MKL_LIBS/libmkl_core.a;$MKL_LIBS/libmkl_blacs_openmpi_ilp64.a;$INTEL_ROOT/linux/compiler/lib/intel64/libiomp5.a;-lpthread;-ldl"

cmake \
-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran \
-DCBLAS_INCLUDE_DIRS=$MKL_INC \
-DLAPACKE_INCLUDE_DIRS=$MKL_INC \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH/TAMMGCCMKL \
-DCMAKE_PREFIX_PATH=$TAMM_INSTALL_PATH/CMakeBuild \
-DTAMM_CXX_FLAGS="-DMKL_ILP64 -m64 -DSCALAPACK" \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS \
-DSCALAPACK_LIBRARIES=$TAMM_BLASLIBS ..
```

```
make -j3
make install
```

Build instructions for Summit (using GCC+ESSL)
----------------------------------------------

```
export TAMM_INSTALL_PATH=/opt/NWChemEx/install
export ESSL_INC=/sw/summit/essl/6.1.0-2/essl/6.1/include
export TAMM_BLASLIBS="/sw/summit/essl/6.1.0-2/essl/6.1/lib64/libesslsmp6464.so"
export NETLIB_BLAS_LIBS="/opt/lapacke/lib64"

cmake \
-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran \
-DCBLAS_INCLUDE_DIRS=$ESSL_INC \
-DLAPACKE_INCLUDE_DIRS=$ESSL_INC \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH/TAMMESSL \
-DCMAKE_PREFIX_PATH=$TAMM_INSTALL_PATH/CMakeBuild \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS \
-DTAMM_CXX_FLAGS="-m64 -mtune=native -ffast-math" \
-DTAMM_EXTRA_LIBS="$NETLIB_BLAS_LIBS/liblapacke.a;$NETLIB_BLAS_LIBS/liblapack.a" ..

```

Build instructions for Cori
----------------------------

```
module unload PrgEnv-intel/6.0.4
module load PrgEnv-gnu/6.0.4
module load gcc/7.3.0 cray-mpich/7.7.0
module load cmake/3.11.4
```

- CMakeBuild repository should be built with the following compiler options.
  - Remove the compiler options from the cmake line or change them to:  
 -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_Fortran_COMPILER=ftn

 
```
export TAMM_INSTALL_PATH=/global/homes/p/user/code/NWChemEx/install
export INTEL_ROOT=/opt/intel/compilers_and_libraries_2018.1.163

export MKL_INC=$INTEL_ROOT/linux/mkl/include
export MKL_LIBS=$INTEL_ROOT/linux/mkl/lib/intel64
export TAMM_BLASLIBS="$MKL_LIBS/libmkl_intel_ilp64.a;$MKL_LIBS/libmkl_lapack95_ilp64.a;$MKL_LIBS/libmkl_blas95_ilp64.a;$MKL_LIBS/libmkl_gnu_thread.a;$MKL_LIBS/libmkl_core.a;-lgomp;-lpthread;-ldl"

cmake -DCBLAS_INCLUDE_DIRS=$MKL_INC \
-DLAPACKE_INCLUDE_DIRS=$MKL_INC \
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH/TAMMGCCMKL \
-DCMAKE_PREFIX_PATH=$TAMM_INSTALL_PATH/CMakeBuild \
-DTAMM_CXX_FLAGS="-DMKL_ILP64 -m64" \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS ..

```


