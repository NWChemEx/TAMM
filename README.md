
The prerequisites needed to build TAMM can be found [here](dox/prerequisites.md) 

- TAMM uses [NWChemEx Base Repository](https://github.com/NWChemEx-Project/CMakeBuild) to manage the build process and can be built as follows:

```
TAMM_INSTALL_PATH=/opt/NWChemEx/install

git clone https://github.com/NWChemEx-Project/CMakeBuild.git
cd CMakeBuild
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH/CMakeBuild \
-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran

make -j2
make install
```

```
git clone https://github.com/NWChemEx-Project/TAMM.git
cd TAMM
# Checkout the branch you want to build
mkdir build && cd build

cmake .. \ 
-DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH/TAMM \
-DCMAKE_PREFIX_PATH=$TAMM_INSTALL_PATH/CMakeBuild \
-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran

#GlobalArrays options
[-DARMCI_NETWORK=MPI3] #Default is MPI-PR

#CMake options for developers (optional)
-DCMAKE_CXX_FLAGS "-fsanitize=address -fsanitize=leak -fsanitize=pointer-compare -fsanitize=pointer-subtract -fsanitize=undefined"
[-DCMAKE_BUILD_TYPE=Debug (OR) -DCMAKE_BUILD_TYPE=RELWITHDEBINFO]

make -j3
make install
```

Building TAMM using GCC+MKL
----------------------------
```
export INTEL_ROOT=/opt/intel/compilers_and_libraries_2019.0.117
export MKL_INC=$INTEL_ROOT/linux/mkl/include
export MKL_LIBS=$INTEL_ROOT/linux/mkl/lib/intel64
export TAMM_BLASLIBS="$MKL_LIBS/libmkl_intel_ilp64.a;$MKL_LIBS/libmkl_lapack95_ilp64.a;$MKL_LIBS/libmkl_blas95_ilp64.a;$MKL_LIBS/libmkl_intel_thread.a;$MKL_LIBS/libmkl_core.a;$INTEL_ROOT/linux/compiler/lib/intel64/libiomp5.a;-lpthread;-ldl"

cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran -DCBLAS_INCLUDE_DIRS=$MKL_INC -DLAPACKE_INCLUDE_DIRS=$MKL_INC -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH/TAMMGCCMKL -DCMAKE_PREFIX_PATH=$TAMM_INSTALL_PATH/CMakeBuild -DTAMM_CXX_FLAGS="-DMKL_ILP64 -m64" -DCBLAS_LIBRARIES=$TAMM_BLASLIBS -DLAPACKE_LIBRARIES=$TAMM_BLASLIBS

make -j3
make install
```