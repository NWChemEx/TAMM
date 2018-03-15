
Prerequisites
------------
- Git
- autotools
- cmake >= 3.7
- C++14 compiler
- MPI Library

**Please see [CMake Build Notes](CMake_Build_Notes.md) for more details on installing Prerequisites**

Supported Compilers
--------------------
- GCC versions >= 6.0
- Intel 18 compilers
- Apple Clang 
- LLVM Clang >=4.0 (Linux Only): Please see [Clang Compiler Support](CMake_Build_Notes.md#clang-compiler-support)

Supported Configurations
-------------------------
- The following configurations are recommended since they are tested and are known to work:
  - GCC6 + OpenMPI-2.x/MPICH-3.x built using GCC6
  - GCC7 + OpenMPI-2.x/MPICH-3.x built using GCC7
  - Intel 18 + Intel MPI Library
  - Apple Clang/LLVM Clang + OpenMPI-2.x/MPICH-3.x 

BUILD
-----

- Make sure the compiler and MPI executables (gcc,mpirun,etc) you want to use are in your system's standard search PATH.

```
TAMM_ROOT=/opt/TAMM  
git clone https://github.com/NWChemEx-Project/TAMM.git $TAMM_ROOT  
git checkout devel  
TAMM_ROOT=/opt/TAMM/old_files
```

- Optionally modify any toolchain file (*except old-tamm-config.cmake*) in ${TAMM_ROOT}/cmake/toolchains to adjust the following:
  - GA Configure Options.
  - BLAS include & library paths.
  - TAMM_PROC_COUNT, EIGEN3_INSTALL_PATH, LIBINT_INSTALL_PATH,
  & ANTLR_CPPRUNTIME_PATH.

  **NOTE:** Eigen3, Netlib blas+lapack, Libint, ANTLR, googletest will be
  built if they do not exist. GA will be built by default. Pre-exisiting GA setup
  cannot be specified.


```
cd ${TAMM_ROOT}/external  
mkdir build && cd build  
cmake .. -DCMAKE_TOOLCHAIN_FILE=${TAMM_ROOT}/cmake/toolchains/gcc-openmpi-netlib.cmake
make  
```

- After missing dependencies are built:

```
cd ${TAMM_ROOT}  
mkdir build && cd build  
cmake ..  -DCMAKE_TOOLCHAIN_FILE=${TAMM_ROOT}/external/build/tamm_build.cmake  
make install
```

