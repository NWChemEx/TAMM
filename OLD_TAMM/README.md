
Requirements
------------
- Git
- autotools 
- cmake >= 3.7
- C++14 compiler

**NOTE:** The current TAMM code has only been tested with gcc versions >= 6.0.
It also works with Intel 18 compilers, but does not build with Intel Compiler versions <= 17.0.  The devel branch does not support building with Clang.


BUILD
-----

```
TAMM_ROOT=/opt/nwx_sandbox  
git clone https://github.com/NWChemEx-Project/NWX_Sandbox.git $TAMM_ROOT  
git checkout devel
```

- Modify any toolchain file in ${TAMM_ROOT}/dependencies/cmake/toolchains to  
  adjust compilers, NWCHEM_TOP (nwchem repo), GA_CONFIG (path to ga_config) and NWCHEM_BUILD_DIR.
  Optionally adjust TAMM_PROC_COUNT, EIGEN3_INSTALL_PATH, LIBINT_INSTALL_PATH, 
  BLAS_INCLUDE_PATH, BLAS_LIBRARY_PATH & ANTLR_CPPRUNTIME_PATH. 
  Eigen3, Netlib blas+lapack, Libint, ANTLR, googletest will be
  built if they do not exist.

```
cd ${TAMM_ROOT}/dependencies  
mkdir build && cd build  
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/macosx-gcc-openmpi-netlib.cmake
make  
```

- After missing dependencies are built:

```
cd ${TAMM_ROOT}/tamm_old  
mkdir build && cd build  
cmake .. -DCMAKE_TOOLCHAIN_FILE=${TAMM_ROOT}/dependencies/build/tamm_build.cmake  
make install

As needed:
make patch
make link
make unpatch
```
