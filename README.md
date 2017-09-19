
Requirements
------------
- Git
- cmake >= 3.7
- C++14 compiler

**NOTE:** The current TAMM code has only been tested with gcc versions >= 6.0.
It also works with Intel v18.0 Beta compilers, but does not build with Intel Compiler versions <= 17.0.  The devel branch does not support building with Clang.


BUILD
-----

```
TAMM_ROOT=/opt/nwx_sandbox  
git clone https://github.com/NWChemEx-Project/NWX_Sandbox.git $TAMM_ROOT  
git checkout devel
```

- Modify any toolchain file in ${TAMM_ROOT}/dependencies/cmake/toolchains to  
 adjust BLAS_INCLUDE_PATH, NWCHEM_TOP. Optionally adjust TAMM_PROC_COUNT,
 LIBINT_INSTALL_PATH & ANTLR_CPPRUNTIME_PATH. Libint, ANTLR and googletest will be
 built if they do not exist. Eigen3 will be built by default.


```
cd ${TAMM_ROOT}/dependencies  
mkdir build && cd build  
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/macosx-gcc-openmpi-netlib.cmake
make  
```

- After missing dependencies are built:

```
cd ${TAMM_ROOT}  
mkdir build && cd build  
cmake .. -DCMAKE_TOOLCHAIN_FILE=${TAMM_ROOT}/dependencies/build/tamm_build.cmake  
make install
make patch
make link

As needed: make unpatch
```
