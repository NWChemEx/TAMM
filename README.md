
Requirements
------------
- Git
- autotools
- cmake >= 3.7
- C++14 compiler

**NOTE:** The current TAMM code has only been tested with gcc versions >= 6.0.
It also works with Intel 18 compilers, but does not build with Intel Compiler versions <= 17.0.  


BUILD
-----

```
TAMM_ROOT=/opt/TAMM  
git clone https://github.com/NWChemEx-Project/TAMM.git $TAMM_ROOT  
git checkout devel
```

- Modify any toolchain file (*except old-tamm-config.cmake*) in ${TAMM_ROOT}/dependencies/cmake/toolchains to  
  adjust compilers and MPI_INCLUDE_PATH, MPI_LIBRARY_PATH, MPI_LIBRARIES.

  Following are optional:
  - GA & BLAS OPTIONS.
  - TAMM_PROC_COUNT, EIGEN3_INSTALL_PATH, LIBINT_INSTALL_PATH,
  & ANTLR_CPPRUNTIME_PATH.


  **NOTE:** GA, Eigen3, Netlib blas+lapack, Libint, ANTLR, googletest will be
  built if they do not exist.


```
cd ${TAMM_ROOT}/dependencies  
mkdir build && cd build  
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/gcc-openmpi-netlib.cmake
make  
```

- After missing dependencies are built:

```
cd ${TAMM_ROOT}  
mkdir build && cd build  
cmake ..  -DCMAKE_TOOLCHAIN_FILE=${TAMM_ROOT}/dependencies/build/tamm_build.cmake  
make install
```


BUILD OLD TAMM CODE (OPTIONAL)
------------------------------

```
TAMM_ROOT=/opt/TAMM  
git clone https://github.com/NWChemEx-Project/TAMM.git $TAMM_ROOT  
git checkout devel
```

 - Modify old-tamm-config.cmake in ${TAMM_ROOT}/dependencies/cmake/toolchains to  
  adjust compilers, NWCHEM_TOP (path to nwchem root folder), GA_CONFIG (path to ga_config)
  and NWCHEM_BUILD_TARGET/NWCHEM_BUILD_DIR.

```
cd ${TAMM_ROOT}/dependencies  
mkdir build && cd build  
cmake .. -DBUILD_OLD_TAMM=ON -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/gcc-openmpi-netlib.cmake
make  
```

- After missing dependencies are built:

```
cd ${TAMM_ROOT}  
mkdir build && cd build  
cmake ..  -DBUILD_OLD_TAMM=ON -DCMAKE_TOOLCHAIN_FILE=${TAMM_ROOT}/dependencies/build/tamm_build.cmake  
make install
```
