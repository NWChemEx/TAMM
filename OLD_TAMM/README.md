
Requirements
------------
- Git
- autotools
- cmake >= 3.7
- C++14 compiler

**NOTE:** The OLD TAMM code has only been tested with gcc versions >= 6.0 and Intel compiler versions 16 & 17.

BUILD
-----

```
OLD_TAMM_ROOT=/opt/old_tamm  
git clone https://github.com/NWChemEx-Project/OLD_TAMM.git $OLD_TAMM_ROOT  
git checkout devel
```

 Modify the sample toolchain file in ${OLD_TAMM_ROOT}/sample-toolchain.cmake to adjust:
  - compilers (use the same compilers used to build nwchem)
  - NWCHEM_TOP (path to nwchem root folder),
  - GA_CONFIG (path to ga_config),
  - NWCHEM_BUILD_DIR (path to nwchem build folder)
  - ANTLR_CPPRUNTIME (path to ANTLR CPP runtime lib)

```
cd ${OLD_TAMM_ROOT}
mkdir build && cd build  
cmake .. -DCMAKE_TOOLCHAIN_FILE=${OLD_TAMM_ROOT}/sample-toolchain.cmake  
make install

As needed:  
- make patch
- make link
- make unpatch
```
