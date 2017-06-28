
Requirements
------------
- Git
- cmake >= 3.7
- C++14 compiler
- Required to build ANTLR - libuuid (uuid-dev or uuid-devel package on Debian/RedHat)


NOTE: The current TAMM code has only been tested with gcc versions >= 6.0 and clang versions >= 3.8.  
It also works with Intel v18.0 Beta compilers, but does not build with Intel Compiler versions <= 17.0.  


Code 
----
The latest code is in the tammx sub-directory. Hartree-Fock code can be found at nwchem_main/hartree_fock.cc  

BUILD
-----

TAMM_ROOT=/opt/nwx_sandbox  
git clone https://github.com/NWChemEx-Project/NWX_Sandbox.git $TAMM_ROOT  


- If you have Eigen3 and Libint already built on your machine, use  

cd ${TAMM_ROOT}  
mkdir build && cd build  
cmake ..   
-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++  
-DEIGEN3_INSTALL_PATH=/opt/eigen3  
-DLIBINT_INSTALL_PATH=/opt/libint  
make install  


- Alternatively, if you want the TAMM cmake setup to download and build Libint, Eigen3, use  

cd ${TAMM_ROOT}/dependencies  
mkdir build && cd build  
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++  
make  

- Once Libint and Eigen are built,  

cd ${TAMM_ROOT}  
mkdir build && cd build  
cmake .. -DCMAKE_TOOLCHAIN_FILE=${TAMM_ROOT}/dependencies/build/tamm_build.cmake  
make install
