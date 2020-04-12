CMAKE Build Notes
=================

Installing Prerequisites
-------------------------

On Mac OSX:
- brew install cmake
- brew install gcc openmpi
- brew install lzlib wget flex bison doxygen autoconf automake libtool

On Linux:

We recommend using the [Spack package manager](https://spack.io) to install and manage the Prerequisites if the versions
avaiable via the OS package manager are not sufficient.

Clang Compiler Support
----------------------
 - Tested on Linux only with Clang >= 7.x
 - Still requires GCC compilers >= 8.3 to be present (Fortran code is compiled using gfortran)
 - Works only with LLVM Clang built with OpenMP support.
 - Install LLVM Clang using the script below:

```
version=7.0.0
current_dir=`pwd`

mkdir stage-$version
cd stage-$version

bases="llvm-${version}.src cfe-${version}.src compiler-rt-${version}.src"
bases="${bases} openmp-${version}.src polly-${version}.src"
bases="${bases} clang-tools-extra-${version}.src"

for base in ${bases}
do
  wget -t inf -c http://llvm.org/releases/${version}/${base}.tar.xz
  tar xvf ${base}.tar.xz
done

llvm_root=llvm-${version}.src
mv -v cfe-${version}.src ${llvm_root}/tools/clang
mv -v clang-tools-extra-${version}.src ${llvm_root}/tools/clang/tools/extra
mv -v compiler-rt-${version}.src ${llvm_root}/projects/compiler-rt
mv -v openmp-${version}.src ${llvm_root}/projects/openmp

mkdir ${llvm_root}/build
cd ${llvm_root}/build
cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_INSTALL_PREFIX=/opt/llvm7 -DCMAKE_BUILD_TYPE=Release
make -j16
make install
```


