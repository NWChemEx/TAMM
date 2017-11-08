CMAKE Build Notes
=================

Installing Prerequisites
-------------------------

On Mac OSX:
- brew install lzlib wget flex bison doxygen autoconf automake libtool
- brew install gcc openmpi
- brew install cmake

On Linux, use the following script to build GCC and OpenMPI from sources if they are not available through a package manager (usually happens when using an older Linux OS). It can be used on Mac OSX as well:

```

wget http://mirrors-usa.go-parts.com/gcc/releases/gcc-6.3.0/gcc-6.3.0.tar.gz
tar xf gcc-6.3.0.tar.gz
cd gcc-6.3.0
./contrib/download_prerequisites
./configure --prefix=/opt/gcc-6.3 --disable-multilib --enable-languages=c,c++,fortran
make -j16
make install
cd ../

wget https://ftp.gnu.org/gnu/binutils/binutils-2.28.tar.gz
tar xf binutils-2.28.tar.gz
cd binutils-2.28
mkdir staging && cd staging
../configure --prefix=/opt/binutils-2.28 --enable-gold --enable-ld=default --enable-plugins --enable-shared  --disable-werror --with-system-zlib
make -j16
make install
cd ../

wget https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.2.tar.gz
tar xf openmpi-2.1.2.tar.gz
echo "Building depend mpi"
cd openmpi-2.1.0
./configure --prefix=/opt/openmpi-2.1 --enable-mpi-cxx --enable-mpi-fortran --enable-mpi-thread-multiple
make -j 16
make install
```

PGI Compiler support
--------------------
  - Eigen/ANTLR Cpp runtime do not build with PGI compilers - use GCC here.
  - GA - openmpi has to be built manually with PGI compilers, openmpi bundled with PGI install does not seem to work
  - CC=pgcc CXX=pgc++ FC=pgfortran ./configure --prefix=/opt/openmpi-2.1 --enable-mpi-cxx --enable-mpi-fortran

  - TAMM code compiles fine, but link line fails due to some (PGI compiler) incompatibility with Eigen

Note: When using GNU compilers, adding pgi-install-path/lib directory to LD_LIBRARY_PATH causes link errors like `libhwloc.so.5: undefined reference to move_pages@libnuma_1.2`

Clang Compiler Support
----------------------
 - Tested on Linux only with Clang >= 4.0
 - GA is still built with GNU compilers due to some issues when mixing clang and gfortran.
 - Works only with LLVM Clang built with OpenMP support and configured to use GNU libstdc++ instead of Clang libc++
 - Install LLVM Clang using the script below:

```
version=4.0.0
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
cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_INSTALL_PREFIX=/opt/llvm4 -DCMAKE_BUILD_TYPE=Release
make -j2
make install
```
