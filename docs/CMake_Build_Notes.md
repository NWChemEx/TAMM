CMAKE Build Notes
=================

Installing Prerequisites
-------------------------

## On Mac OSX:
- brew install cmake
- brew install gcc openmpi
- brew install lzlib wget flex bison doxygen autoconf automake libtool

## On Linux:

We recommend using the [Spack package manager](https://spack.io) to install and manage the Prerequisites if the versions
avaiable via the OS package manager are not sufficient.

Clang Compiler Support
----------------------
 - Tested on Linux only with Clang >= 7.x
 - Still requires GCC compilers >= 8.3 to be present (Fortran code is compiled using gfortran)


