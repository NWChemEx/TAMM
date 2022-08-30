
Prerequisites
-------------
- Git
- autotools
- cmake >= 3.22
- MPI Library
- C++17 compiler
- CUDA >= 11.1 (OR) ROCM >= 5.0 (when building with Nvidia/AMD GPU support)

Supported Compilers
--------------------
- GCC versions >= 9.1
- LLVM Clang >= 9 
- Apple Clang >= 13

Supported Configurations
-------------------------
- The following configurations are recommended since they are tested and are known to work:
  - GCC versions >= 9.1 + OpenMPI-2.x/MPICH-3.x built using the corresponding GCC versions.
  - LLVM Clang versions >= 9 + OpenMPI-2.x/MPICH-3.x 
  - Apple Clang versions >= 9 + OpenMPI-2.x/MPICH-3.x 

Installing Prerequisites
-------------------------

## On Mac OSX:
- brew install gcc openmpi cmake autoconf automake libtool

## On Linux:

We recommend using the [Spack package manager](https://spack.io) to install and manage the prerequisites if the versions
avaiable via the OS package manager are not sufficient.
