
Prerequisites
-------------
- Git
- autotools
- cmake >= 3.11
- MPI Library
- C++17 compiler
- CUDA >= 9.2 (only if building with GPU support)

**Please see [CMake Build Notes](CMake_Build_Notes.md) for more details on installing Prerequisites**

Supported Compilers
--------------------
- GCC versions >= 7.2
- LLVM Clang >= 5.x (Tested on Linux Only): Please see [Clang Compiler Support](CMake_Build_Notes.md#clang-compiler-support)


Supported Configurations
-------------------------
- The following configurations are recommended since they are tested and are known to work:
  - GCC versions 7.2,7.3,8.x + OpenMPI-2.x/MPICH-3.x built using corresponding gcc versions.
  - LLVM Clang versions 5.x,6.x + OpenMPI-2.x/MPICH-3.x 
  <!-- - Intel 19 + Intel MPI Library (work in progress) -->


