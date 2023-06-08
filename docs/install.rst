The prerequisites needed to build this repository can be found

Build Instructions
==================

Depdendencies
-------------

**External depdendencies**

* cmake >= 3.22
* MPI 
* C++17 compiler (information on supported compilers here :doc:`here <prerequisites>`.)
* CUDA >= 11.4 (Required only for CUDA builds)
* ROCM >= 5.4  (Required only for ROCM builds)

**The remaining depdendencies are automatically built and do not need to be installed explicitly:**

* GlobalArrays
* HPTT, Librett
* HDF5
* BLAS/LAPACK (BLIS and netlib-lapack are automatically built if no vendor BLAS libraries are provided)
* BLAS++ and LAPACK++
* Eigen3, doctest


Choose Build Options
--------------------

CUDA Options
~~~~~~~~~~~~

::

   -DUSE_CUDA=ON (OFF by default)  
   -DGPU_ARCH=70 (GPU arch is detected automatically, only set this option if need to override)

HIP Options
~~~~~~~~~~~~

::

   -DUSE_HIP=ON (OFF by default) 
   -DROCM_ROOT=$ROCM_PATH
   -DGPU_ARCH=gfx90a (GPU arch is detected automatically, only set this option if need to override)


DPCPP options
~~~~~~~~~~~~~~

::

   -DUSE_DPCPP=ON

CMake options for developers (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   -DUSE_GA_PROFILER=ON #Enable GA's profiling feature (GCC Only).

   -DMODULES="CC" (empty by default)

Building TAMM
--------------

::

   export REPO_ROOT_PATH=$HOME/TAMM
   export REPO_INSTALL_PATH=$HOME/tamm_install

::

   git clone <repo-url> $REPO_ROOT_PATH
   cd $REPO_ROOT_PATH
   # Checkout the branch you want to build
   mkdir build && cd build

In addition to the build options chosen, there are various build configurations depending on the BLAS library one wants to use.

- :ref:`Default build using BLIS and NETLIB LAPACK <default-build-using-blis-and-netlib-lapack>`

- :ref:`Default build on MACOS <default-build-on-macos>`

- :ref:`Build using Intel MKL <build-using-intel-mkl>`

- :ref:`Build instructions for Summit using ESSL <build-summit-using-essl>`

- :ref:`Build instructions for Summit using ESSL and UPC++ <build-summit-using-essl-and-upc++>`

- :ref:`Build instructions for Frontier <build-frontier>`

- :ref:`Build instructions for Perlmutter and Polaris <build-perlmutter-and-polaris>`

- :ref:`Build instructions for Theta <build-theta>`

- :ref:`SYCL build instructions <build-sycl>`

- :ref:`Build instructions for Sunspot <build-sunspot>`



.. _default-build-using-blis-and-netlib-lapack:

Default build using BLIS and NETLIB LAPACK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable CUDA build, add ``-DUSE_CUDA=ON``


::

   cd $REPO_ROOT_PATH/build 
   CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH ..

   make -j3
   make install

.. _default-build-on-macos:

Default build on MACOS
~~~~~~~~~~~~~~~~~~~~~~

.. note::
   The prerequisites for ``MACOS`` can be installed using ``brew`` as detailed :doc:`here <prerequisites>`.

::

   cd $REPO_ROOT_PATH/build 
   CC=gcc-10 CXX=g++-10 FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH ..

   make -j3
   make install

.. _build-using-intel-mkl:

Build using Intel MKL
~~~~~~~~~~~~~~~~~~~~~~

.. _to-enable-cuda-build-add--duse_cudaon-1:

To enable CUDA build, add ``-DUSE_CUDA=ON``

::

   cd $REPO_ROOT_PATH/build 

   CC=gcc CXX=g++ FC=gfortran cmake -DLINALG_VENDOR=IntelMKL \
   -DLINALG_PREFIX=/opt/intel/mkl \
   -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH ..

   make -j3
   make install

.. _build-summit-using-essl:

Build instructions for Summit using ESSL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   module load gcc
   module load cmake
   module load essl/6.3.0
   module load cuda

::

   cd $REPO_ROOT_PATH/build

   CC=gcc CXX=g++ FC=gfortran cmake \
   -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH \
   -DBLIS_CONFIG=power9 \
   -DLINALG_VENDOR=IBMESSL -DUSE_CUDA=ON \
   -DLINALG_PREFIX=/sw/summit/essl/6.3.0/essl/6.3 ..

   make -j3
   make install

.. _build-summit-using-essl-and-upc++:

Build instructions for Summit using ESSL and UPC++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: UPC++ support is currently experimental.

::

   module load gcc
   module load cmake
   module load essl/6.3.0
   module load cuda
   module load upcxx

::

   cd $REPO_ROOT_PATH/build

   UPCXX_CODEMODE=O3 CC=gcc CXX=upcxx FC=gfortran cmake \
   -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH \
   -DBLIS_CONFIG=power9 \
   -DLINALG_VENDOR=IBMESSL \
   -DLINALG_PREFIX=/sw/summit/essl/6.3.0/essl/6.3 \
   -DUSE_CUDA=ON \
   -DUSE_UPCXX=ON ..

   UPCXX_CODEMODE=O3 make -j3
   UPCXX_CODEMODE=O3 make install

.. _build-frontier:

Build instructions for Frontier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   module load cray-python cmake amd-mixed 
   module load cray-hdf5-parallel
   export CRAYPE_LINK_TYPE=dynamic
   export HDF5_USE_FILE_LOCKING=FALSE

::

   cd $REPO_ROOT_PATH/build

   CC=cc CXX=CC FC=ftn cmake \
   -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH \
   -DGPU_ARCH=gfx90a \
   -DUSE_HIP=ON -DROCM_ROOT=$ROCM_PATH \
   -DGCCROOT=/opt/cray/pe/gcc/10.3.0/snos \
   -DHDF5_ROOT=$HDF5_ROOT ..

   make -j3
   make install


.. _build-perlmutter-and-polaris:

Build instructions for Perlmutter and Polaris
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   module purge
   module load PrgEnv-gnu
   module load craype-x86-milan
   module load cmake
   module load cpe-cuda

   module load cpe gpu (Perlmutter Only)
   module load cudatoolkit-standalone (Polaris Only)

   export CRAYPE_LINK_TYPE=dynamic

.. note:: Currently need to add ``-DUSE_CRAYSHASTA=ON`` to the cmake line below for Polaris builds

::

   cd $REPO_ROOT_PATH/build

   cmake -DUSE_CUDA=ON -DBLIS_CONFIG=generic \
   -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH ..

   make -j3
   make install


.. _build-theta:

Build instructions for Theta
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   module unload PrgEnv-intel/6.0.7
   module load PrgEnv-gnu/6.0.7
   module unload cmake
   module load cmake
   export CRAYPE_LINK_TYPE=dynamic

::

   cd $REPO_ROOT_PATH/build

   CC=cc CXX=CC FC=ftn cmake -DLINALG_VENDOR=IntelMKL \
   -DLINALG_PREFIX=/opt/intel/mkl \
   -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH ..

   make -j3
   make install

.. _build-sycl:

SYCL build instructions using Intel OneAPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``MPI:`` Only tested using ``MPICH``.
-  Set ROOT dir of the GCC installation (need gcc >= v9.1)

::

   export GCC_ROOT_PATH=/opt/gcc-9.1.0

::

   cd $REPO_ROOT_PATH/build 

   CC=icx CXX=icpx FC=ifx cmake \
   -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH \
   -DLINALG_VENDOR=IntelMKL -DLINALG_PREFIX=/opt/oneapi/mkl/latest \
   -DUSE_DPCPP=ON -DGCCROOT=$GCC_ROOT_PATH \
   -DTAMM_CXX_FLAGS="-fma -ffast-math -fsycl -fsycl-default-sub-group-size 16 -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel -sycl-std=2020"

   make -j3
   make install

.. _build-sunspot:

Build instructions for Sunspot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   module load spack cmake
   module load mpich
   ONEAPI_MPICH_GPU=NO_GPU module load oneapi/eng-compiler/2022.12.30.003
   module load tools/xpu-smi/1.2.1
   export GCC_ROOT_PATH=/opt/cray/pe/gcc/11.2.0/snos

::

   unset EnableWalkerPartition
   export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
   export ONEAPI_MPICH_GPU=NO_GPU
   export MPIR_CVAR_ENABLE_GPU=0

   export FI_CXI_DEFAULT_CQ_SIZE=131072
   export FI_CXI_CQ_FILL_PERCENT=20

   export SYCL_PROGRAM_COMPILE_OPTIONS=" -ze-opt-large-register-file -ze-opt-greater-than-4GB-buffer-required"
   export SYCL_PI_LEVEL_ZERO_SINGLE_THREAD_MODE=1
   export ZES_ENABLE_SYSMAN=1
   export SYCL_CACHE_PERSISTENT=1
   unset SYCL_DEVICE_FILTER
   export ONEAPI_DEVICE_SELECTOR=level_zero:*
   export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

::

   cd $REPO_ROOT_PATH/build

   CC=icx CXX=icpx FC=ifx cmake \
   -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH \
   -DLINALG_VENDOR=IntelMKL -DLINALG_PREFIX=$MKLROOT \
   -DUSE_DPCPP=ON -DGCCROOT=$GCC_ROOT_PATH \
   -DTAMM_CXX_FLAGS="-fma -ffast-math -fsycl -fsycl-default-sub-group-size 16 -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel -sycl-std=2020"

   make -j3
   make install