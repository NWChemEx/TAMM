The prerequisites needed to build this repository can be found

Build Instructions
==================

Dependencies
------------

**External dependencies**

* cmake >= 3.22
* MPI 
* C++17 compiler (information on supported compilers here :doc:`here <prerequisites>`.)
* CUDA >= 11.4 (Required only for CUDA builds)
* ROCM >= 5.5  (Required only for ROCM builds)

**The remaining dependencies are automatically built and do not need to be installed explicitly:**

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

- :ref:`SYCL build instructions <build-sycl>`

- :ref:`Build instructions for Aurora <build-aurora>`



.. _default-build-using-blis-and-netlib-lapack:

Default build using BLIS and NETLIB LAPACK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable CUDA build, add ``-DUSE_CUDA=ON`` and ``-DGPU_ARCH=<value>``


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
   CC=gcc-12 CXX=g++-12 FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH ..

   make -j3
   make install

.. _build-using-intel-mkl:

Build using Intel MKL
~~~~~~~~~~~~~~~~~~~~~~

.. _to-enable-cuda-build-add--duse_cudaon-1:

To enable CUDA build, add ``-DUSE_CUDA=ON`` and ``-DGPU_ARCH=<value>``

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

   module load cray-python cmake 
   module load cray-hdf5-parallel
   module load cpe/23.05
   module load rocm/5.5.1
   export CRAYPE_LINK_TYPE=dynamic

::

   cd $REPO_ROOT_PATH/build

   CC=cc CXX=CC FC=ftn cmake \
   -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH \
   -DGPU_ARCH=gfx90a \
   -DUSE_HIP=ON -DROCM_ROOT=$ROCM_PATH \
   -DGCCROOT=/opt/gcc/12.2.0/snos \
   -DHDF5_ROOT=$HDF5_ROOT ..

   make -j3
   make install


.. _build-perlmutter-and-polaris:

Build instructions for Perlmutter and Polaris
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   module load PrgEnv-gnu
   module load craype-x86-milan
   module load cmake
   module load cpe-cuda

   module load cudatoolkit (Perlmutter Only)
   module load cudatoolkit-standalone (Polaris Only)

   module unload craype-accel-nvidia80

   export CRAYPE_LINK_TYPE=dynamic
   export MPICH_GPU_SUPPORT_ENABLED=0

.. note:: Currently need to add ``-DUSE_CRAYSHASTA=ON`` to the cmake line below only for Polaris builds.

::

   cd $REPO_ROOT_PATH/build

   cmake -DUSE_CUDA=ON -DGPU_ARCH=80 -DBLIS_CONFIG=generic \
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

.. _build-aurora:

Build instructions for Aurora
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:: 

   module use /soft/modulefiles/
   module load spack-pe-gcc/0.4-rc1 numactl/2.0.14-gcc-testing cmake
   module load oneapi/release/2023.12.15.001
   export MPIR_CVAR_ENABLE_GPU=0
   export GCC_ROOT_PATH=/opt/cray/pe/gcc/11.2.0/snos

::

   cd $REPO_ROOT_PATH/build

   CC=icx CXX=icpx FC=ifx cmake \
   -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH \
   -DLINALG_VENDOR=IntelMKL -DLINALG_PREFIX=$MKLROOT \
   -DUSE_DPCPP=ON -DUSE_MEMKIND=ON -DGCCROOT=$GCC_ROOT_PATH \
   -DTAMM_CXX_FLAGS="-march=sapphirerapids -mtune=sapphirerapids -ffast-math -fsycl -fsycl-default-sub-group-size 16 -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel -sycl-std=2020"

   make -j12
   make install
