The prerequisites needed to build this repository can be found

Build Instructions
==================

Dependencies
------------

**External dependencies**

* cmake >= 3.26
* MPI 
* C++17 compiler (information on supported compilers here :doc:`here <prerequisites>`.)
* CUDA >= 11.7 (Required only for CUDA builds)
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

   -DTAMM_ENABLE_CUDA=ON (OFF by default)  
   One of -DGPU_ARCH=X (OR) -DCMAKE_CUDA_ARCHITECTURES=X is required. Set the arch value X to 70 for Volta, 80 for Ampere, 90 for Hopper and 95 for Blackwell.

HIP Options
~~~~~~~~~~~~

::

   -DTAMM_ENABLE_HIP=ON (OFF by default) 
   -DROCM_ROOT=$ROCM_PATH
   One of -DGPU_ARCH=gfx90a (OR) -DCMAKE_HIP_ARCHITECTURES=gfx90a is required.


DPCPP options
~~~~~~~~~~~~~~

::

   -DTAMM_ENABLE_DPCPP=ON (OFF by default)

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

- :ref:`Build instructions for Frontier <build-frontier>`

- :ref:`Build instructions for Perlmutter and Polaris <build-perlmutter-and-polaris>`

- :ref:`SYCL build instructions <build-sycl>`

- :ref:`Build instructions for Aurora <build-aurora>`



.. _default-build-using-blis-and-netlib-lapack:

Default build using BLIS and NETLIB LAPACK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable CUDA build, add ``-DTAMM_ENABLE_CUDA=ON`` and ``-DGPU_ARCH=<value>``


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
   FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH ..

   make -j3
   make install

.. _build-using-intel-mkl:

Build using Intel MKL
~~~~~~~~~~~~~~~~~~~~~~

To enable CUDA build, add ``-DTAMM_ENABLE_CUDA=ON`` and ``-DGPU_ARCH=<value>``

::

   cd $REPO_ROOT_PATH/build 

   CC=gcc CXX=g++ FC=gfortran cmake -DLINALG_VENDOR=IntelMKL \
   -DLINALG_PREFIX=/opt/intel/mkl \
   -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH ..

   make -j3
   make install

.. _build-frontier:

Build instructions for Frontier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   module load cpe
   module load cray-python cmake cray-hdf5-parallel
   module load cce
   module load cray-mpich
   module load rocm
   export CRAYPE_LINK_TYPE=dynamic

::

   cd $REPO_ROOT_PATH/build

   CC=cc CXX=CC FC=ftn cmake \
   -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH \
   -DGPU_ARCH=gfx90a \
   -DTAMM_ENABLE_HIP=ON -DROCM_ROOT=$ROCM_PATH \
   -DGCCROOT=/opt/gcc/12.2.0/snos \
   -DHDF5_ROOT=$HDF5_ROOT ..

   make -j3
   make install


.. _build-perlmutter-and-polaris:

Build instructions for Perlmutter and Polaris
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perlmutter modules and env

::

   module load PrgEnv-gnu
   module load cmake
   module load cpe-cuda
   module load cudatoolkit
   module unload craype-accel-nvidia80

   export CRAYPE_LINK_TYPE=dynamic
   export MPICH_GPU_SUPPORT_ENABLED=0

Polaris modules and env

:: 

   module use /soft/modulefiles/
   module load PrgEnv-gnu
   module load cudatoolkit-standalone/12.6.1 spack-pe-base cmake
   module unload craype-accel-nvidia80

   export CRAYPE_LINK_TYPE=dynamic
   export MPICH_GPU_SUPPORT_ENABLED=0   

Common build steps

::

   cd $REPO_ROOT_PATH/build

   cmake -DTAMM_ENABLE_CUDA=ON -DGPU_ARCH=80 -DBLIS_CONFIG=generic \
   -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH ..

   make -j3
   make install

.. _build-sycl:

SYCL build instructions using Intel OneAPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``MPI:`` Only tested using ``MPICH``.
-  Set ROOT dir of the GCC installation (need gcc >= v9.1)

::

   cd $REPO_ROOT_PATH/build 

   CC=icx CXX=icpx FC=ifx cmake \
   -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH \
   -DLINALG_VENDOR=IntelMKL -DLINALG_PREFIX=/opt/oneapi/mkl/latest \
   -DTAMM_ENABLE_DPCPP=ON -DGCCROOT=$GCC_ROOT \
   -DTAMM_CXX_FLAGS="-fma -ffast-math -fsycl -fsycl-default-sub-group-size 16 -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel -sycl-std=2020"

   make -j3
   make install

.. _build-aurora:

Build instructions for Aurora
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:: 

   module restore
   module load cmake python

::

   cd $REPO_ROOT_PATH/build

   CC=icx CXX=icpx FC=ifx cmake \
   -DCMAKE_INSTALL_PREFIX=$REPO_INSTALL_PATH \
   -DLINALG_VENDOR=IntelMKL -DLINALG_PREFIX=$MKLROOT \
   -DTAMM_ENABLE_DPCPP=ON -DGCCROOT=$GCC_ROOT \
   -DTAMM_CXX_FLAGS="-march=sapphirerapids -mtune=sapphirerapids -ffast-math -fsycl -fsycl-device-code-split=per_kernel -fsycl-targets=intel_gpu_pvc -sycl-std=2020"

   make -j12
   make install
