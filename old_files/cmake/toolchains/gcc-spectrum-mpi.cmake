# On Summitdev following modules are required
# module load xl/20170508-beta
# module load gcc/6.3.1-20170301
# module load spectrum-mpi/10.1.0.4-20170915
# module load essl/5.5.0-20161110
# module load cuda/9.0.69
# module load cmake/3.9.2

set(CMAKE_BUILD_TYPE Release)

# Compilers (assuming the compilers are in the PATH)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_Fortran_COMPILER gfortran)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The MPI settings below will soon be made optional, but they have to be specified for now.
set(MPI_INCLUDE_PATH /autofs/nccs-svm1_sw/summitdev/.swci/1-compute/opt/spack/20171006/linux-rhel7-ppc64le/gcc-6.3.1/spectrum-mpi-10.1.0.4-20170915-mwst4ujoupnioe3kqzbeqh2efbptssqz/include)
set(MPI_LIBRARY_PATH /autofs/nccs-svm1_sw/summitdev/.swci/1-compute/opt/spack/20171006/linux-rhel7-ppc64le/gcc-6.3.1/spectrum-mpi-10.1.0.4-20170915-mwst4ujoupnioe3kqzbeqh2efbptssqz/lib)
set(MPI_LIBRARIES "-lmpiprofilesupport -lmpi_usempi -lmpi_mpifh -lmpi_ibm")

#------------------ OPTIONAL ---------------------

#Enable/Disable GPU support
set(TAMM_ENABLE_GPU OFF)

# GA CONFIGURATION
#set(ARMCI_NETWORK OPENIB)
#set(USE_OFFLOAD "OFFLOAD" ON)

# BLAS, LAPACK & SCALAPACK. Support only 8-byte integers for now.
#set(BLAS_INCLUDE_PATH /opt/intel/mkl/include)
#set(BLAS_LIBRARY_PATH /opt/intel/mkl/lib/intel64)
#set(BLAS_LIBRARIES "-mkl -lmkl_lapack95_ilp64 -lmkl_blas95_ilp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl" CACHE STRING "BLAS linker flags")
#set(LAPACK_LIBRARIES "${BLAS_LIBRARIES}" CACHE STRING "LAPACK linker flags")
#set(SCALAPACK_LIBRARIES "-mkl -lmkl_scalapack_ilp64 -lmkl_blacs_openmpi_ilp64 -lmkl_intel_thread -lpthread -lm -ldl" CACHE STRING "SCALAPACK linker flags"))

#Location where the dependencies need to be installed.
#Default location: cmake_build_folder/tamm-deps
#set(CMAKE_INSTALL_PREFIX /opt/tamm-deps)

#set(LIBINT_INSTALL_PATH /opt/libint2) 
#set(ANTLR_CPPRUNTIME_PATH /opt/ANTLR/CppRuntime) 
#set(EIGEN3_INSTALL_PATH /opt/eigen3) 
#set(GTEST_INSTALL_PATH /opt/googletest) 

#set(TALSH_INSTALL_PATH /opt/TAL_SH)

