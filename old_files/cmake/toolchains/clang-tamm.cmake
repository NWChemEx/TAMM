
set(CMAKE_BUILD_TYPE Release)

# Compilers (assuming the compilers are in the PATH)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_Fortran_COMPILER gfortran)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set(BUILD_NETLIB_BLAS_LAPACK ON)
# The MPI settings below will soon be made optional, but they have to be specified for now.
set(MPI_INCLUDE_PATH /usr/local/Cellar/open-mpi/3.0.0/include)
set(MPI_LIBRARY_PATH /usr/local/Cellar/open-mpi/3.0.0/lib)
set(MPI_LIBRARIES "-lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi")

#------------------ OPTIONAL ---------------------

#Enable/Disable GPU support
set(TAMM_ENABLE_GPU OFF)

# GA CONFIGURATION
#set(ARMCI_NETWORK OPENIB)
#set(USE_OFFLOAD "OFFLOAD" ON)

# BLAS, LAPACK & SCALAPACK. Support only 8-byte integers for now.
#set(BLAS_INCLUDE_PATH /opt/blas/include)
#set(BLAS_LIBRARY_PATH /opt/blas/lib)
# set(BLAS_LIBRARIES "-lblas -llapack" CACHE STRING "BLAS linker flags")
# set(LAPACK_LIBRARIES "${BLAS_LIBRARIES}" CACHE STRING "LAPACK linker flags")
#set(SCALAPACK_LIBRARIES "-mkl -lmkl_scalapack_ilp64 -lmkl_blacs_openmpi_ilp64 -lmkl_intel_thread -lpthread -lm -ldl" CACHE STRING "SCALAPACK linker flags")

# set(LIBINT_ROOT_DIR /opt/tamm-deps) 
# set(ANTLRCPPRUNTIME_ROOT_DIR /opt/tamm-deps) 
# set(EIGEN3_ROOT_DIR /opt/tamm-deps) 
# set(GTEST_ROOT_DIR /opt/tamm-deps) 
# set(BLAS_ROOT_DIR /opt/tamm-deps) 
# set(GLOBALARRAYS_ROOT_DIR /opt/tamm-deps) 

#set(TALSH_ROOT_DIR /opt/TAL_SH)

