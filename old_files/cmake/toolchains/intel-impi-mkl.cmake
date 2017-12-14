
set(CMAKE_BUILD_TYPE Release)

# Compilers (assuming the compilers are in the PATH)
set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpc)
set(CMAKE_Fortran_COMPILER ifort)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The MPI settings below will soon be made optional, but they have to be specified for now.
set(MPI_INCLUDE_PATH /usr/lib/x86_64-linux-gnu/openmpi/include)
set(MPI_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/openmpi/lib)
set(MPI_LIBRARIES "-lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi")

#------------------ OPTIONAL ---------------------

#Number of cores to be used for the build
set(NWX_PROC_COUNT 2)

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

