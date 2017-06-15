
set(CMAKE_BUILD_TYPE Release)

# Compilers (assuming the compilers are in the PATH)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_Fortran_COMPILER gfortran)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#Location where the dependencies need to be installed.
#Default location: cmake_build_folder/tamm-deps
#set(CMAKE_INSTALL_PREFIX /opt/tamm-deps)

#Number of cores to be used for the build
set(TAMM_PROC_COUNT 2)

# GA Options
#set(ARMCI_NETWORK OPENIB)
set(USE_OFFLOAD "OFFLOAD" ON)

# MPI 
set(MPI_INCLUDE_PATH /usr/lib/openmpi/include/)
set(MPI_LIBRARY_PATH /usr/lib/openmpi/lib/)
set(MPI_LIBRARIES "-lmpi_f77 -lmpi -ldl -lhwloc")

# Optionally set BLAS, LAPACK & SCALAPACK. Support only 8-byte integers for now.
# If not set, will build BLAS+LAPACK automatically.
#set(BLAS_INCLUDE_PATH /opt/intel/mkl/include/)
##set(BLAS_LIBRARY_PATH /opt/intel/mkl/lib/intel64)
#
#set(BLAS_LIBRARIES "-mkl -lmkl_lapack95_ilp64 -lmkl_blas95_ilp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl" CACHE STRING "BLAS linker flags")
#set(LAPACK_LIBRARIES "${BLAS_LIBRARIES}" CACHE STRING "LAPACK linker flags")
#set(SCALAPACK_LIBRARIES "-mkl -lmkl_scalapack_ilp64 -lmkl_blacs_openmpi_ilp64 -lmkl_intel_thread -lpthread -lm -ldl" CACHE STRING "SCALAPACK linker flags")

#set(TALSH_INSTALL_PATH /home/gawa722/Cascade/nwchem/TAL_SH)

