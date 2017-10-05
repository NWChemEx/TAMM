
set(CMAKE_BUILD_TYPE Release)

# Compilers (assuming the compilers are in the PATH)
set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpc)
set(CMAKE_Fortran_COMPILER ifort)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# General NWCHEM Options
# Set the following when using the NWCHEM CMAKE build
set(NWCHEM_TOP /opt/nwchem-cmake/nwchem-devel)
set(GA_CONFIG ${NWCHEM_TOP}/build/nwchem-install/external/ga/bin)
set(NWCHEM_BUILD_DIR ${NWCHEM_TOP}/build/)
option(NWCHEM_CMAKE_BUILD "Using NWCHEM CMAKE build" ON)

# Set the following when using the regular Makefile-based NWCHEM build
# set(NWCHEM_TOP /opt/libraries/nwchem-devel)
# set(NWCHEM_BUILD_TARGET LINUX64)
# set(GA_CONFIG ${NWCHEM_TOP}/src/tools/install/bin)
# option(NWCHEM_CMAKE_BUILD "Using NWCHEM CMAKE build" OFF)


#------------------ OPTIONAL ---------------------

#Number of cores to be used for the build
set(TAMM_PROC_COUNT 2)

#Location where the dependencies need to be installed.
#Default location: cmake_build_folder/tamm-deps
#set(CMAKE_INSTALL_PREFIX /opt/tamm-deps)

#set(LIBINT_INSTALL_PATH /opt/libint2) 
#set(ANTLR_CPPRUNTIME_PATH /opt/ANTLR/CppRuntime) 
#set(EIGEN3_INSTALL_PATH /opt/eigen3) 
#set(BLAS_INCLUDE_PATH /opt/blas_lapack/include) 
#set(BLAS_LIBRARY_PATH /opt/blas_lapack/lib) 
#set(GTEST_INSTALL_PATH /opt/googletest) 

#set(TALSH_INSTALL_PATH /opt/TAL_SH)


#-------------------- NOT NEEDED FOR NOW --------------------------

# Optionally set BLAS, LAPACK & SCALAPACK. Support only 8-byte integers for now.
# If not set, will build BLAS+LAPACK automatically.
#set(BLAS_INCLUDE_PATH /opt/intel/mkl/include)
#set(BLAS_LIBRARY_PATH /opt/intel/mkl/lib/intel64)

# GA Options
#set(ARMCI_NETWORK OPENIB)
#set(USE_OFFLOAD "OFFLOAD" ON)

# MPI 
#set(MPI_INCLUDE_PATH /opt/openmpi/include)
#set(MPI_LIBRARY_PATH /opt/openmpi/lib)
#set(MPI_LIBRARIES "-lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi")

#set(BLAS_LIBRARIES "-mkl -lmkl_lapack95_ilp64 -lmkl_blas95_ilp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl" CACHE STRING "BLAS linker flags")
#set(LAPACK_LIBRARIES "${BLAS_LIBRARIES}" CACHE STRING "LAPACK linker flags")
#set(SCALAPACK_LIBRARIES "-mkl -lmkl_scalapack_ilp64 -lmkl_blacs_openmpi_ilp64 -lmkl_intel_thread -lpthread -lm -ldl" CACHE STRING "SCALAPACK linker flags")


