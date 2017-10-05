
set(CMAKE_BUILD_TYPE Release)

# Compilers (assuming the compilers are in the PATH)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_Fortran_COMPILER gfortran)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(BLAS_INCLUDE_PATH /home/panyala/intel/mkl/include)

# General NWCHEM Options
# Set the following when using the NWCHEM CMAKE build
set(NWCHEM_TOP /home/panyala/git/nwchem-cmake/nwchem-devel)
set(GA_CONFIG ${NWCHEM_TOP}/build/nwchem-install/external/ga/bin)
set(NWCHEM_BUILD_DIR ${NWCHEM_TOP}/build/)
option(NWCHEM_CMAKE_BUILD "Using NWCHEM CMAKE build" ON)

# Set the following when using the old Makefile-based NWCHEM build
# set(NWCHEM_TOP /opt/libraries/nwchem-devel)
# set(NWCHEM_BUILD_TARGET LINUX64)
# set(GA_CONFIG ${NWCHEM_TOP}/src/tools/install/bin)
# option(NWCHEM_CMAKE_BUILD "Using NWCHEM CMAKE build" OFF)


#------------------ OPTIONAL ---------------------
#Location where the dependencies need to be installed.
#Default location: cmake_build_folder/tamm-deps
#set(CMAKE_INSTALL_PREFIX /opt/tamm-deps)

#Number of cores to be used for the build
set(TAMM_PROC_COUNT 2)

set(EIGEN3_INSTALL_PATH /home/panyala/git/tamm/dependencies/b1/tamm-deps/eigen3)
set(LIBINT_INSTALL_PATH /home/panyala/git/tamm/dependencies/b1/tamm-deps/libint2)
#set(ANTLR_CPPRUNTIME_PATH /opt/libraries/ANTLR4/antlr4-cpp-runtime)


#-------------------- NOT NEEDED FOR NOW --------------------------

# Optionally set BLAS, LAPACK & SCALAPACK. Support only 8-byte integers for now.
# If not set, will build BLAS+LAPACK automatically.
#set(BLAS_INCLUDE_PATH /opt/intel/mkl/include)
#set(BLAS_LIBRARY_PATH /opt/intel/mkl/lib/intel64)

# GA Options
#set(ARMCI_NETWORK OPENIB)
#set(USE_OFFLOAD "OFFLOAD" ON)

# MPI 
#set(MPI_INCLUDE_PATH /Users/panyala/software/libraries/openmpi/include)
#set(MPI_LIBRARY_PATH /Users/panyala/software/libraries/openmpi/lib)
#set(MPI_LIBRARIES "-lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi")

#set(TALSH_INSTALL_PATH /home/gawa722/Cascade/nwchem/TAL_SH)

# Optionally set BLAS, LAPACK & SCALAPACK. Support only 8-byte integers for now.
# If not set, will build BLAS+LAPACK automatically.
#set(BLAS_INCLUDE_PATH /opt/intel/mkl/include/)
##set(BLAS_LIBRARY_PATH /opt/intel/mkl/lib/intel64)
#
#set(BLAS_LIBRARIES "-mkl -lmkl_lapack95_ilp64 -lmkl_blas95_ilp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl" CACHE STRING "BLAS linker flags")
#set(LAPACK_LIBRARIES "${BLAS_LIBRARIES}" CACHE STRING "LAPACK linker flags")
#set(SCALAPACK_LIBRARIES "-mkl -lmkl_scalapack_ilp64 -lmkl_blacs_openmpi_ilp64 -lmkl_intel_thread -lpthread -lm -ldl" CACHE STRING "SCALAPACK linker flags")


