
set(CMAKE_BUILD_TYPE Release)

# Compilers (assuming the compilers are in the PATH)
set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpc)
set(CMAKE_Fortran_COMPILER ifort)
# set(MPI_C_COMPILER mpicc)
# set(MPI_CXX_COMPILER mpicxx)

set(TAMM_PROC_COUNT 2)

# BLAS
set(TAMM_BLAS_INC /opt/intel/mkl/include/)
set(TAMM_BLAS_LIB /opt/intel/mkl/lib/intel64)

# BLAS. Only 8-byte BLAS is supported for now. 
# Set BLAS_LIBRARIES only when using nwchem internal blas - All other cases these come from ga_config
#set(BLAS_LIBRARIES "-mkl=parallel -lmkl_blacs_openmpi_ilp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl" CACHE STRING "BLAS linker flags")

set(ANTLR_CPPRUNTIME /opt/libraries/ANTLR4/antlr4-cpp-runtime)

# General NWCHEM Options
#Using NWCHEM CMAKE build
set(NWCHEM_TOP /home/panyala/git/nwchem-cmake/nwchem-devel)
set(GA_CONFIG ${NWCHEM_TOP}/buildIntel/nwc-install/external/ga/bin)
set(NWCHEM_BUILD_DIR ${NWCHEM_TOP}/buildIntel/)
option(NWCHEM_CMAKE_BUILD "Using NWCHEM CMAKE build" ON)

#When using the old Makefile-based NWCHEM build
# set(NWCHEM_TOP /opt/libraries/nwchem-devel)
# set(NWCHEM_BUILD_DIR ${NWCHEM_TOP}/src/)
# set(GA_CONFIG ${NWCHEM_TOP}/src/tools/install/bin)
# option(NWCHEM_CMAKE_BUILD "Using NWCHEM CMAKE build" OFF)
