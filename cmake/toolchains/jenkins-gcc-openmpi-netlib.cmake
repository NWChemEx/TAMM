set(CMAKE_BUILD_TYPE Release)

# Compilers (assuming the compilers are in the PATH)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_Fortran_COMPILER gfortran)
# set(MPI_C_COMPILER mpicc)
# set(MPI_CXX_COMPILER mpicxx)

set(TAMM_PROC_COUNT 8)

# BLAS
set(TAMM_BLAS_INC /opt/intel/mkl/include/)
set(TAMM_BLAS_LIB /opt/intel/mkl/lib/intel64)

# BLAS. Only 8-byte BLAS is supported for now. 
# Set BLAS_LIBRARIES only when using nwchem internal blas - All other cases these come from ga_config
set(BLAS_LIBRARIES "-lnwcblas" CACHE STRING "BLAS linker flags")

set(ANTLR_CPPRUNTIME /opt/libraries/ANTLR4/antlr4-cpp-runtime)

# General NWCHEM Options
set(NWCHEM_TOP /home/jenkins/nwchem-cmake/nwchem-devel)
set(GA_CONFIG ${NWCHEM_TOP}/build/nwc-install/external/ga/bin)
set(NWCHEM_BUILD_DIR ${NWCHEM_TOP}/build/)
option(NWCHEM_CMAKE_BUILD "Using NWCHEM CMAKE build" ON)