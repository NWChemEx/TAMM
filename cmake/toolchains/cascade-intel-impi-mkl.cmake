
set(CMAKE_BUILD_TYPE Release)

# Compilers (assuming the compilers are in the PATH)
set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpc)
set(CMAKE_Fortran_COMPILER ifort)
# set(MPI_C_COMPILER mpicc)
# set(MPI_CXX_COMPILER mpicxx)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TAMM_PROC_COUNT 16)

# BLAS 
set(TAMM_BLAS_INC /msc/apps/compilers/IPS_2017_U1/compilers_and_libraries_2017.1.132/linux/mkl/include)
#set(TAMM_BLAS_LIB /msc/apps/compilers/IPS_2017_U1/compilers_and_libraries_2017.1.132/linux/mkl/lib/intel64)

# BLAS. Only 8-byte BLAS is supported for now. 
# Set BLAS_LIBRARIES only when using nwchem internal blas - All other cases these come from ga_config
#set(BLAS_LIBRARIES "-mkl=parallel -lmkl_lapack95_ilp64 -lmkl_blas95_ilp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl" CACHE STRING "BLAS linker flags")

set(ANTLR_CPPRUNTIME /home/panyala/software/ANTLR/antlr4-cpp-runtime)
set(TALSH_INSTALL_PATH /home/gawa722/Cascade/nwchem/TAL_SH)
# General NWCHEM Options
#Using NWCHEM CMAKE build
# set(NWCHEM_TOP /home/panyala/nwchem-cmake/nwchem-devel)
# set(GA_CONFIG ${NWCHEM_TOP}/build/nwc-install/external/ga/bin)
# set(NWCHEM_BUILD_DIR ${NWCHEM_TOP}/build/)
# option(NWCHEM_CMAKE_BUILD "Using NWCHEM CMAKE build" ON)


#When using the old Makefile-based NWCHEM build
set(NWCHEM_TOP /home/panyala/software/nwchem-devel)
set(GA_CONFIG ${NWCHEM_TOP}/src/tools/install/bin)
set(NWCHEM_BUILD_DIR ${NWCHEM_TOP}/src/)
option(NWCHEM_CMAKE_BUILD "Using NWCHEM CMAKE build" OFF)
