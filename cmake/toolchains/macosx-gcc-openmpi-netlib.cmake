
set(CMAKE_BUILD_TYPE Release)

# Compilers (assuming the compilers are in the PATH)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_Fortran_COMPILER gfortran)
# set(MPI_C_COMPILER mpicc)
# set(MPI_CXX_COMPILER mpicxx)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TAMM_PROC_COUNT 2)

# BLAS 
set(TAMM_BLAS_INC /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/)
set(TAMM_BLAS_LIB /Users/panyala/software/libraries/nwchem-cmake/nwchem-devel/build/src/blas)

# BLAS. Only 8-byte BLAS is supported for now. 
# Set BLAS_LIBRARIES only when using nwchem internal blas - All other cases these come from ga_config
set(BLAS_LIBRARIES "-lnwcblas" CACHE STRING "BLAS linker flags")

set(ANTLR_CPPRUNTIME /Users/panyala/software/libraries/ANTLR4/antlr4-cpp-runtime)

# General NWCHEM Options
set (NWCHEM_TOP /Users/panyala/software/libraries/nwchem-cmake/nwchem-devel)
set(GA_CONFIG ${NWCHEM_TOP}/build/nwc-install/external/ga/bin)
set(NWCHEM_BUILD_DIR ${NWCHEM_TOP}/build/)
option(NWCHEM_CMAKE_BUILD "Using NWCHEM CMAKE build" ON)


