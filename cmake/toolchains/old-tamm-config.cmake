#------------------------------------------------------------------------
# NOTE: DO NOT USE THIS FILE UNLESS YOU WANT TO BUILD THE OLD TAMM CODE.
#------------------------------------------------------------------------

set(CMAKE_BUILD_TYPE Release)

# Compilers (assuming the compilers are in the PATH)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_Fortran_COMPILER gfortran)
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

#Enable/Disable GPU support
set(TAMM_ENABLE_GPU OFF)

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

