
set(CMAKE_BUILD_TYPE Release)

# Compilers (assuming the compilers are in the PATH)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_Fortran_COMPILER gfortran)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ANTLR_CPPRUNTIME /opt/ANTLR/CppRuntime)

# General NWCHEM Options
# Set the following when using the NWCHEM CMAKE build
set(NWCHEM_TOP /opt/nwchem-cmake/nwchem-devel)
set(GA_CONFIG ${NWCHEM_TOP}/build/nwchem-install/external/ga/bin)
set(NWCHEM_BUILD_DIR ${NWCHEM_TOP}/buildGCC/)
option(NWCHEM_CMAKE_BUILD "Using NWCHEM CMAKE build" ON)

# Set the following when using the old Makefile-based NWCHEM build
# set(NWCHEM_TOP /opt/libraries/nwchem-devel)
# set(NWCHEM_BUILD_DIR ${NWCHEM_TOP}/src/)
# set(GA_CONFIG ${NWCHEM_TOP}/src/tools/install/bin)
# option(NWCHEM_CMAKE_BUILD "Using NWCHEM CMAKE build" OFF)


#Number of cores to be used for the build
#set(TAMM_PROC_COUNT 2)




