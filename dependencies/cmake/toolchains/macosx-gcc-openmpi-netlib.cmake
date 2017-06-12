
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
set(TAMM_PROC_COUNT 4)

# GA Options
#set(ARMCI_NETWORK OPENIB)
set(USE_OFFLOAD "OFFLOAD" ON)


# MPI 
set(MPI_INCLUDE_PATH /Users/panyala/software/libraries/openmpi/include)
set(MPI_LIBRARY_PATH /Users/panyala/software/libraries/openmpi/lib)
set(MPI_LIBRARIES "-lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi")



#set(TALSH_INSTALL_PATH /home/gawa722/Cascade/nwchem/TAL_SH)
# General NWCHEM Options
#Using NWCHEM CMAKE build
set(NWCHEM_TOP /home/panyala/git/nwchem-cmake/nwchem-devel)
set(GA_CONFIG ${NWCHEM_TOP}/build/nwchem-install/external/ga/bin)
set(NWCHEM_BUILD_DIR ${NWCHEM_TOP}/build/)
option(NWCHEM_CMAKE_BUILD "Using NWCHEM CMAKE build" ON)