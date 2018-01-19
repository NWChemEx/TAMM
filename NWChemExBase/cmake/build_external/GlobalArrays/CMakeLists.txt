cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(GlobalArrays VERSION 5.6.2 LANGUAGES C Fortran)

include(DependencyMacros)
include(ExternalProject)
foreach(depend NWX_MPI CBLAS LAPACKE )#ScaLAPACK)
    find_dependency(${depend} ${depend}_INCLUDE_DIRS
                                  ${depend}_LIBRARIES
                                  ${depend}_DEFINITIONS
                                  ${depend}_LINK_FLAGS
                                  _${depend}_FOUND)
    assert(_${depend}_FOUND)
endforeach()
string (REPLACE ";" " " BLAS_LIBRARIES "${NWX_CBLAS_LIBRARIES}")
set(GA_BLAS "--with-blas8=${BLAS_LIBRARIES}")
string(REPLACE ";" " " LAPACK_LIBRARIES "${LAPACKE_LIBRARIES}")
set(GA_LAPACK "--with-lapack=${LAPACK_LIBRARIES}")
#string(REPLACE ";" " " GA_ScaLAPACK_LIBRARIES "${ScaLAPACK_LIBRARIES}")
#set(GA_SCALAPACK "--with-scalapack8=${GA_ScaLAPACK_LIBRARIES}")

# GA has to be difficult when it comes to MPI...
string(REPLACE ";" " -I" GA_MPI_INCLUDES "${NWX_MPI_INCLUDE_DIRS}")
message(STATUS "${GA_MPI_INCLUDES}")
foreach(_nwx_lib ${NWX_MPI_LIBRARIES})
    # Remove the actual library from the path
    get_filename_component(_lib_path ${_nwx_lib} DIRECTORY)
    # Get only library name with extension
    get_filename_component(_name_lib ${_nwx_lib} NAME_WE)
    # Strip the lib prefix
    string(SUBSTRING ${_name_lib} 3 -1 _name_lib)
    set(GA_MPI "${GA_MPI} -L${_lib_path} -l${_name_lib}")
endforeach()
set(GA_MPI "--with-mpi=${GA_MPI_INCLUDES} ${GA_MPI}")

set(GA_SYSVSHMEM "ARMCI_DEFAULT_SHMMAX_UBOUND=131072")

if (USE_OFFLOAD)
    set(GA_OFFLOAD "INTEL_64ALIGN=1")
endif()

set(GA_CXXFLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(GA_CFLAGS "${CMAKE_C_FLAGS} -fPIC")
set(GA_FFLAGS "${CMAKE_Fortran_Flags} -fPIC")


# Build GA
set(GA_INSTALL_DIR ${STAGE_DIR}${CMAKE_INSTALL_PREFIX})
ExternalProject_Add(GlobalArrays_External
        URL https://github.com/GlobalArrays/ga/releases/download/v5.6.3/ga-5.6.3.tar.gz
        #Pass location where autotools needs to be built
        #CONFIGURE_COMMAND ./autogen.sh
        CONFIGURE_COMMAND ./configure --with-tcgmsg
                                      ${GA_MPI}
                                      --enable-underscoring
                                      --disable-mpi-tests
                                      --prefix=${GA_INSTALL_DIR}
                                      ${GA_SCALAPACK}
                                      ${GA_BLAS}
                                      ${GA_LAPACK}
                                      ${ARMCI_NETWORK}
                                      ${GA_OFFLOAD}
                                      ${GA_SYSVSHMEM}
                                      CC=${CMAKE_C_COMPILER}
                                      CFLAGS=${GA_CFLAGS}
                                      CXX=${CMAKE_CXX_COMPILER}
                                      CXXFLAGS=${GA_CXXFLAGS}
                                      F77=${CMAKE_Fortran_COMPILER}
                                      FFLAGS=${GA_FFLAGS}
        BUILD_IN_SOURCE 1
        #LOG_CONFIGURE 1
        #LOG_BUILD 1
        )
