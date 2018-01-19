#
# This file actually builds Netlib's LAPACK 3.8.0
#
cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(LAPACK VERSION 3.8.0 LANGUAGES C Fortran)
include(DependencyMacros)
include(ExternalProject)
find_dependency(BLAS LAPACK_BLAS_INCLUDE_DIR
                     LAPACK_BLAS_LIBRARIES
                     LAPACK_BLAS_DEFINITIONS
                     LAPACK_BLAS_LINK_FLAGS
                     _LAPACK_BLAS_FOUND)

set(LAPACK_VERSION 3.8.0)
set(LAPACK_URL http://www.netlib.org/lapack/lapack-${LAPACK_VERSION}.tar.gz)
set(LAPACK_MD5 96591AFFDBF58C450D45C1DAA540DBD2)

#It appears somehow the external project doesn't honor
#CMAKE_POSITION_INDEPENDENT_CODE
if(CMAKE_POSITION_INDEPENDENT_CODE)
    list(APPEND LAPACK_FLAGS "-fPIC")
endif()

set(STAGE_DIR ${CMAKE_BINARY_DIR}/stage)
ExternalProject_Add(LAPACK
        DOWNLOAD_NO_PROGRESS TRUE
        URL ${LAPACK_URL}
        URL_MD5 ${LAPACK_MD5}
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE
                   -DUSE_OPTIMIZED_BLAS=ON
                   -DBLAS_LIBRARIES=${BLAS_LIBRARIES}
                   -DBUILD_TESTING=OFF
                   -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                   -DCMAKE_C_FLAGS=${LAPACK_FLAGS}
                   -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
                   -DCMAKE_Fortran_FLAGS=${LAPACK_FLAGS}
                   -DLAPACKE=OFF
                   -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        MAKE_COMMAND $(MAKE)
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
)

#This is primarily for testing our LAPACK build, actual code should use LAPACKE
#as this header will likely only exist if we built LAPACK
include(FortranCInterface)
set(FC_MANGLE_INSTALL ${STAGE_DIR}${CMAKE_INSTALL_PREFIX}/include)
FortranCInterface_HEADER(${FC_MANGLE_INSTALL}/FCMangleLAPACK.h
        MACRO_NAMESPACE
        "FCLAPACK_")
install(DIRECTORY ${STAGE_DIR}${CMAKE_INSTALL_PREFIX}/
        DESTINATION ${CMAKE_INSTALL_PREFIX} USE_SOURCE_PERMISSIONS)
