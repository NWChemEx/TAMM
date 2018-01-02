#
# This file actually builds Netlib's LAPACKE 3.8.0
#
cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(LAPACKE VERSION 3.8.0 LANGUAGES C Fortran)
include(DependencyMacros)
include(ExternalProject)
find_dependency(BLAS NWX_BLAS_INCLUDE_DIRS
                          NWX_BLAS_LIBRARIES
                          NWX_BLAS_DEFINITIONS
                          NWX_BLAS_LINK_FLAGS
                          _BLAS_FOUND)
find_dependency(LAPACK NWX_LAPACK_INCLUDE_DIRS
                           NWX_LAPACK_LIBRARIES
                           NWX_LAPACK_DEFINITIONS
                           NWX_LAPACK_LINK_FLAGS
                           _LAPACK_FOUND)
set(LAPACKE_VERSION 3.8.0)

set(LAPACKE_URL http://www.netlib.org/lapack/lapack-${LAPACKE_VERSION}.tar.gz)
set(LAPACKE_MD5 96591AFFDBF58C450D45C1DAA540DBD2)

if(CMAKE_POSITION_INDEPENDENT_CODE)
    list(APPEND LAPACK_FLAGS "-fPIC")
endif()

ExternalProject_Add(LAPACKE
        DOWNLOAD_NO_PROGRESS TRUE
        URL ${LAPACKE_URL}
        URL_MD5 ${LAPACKE_MD5}
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE
                   -DUSE_OPTIMIZED_BLAS=ON
                   -DBLAS_LIBRARIES=${NWX_BLAS_LIBRARIES}
                   -DUSE_OPTIMIZED_LAPACK=ON

                   -DBUILD_TESTING=OFF
                   -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                   -DCMAKE_C_FLAGS=${LAPACK_FLAGS}
                   -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
                   -DCMAKE_Fortran_FLAGS=${LAPACK_FLAGS}
                   -DLAPACKE=ON
                   -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        MAKE_COMMAND $(MAKE)
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
        CMAKE_CACHE_ARGS -DLAPACK_LIBRARIES:LIST=${NWX_LAPACK_LIBRARIES}
        )
