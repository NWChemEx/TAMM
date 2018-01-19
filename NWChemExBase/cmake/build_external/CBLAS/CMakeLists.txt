#The actual file that builds CBLAS
cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(CBLAS VERSION 0.0.0 LANGUAGES C Fortran)

include(DependencyMacros)
include(ExternalProject)
find_dependency(BLAS NWX_BLAS_INCLUDE_DIRS
                     NWX_BLAS_LIBRARIES
                     NWX_BLAS_DEFINITIONS
                     NWX_BLAS_LINK_FLAGS
                     _BLAS_FOUND)

set(CBLAS_URL http://www.netlib.org/blas/blast-forum/cblas.tgz)
set(CBLAS_MD5 1E8830F622D2112239A4A8A83B84209A)

if(CMAKE_POSITION_INDEPENDENT_CODE)
    set(CBLAS_C_FLAGS "-fPIC")
    set(CBLAS_Fortran_FLAGS "-fPIC")
endif()
set(CBLAS_C_FLAGS "${CBLAS_C_FLAGS} -O3 -DADD_")


#CBLAS want's it's options in Makefile.in
set(CBLAS_MAKEFILE "${CMAKE_BINARY_DIR}/Makefile.in")
set(CBLAS_BUILD_DIR "${CMAKE_BINARY_DIR}/NWX_CBLAS-prefix/src/NWX_CBLAS")
set(MAKEFILE_DEST "${CBLAS_BUILD_DIR}/Makefile.in")

set(CBLAS_LIB "libcblas${CMAKE_STATIC_LIBRARY_SUFFIX}")
file(WRITE  ${CBLAS_MAKEFILE} "SHELL = /bin/sh\n")
file(APPEND ${CBLAS_MAKEFILE} "PLAT = ${CMAKE_SYSTEM_NAME}\n")
file(APPEND ${CBLAS_MAKEFILE} "BLLIB = ${NWX_BLAS_LIBRARIES}\n")
file(APPEND ${CBLAS_MAKEFILE} "CBLIB = ../lib/${CBLAS_LIB}\n")
file(APPEND ${CBLAS_MAKEFILE} "CC = ${CMAKE_C_COMPILER}\n")
file(APPEND ${CBLAS_MAKEFILE} "FC = ${CMAKE_Fortran_COMPILER}\n")
file(APPEND ${CBLAS_MAKEFILE} "LOADER = \$(FC)\n")
file(APPEND ${CBLAS_MAKEFILE} "CFLAGS = ${CBLAS_C_FLAGS}\n")
file(APPEND ${CBLAS_MAKEFILE} "FFLAGS = ${CBLAS_Fortran_FLAGS}\n")
file(APPEND ${CBLAS_MAKEFILE} "ARCH = ${CMAKE_AR}\n")
file(APPEND ${CBLAS_MAKEFILE} "ARCHFLAGS = cr\n")
file(APPEND ${CBLAS_MAKEFILE} "RANLIB = echo\n")

set(CBLAS_INSTALL_DIR ${STAGE_DIR}${CMAKE_INSTALL_PREFIX})
ExternalProject_Add(NWX_CBLAS
        DOWNLOAD_NO_PROGRESS TRUE
        URL ${CBLAS_URL}
        URL_MD5 ${CBLAS_MD5}
        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E copy ${CBLAS_MAKEFILE}
                                                   ${MAKEFILE_DEST}
        BUILD_COMMAND $(MAKE) alllib
        BUILD_IN_SOURCE TRUE
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy
                                        ${CBLAS_BUILD_DIR}/lib/${CBLAS_LIB}
                                        ${CBLAS_INSTALL_DIR}/lib/${CBLAS_LIB}
                        COMMAND ${CMAKE_COMMAND} -E copy
                                ${CBLAS_BUILD_DIR}/include/cblas_f77.h
                                ${CBLAS_INSTALL_DIR}/include/cblas_f77.h
                        COMMAND ${CMAKE_COMMAND} -E copy
                                ${CBLAS_BUILD_DIR}/include/cblas.h
                                ${CBLAS_INSTALL_DIR}/include/cblas.h
        )
