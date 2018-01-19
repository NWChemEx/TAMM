#
# This file will build Netlib's BLAS 3.8.0 distribution
#

include(ExternalProject)
enable_language(C Fortran)

#Settings for the current version of BLAS
set(BLAS_VERSION 3.8.0)
set(BLAS_URL http://www.netlib.org/blas/blas-${BLAS_VERSION}.tgz)
set(BLAS_MD5 3E6E783ECEFC3B0B461722A939A16D9B)

#This is the root of the source tree created by External Project
set(BLAS_PREFIX ${CMAKE_BINARY_DIR}/NWX_BLAS_External)
#This is where the BLAS tar-ball will be downloaded to
set(BLAS_DOWNLOAD ${BLAS_PREFIX}/src)
#This is where BLAS will be extracted to
set(BLAS_SRC_DIR  ${BLAS_DOWNLOAD}/BLAS/BLAS-${BLAS_VERSION})
#This is where we need to put the Makefile settings
set(MAKEFILE_DEST ${BLAS_SRC_DIR}/make.inc)
#This is the name of the created library
set(BLAS_LIBRARY "libblas${CMAKE_STATIC_LIBRARY_SUFFIX}")
#This is the full set of compiler flags
set(BLAS_FLAGS ${CMAKE_Fortran_FLAGS_RELEASE})
if(CMAKE_POSITION_INDEPENDENT_CODE)
    list(APPEND BLAS_FLAGS -fPIC)
endif()
clean_flags(BLAS_FLAGS BLAS_FLAGS)

#BLAS wants it's options in Makefile.inc
set(BLAS_MAKEFILE ${CMAKE_BINARY_DIR}/make.inc)
file(WRITE ${BLAS_MAKEFILE} "SHELL = /bin/sh\n")
file(APPEND ${BLAS_MAKEFILE} "PLAT = _${CMAKE_SYSTEM_NAME}\n" )
file(APPEND ${BLAS_MAKEFILE} "FORTRAN  = ${CMAKE_Fortran_COMPILER}\n")
file(APPEND ${BLAS_MAKEFILE} "OPTS     = ${BLAS_FLAGS}\n")
file(APPEND ${BLAS_MAKEFILE} "DRVOPTS  = ${BLAS_FLAGS}\n")
file(APPEND ${BLAS_MAKEFILE} "NOOPT    = \n")
file(APPEND ${BLAS_MAKEFILE} "LOADER   = ${CMAKE_Fortran_COMPILER}\n")
file(APPEND ${BLAS_MAKEFILE} "LOADOPTS = \n")
file(APPEND ${BLAS_MAKEFILE} "ARCH     = ${CMAKE_AR}\n")
file(APPEND ${BLAS_MAKEFILE} "ARCHFLAGS= cr\n")
file(APPEND ${BLAS_MAKEFILE} "RANLIB   = echo\n")
file(APPEND ${BLAS_MAKEFILE} "BLASLIB  = ${BLAS_LIBRARY}\n")

set(BLAS_INSTALL ${STAGE_DIR}${CMAKE_INSTALL_PREFIX}/lib/${BLAS_LIBRARY})
ExternalProject_Add(BLAS_External
        PREFIX ${BLAS_PREFIX}
        DOWNLOAD_NO_PROGRESS TRUE
        DOWNLOAD_DIR ${BLAS_DOWNLOAD}
        SOURCE_DIR ${BLAS_DOWNLOAD}/BLAS
        URL ${BLAS_URL}
        URL_MD5 ${BLAS_MD5}
        BINARY_DIR ${BLAS_SRC_DIR}
        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E copy ${BLAS_MAKEFILE}
                                                   ${MAKEFILE_DEST}
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy
                                         ${BLAS_SRC_DIR}/${BLAS_LIBRARY}
                                         ${BLAS_INSTALL}
)

#This is primarily for testing our BLAS build, actual code should use CBLAS as
#this header will likely only exist if we built BLAS
include(FortranCInterface)
set(FC_MANGLE_INSTALL ${STAGE_DIR}${CMAKE_INSTALL_PREFIX}/include)
FortranCInterface_HEADER(${FC_MANGLE_INSTALL}/FCMangleBLAS.h
        MACRO_NAMESPACE
        "FCBLAS_")
