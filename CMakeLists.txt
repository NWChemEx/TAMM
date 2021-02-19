cmake_minimum_required(VERSION 3.18.0)

include(FetchContent)

project(tamm VERSION 0.0.0 LANGUAGES C CXX Fortran)
include(${PROJECT_SOURCE_DIR}/cmake/check_user_env.cmake)

find_package( CMakeBuild QUIET )

if( NOT ${CMakeBuild_FOUND} )
  if (NOT CMSB_TAG)
    set(CMSB_TAG master)
  endif()

  FetchContent_Declare(
    CMakeBuild
    GIT_REPOSITORY https://github.com/NWChemEx-Project/CMakeBuild.git
    GIT_TAG ${CMSB_TAG}
  )

  FetchContent_MakeAvailable( CMakeBuild )
  set(CMSB_MACROS ${CMakeBuild_SOURCE_DIR}/cmake/macros)
endif()

set(CMSB_PROJECTS tamm)

set(tamm_DEPENDENCIES MSGSL Eigen3 LibInt2 HPTT HDF5) # GlobalArrays BLAS LAPACK

if(USE_CUDA) 
   list(APPEND tamm_DEPENDENCIES TALSH)
endif()

if(DEFINED LINALG_VENDOR)
    if(NOT ${LINALG_VENDOR} STREQUAL "BLIS")
        list(APPEND tamm_DEPENDENCIES BLIS)
    endif()
endif()

if(USE_GA_AT)
    if(USE_DPCPP)
        list(APPEND tamm_DEPENDENCIES IntelSYCL)
    endif()
    list(APPEND tamm_DEPENDENCIES GlobalArrays)
    if(USE_SCALAPACK)
        list(APPEND tamm_DEPENDENCIES ScaLAPACK LAPACK BLAS)
    else()
        list(APPEND tamm_DEPENDENCIES LAPACK BLAS)
    endif()
else()
    list(APPEND tamm_DEPENDENCIES GlobalArrays)
endif()

set(tamm_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tamm)
set(tamm_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/)
set(tamm_METHODS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/methods/)

if(NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif()

set(CMSB_BASISSET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/methods/basis)
build_cmsb_module(${CMAKE_CURRENT_LIST_DIR})

