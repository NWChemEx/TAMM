cmake_minimum_required(VERSION 3.18.0)

include(FetchContent)

project(TAMM VERSION 0.0.0 LANGUAGES C CXX Fortran)
include(${PROJECT_SOURCE_DIR}/cmake/check_user_env.cmake)

find_package( CMakeBuild QUIET )

if( NOT ${CMakeBuild_FOUND} )
  if (NOT CMSB_TAG)
    set(CMSB_TAG 3eee6e63d897b37bba021ab730b26c60b7169f4d)
    if (ENABLE_DEV_MODE)
      set(CMSB_TAG master)
    endif()
  endif()

  FetchContent_Declare(
    CMakeBuild
    GIT_REPOSITORY https://github.com/NWChemEx-Project/CMakeBuild.git
    GIT_TAG ${CMSB_TAG}
  )

  FetchContent_MakeAvailable( CMakeBuild )
  set(CMSB_MACROS ${CMakeBuild_SOURCE_DIR}/cmake/macros)

  FetchContent_MakeAvailable( linalg-cmake-modules )
  set(CMSB_LAM_PATH ${linalg-cmake-modules_SOURCE_DIR})
endif()

set(CMSB_PROJECTS TAMM)

set(TAMM_DEPENDENCIES NJSON MSGSL DOCTEST Eigen3 HPTT HDF5)

if(NOT DEFINED BUILD_METHODS)
  list(APPEND TAMM_DEPENDENCIES LibInt2)
else()
  if(BUILD_METHODS)
    list(APPEND TAMM_DEPENDENCIES LibInt2)
  endif()
endif()

if(USE_CUDA OR USE_HIP) 
  if(NOT DEFINED USE_TALSH)
    list(APPEND TAMM_DEPENDENCIES TALSH)
  elseif(USE_TALSH)
    list(APPEND TAMM_DEPENDENCIES TALSH)
  endif()
endif()

if(DEFINED LINALG_VENDOR)
    if(NOT ${LINALG_VENDOR} STREQUAL "BLIS")
        list(APPEND TAMM_DEPENDENCIES BLIS)
    endif()
endif()

if(USE_GA_AT)
    if(USE_DPCPP)
        list(APPEND TAMM_DEPENDENCIES IntelSYCL)
    endif()
    list(APPEND TAMM_DEPENDENCIES GlobalArrays)
    if(USE_SCALAPACK)
        list(APPEND TAMM_DEPENDENCIES ScaLAPACK LAPACK BLAS)
    else()
        list(APPEND TAMM_DEPENDENCIES LAPACK BLAS)
    endif()
else()
    list(APPEND TAMM_DEPENDENCIES GlobalArrays) #BLAS LAPACK
endif()

if(USE_GAUXC)
  list(APPEND TAMM_DEPENDENCIES GauXC)
endif()

set(TAMM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tamm)
set(TAMM_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/)
set(TAMM_METHODS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/methods/)

if(NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release)
endif()

set(CMSB_BASISSET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/methods/basis)
build_cmsb_module(${CMAKE_CURRENT_LIST_DIR})

