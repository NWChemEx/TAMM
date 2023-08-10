cmake_minimum_required(VERSION 3.22.0)

include(FetchContent)

project(TAMM VERSION 0.0.0 LANGUAGES C CXX Fortran)
include(${PROJECT_SOURCE_DIR}/cmake/check_user_env.cmake)

find_package( CMakeBuild QUIET )

if( NOT ${CMakeBuild_FOUND} )
  if (NOT CMSB_TAG)
    set(CMSB_TAG main)
    if (ENABLE_DEV_MODE)
      set(CMSB_TAG develop)
    endif()
  endif()

  if(ENABLE_OFFLINE_BUILD)
    FetchContent_Declare(
      CMakeBuild
      URL ${DEPS_LOCAL_PATH}/CMakeBuild
    )
  else()
    FetchContent_Declare(
      CMakeBuild
      GIT_REPOSITORY https://github.com/NWChemEx-Project/CMakeBuild.git
      GIT_TAG ${CMSB_TAG}
    )
  endif()

  FetchContent_MakeAvailable( CMakeBuild )
  set(CMSB_MACROS ${CMakeBuild_SOURCE_DIR}/cmake/macros)

  FetchContent_MakeAvailable( linalg-cmake-modules )
  set(CMSB_LAM_PATH ${linalg-cmake-modules_SOURCE_DIR})
endif()

set(CMSB_PROJECTS TAMM)

set(TAMM_DEPENDENCIES NJSON MSGSL DOCTEST SPDLOG Eigen3 HPTT HDF5)

if(USE_CUDA OR USE_HIP OR USE_DPCPP)
  list(APPEND TAMM_DEPENDENCIES Librett)
endif()

list(APPEND TAMM_DEPENDENCIES GlobalArrays) #BLAS LAPACK

# Libint2, GauXC, FCI
if(DEFINED MODULES AND (NOT "${MODULES}" STREQUAL ""))
  list(TRANSFORM MODULES TOUPPER)
  list(APPEND TAMM_DEPENDENCIES EcpInt LibInt2)
  message(STATUS "MODULES specified = ${MODULES}")
  if ("DFT" IN_LIST MODULES)
    list(APPEND TAMM_DEPENDENCIES GauXC)
  endif()
  if ("FCI" IN_LIST MODULES)
    list(APPEND TAMM_DEPENDENCIES MACIS)
  endif()
endif()

set(BUILD_METHODS OFF)
set(TAMM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tamm)
set(TAMM_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/)

if(NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release)
endif()

build_cmsb_module(${CMAKE_CURRENT_LIST_DIR})

