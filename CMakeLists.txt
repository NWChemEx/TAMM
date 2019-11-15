cmake_minimum_required(VERSION 3.13)
project(tamm VERSION 0.0.0 LANGUAGES C CXX Fortran)
include(${PROJECT_SOURCE_DIR}/cmake/check_user_env.cmake)
set(NWX_PROJECTS tamm)

set(tamm_DEPENDENCIES NWX_MPI MSGSL Eigen3 GlobalArrays LibInt2 CBLAS LAPACKE HPTT)
if(USE_CUDA) 
   list(APPEND tamm_DEPENDENCIES TALSH)
endif()
if(USE_SCALAPACK)
    list(APPEND tamm_DEPENDENCIES ScaLAPACK CXXBLACS)
endif()
if(USE_BLIS)
    list(APPEND tamm_DEPENDENCIES BLIS)
endif()

set(tamm_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tamm)
set(tamm_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/)

if(NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif()

find_package(CMakeBuild)
build_nwchemex_module(${CMAKE_CURRENT_LIST_DIR})

