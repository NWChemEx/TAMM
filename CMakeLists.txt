cmake_minimum_required(VERSION 3.8)
project(tamm VERSION 0.0.0 LANGUAGES CXX)
set(NWX_PROJECTS tamm)
set(tamm_DEPENDENCIES GlobalArrays NWX_MPI CBLAS LAPACKE )

set(tamm_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tamm)
set(tamm_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/)

find_package(CMakeBuild)
build_nwchemex_module(${CMAKE_CURRENT_LIST_DIR})

