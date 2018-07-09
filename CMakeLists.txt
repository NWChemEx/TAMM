cmake_minimum_required(VERSION 3.6)
project(tamm VERSION 0.0.0 LANGUAGES CXX)
set(NWX_PROJECTS tamm ccsd)
set(tamm_DEPENDENCIES NWX_MPI CBLAS LAPACKE GlobalArrays)

set(tamm_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tamm)
set(tamm_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/)

set(ccsd_DEPENDENCIES tamm)
set(ccsd_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/ccsd)
set(ccsd_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/)

find_package(CMakeBuild)
build_nwchemex_module(${CMAKE_CURRENT_LIST_DIR})

