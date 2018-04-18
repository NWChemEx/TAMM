cmake_minimum_required(VERSION 3.6)
project(TAMM VERSION 0.0.0 LANGUAGES CXX)
set(NWX_PROJECTS TAMM CCSD)
set(TAMM_DEPENDENCIES NWX_MPI CBLAS LAPACKE Eigen3 GlobalArrays LibInt)

set(TAMM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tamm)
set(TAMM_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/tamm)

find_package(CMakeBuild)
build_nwchemex_module(${CMAKE_CURRENT_LIST_DIR})

