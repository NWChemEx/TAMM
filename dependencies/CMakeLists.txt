cmake_minimum_required(VERSION 3.7.0 FATAL_ERROR)
project (TAMM_DEPS C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/tammdeps-install" CACHE PATH "default install path" FORCE )
endif()

include(libint)
include(eigen)
#include(tiledarray)
#include(mpqc)
#include(globalarrays)

add_executable(tamm_deps ${PROJECT_SOURCE_DIR}/dummy.cpp)
add_dependencies(tamm_deps EIGEN3 LIBINT)
