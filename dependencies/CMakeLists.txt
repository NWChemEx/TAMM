cmake_minimum_required(VERSION 3.7.0 FATAL_ERROR)
project (TAMM_DEPENDENCIES C CXX Fortran)

include(ExternalProject)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/tamm-deps" CACHE PATH "default install path" FORCE)
endif()

if(DEFINED TAMM_PROC_COUNT)
    set(TAMM_PROC_COUNT ${TAMM_PROC_COUNT})
else()
    include(ProcessorCount)
    ProcessorCount(TAMM_PROC_COUNT)
    message(STATUS "Number of cores detected = ${TAMM_PROC_COUNT}")
endif()

include(libint)
include(eigen)
include(globalarrays)
