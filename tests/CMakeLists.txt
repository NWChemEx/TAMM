cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TAMM-Test VERSION 0.0.0 LANGUAGES CXX)
find_package(TAMM REQUIRED)
include(TargetMacros)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")

include(${CMAKE_CURRENT_LIST_DIR}/tamm/test_tamm.cmake)
if(NOT USE_UPCXX)
  include(${CMAKE_CURRENT_LIST_DIR}/ga/test_ga.cmake)
endif()

