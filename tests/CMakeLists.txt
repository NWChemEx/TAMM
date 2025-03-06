cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TAMM-Test VERSION 0.0.0 LANGUAGES CXX)
find_package(TAMM REQUIRED)
include(TargetMacros)

include(${CMAKE_CURRENT_LIST_DIR}/tamm/test_tamm.cmake)
if(NOT USE_UPCXX)
  include(${CMAKE_CURRENT_LIST_DIR}/ga/test_ga.cmake)
endif()

