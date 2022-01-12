cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TAMM-Test VERSION 0.0.0 LANGUAGES CXX)
find_package(TAMM REQUIRED)
include(TargetMacros)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../methods/util/external)
include(${CMAKE_CURRENT_LIST_DIR}/tamm/test_tamm.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/ga/test_ga.cmake)

