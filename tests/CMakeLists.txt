cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(tamm-Test VERSION 0.0.0 LANGUAGES CXX)
find_package(tamm REQUIRED)
include(TargetMacros)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../utils/external)
include(${CMAKE_CURRENT_LIST_DIR}/tamm/test_tamm.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/ga/test_ga.cmake)
