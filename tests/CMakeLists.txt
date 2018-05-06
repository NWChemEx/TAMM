cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(tamm-Test VERSION 0.0.0 LANGUAGES CXX)
find_package(tamm REQUIRED)
include(TargetMacros)

include(${CMAKE_CURRENT_LIST_DIR}/tamm/test_tamm.cmake)
