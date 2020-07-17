cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(tamm-Methods VERSION 0.0.0 LANGUAGES CXX)
find_package(tamm REQUIRED)
include(TargetMacros)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include(${CMAKE_CURRENT_LIST_DIR}/scf/scf.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cc/cc.cmake)
