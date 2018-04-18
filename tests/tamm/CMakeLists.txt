cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TAMM-Test VERSION 0.0.0 LANGUAGES CXX)
find_package(tamm REQUIRED)
include(TargetMacros)

add_cxx_unit_test(test_index_space)
