cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TestBuildLibInt-Test VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros)
add_cxx_unit_test(TestBuildLibInt)
