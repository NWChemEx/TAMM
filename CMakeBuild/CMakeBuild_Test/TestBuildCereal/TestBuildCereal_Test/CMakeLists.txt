cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TestBuildCereal-Test VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros)
add_cxx_unit_test(TestBuildCereal)
