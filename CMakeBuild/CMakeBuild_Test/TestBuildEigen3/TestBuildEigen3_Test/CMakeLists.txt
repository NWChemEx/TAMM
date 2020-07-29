cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TestBuildEigen3-Test VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros)
add_cxx_unit_test(TestBuildEigen3)
