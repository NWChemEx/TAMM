cmake_minimum_required(VERSION 3.6)
project(TestSharedLibrary VERSION 1.0.0 LANGUAGES CXX)
find_package(CMakeBuild)
build_nwchemex_module(${CMAKE_CURRENT_LIST_DIR})
