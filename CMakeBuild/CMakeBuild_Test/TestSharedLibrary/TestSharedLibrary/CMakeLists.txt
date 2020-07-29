cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TestSharedLibrary VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros)
set(TestSharedLibrarySRCS TestSharedLibrary.cpp)
set(TestSharedLibraryHEADERS TestSharedLibrary.hpp)
nwchemex_add_library(TestSharedLibrary
                     TestSharedLibrarySRCS
                     TestSharedLibraryHEADERS
                     "" "")
