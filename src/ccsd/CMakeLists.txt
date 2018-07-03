cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(ccsd_driver VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) 

set(ccsd_driver_SRCS ccsd_driver.cpp)
set(ccsd_driver_DEFINITONS )
set(ccsd_driver_LFLAGS )

nwchemex_add_executable(ccsd_driver ccsd_driver_SRCS ccsd_driver_DEFINTIONS
                                    ccsd_driver_LFLAGS)
