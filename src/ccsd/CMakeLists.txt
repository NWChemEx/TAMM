cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(CCSD_DRIVER VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) 

set(CCSD_DRIVER_SRCS ccsd_driver.cc)
set(CCSD_DRIVER_DEFINITONS )
set(CCSD_DRIVER_LFLAGS )

nwchemex_add_executable(ccsd_driver CCSD_DRIVER_SRCS CCSD_DRIVER_DEFINTIONS
                                    CCSD_DRIVER_LFLAGS)
