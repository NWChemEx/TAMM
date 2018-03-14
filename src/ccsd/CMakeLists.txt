cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(CCSD_DRIVER VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) #For nwchemex_add_executable

set(CCSD_DRIVER_SRCS ccsd_driver.cc)
set(CCSD_DRIVER_DEFINITONS )
set(CCSD_DRIVER_LFLAGS "-lpthread")

if(NOT "${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Darwin")
    set(CCSD_DRIVER_LFLAGS "${CCSD_DRIVER_LFLAGS} -lrt")
endif()

nwchemex_add_executable(ccsd_driver CCSD_DRIVER_SRCS CCSD_DRIVER_DEFINTIONS
                                    CCSD_DRIVER_LFLAGS)
