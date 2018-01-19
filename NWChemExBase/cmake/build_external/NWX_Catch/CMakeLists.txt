################################################################################
#
#  Now we build NWX_Catch, which is our installed version of Catch. It
#  differs from the default one in that we've installed the header and we've
#  taken the liberty of compiling the main function into a library to help
#  with compile times.
#
################################################################################
cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(CATCH VERSION 0.0.0 LANGUAGES CXX)
include(ExternalProject)

#The root of the catch project from our perspective
set(CATCH_PREFIX ${CMAKE_BINARY_DIR}/catch)
set(CATCH_URL https://github.com/catchorg/Catch2.git)
#Where CMake will clone Catch to
set(CATCH_DOWNLOAD_DIR ${CATCH_PREFIX}/src/Catch_Headers)
#The path to the single header file in the clone
set(CATCH_HEADER_FILE ${CATCH_DOWNLOAD_DIR}/single_include/catch.hpp)
#Effective installation path for staging the header
set(CATCH_STAGE_INSTALL ${CMAKE_BINARY_DIR}/stage${CMAKE_INSTALL_PREFIX})
#Staged include install path
set(CATCH_STAGE_INCLUDE ${CATCH_STAGE_INSTALL}/include)
#Header's full staged path
set(CATCH_HEADER_INSTALL ${CATCH_STAGE_INCLUDE}/catch/catch.hpp)

#Actually get Catch's header and install it
ExternalProject_Add(Catch_Headers
        PREFIX ${CATCH_PREFIX}
        GIT_REPOSITORY ${CATCH_URL}
        DOWNLOAD_DIR ${CATCH_DOWNLOAD_DIR}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy ${CATCH_HEADER_FILE}
                                                 ${CATCH_HEADER_INSTALL}
        )

#Now make a library for the main function

#Generate source file
set(CATCH_MAIN ${CMAKE_BINARY_DIR}/catch_main.cpp)
file(WRITE ${CATCH_MAIN} "#define CATCH_CONFIG_MAIN\n")
file(APPEND ${CATCH_MAIN} "#include \"catch/catch.hpp\"\n")

#Make library target
add_library(catch ${CATCH_MAIN})
target_include_directories(catch PRIVATE ${CATCH_STAGE_INCLUDE})
install(TARGETS catch DESTINATION lib)
add_dependencies(catch Catch_Headers)
install(DIRECTORY ${CATCH_STAGE_INSTALL}/
        DESTINATION ${CMAKE_INSTALL_PREFIX} USE_SOURCE_PERMISSIONS)
