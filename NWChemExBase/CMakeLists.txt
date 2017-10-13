################################################################################
#
# This is the effective top-level CMakeLists.txt file for the current NWChemEx
# library.  You shouldn't have to modify it.
#
################################################################################

#We require C++14
set(CMAKE_CXX_STANDARD 14)

#To "namespace" protect our targets we append this to them
set(TARGET_SUFFIX _nwx)

#Source our macros
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/macros")

include(OptionMacros)
include(Flags)

################################################################################
#
# Read some built-in CMake variables, report their values.
#
################################################################################

option_w_default(CMAKE_BUILD_TYPE Release)
option_w_default(BUILD_SHARED_LIBS TRUE)

bundle_cmake_args(CORE_CMAKE_OPTIONS CMAKE_CXX_COMPILER CMAKE_C_COMPILER
    CMAKE_BUILD_TYPE BUILD_SHARED_LIBS CMAKE_INSTALL_PREFIX CMAKE_CXX_STANDARD)

bundle_cmake_list(CORE_CMAKE_LISTS CMAKE_PREFIX_PATH CMAKE_INSTALL_RPATH
    CMAKE_MODULE_PATH)

bundle_cmake_strings(CORE_CMAKE_STRINGS CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG)

################################################################################
#
# Handle NWChemEx's special options, report their values.
#
################################################################################

option_w_default(BUILD_TESTS TRUE)

################################################################################
#
# Configure some options that tell us where things for the super project are
#
################################################################################

get_filename_component(SUPER_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
set(SUPER_PROJECT_SRC ${SUPER_PROJECT_ROOT}/${PROJECT_NAME})
set(SUPER_PROJECT_TEST ${SUPER_PROJECT_ROOT}/${PROJECT_NAME}_Test)
set(STAGE_DIR ${CMAKE_BINARY_DIR}/stage)
set(STAGE_INSTALL_DIR ${STAGE_DIR}${CMAKE_INSTALL_PREFIX})
#We use a different dir so tests don't get installed
set(TEST_STAGE_DIR ${CMAKE_BINARY_DIR}/test_stage)

################################################################################
#
# Find the dependencies of the project.  Assign their libraries to
# NWCHEMEX_LIBRARY_LIBRARIES and their includes to NWCHEMEX_LIBRARY_INCLUDE_DIRS
#
################################################################################
include(DependencyMacros)
include(ExternalProject)
foreach(depend ${NWCHEMEX_LIBRARY_DEPENDS})
    find_or_build_dependency(${depend})
endforeach()

################################################################################
#
# Build the main library
#
################################################################################

ExternalProject_Add(${PROJECT_NAME}${TARGET_SUFFIX}
    SOURCE_DIR ${SUPER_PROJECT_SRC}
    CMAKE_ARGS -DSUPER_PROJECT_ROOT=${SUPER_PROJECT_ROOT}
               ${CORE_CMAKE_OPTIONS}
    BUILD_ALWAYS 1
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
    CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
                     ${CORE_CMAKE_STRINGS}
                     -DNWCHEMEX_LIBRARY_DEPENDS:STRING=${NWCHEMEX_LIBRARY_DEPENDS}
)
foreach(depend ${NWCHEMEX_LIBRARY_DEPENDS})
    add_dependencies(${PROJECT_NAME}${TARGET_SUFFIX} ${depend}${TARGET_SUFFIX})
endforeach()

################################################################################
#                                                                              #
# The generated ProjectConfig.cmake file will need access to some of the CMake #
# tools packaged with NWChemExBase.  Rather than installing it under that      #
# library's cmake folder (and cluttering the install dir since presumably      #
# other libraries will use NWChemExBase too) we install ourselves.  Before we  #
# do that though we look for an installed version.                             #
#                                                                              #
################################################################################

find_package(NWChemExBase QUIET)
if(NOT NWCHEMEXLIBRARYBASE_FOUND)
    ExternalProject_Add(NWChemExBase${TARGET_SUFFIX}
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/NWChemExBase
        CMAKE_ARGS -DPLB_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}
                   -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
                   -DTARGET_SUFFIX=${TARGET_SUFFIX}
        BUILD_ALWAYS 1
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
    )
else() # Add a fake target so we don't need if statements
    add_library(NWChemExBase${TARGET_SUFFIX} INTERFACE)
endif()
add_dependencies(NWChemExBase${TARGET_SUFFIX} ${PROJECT_NAME}${TARGET_SUFFIX})

################################################################################
#                                                                              #
# Build the main library's tests if the user requested them to be built.  Will #
# use the staged ProjectConfig.cmake file.                                     #
#                                                                              #
################################################################################
if(${BUILD_TESTS})
    list(APPEND CMAKE_PREFIX_PATH "${STAGE_INSTALL_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${STAGE_INSTALL_DIR}")
    ExternalProject_Add(${PROJECT_NAME}_TEST
        SOURCE_DIR ${SUPER_PROJECT_TEST}
        CMAKE_ARGS -DSUPER_PROJECT_ROOT=${SUPER_PROJECT_TEST}
                   ${CORE_CMAKE_OPTIONS}
        BUILD_ALWAYS 1
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${TEST_STAGE_DIR}
        CMAKE_CACHE_ARGS -DCMAKE_PREFIX_PATH:LIST=${CMAKE_PREFIX_PATH}
                         -DCMAKE_INSTALL_RPATH:LIST=${CMAKE_INSTALL_RPATH}
                         -DCMAKE_MODULE_PATH:LIST=${CMAKE_MODULE_PATH}
                         ${CORE_CMAKE_STRINGS}
    )
    add_dependencies(${PROJECT_NAME}_TEST NWChemExBase${TARGET_SUFFIX})

    # This file will allow us to run ctest in the top-level build dir
    # Basically it just defers to the actual top-level CTestTestfile.cmake in the
    # build directory for this project
    file(WRITE ${CMAKE_BINARY_DIR}/CTestTestfile.cmake
               "subdirs(test_stage${CMAKE_INSTALL_PREFIX}/tests)")
endif()

# Install the staging directory
install(DIRECTORY ${STAGE_DIR}/${CMAKE_INSTALL_PREFIX}/
        DESTINATION ${CMAKE_INSTALL_PREFIX} USE_SOURCE_PERMISSIONS)



#####################
# uninstall target
#####################
#configure_file(
#"${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
#"${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
#IMMEDIATE @ONLY)
#
#add_custom_target(uninstall
#  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
