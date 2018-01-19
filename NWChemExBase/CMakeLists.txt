################################################################################
#
# This is the effective top-level CMakeLists.txt file for the current NWChemEx
# library.  You shouldn't have to modify it.
#
################################################################################

#We require C++14 get it out of the way early
set(CMAKE_CXX_STANDARD 14)

#Set the environment up and pull-in macros we'll need
include(cmake/macros/SetPaths.cmake)
set_paths()
include(OptionMacros)
include(Flags)
include(DependencyMacros)
include(ExternalProject)

################################################################################
#
# Read some built-in CMake variables, report their values.
#
################################################################################

option_w_default(CMAKE_BUILD_TYPE Release)
option_w_default(BUILD_SHARED_LIBS TRUE)
option_w_default(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

#Make a list of all CMake variables that should be passed to all dependencies
bundle_cmake_args(CORE_CMAKE_OPTIONS CMAKE_CXX_COMPILER CMAKE_C_COMPILER
    CMAKE_BUILD_TYPE BUILD_SHARED_LIBS CMAKE_INSTALL_PREFIX CMAKE_CXX_STANDARD
    CMAKE_VERSION PROJECT_VERSION CMAKE_POSITION_INDEPENDENT_CODE)

bundle_cmake_list(CORE_CMAKE_LISTS CMAKE_PREFIX_PATH CMAKE_INSTALL_RPATH
    CMAKE_MODULE_PATH)

bundle_cmake_strings(CORE_CMAKE_STRINGS CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG)

################################################################################
#
# Handle NWChemEx's special options, report their values.
#
################################################################################

#What are the names of the subprojects you want us to build?
option_w_default(NWX_PROJECTS ${PROJECT_NAME})
foreach(__project ${NWX_PROJECTS})
    #Directory where the sub-project's source is located
    option_w_default(${__project}_SRC_DIR ${SUPER_PROJECT_ROOT}/${__project})
    #Includes should be relative to NWX_SRC_DIR without last directory
    get_filename_component(${__project}_INCLUDE_DIR "${${__project}_SRC_DIR}"
                           DIRECTORY)
    #Directory where your tests are
    option_w_default(${__project}_TEST_DIR
            ${SUPER_PROJECT_ROOT}/${__project}_Test)
    #Name of variable containing your project's dependencies
    option_w_default(${__project}_DEPENDENCIES "")
endforeach()

#Should we build the tests?
option_w_default(BUILD_TESTS TRUE)
#Do you want verbose printing on the CMake configuration?
option_w_default(NWX_DEBUG_CMAKE TRUE)

################################################################################
#
# Add the subprojects, their dependencies, and their tests
#
################################################################################

foreach(__project ${NWX_PROJECTS})
    ExternalProject_Add(${__project}_External
        SOURCE_DIR ${${__project}_SRC_DIR}
        CMAKE_ARGS -DNWX_INCLUDE_DIR=${${__project}_INCLUDE_DIR}
                   -DNWX_DEBUG_CMAKE=${NWX_DEBUG_CMAKE}
                   ${CORE_CMAKE_OPTIONS}
        BUILD_ALWAYS 1
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
        CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
                         ${CORE_CMAKE_STRINGS}
        -DNWX_DEPENDENCIES:STRING=${${__project}_DEPENDENCIES}
    )

    foreach(depend NWChemExBase ${${__project}_DEPENDENCIES})
        find_or_build_dependency(${depend} was_found)
        if(was_found)
            list(APPEND DEPENDS_WE_FOUND ${depend})
        else()
            list(APPEND DEPENDS_WERE_BUILDING ${depend})
        endif()
        add_dependencies(${__project}_External ${depend}_External)
    endforeach()

    if(${BUILD_TESTS})
        ExternalProject_Add(${__project}_Tests_External
            SOURCE_DIR ${${__project}_TEST_DIR}
            CMAKE_ARGS -DSUPER_PROJECT_ROOT=${SUPER_PROJECT_TEST}
                       -DNWX_DEBUG_CMAKE=${NWX_DEBUG_CMAKE}
                       ${CORE_CMAKE_OPTIONS}
                       ${CORE_CMAKE_STRINGS}
            BUILD_ALWAYS 1
            INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${TEST_STAGE_DIR}
            CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
                             -DNWX_DEPENDENCIES:STRING=
                "NWChemExBase;${__project}"
    )
    add_dependencies(${__project}_Tests_External ${__project}_External)

    # This file will allow us to run ctest in the top-level build dir
    # Basically it just defers to the actual top-level CTestTestfile.cmake in the
    # build directory for this project
    file(WRITE ${CMAKE_BINARY_DIR}/CTestTestfile.cmake
               "subdirs(test_stage${CMAKE_INSTALL_PREFIX}/tests)")
    endif()
endforeach()

# Install the staging directory
install(DIRECTORY ${STAGE_INSTALL_DIR}/
        DESTINATION ${CMAKE_INSTALL_PREFIX} USE_SOURCE_PERMISSIONS)

################################################################################
#
# Let the user know all the settings we worked out
#
################################################################################
message(STATUS
        "----------------------------------------------------------------------"
        "--------")
message(STATUS "Summary of ${PROJECT_NAME} Configuration Settings:")
message(STATUS
        "----------------------------------------------------------------------"
        "--------")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Found the following dependencies: ")
foreach(__depend ${DEPENDS_WE_FOUND})
    message(STATUS "    ${__depend}")
endforeach()
message(STATUS "Will build: ")
foreach(__depend ${DEPENDS_WERE_BUILDING})
    message(STATUS "    ${__depend}")
endforeach()

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
