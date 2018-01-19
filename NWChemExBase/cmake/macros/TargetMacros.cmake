################################################################################
#
# Macros for defining new targets.
#
# General definitions so they aren't defined for each function:
#     name      : The name of the target
#     flags     : These are compile-time flags to pass to the compiler
#     includes  : The directories containing include files for the target
#     libraries : The libraries the target needs to link against
#
################################################################################

include(CTest)
enable_testing()

include(UtilityMacros)
include(DependencyMacros)
include(AssertMacros)

#Little trick so we always know this directory even when we are in a function
set(DIR_OF_TARGET_MACROS ${CMAKE_CURRENT_LIST_DIR})

#
# This is code factorization for the next few functions and only used internally
#
#  To make hide a lot of complexity from the user we transport the following:
#     - NWX_INCLUDE_DIR : path the root of source tree, should be the include
#                         path for all headers included with target
#     - NWX_DEPENDENCIES : A list of all dependencies this target depends on
#                          will be passed to find_dependencies so names must be
#                          find_packgage-able
#
# Syntax: nwchemex_set_up_target(<name> <flags> <lflags> <includes> <install>
#         name : the name of the target
#         flags : a list of compile flags needed to compile the target in
#                 addition to those added by dependencies
#         lflags : a list of flags needed to link the target in addition to
#                  those provided by dependencies
#         install : The path to install the target to (relative to
#                   CMAKE_INSTALL_PREFIX
#
function(nwchemex_set_up_target __name __flags __lflags __install)
    set(__headers_copy ${${__includes}})
    make_full_paths(__headers_copy)
    foreach(__depend ${NWX_DEPENDENCIES})
        find_dependency(${__depend} __DEPEND_INCLUDES
                                    __DEPEND_LIBRARIES
                                    __DEPEND_FLAG
                                    __DEPEND_LFLAG
                                    __${__depend}_found)
        assert(__${__depend}_found)
    endforeach()
    list(APPEND __all_flags ${__flags} ${__DEPEND_FLAG})
    list(APPEND __all_lflags ${__lflags} ${__DEPEND_LFLAG})
    list(APPEND __DEPEND_INCLUDES ${NWX_INCLUDE_DIR})
    debug_message("Adding target ${__name}:")
    debug_message("    Include Directories: ${__DEPEND_INCLUDES}")
    debug_message("    Compile Flags: ${__all_flags}")
    debug_message("    Link Libraries: ${__DEPEND_LIBRARIES}")
    debug_message("    Link Flags: ${__all_lflags}")
    target_link_libraries(${__name} PRIVATE "${__DEPEND_LIBRARIES}")
    target_compile_options(${__name} PRIVATE "${__all_flags}")
    target_include_directories(${__name} PRIVATE ${__DEPEND_INCLUDES})
    set_property(TARGET ${__name} PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
    set_property(TARGET ${__name} PROPERTY LINK_FLAGS "${__all_lflags}")
    install(TARGETS ${__name} DESTINATION ${__install})
endfunction()

#
# Macro for building an executable
#
# Syntax: nwchemex_add_executable(<Name> <Sources> <Headers> <Flags> <LFlags>
#     - Name : The name to use for the executable
#     - Sources : A list of source files to compile to form the executable.
#                 Should be relative paths.
#     - Flags   : These are flags needed to compile the executable in
#                 addition to those provided by dependencies.
#     - LFlags  " These are the flags needed to link this executable in addition
#                 to those provided by the dependencies
#
function(nwchemex_add_executable __name __srcs __flags __lflags)
    set(__srcs_copy ${${__srcs}})
    make_full_paths(__srcs_copy)
    add_executable(${__name} ${__srcs_copy})
    nwchemex_set_up_target(${__name}
                           "${${__flags}}"
                           "${${__lflags}}"
                           bin/${__name})
endfunction()

#
# Macro for building a library
#
# Syntax: nwchemex_add_library(<Name> <Sources> <Headers> <Flags> <LFlags>
#     - Name : The name to use for the library, result will be libName.so
#     - Sources : A list of source files to compile to form the library.  Should
#                 be relative paths.
#     - Headers : A list of header files that should be considered the public
#                 API of the library.  Should be relative paths.
#     - Flags   : These are flags needed to compile the library in addition to
#                 those provided by this library's dependencies.
#     - LFlags  " These are the flags needed to link this library in addition
#                 to those provided by the dependencies
#
function(nwchemex_add_library __name __srcs __headers __flags __lflags)
    set(__srcs_copy ${${__srcs}})
    make_full_paths(__srcs_copy)
    is_valid(__srcs_copy HAS_LIBRARY)
    if(HAS_LIBRARY)
        add_library(${__name} ${__srcs_copy})
        nwchemex_set_up_target(${__name}
                "${${_flags}}"
                "${${__lflags}}"
                lib/${__name})
    endif()
    set(NWCHEMEX_LIBRARY_NAME ${__name})
    set(NWCHEMEX_LIBRARY_HEADERS ${${__headers}})
    get_filename_component(__CONFIG_FILE ${DIR_OF_TARGET_MACROS} DIRECTORY)
    configure_file("${__CONFIG_FILE}/NWChemExTargetConfig.cmake.in"
                    ${__name}Config.cmake @ONLY)
    install(FILES ${CMAKE_BINARY_DIR}/${__name}Config.cmake
            DESTINATION share/cmake/${__name})
    foreach(__header_i ${${__headers}})
        #We want to preserve structure so get directory (if it exists)
        get_filename_component(__header_i_dir ${__header_i} DIRECTORY)
        install(FILES ${__header_i}
                DESTINATION include/${__name}/${__header_i_dir})
    endforeach()
endfunction()

#
# Defines a test.  This is the base call.  You'll likely be using one of the
#    functions that follows this declaration.
#
# Syntax: nwchemex_add_test(<name> <test_file>)
#    - name      : the name of the test
#    - test_file : This is the file containing the test
function(nwchemex_add_test __name __test_file)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(__file_copy ${__test_file})
    make_full_paths(__file_copy)
    add_executable(${__name} ${__file_copy})
    nwchemex_set_up_target(${__name} "" "" "tests")
    add_test(NAME ${__name} COMMAND ./${__name})
    #target_include_directories(${__name} PRIVATE ${NWCHEMEXBASE_INCLUDE_DIRS})
    install(FILES ${CMAKE_BINARY_DIR}/CTestTestfile.cmake DESTINATION tests)
endfunction()

#
# Specializes add_test to C++ unit tests.  Assumes the test is coded up in a
# file with the same name as the test and a ".cpp" extension.
#
function(add_cxx_unit_test __name)
    nwchemex_add_test(${__name} ${__name}.cpp)
    set_tests_properties(${__name} PROPERTIES LABELS "UnitTest")
endfunction()
