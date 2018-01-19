# - Try to find Antlr CppRuntime Library
#
#  To aid find_package in locating the Antlr CppRuntime the user may set the
#  variable ANTLRCPPRUNTIME_ROOT_DIR to the root of the Antlr CppRuntime install
#  directory.
#
#  Once done this will define
#  ANTLRCPPRUNTIME_FOUND - System has Antlr CppRuntime
#  ANTLRCPPRUNTIME_INCLUDE_DIR - The Antlr CppRuntime include directories
#  ANTLRCPPRUNTIME_LIBRARY - The library needed to use Antlr CppRuntime

if(NOT DEFINED ANTLRCPPRUNTIME_ROOT_DIR)
    find_package(PkgConfig)
    pkg_check_modules(PC_ANTLRCPPRUNTIME QUIET libantlr4-runtime)
endif()

find_path(ANTLRCPPRUNTIME_INCLUDE_DIR antlr4-runtime.h
          HINTS ${PC_ANTLRCPPRUNTIME_INCLUDEDIR}
                ${PC_ANTLRCPPRUNTIME_INCLUDE_DIRS}
          PATHS ${ANTLRCPPRUNTIME_ROOT_DIR}
          PATH_SUFFIXES antlr4-runtime)

find_library(ANTLRCPPRUNTIME_LIBRARY
             NAMES libantlr4-runtime${CMAKE_STATIC_LIBRARY_SUFFIX}
             HINTS ${PC_ANTLRCPPRUNTIME_LIBDIR}
                   ${PC_ANTLRCPPRUNTIME_LIBRARY_DIRS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AntlrCppRuntime DEFAULT_MSG
                                  ANTLRCPPRUNTIME_LIBRARY
                                  ANTLRCPPRUNTIME_INCLUDE_DIR)
