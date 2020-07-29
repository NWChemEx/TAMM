# - Try to find CXXBLACS
#
#  To aid find_package in locating CXXBLACS, the user may set the
#  variable CXXBLACS_ROOT_DIR to the root of the CXXBLACS install
#  directory.
#
#  Once done this will define
#  CXXBLACS_FOUND - System has CXXBLACS
#  CXXBLACS_INCLUDE_DIR - The CXXBLACS include directories

if(NOT DEFINED CXXBLACS_ROOT_DIR)
    find_package(PkgConfig)
    pkg_check_modules(PC_CXXBLACS QUIET cxxblacs)
endif()

find_path(CXXBLACS_INCLUDE_DIR cxxblacs/blacs.hpp
          HINTS ${PC_CXXBLACS_INCLUDEDIR}
                ${PC_CXXBLACS_INCLUDE_DIRS}
          PATHS ${CXXBLACS_ROOT_DIR}
          )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CXXBLACS DEFAULT_MSG
                                  CXXBLACS_INCLUDE_DIR)

set(CXXBLACS_FOUND ${CXXBLACS_FOUND})
set(CXXBLACS_INCLUDE_DIRS ${CXXBLACS_INCLUDE_DIR})
