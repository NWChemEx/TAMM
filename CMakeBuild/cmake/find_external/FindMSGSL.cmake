# - Try to find Microsoft GSL Library
#
#  To aid find_package in locating MS GSL, the user may set the
#  variable MSGSL_ROOT_DIR to the root of the GSL install
#  directory.
#
#  Once done this will define
#  MSGSL_FOUND - System has MS GSL
#  MSGSL_INCLUDE_DIR - The MS GSL include directories

if(NOT DEFINED MSGSL_ROOT_DIR)
    find_package(PkgConfig)
    pkg_check_modules(PC_MSGSL QUIET gsl)
endif()

find_path(MSGSL_INCLUDE_DIR gsl/gsl
          HINTS ${PC_MSGSL_INCLUDEDIR}
                ${PC_MSGSL_INCLUDE_DIRS}
          PATHS ${MSGSL_ROOT_DIR}
          )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MSGSL DEFAULT_MSG
                                  MSGSL_INCLUDE_DIR)

set(MSGSL_FOUND ${MSGSL_FOUND})
set(MSGSL_INCLUDE_DIRS ${MSGSL_INCLUDE_DIR})
