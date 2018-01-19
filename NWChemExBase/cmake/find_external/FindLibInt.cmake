# - Try to find LibInt
#
#  In order to aid find_package the user may set LIBINT_ROOT_DIR to the root of
#  the installed libint.
#
#  Once done this will define
#  LIBINT_FOUND - System has Libint
#  LIBINT_INCLUDE_DIR - The Libint include directories
#  LIBINT_LIBRARY - The libraries needed to use Libint
include(DependencyMacros)
find_package(Eigen3 REQUIRED)

#Prefer LIBINT_ROOT_DIR if the user specified it
if(NOT DEFINED LIBINT_ROOT_DIR)
    find_package(PkgConfig)
    pkg_check_modules(PC_LIBINT QUIET libint2)
endif()

find_path(LIBINT_INCLUDE_DIR libint2.hpp
          HINTS ${PC_LIBINT_INCLUDEDIR}
                ${PC_LIBINT_INCLUDE_DIRS}
          PATHS ${LIBINT_ROOT_DIR}
          PATH_SUFFIXES libint2)

find_library(LIBINT_LIBRARY NAMES int2
             HINTS ${PC_LIBINT_LIBDIR}
                   ${PC_LIBINT_LIBRARY_DIRS}
             PATHS ${LIBINT_ROOT_DIR}
        )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibInt DEFAULT_MSG
                                  LIBINT_LIBRARY LIBINT_INCLUDE_DIR)
set(LIBINT_LIBRARIES ${LIBINT_LIBRARY})
set(LIBINT_INCLUDE_DIRS ${LIBINT_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS})
