# - Try to find HPTT Library
#
#  To aid find_package in locating HPTT, the user may set the
#  variable HPTT_ROOT_DIR to the root of the HPTT install
#  directory.
#
#  Once done this will define
#  HPTT_FOUND - System has HPTT
#  HPTT_INCLUDE_DIR - The HPTT include directories
#  HPTT_LIBRARY - The library needed to use HPTT

if(NOT DEFINED HPTT_ROOT_DIR)
    find_package(PkgConfig)
    pkg_check_modules(PC_HPTT QUIET libhptt)
endif()

find_path(HPTT_INCLUDE_DIR hptt.h
          HINTS ${PC_HPTT_INCLUDEDIR}
                ${PC_HPTT_INCLUDE_DIRS}
          PATHS ${HPTT_ROOT_DIR}
          )

find_library(HPTT_LIBRARY
             NAMES hptt
             HINTS ${PC_HPTT_LIBDIR}
                   ${PC_HPTT_LIBRARY_DIRS}
	     PATHS ${HPTT_ROOT_DIR}
	  )


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HPTT DEFAULT_MSG
                                  HPTT_LIBRARY
                                  HPTT_INCLUDE_DIR)

if(USE_OPENMP)                                  
      find_package(OpenMP REQUIRED)
endif()

set(HPTT_LIBRARIES ${HPTT_LIBRARY} ${OpenMP_CXX_FLAGS})
set(HPTT_INCLUDE_DIRS ${HPTT_INCLUDE_DIR})
