# - Try to find Global Arrays
#
#  The user may specify GLOBALARRAYS_ROOT_DIR to aid find_packge in
#  finding an already installed Global Arrays
#
#  Once done this will define
#  GLOBALARRAYS_FOUND - System has Global Arrays
#  GLOBALARRAYS_CONFIG - The ga-config binary path
#  GLOBALARRAYS_INCLUDE_DIR - The Global Arrays include directories
#  GLOBALARRAYS_LIBRARIES - The libraries needed to use Global Arrays

if(NOT DEFINED GLOBALARRAYS_ROOT_DIR)
    find_package(PkgConfig)
    pkg_check_modules(PC_GLOBALARRAYS QUIET ga)
endif()

find_path(GLOBALARRAYS_INCLUDE_DIR ga.h
          HINTS ${PC_GLOBALARRAYS_INCLUDEDIR}
                ${PC_GLOBALARRAYS_INCLUDE_DIRS}
          PATHS ${GLOBALARRAYS_ROOT_DIR}
         )

find_path(GLOBALARRAYS_CONFIG ga-config
         HINTS ${PC_GLOBALARRAYS_BINDIR}
               ${PC_GLOBALARRAYS_BIN_DIRS}
         PATHS ${GLOBALARRAYS_ROOT_DIR} 
         PATH_SUFFIXES bin
        )         

find_library(GLOBALARRAYS_C_LIBRARY NAMES libga${CMAKE_STATIC_LIBRARY_SUFFIX}
             HINTS ${PC_GLOBALARRAYS_LIBDIR}
                   ${PC_GLOBALARRAYS_LIBRARY_DIRS}
             PATHS ${GLOBALARRAYS_ROOT_DIR}
        )

find_library(GLOBALARRAYS_ARMCI_LIBRARY libarmci${CMAKE_STATIC_LIBRARY_SUFFIX}
             HINTS ${PC_GLOBALARRAYS_LIBDIR}
                   ${PC_GLOBALARRAYS_LIBRARY_DIRS}
             PATHS ${GLOBALARRAYS_ROOT_DIR}
        )
set(GLOBALARRAYS_LIBRARIES ${GLOBALARRAYS_C_LIBRARY}
                           ${GLOBALARRAYS_ARMCI_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GlobalArrays DEFAULT_MSG
                                  GLOBALARRAYS_LIBRARIES
                                  GLOBALARRAYS_INCLUDE_DIR
                                  GLOBALARRAYS_CONFIG
)
set(GLOBALARRAYS_INCLUDE_DIRS ${GLOBALARRAYS_INCLUDE_DIR})
set(GLOBALARRAYS_FOUND ${GlobalArrays_FOUND})

