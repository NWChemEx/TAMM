#
# Attempts to find a LAPACK distribution.  This module will override the one
# that comes with CMake.
#
# After this module runs the following variables are set:
#   LAPACK_LIBRARY   : The LAPACK library(s)
#   LAPACK_LIBRARIES : The LAPACK library(s) plus those of any dependencies
#   LAPACK_FOUND     : True if we found a LAPACK library
#
# Users can override paths found by this module in several ways:
# 1. Specify LAPACK_LIBRARIES
#    - This will prevent attempts to find LAPACK or its dependencies and will
#      use the supplied paths instead
# 2. Specify LAPACK_LIBRARY
#    - This will not look for LAPACK, but will look for its dependencies
include(FindPackageHandleStandardArgs)

is_valid(LAPACK_LIBRARIES FINDLAPACK_LIBS_SET)
if(NOT FINDLAPACK_LIBS_SET)
    find_library(LAPACK_LIBRARY NAMES lapack NO_CMAKE_SYSTEM_PATH)
    find_package_handle_standard_args(LAPACK DEFAULT_MSG LAPACK_LIBRARY)

    #Now we need to find a BLAS library that hopefully is compatible with our LAPACK
    find_package(BLAS)

    #Here we test that it is, which as you can see we actually don't...
    set(LAPACK_LIBRARIES ${LAPACK_LIBRARY} ${BLAS_LIBRARIES})

    #Now if we built LAPACK we need to find the standard Fortran libraries,
    #specifically the ones our LAPACK was compiled with
    find_package(StandardFortran REQUIRED)

    #Here's where we test that it worked, which as you can see we don't
    list(APPEND LAPACK_LIBRARIES ${STANDARDFORTRAN_LIBRARIES})
endif()

find_package_handle_standard_args(LAPACK DEFAULT_MSG LAPACK_LIBRARIES)
