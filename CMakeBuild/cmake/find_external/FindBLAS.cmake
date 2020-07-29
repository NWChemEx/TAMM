#
# Attempts to find a BLAS distribution.  This module is meant to replace the one
# distributed with CMake.
#
# After this module runs the following variables are set:
#   BLAS_LIBRARY   : The BLAS library(s)
#   BLAS_LIBRARIES : The BLAS library(s) plus those of all dependencies
#   BLAS_FOUND     : True if we found a BLAS library or if the user provided one
#
# Users can override paths found by this module in several ways:
# 1. Specify BLAS_LIBRARIES
#    - This will prevent attempts to find BLAS or its dependencies and will use
#      the supplied paths instead
# 2. Specify BLAS_LIBRARY
#    - This will not look for BLAS, but will look for its dependencies
include(FindPackageHandleStandardArgs)
is_valid(BLAS_LIBRARIES FINDBLAS_LIBS_SET)
if(NOT FINDBLAS_LIBS_SET)
    find_library(BLAS_LIBRARY NAMES blas NO_CMAKE_SYSTEM_PATH)
    find_package_handle_standard_args(BLAS DEFAULT_MSG BLAS_LIBRARY)
    set(BLAS_LIBRARIES ${BLAS_LIBRARY})

endif()
find_package_handle_standard_args(BLAS DEFAULT_MSG BLAS_LIBRARIES)
