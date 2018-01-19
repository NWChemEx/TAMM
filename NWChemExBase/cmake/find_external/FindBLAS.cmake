#
# Attempts to find a BLAS distribution.
#
# After this module runs the following variables are set:
#   BLAS_LIBRARIES : The literal BLAS library(s) to link against
#   BLAS_FOUND     : True if we found a BLAS library or if the user provided one
#
include(FindPackageHandleStandardArgs)
find_library(BLAS_LIBRARIES libblas${CMAKE_STATIC_LIBRARY_SUFFIX})
find_package_handle_standard_args(BLAS DEFAULT_MSG BLAS_LIBRARIES)
