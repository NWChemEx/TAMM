#
# Attempts to find a LAPACK distribution.
#
# After this module runs the following variables are set:
#   LAPACK_LIBRARIES : The literal LAPACK library(s) to link against
#   LAPACK_FOUND     : True if we found a LAPACK library
#
include(FindPackageHandleStandardArgs)
find_library(LAPACK_LIBRARIES liblapack${CMAKE_STATIC_LIBRARY_SUFFIX})

#Now we need to find a BLAS library that hopefully is compatible with our LAPACK
find_package(BLAS)

#Here we test that it is, which as you can see we actually don't...
list(APPEND LAPACK_LIBRARIES ${BLAS_LIBRARIES})

#Now if we built LAPACK we need to find the standard Fortran libraries,
#specifically the ones our LAPACK was compiled with
find_package(StandardFortran REQUIRED)

#Here's where we test that it worked, which as you can see we don't

list(APPEND LAPACK_LIBRARIES ${STANDARDFORTRAN_LIBRARIES})
find_package_handle_standard_args(LAPACK DEFAULT_MSG LAPACK_LIBRARIES)
