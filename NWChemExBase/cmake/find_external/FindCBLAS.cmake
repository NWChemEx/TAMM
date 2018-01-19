# Find CBLAS
#
# At the moment this is a relatively thin wrapper.  Given the variability in
# BLAS distributions *cough* mkl *cough* find module attempts to provide the
# user a unified API to the returned CBLAS implementation by defining a C macro
# CBLAS_HEADER, which can be used in your source files like:
# `#include CBLAS_HEADER`, it's passed to your library via the
# `CBLAS_DEFINITIONS` cmake variable
#
# This module defines
#  CBLAS_INCLUDE_DIRS, where to find cblas.h or mkl.h
#  CBLAS_LIBRARIES, the libraries to link against for CBLAS support
#  CBLAS_DEFINITIONS, flags to include when compiling against CBLAS
#  CBLAS_LINK_FLAGS, flags to include when linking against CBLAS
#  CBLAS_FOUND, True if we found CBLAS

include(FindPackageHandleStandardArgs)
set(FINDCBLAS_is_mkl FALSE)
set(FINDCBLAS_HEADER cblas.h)
find_library(CBLAS_LIBRARIES libcblas${CMAKE_STATIC_LIBRARY_SUFFIX})

#Let's see if it's MKL. Intel likes their branding, which we can use to our
#advantage by looking if the string "mkl" appears in any of the library names
string(FIND "${BLAS_LIBRARIES}" "mkl" FINDCBLAS_substring_found)
is_valid_and_true(FINDNWXCBLAS_substring_found FINDCBLAS_is_mkl)
if(FINDCBLAS_is_mkl)
    set(FINDCBLAS_HEADER mkl.h)
endif()
find_path(CBLAS_INCLUDE_DIRS ${FINDCBLAS_HEADER})
list(APPEND CBLAS_DEFINITIONS "-DCBLAS_HEADER=\<${FINDCBLAS_HEADER}\>")

#Now we have to find a BLAS library compatible with our CBLAS implementation
find_package(BLAS)

#This is where'd we check that it's compatible, but as you can see we don't

list(APPEND CBLAS_LIBRARIES ${BLAS_LIBRARIES})
find_package_handle_standard_args(CBLAS DEFAULT_MSG CBLAS_INCLUDE_DIRS
                                                    CBLAS_LIBRARIES)
