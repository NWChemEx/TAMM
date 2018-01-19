# Find LAPACKE
#
# At the moment this is a relatively thin wrapper.  Given the variability in
# BLAS distributions *cough* mkl *cough* find module attempts to provide the
# user a unified API to the returned LAPACKE implementation by defining a C
# macro LAPACKE_HEADER, which can be used in your source files like:
# `#include LAPACKE_HEADER`, it's passed to your library via the
# `LAPACKE_DEFINITIONS` cmake variable
#
# This module defines
#  LAPACKE_INCLUDE_DIRS, where to find cblas.h or mkl.h
#  LAPACKE_LIBRARIES, the libraries to link against for LAPACKE support
#  LAPACKE_DEFINITIONS, flags to include when compiling against LAPACKE
#  LAPACKE_LINK_FLAGS, flags to include when linking against LAPACKE
#  LAPACKE_FOUND, True if we found LAPACKE

include(FindPackageHandleStandardArgs)
set(FINDLAPACKE_is_mkl FALSE)
set(FINDLAPACKE_HEADER lapacke.h)
is_valid(LAPACKE_LIBRARIES FINDLAPACKE_LIBS_SET)
if(NOT FINDLAPACKE_LIBS_SET)
    find_library(LAPACKE_LIBRARIES liblapacke${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

is_valid(LAPACKE_INCLUDE_DIRS FINDLAPACKE_INCLUDES_SET)
if(FINDLAPACKE_INCLUDES_SET)
    #Let's see if it's MKL. Intel likes their branding, which we can use
    #to our advantage by looking if the string "mkl" appears in any of the
    #library names
    string(FIND "${BLAS_LIBRARIES}" "mkl" FINDLAPACKE_substring_found)
    is_valid_and_true(FINDNWXLAPACKE_substring_found FINDLAPACKE_is_mkl)
    if(FINDLAPACKE_is_mkl)
        set(FINDLAPACKE_HEADER mkl.h)
    endif()
    #For sanity could make sure header is actually located in that path, but not
    #typical CMake behavior...
    #find_path(LAPACKE_INCLUDE_DIR ${FINDLAPACKE_HEADER}
    #          HINTS ${LAPACKE_INCLUDE_DIRS})
    #assert_strings_are_equal("${LAPACKE_INCLUDE_DIR}" "${LAPACKE_INCLUDE_DIRS}")
else()
    find_path(LAPACKE_INCLUDE_DIRS ${FINDLAPACKE_HEADER})
endif()
list(APPEND LAPACKE_DEFINITIONS "-DLAPACKE_HEADER=\"${FINDLAPACKE_HEADER}\"")

#Now we have to find a LAPACK library compatible with our CBLAS implementation
find_package(LAPACK)

#This is where'd we check that it's compatible, but as you can see we don't

#Now we have to find a BLAS library compatible with our LAPACK implementation
find_package(BLAS)

#This is where'd we check that it's compatible, but as you can see we don't

list(APPEND LAPACKE_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

find_package_handle_standard_args(LAPACKE DEFAULT_MSG LAPACKE_INCLUDE_DIRS
                                                      LAPACKE_LIBRARIES)
