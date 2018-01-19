# Find NWX_MPI
#
# This is a thin wrapper around the FindMPI.cmake packaged with CMake.  This
# version looks only for the C and CXX libraries and includes
#
# This module defines
#  NWX_MPI_INCLUDE_DIRS, where to find mpi.h
#  NWX_MPI_LIBRARIES, the libraries to link against for MPI support
#  NWX_MPI_DEFINITIONS the flags needed to compile with MPI
#  NWX_MPI_LINK_FLAGS the flags needed to link with MPI
#  NWX_MPI_FOUND, True if we found NWX_MPI
include(UtilityMacros)

find_package(MPI QUIET)
foreach(FindNWX_MPI_lang C CXX)
    is_valid_and_true(MPI_${FindNWX_MPI_lang}_FOUND FINDNWX_MPI_was_found)
    if(FINDNWX_MPI_was_found)
        list(APPEND NWX_MPI_LIBRARIES
                    ${MPI_${FindNWX_MPI_lang}_LIBRARIES})
        list(APPEND NWX_MPI_INCLUDE_DIRS
                    ${MPI_${FindNWX_MPI_lang}_INCLUDE_PATH})
        list(APPEND NWX_MPI_DEFINITIONS
                    ${MPI_${FindNWX_MPI_lang}_COMPILE_FLAGS})
        list(APPEND NWX_MPI_LINK_FLAGS
                    ${MPI_${FindNWX_MPI_lang}_LINK_FLAGS})
        set(NWX_MPI_FOUND TRUE)
    endif()
endforeach()
