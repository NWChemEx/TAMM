set(mpi_languages C CXX)
foreach(mpi_lang ${mpi_languages})
    _cpp_is_true(have_lang ${mpi_lang})
    if(have_lang)
       list(APPEND MPI_LIBRARIES ${MPI_${mpi_lang}_LIBRARIES})
       list(APPEND MPI_INCLUDE_DIRS ${MPI_${mpi_lang}_INCLUDE_DIRS})
       set(comp_flags MPI_${mpi_lang}_COMPILE_FLAGS)
       if(NOT "${comp_flags}" STREQUAL "")
           set()
           string(REPLACE " " ";" ${comp_flags} ${comp_flags})
       endif()
       list(APPEND MPI_LINK_FLAGS ${MPI_${mpi_lang}_LINK_FLAGS})
    endif()
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    MPI
    DEFAULT_MSG
    MPI_LIBRARIES MPI_INCLUDE_DIRS
)
