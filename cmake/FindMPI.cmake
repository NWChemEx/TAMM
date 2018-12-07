find_path(MPI_INCLUDE_DIR mpi.h)
find_library(MPI_LIBRARY NAMES mpi)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    MPI DEFAULT_MSG
    MPI_LIBRARY MPI_INCLUDE_DIR
)

set(MPI_LIBRARIES ${MPI_LIBRARY})
set(MPI_INCLUDE_DIRS ${MPI_INCLUDE_DIR})

