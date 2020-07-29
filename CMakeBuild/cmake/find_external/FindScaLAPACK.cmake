# Find ScaLAPACK
#
# Once done this will define
# SCALAPACK_FOUND - System has ScaLAPACK
# SCALAPACK_LIBRARIES - The ScaLAPACK include directories
#

include(UtilityMacros)
include(FindPackageHandleStandardArgs)

is_valid(SCALAPACK_LIBRARIES FINDSCA_LIBS_SET)
if(NOT FINDSCA_LIBS_SET)
    find_library(SCALAPACK_LIBRARY
        NAMES libscalapack.a libscalapack.so mkl_scalapack_lp64 mkl_scalapack_ilp64
        PATHS ${SCALAPACK_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib32
        DOC "ScaLAPACK Libraries"
    )
    find_package_handle_standard_args(ScaLAPACK DEFAULT_MSG
                                      SCALAPACK_LIBRARY)
    set(SCALAPACK_FOUND ${SCALAPACK_FOUND})
    set(SCALAPACK_LIBRARIES ${SCALAPACK_LIBRARY})
endif()

find_package_handle_standard_args(ScaLAPACK DEFAULT_MSG
                                  SCALAPACK_LIBRARIES)