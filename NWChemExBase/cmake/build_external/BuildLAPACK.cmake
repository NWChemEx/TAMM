#
# This file will build Netlib's LAPACK distribution over an existing BLAS
# installation. To do this we use a mock superbuild in case we need to build
# BLAS for the user.
#
find_or_build_dependency(BLAS _was_Found)
enable_language(C Fortran)

ExternalProject_Add(LAPACK_External
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/LAPACK
        CMAKE_ARGS -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
                   -DCMAKE_C_COMILER=${CMAKE_C_COMPILER}
                   ${CORE_CMAKE_OPTIONS}
        BUILD_ALWAYS 1
        INSTALL_COMMAND $(MAKE) install DESTDIR=${STAGE_DIR}
        CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
                         ${CORE_CMAKE_STRINGS}
        )
add_dependencies(LAPACK_External BLAS_External)
