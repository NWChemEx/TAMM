
include(ExternalProject)

# TODO: Use find_package
# find_package(BLAS)
# find_package(LAPACK)
# if (BLAS_FOUND AND LAPACK_FOUND)

#Keep it simple for now, build if user does not provide the following cmake variables
if (BLAS_INCLUDE_PATH AND BLAS_LIBRARY_PATH AND BLAS_LIBRARIES AND LAPACK_LIBRARIES)
    add_library(blas_lapack${TARGET_SUFFIX} INTERFACE)
else()
    set(BLAS_LAPACK_ROOT_DIR ${CMAKE_INSTALL_PREFIX}/blas_lapack)
    message(STATUS "Building BLAS+LAPACK at: ${BLAS_LAPACK_ROOT_DIR}")

    ExternalProject_Add(blas_lapack${TARGET_SUFFIX}
        URL http://www.netlib.org/lapack/lapack-3.7.1.tgz
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DCBLAS=ON -DBUILD_TESTING=OFF -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
        -DLAPACKE=ON -DCMAKE_INSTALL_PREFIX=${BLAS_LAPACK_ROOT_DIR}
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install
    )

    # set(BLAS_INCLUDE_PATH ${BLAS_LAPACK_ROOT_DIR}/include)
    # set(BLAS_LIBRARY_PATH ${BLAS_LAPACK_ROOT_DIR}/lib)

endif()
