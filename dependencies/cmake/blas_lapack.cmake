
if(EXISTS ${CMAKE_INSTALL_PREFIX}/blas_lapack/lib/libblas.a)
#    set(BLAS_INCLUDE_PATH ${CMAKE_INSTALL_PREFIX}/blas_lapack/include)
#    set(BLAS_LIBRARY_PATH ${CMAKE_INSTALL_PREFIX}/blas_lapack/lib)
    add_custom_target(BLAS_LAPACK ALL)
else()
    message("Building BLAS+LAPACK")
    include(ExternalProject)
    ExternalProject_Add(BLAS_LAPACK
        PREFIX BLAS_LAPACK
        URL http://www.netlib.org/lapack/lapack-3.7.1.tgz
        SOURCE_DIR ${PROJECT_BINARY_DIR}/external/blas_lapack
        UPDATE_COMMAND mkdir -p "${PROJECT_BINARY_DIR}/external/blas_lapack/build"
        BINARY_DIR ${PROJECT_BINARY_DIR}/external/blas_lapack/build
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DCBLAS=ON -DBUILD_TESTING=OFF -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
        -DLAPACKE=ON -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/blas_lapack
        BUILD_COMMAND make -j${TAMM_PROC_COUNT}
        INSTALL_COMMAND make install
    )

#    set(BLAS_INCLUDE_PATH ${CMAKE_INSTALL_PREFIX}/blas_lapack/include)
#    set(BLAS_LIBRARY_PATH ${CMAKE_INSTALL_PREFIX}/blas_lapack/lib)
endif()
