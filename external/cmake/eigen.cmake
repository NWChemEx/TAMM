
if(DEFINED EIGEN3_INSTALL_PATH AND EXISTS ${EIGEN3_INSTALL_PATH}/include/eigen3/Eigen/Core)

message(STATUS "Eigen3 found at: ${EIGEN3_INSTALL_PATH}")
ADD_CUSTOM_TARGET(EIGEN3 ALL)

else()

set (EIGEN3_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/eigen3)

if(EXISTS ${CMAKE_INSTALL_PREFIX}/eigen3/include/eigen3/Eigen/Core)

    ADD_CUSTOM_TARGET(EIGEN3 ALL)

else()

    message("Building Eigen")
    include(ExternalProject)
    ExternalProject_Add(EIGEN3
        PREFIX EIGEN3
        URL http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz
        SOURCE_DIR ${PROJECT_BINARY_DIR}/external/eigen
        UPDATE_COMMAND mkdir -p "${PROJECT_BINARY_DIR}/external/eigen/build"
        BINARY_DIR ${PROJECT_BINARY_DIR}/external/eigen/build
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
            -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/eigen3
        #BUILD_COMMAND make -j${TAMM_PROC_COUNT} blas
        INSTALL_COMMAND make install
    )
endif()

endif()
