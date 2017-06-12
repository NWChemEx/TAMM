
set (EIGEN3_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/eigen3)

if(EXISTS ${CMAKE_INSTALL_PREFIX}/eigen3/include/eigen3/Eigen/Core)

    ADD_CUSTOM_TARGET(EIGEN3 ALL)

else()

    message("Building Eigen")
    include(ExternalProject)
    ExternalProject_Add(EIGEN3
        PREFIX EIGEN3
        HG_REPOSITORY https://bitbucket.org/eigen/eigen
        SOURCE_DIR ${PROJECT_BINARY_DIR}/external/eigen
        UPDATE_COMMAND mkdir -p "${PROJECT_BINARY_DIR}/external/eigen/build"
        BINARY_DIR ${PROJECT_BINARY_DIR}/external/eigen/build
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}  
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/eigen3
        INSTALL_COMMAND make install
    )
endif()
