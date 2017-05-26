unset(BUILD)

if(NOT BUILD_EIGEN)
    find_path(EIGEN3 eigen3/Eigen/src/Core/util/Macros.h)
    if(NOT EIGEN3)
        set(BUILD 1)
    else()
        message("Found Eigen at ${EIGEN3}")
        add_custom_target(EIGEN3)
    endif()
else()
    set(BUILD 1)
endif()

# Build EIGEN 
if(BUILD)
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
