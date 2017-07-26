
if(DEFINED GTEST_INSTALL_PATH AND EXISTS ${GTEST_INSTALL_PATH}/include/gtest/gtest.h)

message(STATUS "GTEST found at: ${GTEST_INSTALL_PATH}")
ADD_CUSTOM_TARGET(GTEST ALL)

else()

set (GTEST_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/googletest)

if(EXISTS ${CMAKE_INSTALL_PREFIX}/googletest/include/gtest/gtest.h)

    ADD_CUSTOM_TARGET(GTEST ALL)

else()

    message("Building GOOGLETEST")
    include(ExternalProject)
    ExternalProject_Add(GTEST
        PREFIX GTEST
        URL https://github.com/google/googletest/archive/release-1.8.0.tar.gz
        SOURCE_DIR ${PROJECT_BINARY_DIR}/external/googletest
        UPDATE_COMMAND mkdir -p "${PROJECT_BINARY_DIR}/external/googletest/build"
        BINARY_DIR ${PROJECT_BINARY_DIR}/external/googletest/build
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/googletest
        BUILD_COMMAND make -j${TAMM_PROC_COUNT}
        INSTALL_COMMAND make install
    )
endif()

endif()
