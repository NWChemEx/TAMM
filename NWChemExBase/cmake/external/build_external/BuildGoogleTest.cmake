ExternalProject_Add(GTest${TARGET_SUFFIX}
    URL https://github.com/google/googletest/archive/release-1.8.0.tar.gz
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install -DESTDIR=${STAGE_DIR}
)

