set(LIBINT_FLAGS ${CMAKE_CXX_FLAGS} -fPIC)

ExternalProject_Add(Libint${TARGET_SUFFIX}
    URL https://github.com/evaleev/libint/releases/download/v2.3.1/libint-2.3.1.tgz
    CONFIGURE_COMMAND ./configure --prefix=${CMAKE_INSTALL_PREFIX}
        CXX=${CMAKE_CXX_COMPILER}
        CC=${CMAKE_C_COMPILER}
        CXXFLAGS=${LIBINT_FLAGS}
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR]
    BUILD_IN_SOURCE 1
)
