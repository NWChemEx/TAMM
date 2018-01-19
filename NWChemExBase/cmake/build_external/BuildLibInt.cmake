set(LIBINT_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(LIBINT_MAJOR 2)
set(LIBINT_MINOR 4)
set(LIBINT_PATCH 2)
set(LIBINT_VERSION "${LIBINT_MAJOR}.${LIBINT_MINOR}.${LIBINT_PATCH}")
set(LIBINT_URL https://github.com/evaleev/libint)
set(LIBINT_TAR ${LIBINT_URL}/releases/download/v${LIBINT_VERSION}/)
set(LIBINT_TAR ${LIBINT_TAR}/libint-${LIBINT_VERSION})

if(${PROJECT_NAME} STREQUAL "BuildLibInt")
    #Grab the small version of libint for testing purposes
    set(LIBINT_TAR ${LIBINT_TAR}-test-mpqc4.tgz)
else()
    set(LIBINT_TAR ${LIBINT_TAR}.tgz)
endif()

find_or_build_dependency(Eigen3 was_found)
ExternalProject_Add(LibInt_External
    URL ${LIBINT_TAR}
    CONFIGURE_COMMAND ./configure --prefix=${CMAKE_INSTALL_PREFIX}
        CXX=${CMAKE_CXX_COMPILER}
        CC=${CMAKE_C_COMPILER}
        CXXFLAGS=${LIBINT_FLAGS}
        ${LIBINT_CONFIG_OPTIONS}
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
    BUILD_IN_SOURCE 1
)
add_dependencies(LibInt_External Eigen3_External)
