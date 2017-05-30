unset(BUILD)

if(NOT BUILD_LIBINT)
    find_library(LIBINT int2)
    if(NOT LIBINT)
        set(BUILD 1)
    else()
        message("Found LibtInt at ${LIBINT}")
        add_custom_target(LIBINT)
    endif()
else()
    set(BUILD 1)
endif()
# Build LIBINT2
if(BUILD)
    message("Building LibtInt")
    set(LIBINT_FLAGS ${CMAKE_CXX_FLAGS} -fPIC)
    include(ExternalProject)
    ExternalProject_Add(LIBINT
        PREFIX LIBINT
        #GIT_REPOSITORY https://github.com/evaleev/libint.git
        URL https://github.com/evaleev/libint/releases/download/v2.3.1/libint-2.3.1.tgz
        SOURCE_DIR ${PROJECT_BINARY_DIR}/external/libint
        #BINARY_DIR ${CMAKE_BINARY_DIR}/external/libint-build
        CONFIGURE_COMMAND ./configure
                --prefix=${CMAKE_INSTALL_PREFIX}/libint2
		CXX=${CMAKE_CXX_COMPILER}
                CC=${CMAKE_C_COMPILER}
                CXXFLAGS=${LIBINT_FLAGS}
        BUILD_COMMAND make -j${TAMM_PROC_COUNT}
        INSTALL_COMMAND make install
        BUILD_IN_SOURCE 1
        #LOG_CONFIGURE 1
        #LOG_BUILD 1
    )
endif()
