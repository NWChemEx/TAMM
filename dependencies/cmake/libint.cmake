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
    set(LIBINT_PREFIX ${CMAKE_BINARY_DIR}/libint)
    set(LIBINT_FLAGS ${CMAKE_CXX_FLAGS} -fPIC)
    include(ExternalProject)
    ExternalProject_Add(LIBINT
        PREFIX ${LIBINT_PREFIX}
        GIT_REPOSITORY https://github.com/evaleev/libint
        #GIT_TAG 33560a073efa5a0abd88a37486e552c954808f1d
        PATCH_COMMAND ./autogen.sh
        CONFIGURE_COMMAND ${LIBINT_PREFIX}/src/LIBINT/configure
                            CXX=${CMAKE_CXX_COMPILER}
			    CC=${CMAKE_C_COMPILER}
                            CXXFLAGS=${LIBINT_FLAGS}
                            CPPFLAGS=-I${BOOST_INSTALL_PATH}/include
                            LDFLAGS=-L${BOOST_INSTALL_PATH}/lib
			    --prefix=${CMAKE_INSTALL_PREFIX}/libint2
                            --enable-eri=0
                            --enable-eri3=0
                            --enable-eri2=0
                            --with-max-am=4
                            --disable-t1g12-support
        BUILD_COMMAND make -j${TAMM_PROC_COUNT}
        #INSTALL_DIR ${STAGE_INSTALL_PREFIX}
        INSTALL_COMMAND make install
        #LOG_CONFIGURE 1
        #LOG_BUILD 1
    )
endif()
