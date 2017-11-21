
if(DEFINED LIBINT_INSTALL_PATH AND EXISTS ${LIBINT_INSTALL_PATH}/lib/libint2.a)

    message(STATUS "LIBINT found at: ${LIBINT_INSTALL_PATH}")
    ADD_CUSTOM_TARGET(LIBINT ALL)

else()

    set (LIBINT_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/libint2)

    if(EXISTS ${CMAKE_INSTALL_PREFIX}/libint2/lib/libint2.a)

        ADD_CUSTOM_TARGET(LIBINT ALL)

    else()
        message(STATUS "Building LibtInt at: ${LIBINT_INSTALL_PATH}")
        set(LIBINT_FLAGS ${CMAKE_CXX_FLAGS} -fPIC)
        include(ExternalProject)
        ExternalProject_Add(LIBINT
            #URL https://github.com/evaleev/libint/releases/download/v2.4.1/libint-2.4.1.tgz
            URL https://github.com/evaleev/libint/releases/download/v2.3.1/libint-2.3.1.tgz 
            CONFIGURE_COMMAND ./configure --prefix=${LIBINT_INSTALL_PATH}
                CXX=${CMAKE_CXX_COMPILER}
                CC=${CMAKE_C_COMPILER}
                CXXFLAGS=${LIBINT_FLAGS}
                INSTALL_COMMAND make install
                BUILD_IN_SOURCE 1
        )
    endif()

endif()
