include(ExternalProject)
message(${CMAKE_CURRENT_SOURCE_DIR})
ExternalProject_Add(
    GA_External
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    PATCH_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/autogen.sh
    CONFIGURE_COMMAND   CXX=${CMAKE_CXX_COMPILER}
                        CC=${CMAKE_C_COMPILER}
                        ${CMAKE_CURRENT_SOURCE_DIR}/configure
                            --prefix=${CMAKE_INSTALL_PREFIX}
                            --with-mpi-ts
    BUILD_COMMAND make
    INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
    INSTALL_COMMAND make install
)
