

# Set location of the ANTLR C runtime library.
set (ANTLR_CPPRUNTIME_PATH ${CMAKE_INSTALL_PREFIX}/ANTLR/CppRuntime)

if(EXISTS ${CMAKE_INSTALL_PREFIX}/ANTLR/CppRuntime/libantlr4-runtime.a)

    add_custom_target(ANTLR ALL)

else()

    SET(ANTLR_SOURCES ${PROJECT_BINARY_DIR}/external/ANTLR/)
    execute_process(COMMAND mkdir -p "${ANTLR_SOURCES}")
    file(COPY ${PROJECT_SOURCE_DIR}/ANTLR4/ DESTINATION ${ANTLR_SOURCES})

    # Build the ANTLR C Runtime library.
    include(ExternalProject)
    ExternalProject_Add(ANTLR
        PREFIX ANTLR
        SOURCE_DIR ${ANTLR_SOURCES}/antlr4-cpp-runtime
        #INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/ANTLR/CppRuntime
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_CXX_COMPILER=g++  -DWITH_DEMO=OFF
            -DCMAKE_C_COMPILER=gcc -DCMAKE_INSTALL_PREFIX=${ANTLR_CPPRUNTIME_PATH}
        BINARY_DIR ${ANTLR_SOURCES}/antlr4-cpp-runtime/build
        BUILD_COMMAND make -j${TAMM_PROC_COUNT}
        INSTALL_COMMAND make install
    )

endif()

file(COPY ${PROJECT_SOURCE_DIR}/ANTLR4/antlr-4.7-complete.jar DESTINATION ${CMAKE_INSTALL_PREFIX}/ANTLR/)
# Set location of the ANTLR binary.
set(AntlrBinary ${CMAKE_INSTALL_PREFIX}/ANTLR/antlr-4.7-complete.jar)


