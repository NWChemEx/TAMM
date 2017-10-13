set(ANTLR_PATCH_FILE
        ${CMAKE_CURRENT_SOURCE_DIR}/../patches/antlr_cmakelists.patch)

ExternalProject_Add(antlr_cpp_runtime${TARGET_SUFFIX}
        URL http://www.antlr.org/download/antlr4-cpp-runtime-4.7-source.zip
        PATCH_COMMAND patch < ${ANTLR_PATCH_FILE}
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE
                   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                   -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                   -DWITH_DEMO=OFF
                   -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
)

