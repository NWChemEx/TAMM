#
# Building Catch is easiest as a superbuild because we have to generate files
# and then set-up a CMake build for the resulting file.  We do this in the file:
# NWX_Catch/CMakeLists.txt.  This file simply adds NWX_Catch as an eventual
# build target
#

ExternalProject_Add(NWX_Catch_External
        PREFIX ${CMAKE_BINARY_DIR}/NWX_Catch_External
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/NWX_Catch
        CMAKE_ARGS -DCMAKE_CXX_COMILER=${CMAKE_CXX_COMPILER}
                   ${CORE_CMAKE_OPTIONS}
        BUILD_ALWAYS 1
        BINARY_DIR ${CMAKE_BINARY_DIR}/NWX_Catch_External
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
        CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
                         ${CORE_CMAKE_STRINGS}
        )
