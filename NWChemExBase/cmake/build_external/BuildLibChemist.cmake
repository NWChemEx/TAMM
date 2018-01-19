ExternalProject_Add(LibChemist_External
        GIT_REPOSITORY https://github.com/NWChemEx-Project/LibChemist.git
        CMAKE_ARGS ${CORE_CMAKE_OPTIONS}
        BUILD_ALWAYS 1
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
        CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
                         ${CORE_CMAKE_STRINGS}
        )
