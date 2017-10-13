cmake_minimum_required(VERSION 3.1)
project(NWChemExBase VERSION 0.0.0 LANGUAGES NONE)

file(GLOB __MACRO_FILES "${PLB_PREFIX}/cmake/macros/*.cmake")
install(FILES ${__MACRO_FILES}
        DESTINATION share/cmake/NWChemExBase/macros)
file(GLOB __BUILD_FILES "${PLB_PREFIX}/cmake/external/find_external/*.cmake")
install(FILES ${__BUILD_FILES}
        DESTINATION share/cmake/NWChemExBase/external/find_external)

configure_file("${PLB_PREFIX}/cmake/NWChemExBaseConfig.cmake.in"
               NWChemExBaseConfig.cmake @ONLY
)
install(FILES ${CMAKE_BINARY_DIR}/NWChemExBaseConfig.cmake
        DESTINATION share/cmake/NWChemExBase
)

################################################################################
#
#  Now we build NWX_Catch, which is our installed version of Catch. It
#  differs from the default one in that we've installed the header and we've
#  taken the liberty of compiling the main function into a library to help
#  with compile times.
#
################################################################################
#TODO: Move this to a BuildCatch.cmake file

set(CATCH_PREFIX ${CMAKE_BINARY_DIR}/catch)
set(CATCH_HEADER_FILE_INITIAL
    ${CATCH_PREFIX}/src/NWX_Catch${TARGET_SUFFIX}/single_include/catch.hpp)
include(ExternalProject)
ExternalProject_Add(NWX_Catch${TARGET_SUFFIX}
                    PREFIX ${CATCH_PREFIX}
                    GIT_REPOSITORY
                    https://github.com/philsquared/Catch.git
                    UPDATE_COMMAND ""
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND ""
                    )
install(FILES ${CATCH_HEADER_FILE_INITIAL}
        DESTINATION include/catch/)
