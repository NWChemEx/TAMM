cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(NWChemExBase VERSION 0.0.0 LANGUAGES NONE)

set(NWXBASE_MACROS ${NWXBASE_CMAKE}/macros)
set(MACRO_INSTALL share/cmake/NWChemExBase/macros)

file(GLOB __MACRO_FILES "${NWXBASE_MACROS}/*.cmake")
install(FILES ${__MACRO_FILES}  DESTINATION ${MACRO_INSTALL})

file(GLOB __BUILD_FILES "${NWXBASE_CMAKE}/external/find_external/*.cmake")
install(FILES ${__BUILD_FILES}
        DESTINATION share/cmake/NWChemExBase/external/find_external)

configure_file("${NWXBASE_CMAKE}/NWChemExBaseConfig.cmake.in"
               NWChemExBaseConfig.cmake @ONLY
)
install(FILES ${CMAKE_BINARY_DIR}/NWChemExBaseConfig.cmake
        DESTINATION share/cmake/NWChemExBase
)


