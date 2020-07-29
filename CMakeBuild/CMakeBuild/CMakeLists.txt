cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(CMakeBuild VERSION ${PROJECT_VERSION} LANGUAGES NONE)

# We are going to abuse the include directory variable to find out the path to
# cmake directory
set(NWXBASE_CMAKE ${NWX_INCLUDE_DIR}/cmake)
set(NWXBASE_MACROS ${NWXBASE_CMAKE}/macros)
set(MACRO_INSTALL share/cmake/CMakeBuild/macros)

file(GLOB __MACRO_FILES "${NWXBASE_MACROS}/*.cmake")
install(FILES ${__MACRO_FILES}  DESTINATION ${MACRO_INSTALL})

file(GLOB __FIND_FILES "${NWXBASE_CMAKE}/find_external/*.cmake")
install(FILES ${__FIND_FILES}
        DESTINATION share/cmake/CMakeBuild/find_external)

file(GLOB __BUILD_FILES "${NWXBASE_CMAKE}/build_external/*.cmake")
install(DIRECTORY ${NWXBASE_CMAKE}/build_external/
        DESTINATION share/cmake/CMakeBuild/build_external)


configure_file("${NWXBASE_CMAKE}/CMakeBuildConfig.cmake.in"
               CMakeBuildConfig.cmake @ONLY
)
install(FILES ${CMAKE_BINARY_DIR}/CMakeBuildConfig.cmake
        DESTINATION share/cmake/CMakeBuild
)

configure_file(
        "${NWXBASE_CMAKE}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

install(FILES ${NWXBASE_CMAKE}/NWChemExTargetConfig.cmake.in
        ${NWXBASE_CMAKE}/cmake_uninstall.cmake.in
        DESTINATION share/cmake/CMakeBuild)

add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P
        ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

