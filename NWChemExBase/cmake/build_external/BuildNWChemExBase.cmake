#
# In a pretty meta turn of events, this file will build this project...
#

find_or_build_dependency(NWX_Catch was_found)

ExternalProject_Add(NWChemExBase_External
    SOURCE_DIR ${NWXBASE_ROOT}/NWChemExBase
    CMAKE_ARGS -DNWXBASE_CMAKE=${NWXBASE_CMAKE}
               -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
               -DCMAKE_VERSION=${CMAKE_VERSION}
    BUILD_ALWAYS 1
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
)

add_dependencies(NWChemExBase_External NWX_Catch_External)
