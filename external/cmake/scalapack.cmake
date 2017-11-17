
include(ExternalProject)

set(SCALAPACK_LIBRARIES ${CMAKE_INSTALL_PREFIX}/scalapack/lib/libscalapack.a)
if(EXISTS ${SCALAPACK_LIBRARIES})
    add_custom_target(SCALAPACK ALL)
else()
    ExternalProject_Add(SCALAPACK
        URL ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scalapack.tar.gz
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_TESTING=OFF 
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/scalapack
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install        
    )
endif()
