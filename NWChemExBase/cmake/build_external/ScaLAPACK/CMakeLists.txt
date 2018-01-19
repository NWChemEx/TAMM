#The actual file that builds ScaLAPACK
cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(NWX_CBLAS VERSION 2.0.2 LANGUAGES C Fortran)
include(DependencyMacros)
include(ExternalProject)
find_dependency(NWX_CBLAS NWX_CBLAS_INCLUDE_DIRS
                          NWX_CBLAS_LIBRARIES
                          NWX_CBLAS_DEFINITIONS
                          NWX_CBLAS_LINK_FLAGS
                          _CBLAS_FOUND)
find_dependency(LAPACKE LAPACKE_INCLUDE_DIRS
                        LAPACKE_LIBRARIES
                        LAPACKE_DEFINITIONS
                        LAPACKE_LINK_FLAGS
                        _LAPACKE_FOUND)

set(ScaLAPACK_VERSION 2.0.2)
set(ScaLAPACK_URL
        http://www.netlib.org/scalapack/scalapack-${ScaLAPACK_VERSION}.tgz)

enable_language(C Fortran)
ExternalProject_Add(ScaLAPACK
        URL ${ScaLAPACK_URL}
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE
                   -DBUILD_TESTING=OFF
                   -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                   -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
                   -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
                   -DBLAS_LIBRARIES=${NWX_CBLAS_LIBRARIES}
                   -DLAPACK_LIBRARIES=${LAPACKE_LIBRARIES}
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
        )
