cmake_minimum_required(VERSION 3.11)
project(tamm VERSION 0.0.0)
include(${PROJECT_SOURCE_DIR}/cmake/check_user_env.cmake)
find_package(CPP REQUIRED)
cpp_option(BUILD_TESTS ON)

cpp_find_or_build_dependency(
    NAME HPTT
    URL github.com/ajaypanyala/hptt
    FIND_MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindHPTT.cmake
)
cpp_find_or_build_dependency(
    NAME MSGSL
    URL github.com/Microsoft/GSL
    FIND_MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindMSGSL.cmake
)
cpp_find_or_build_dependency(
    NAME LAPACKE
    URL http://www.netlib.org/lapack/lapack-3.8.0.tar.gz
    FIND_MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindLAPACKE.cmake
    CMAKE_ARGS CBLAS=ON LAPACKE=ON
)
cpp_find_or_build_dependency(
    NAME Eigen3
    URL github.com/eigenteam/eigen-git-mirror
)

cpp_find_or_build_dependency(
    NAME GlobalArrays
    FIND_MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindGlobalArrays.cmake
    BUILD_MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildGlobalArrays.cmake
    URL github.com/GlobalArrays/ga
)

cpp_find_dependency(
    NAME MPI
    FIND_MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindMPI.cmake
)

add_subdirectory(src/tamm)

if(BUILD_TESTS)
    cpp_find_or_build_dependency(NAME Catch2 URL github.com/catchorg/Catch2)
    #add_subdirectory(tests)
endif()
