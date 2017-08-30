# ------------------------------------------------------------------------------
#  Copyright (C) 2016-2017, Pacific Northwest National Laboratory
#  This software is subject to copyright protection under the laws of the
#  United States and other countries
#
#  All rights in this computer software are reserved by the
#  Pacific Northwest National Laboratory (PNNL)
#  Operated by Battelle for the U.S. Department of Energy
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
project (TAMM_LIBRARY C CXX)
enable_language (Fortran)

set(CMAKE_BUILD_TYPE Debug)

# The version number.
set (TAMM_VERSION_MAJOR 0)
set (TAMM_VERSION_MINOR 1)

#Cannot set the following since cmake cannot handle clang
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -Ofast")#  -Wextra  -Wall")
set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} -std=c++14 -g -Ofast")# -Wextra  -Wall")
#set(CMAKE_CXX_FLAGS_MINSIZEREL     "")
#set(CMAKE_CXX_FLAGS_RELEASE        "")
#set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "")

INCLUDE(CTest)
# Should a test fail, display the test program output
set (CTEST_OUTPUT_ON_FAILURE TRUE)

if(TAMM_PROC_COUNT)
    set(TAMM_PROC_COUNT ${TAMM_PROC_COUNT})
else()
    include(ProcessorCount)
    ProcessorCount(TAMM_PROC_COUNT)
    message(STATUS "Number of cores detected = ${TAMM_PROC_COUNT}")
endif()
message(STATUS "Using ${TAMM_PROC_COUNT} cores for running TAMM tests.")
message("Please set the TAMM_PROC_COUNT cmake variable to change the cpu count.")

if(EIGEN3_INSTALL_PATH)
    message(STATUS "Found Eigen3: ${EIGEN3_INSTALL_PATH}")
else()
    message(FATAL_ERROR "Eigen3 not found. Please provide Eigen3 insall path")
endif()

if(LIBINT_INSTALL_PATH)
    message(STATUS "Found LIBINT: ${LIBINT_INSTALL_PATH}")
else()
    message(FATAL_ERROR "LIBINT not found. Please provide LIBINT insall path")
endif()

include_directories( ${PROJECT_SOURCE_DIR} )

# Install
if(NOT TAMM_INSTALL_PREFIX)
    set(TAMM_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/tamm-install)
endif()

install(FILES ${PROJECT_SOURCE_DIR}/h2o.xyz DESTINATION ${PROJECT_BINARY_DIR})

# Find Doxygen
FIND_PACKAGE(Doxygen)

if(DOXYGEN_FOUND)
    set(DOCS "${PROJECT_BINARY_DIR}/docs")
    message(STATUS "Documentation will be built at: " ${DOCS})
    execute_process(COMMAND mkdir -p ${DOCS})
    set(DOCOUTPUT ${DOCS}/html)
    # Custom Target to build the Doxygen documentation.
    ADD_CUSTOM_COMMAND(OUTPUT ${DOCOUTPUT}
            COMMAND cmake -E copy ${PROJECT_SOURCE_DIR}/Doxyfile.in ${DOCS}/Doxyfile.in
            COMMAND cmake -E copy_directory ${PROJECT_SOURCE_DIR}/tammx ${DOCS}/tammx
#            COMMAND cmake -E copy_directory ${PROJECT_SOURCE_DIR}/tensor ${DOCS}/tensor
            # Change to docs folder before building the documentation.
            COMMAND cmake -E chdir ${DOCS} doxygen ${DOCS}/Doxyfile.in
            COMMAND cmake -E remove_directory ${DOCS}/tammx
            COMMAND cmake -E remove -f ${DOCS}/Doxyfile.in
            DEPENDS ${PROJECT_SOURCE_DIR}/Doxyfile.in
            COMMENT "Building Doxygen documentation."
            VERBATIM)
    ADD_CUSTOM_TARGET(docs DEPENDS ${DOCOUTPUT})
endif()

find_path (BLAS_INCLUDE_PATH NAMES cblas.h mkl_cblas.h PATHS ${BLAS_INCLUDE_PATH})
message(STATUS "BLAS headers found at: ${BLAS_INCLUDE_PATH}")

#Load the GA_CONFIG variable
if(NOT GA_CONFIG)
    message(FATAL_ERROR "TAMM ERROR: CMAKE variable GA_CONFIG not set")
else()
    set(GA_CONFIG ${GA_CONFIG}/ga-config)
endif()

add_definitions(-DMPICH_IGNORE_CXX_SEEK)
execute_process(COMMAND ${GA_CONFIG} --cc OUTPUT_VARIABLE tamm_c_compiler OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${GA_CONFIG} --f77 OUTPUT_VARIABLE tamm_f77_compiler OUTPUT_STRIP_TRAILING_WHITESPACE)

SET(CMAKE_C_COMPILER ${tamm_c_compiler})
set(CMAKE_Fortran_COMPILER ${tamm_f77_compiler})

#Assume Cmake picks the right c++ compiler for now
#SET(CMAKE_CXX_COMPILER mpicxx)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXT_INT -DGFORTRAN")

execute_process(COMMAND ${GA_CONFIG} --cppflags OUTPUT_VARIABLE tamm_includes OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${GA_CONFIG} --cflags OUTPUT_VARIABLE tamm_cflags OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${GA_CONFIG} --fflags OUTPUT_VARIABLE tamm_fflags OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${GA_CONFIG} --ldflags OUTPUT_VARIABLE tamm_ldflags OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${GA_CONFIG} --libs OUTPUT_VARIABLE tamm_libs OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REPLACE "-I" "" tamm_cmake_inc ${tamm_includes})
#To be recognized as a list, the strings must be seperated by ;
string(REPLACE " " ";" tamm_includes ${tamm_cmake_inc})

execute_process(COMMAND mpicxx --showme:link OUTPUT_VARIABLE mpi_link OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

#Build fortran code for unit testing

set(FORTRAN_DEPS_SRC
nwtest/put_hash_block.F
nwtest/add_block.F
nwtest/get_block.F
nwtest/get_block_ind.F
nwtest/add_hash_block.F
nwtest/get_hash_block.F
nwtest/get_hash_block_ma.F
nwtest/nxtask.F
nwtest/nxtval_ga.F
nwtest/errquit.F
nwtest/put_block.F
nwtest/tce_hash.F
nwtest/tce_hash_ind.F
nwtest/tce_restricted.F
nwtest/tce_sort.F
nwtest/tce_sortacc.F
nwtest/misc.F
nwtest/set_vars.F
nwtest/ccsd_t1.F
nwtest/ccsd_t2.F
nwtest/cc2_t1.F
nwtest/cc2_t2.F
)

set(TESTING_DEPS_SRC
#nwtest/fort_assign_tests.cc
)

#SET(CMAKE_Fortran_COMPILER mpif90)
if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
  set(tamm_fortran_link_flags ${tamm_fflags})
  set(tamm_libs ${tamm_libs} ${mpi_link})
elseif("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
  set(tamm_fortran_link_flags -i8)
else()
    message(FATAL_ERROR "Only GNU/Intel Fortran compilers supported")
endif()

if (EXISTS ${NWCHEM_BUILD_DIR}/src/blas/libnwcblas.a)
set(BLAS_LIBRARIES
    ${NWCHEM_BUILD_DIR}/src/peigs/libpeigsc.a
    ${NWCHEM_BUILD_DIR}/src/peigs/libpeigsf.a
    ${NWCHEM_BUILD_DIR}/src/blas/libnwcblas.a
    ${NWCHEM_BUILD_DIR}/src/lapack/libnwclapack.a)

else()
    set(BLAS_LIBRARIES
            ${NWCHEM_BUILD_DIR}/src/peigs/libpeigsc.a
            ${NWCHEM_BUILD_DIR}/src/peigs/libpeigsf.a)
endif()

add_library(fort_deps STATIC ${FORTRAN_DEPS_SRC})
target_include_directories(fort_deps SYSTEM PUBLIC ${tamm_includes})
target_compile_options(fort_deps PRIVATE ${tamm_fortran_link_flags})

add_library(fort_deps_c STATIC nwtest/util_gnxtval.c)
target_include_directories(fort_deps_c SYSTEM PUBLIC ${tamm_includes})

add_executable(eigen_tests nwtest/test_eigen.cc ${TESTING_DEPS_SRC})
add_dependencies(eigen_tests fort_deps fort_deps_c tammx)
target_include_directories(eigen_tests SYSTEM PUBLIC ${tamm_includes}
         ${GTEST_INSTALL_PATH}/include)
target_compile_options(eigen_tests PUBLIC ${tamm_cflags})
target_link_libraries(eigen_tests fort_deps fort_deps_c tammx ${tamm_cflags}
        ${tamm_ldflags} ${BLAS_LIBRARIES}  ${tamm_libs} ${BLAS_LIBRARIES} -L${GTEST_INSTALL_PATH}/lib -lgtest)

#add_executable(eigentest nwtest/test_eigen.cc ${TESTING_DEPS_SRC})
#add_dependencies(eigentest fort_deps fort_deps_c tamm tammx)
#target_include_directories(eigentest SYSTEM PUBLIC ${tamm_includes}
#        ${ANTLR_CPPRUNTIME_PATH}/include/antlr4-runtime ${GTEST_INSTALL_PATH}/include
#        ${EIGEN3_INSTALL_PATH}/include/eigen3)
#target_compile_options(eigentest PUBLIC ${tamm_cflags})
#target_link_libraries(eigentest fort_deps fort_deps_c tamm tammx ${tamm_cflags}
#        ${tamm_ldflags} ${BLAS_LIBRARIES}  ${tamm_libs} ${BLAS_LIBRARIES} -L${GTEST_INSTALL_PATH}/lib -lgtest)

#BUILD TAMMX
    set(tammxSources
        tammx/tammx.cc
        tammx/index_sort.cc)

add_library(tammx STATIC ${tammxSources})
target_include_directories(tammx SYSTEM PUBLIC ${PROJECT_BINARY_DIR} ${EIGEN3_INSTALL_PATH}/include/eigen3 ${tamm_includes})
target_link_libraries(tammx ${tamm_ldflags} ${BLAS_LIBRARIES}  ${tamm_libs})

add_executable(tammx_main tammx/tammx_main.cc)
target_link_libraries(tammx_main tammx)
#set_property(TARGET tammx_main PROPERTY CXX_STANDARD 14)
#set_property(TARGET tammx_main PROPERTY CXX_STANDARD_REQUIRED ON)

if(NOT "${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Darwin")
  set(TAMM_LINK_EXTRA -lrt)
endif()

add_executable(ccsd_driver tammx/ccsd_driver.cc tammx/hartree_fock.cc)
add_dependencies(ccsd_driver tammx)
target_include_directories(ccsd_driver SYSTEM PUBLIC ${PROJECT_SOURCE_DIR} ${LIBINT_INSTALL_PATH}/include ${EIGEN3_INSTALL_PATH}/include/eigen3)
target_link_libraries(ccsd_driver tammx -L${LIBINT_INSTALL_PATH}/lib -lint2 ${TAMM_LINK_EXTRA})

# add_executable(hf nwchem_main/hartree_fock.cc)
# target_include_directories(hf SYSTEM PUBLIC ${LIBINT_INSTALL_PATH}/include ${EIGEN3_INSTALL_PATH}/include/eigen3)
# target_link_libraries(hf -L${LIBINT_INSTALL_PATH}/lib -lint2)

#set_property(TARGET tamm PROPERTY CXX_STANDARD 14)
#set_property(TARGET tamm PROPERTY CXX_STANDARD_REQUIRED ON)

find_path(cblas_header_name "mkl_cblas.h" ${BLAS_INCLUDE_PATH})
if (NOT cblas_header_name)
    set(cblas_header_name "cblas.h")
else()
    set(cblas_header_name "mkl_cblas.h")
endif()
#message(STATUS "cblas header: " ${cblas_header_name})


message(STATUS "Installing TAMM library at: " ${TAMM_INSTALL_PREFIX})

install(TARGETS tammx
        DESTINATION ${TAMM_INSTALL_PREFIX}/lib)

file(GLOB TAMM_HEADERS RELATIVE ${PROJECT_SOURCE_DIR} "${PROJECT_SOURCE_DIR}/tammx/*.h")
install(FILES ${TAMM_HEADERS} DESTINATION ${TAMM_INSTALL_PREFIX}/include)
