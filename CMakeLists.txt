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
project (TAMM C CXX Fortran)

if (BUILD_OLD_TAMM)
    message(STATUS "Building OLD_TAMM!")
else()
    message(STATUS "Building TAMM!")
endif()

if(NWX_PROC_COUNT)
    set(NWX_PROC_COUNT ${NWX_PROC_COUNT})
else()
    include(ProcessorCount)
    ProcessorCount(NWX_PROC_COUNT)
    message(STATUS "Number of cores detected = ${NWX_PROC_COUNT}")
endif()

message(STATUS "Using ${NWX_PROC_COUNT} cores for building & running TAMM tests.")
message("Please set the NWX_PROC_COUNT cmake variable to change the cpu count.")

if (BUILD_OLD_TAMM)
  include(ExternalProject)
  ExternalProject_Add (OLD_TAMM
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/OLD_TAMM
  BINARY_DIR ${PROJECT_SOURCE_DIR}/OLD_TAMM/build
  CMAKE_ARGS 
  -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DNWCHEM_TOP=${NWCHEM_TOP} 
  -DGA_CONFIG=${GA_CONFIG} -DNWCHEM_BUILD_DIR=${NWCHEM_BUILD_DIR} 
  -DNWCHEM_CMAKE_BUILD=${NWCHEM_CMAKE_BUILD} -DNWX_PROC_COUNT=${NWX_PROC_COUNT} 
  -DANTLR_CPPRUNTIME_PATH=${ANTLR_CPPRUNTIME_PATH}
  -DCMAKE_EXPORT_COMPILE_COMMANDS=${CMAKE_EXPORT_COMPILE_COMMANDS}
  BUILD_COMMAND make -j${NWCHEM_PROC_COUNT}
  INSTALL_COMMAND make install
  )

  ExternalProject_Add (TAMM
    SOURCE_DIR ${PROJECT_SOURCE_DIR}
    BINARY_DIR ${PROJECT_BINARY_DIR}
    DEPENDS OLD_TAMM
    CMAKE_ARGS -DBUILD_OLD_TAMM=OFF -DOLD_TAMM=ON 
    # -DCMAKE_INSTALL_PREFIX=${NWCHEM_INSTALL_DIR}
    # -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    # -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DNWX_PROC_COUNT=${NWX_PROC_COUNT} 
    # -DLIBINT_INSTALL_PATH=${LIBINT_INSTALL_PATH} -DANTLR_CPPRUNTIME_PATH=${ANTLR_CPPRUNTIME_PATH}
    # -DEIGEN3_INSTALL_PATH=${EIGEN3_INSTALL_PATH} -DBLAS_INCLUDE_PATH=${BLAS_INCLUDE_PATH}
    # -DBLAS_LIBRARY_PATH=${BLAS_LIBRARY_PATH} -DBLAS_LIBRARY64_PATH=${BLAS_LIBRARY64_PATH}
    # -DGTEST_INSTALL_PATH=${GTEST_INSTALL_PATH} -DTALSH_INSTALL_PATH=${TALSH_INSTALL_PATH}
    # -DCMAKE_EXPORT_COMPILE_COMMANDS=${CMAKE_EXPORT_COMPILE_COMMANDS}
    # -DNWCHEM_TOP=${NWCHEM_TOP} 
    # -DGA_CONFIG=${GA_CONFIG} -DNWCHEM_BUILD_DIR=${NWCHEM_BUILD_DIR} 
    # -DNWCHEM_CMAKE_BUILD=${NWCHEM_CMAKE_BUILD} 
    BUILD_COMMAND make -j${NWCHEM_PROC_COUNT}
    INSTALL_COMMAND make install
  )
  return() # stop further processing
# else()
#     message(STATUS "Building TAMM!")
endif()

set(CMAKE_BUILD_TYPE Debug)

# The version number.
set (TAMM_VERSION_MAJOR 0)
set (TAMM_VERSION_MINOR 1)

#Cannot set the following since cmake cannot handle clang
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TAMM_OPENMP_FLAGS -fopenmp)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    set(TAMM_OPENMP_FLAGS -mp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --c++14 -g  -O3 -mp --gnu --gnu_extensions")
    set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} --c++14 -mp -g  -O0 --gnu --gnu_extensions")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g  -Ofast ${TAMM_OPENMP_FLAGS}")#  -Wextra  -Wall")
    set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} -std=c++14 -g -Ofast ${TAMM_OPENMP_FLAGS}")# -Wextra  -Wall")
endif()
#set(CMAKE_CXX_FLAGS_MINSIZEREL     "")
#set(CMAKE_CXX_FLAGS_RELEASE        "")
#set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "")

INCLUDE(CTest)
# Should a test fail, display the test program output
set (CTEST_OUTPUT_ON_FAILURE TRUE)

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
include_directories( ${PROJECT_SOURCE_DIR}/src )

if(OLD_TAMM)

    if(NOT NWCHEM_TOP)
        message(FATAL_ERROR "TAMM ERROR: CMAKE variable NWCHEM_TOP not set")
    else()
    if (NOT EXISTS ${NWCHEM_TOP})
        message(FATAL_ERROR "${NWCHEM_TOP} not found. Please make sure the path exists \
        and NWCHEM_TOP is set correctly in the toolchain file provided: ${CMAKE_TOOLCHAIN_FILE}")
    endif()
        message(STATUS "NWCHEM_TOP set to: ${NWCHEM_TOP}")
    endif()

    #Setup required for building OLD TAMM code
    set(OLD_TAMM_PROJECT_DIR "${PROJECT_SOURCE_DIR}/OLD_TAMM")
    include_directories( ${OLD_TAMM_PROJECT_DIR} )
    set(OLD_TAMM_LIB ${PROJECT_SOURCE_DIR}/OLD_TAMM/build/oldtamm-install/lib/libtamm.a)

    if (NWCHEM_CMAKE_BUILD)
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

    else()
        if (EXISTS ${NWCHEM_TOP}/lib/${NWCHEM_BUILD_TARGET}/libnwcblas.a)
        set(BLAS_LIBRARIES        
            ${NWCHEM_TOP}/lib/${NWCHEM_BUILD_TARGET}/libpeigs.a
            ${NWCHEM_TOP}/lib/${NWCHEM_BUILD_TARGET}/libnwcblas.a
            ${NWCHEM_TOP}/lib/${NWCHEM_BUILD_TARGET}/libnwclapack.a)
        else()
            set(BLAS_LIBRARIES
            ${NWCHEM_TOP}/lib/${NWCHEM_BUILD_TARGET}/libpeigs.a)
        endif()
    endif()

else()
    set(GA_CONFIG "${GA_INSTALL_PATH}/bin")
    set(GA_LIB_PATH "${GA_INSTALL_PATH}/lib")
endif()



# Install
if(NOT TAMM_INSTALL_PREFIX)
    set(TAMM_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/tamm-install)
endif()

file(GLOB tamm_inputs "${PROJECT_SOURCE_DIR}/tamm_inputs/*.xyz")

install(FILES ${tamm_inputs} DESTINATION ${PROJECT_BINARY_DIR}/tamm_inputs)

find_path (BLAS_INCLUDE_PATH NAMES cblas.h mkl_cblas.h PATHS ${BLAS_INCLUDE_PATH})
message(STATUS "BLAS headers found at: ${BLAS_INCLUDE_PATH}")

#Load the GA_CONFIG variable
if(NOT GA_CONFIG)
    message(FATAL_ERROR "TAMM ERROR: CMAKE variable GA_CONFIG not set")
else()
    set(GA_CONFIG ${GA_CONFIG}/ga-config)
    if (NOT EXISTS ${GA_CONFIG})
        message(FATAL_ERROR "${GA_CONFIG} not found. Please make sure the path exists \
        and GA_CONFIG is set correctly in the toolchain file provided: ${CMAKE_TOOLCHAIN_FILE}")
    endif()
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
${PROJECT_SOURCE_DIR}/src/nwtest/put_hash_block.F
${PROJECT_SOURCE_DIR}/src/nwtest/add_block.F
${PROJECT_SOURCE_DIR}/src/nwtest/get_block.F
${PROJECT_SOURCE_DIR}/src/nwtest/get_block_ind.F
${PROJECT_SOURCE_DIR}/src/nwtest/add_hash_block.F
${PROJECT_SOURCE_DIR}/src/nwtest/get_hash_block.F
${PROJECT_SOURCE_DIR}/src/nwtest/get_hash_block_ma.F
${PROJECT_SOURCE_DIR}/src/nwtest/nxtask.F
${PROJECT_SOURCE_DIR}/src/nwtest/nxtval_ga.F
#${PROJECT_SOURCE_DIR}/src/nwtest/errquit.F
${PROJECT_SOURCE_DIR}/src/nwtest/put_block.F
${PROJECT_SOURCE_DIR}/src/nwtest/tce_hash.F
${PROJECT_SOURCE_DIR}/src/nwtest/tce_hash_ind.F
${PROJECT_SOURCE_DIR}/src/nwtest/tce_restricted.F
${PROJECT_SOURCE_DIR}/src/nwtest/tce_sort.F
${PROJECT_SOURCE_DIR}/src/nwtest/tce_sortacc.F
${PROJECT_SOURCE_DIR}/src/nwtest/misc.F
${PROJECT_SOURCE_DIR}/src/nwtest/ccsd_t1.F
${PROJECT_SOURCE_DIR}/src/nwtest/ccsd_t2.F
${PROJECT_SOURCE_DIR}/src/nwtest/cc2_t1.F
${PROJECT_SOURCE_DIR}/src/nwtest/cc2_t2.F
)

set(EXTRA_FORT_TAMM ${PROJECT_SOURCE_DIR}/src/nwtest/set_vars_tamm.F)
set(EXTRA_FORT ${PROJECT_SOURCE_DIR}/src/nwtest/set_vars.F)

set(TESTING_DEPS_SRC
#${PROJECT_SOURCE_DIR}/src/nwtest/fort_assign_tests.cc
#${PROJECT_SOURCE_DIR}/src/nwtest/ccsd_t1_test.cc
#${PROJECT_SOURCE_DIR}/src/nwtest/ccsd_t2_test.cc
)

#SET(CMAKE_Fortran_COMPILER mpif90)
if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_Fortran_COMPILER_ID}" STREQUAL "PGI")
  set(tamm_fortran_link_flags ${tamm_fflags})
  set(tamm_libs ${tamm_libs} ${mpi_link})
elseif("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
  set(tamm_fortran_link_flags -i8)
else()
    message(FATAL_ERROR "Only GNU/Intel/PGI Fortran compilers supported")
endif()


SET(CBLAS_LIBRARY_PATH ${BLAS_LIBRARY_PATH})
if (EXISTS ${BLAS_LIBRARY64_PATH})
    set(CBLAS_LIBRARY_PATH ${BLAS_LIBRARY64_PATH})
endif()


if (BLAS_LIBRARIES AND LAPACK_LIBRARIES)
  set(CBLAS_LIBRARIES "${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}")
else()
  set(CBLAS_LIBRARIES
    ${CBLAS_LIBRARY_PATH}/libblas.a
    ${CBLAS_LIBRARY_PATH}/libcblas.a
    ${CBLAS_LIBRARY_PATH}/liblapack.a
    ${CBLAS_LIBRARY_PATH}/liblapacke.a
    )
endif()

if(NOT "${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Darwin")
    set(TAMM_LINK_EXTRA -lrt)
endif()

add_library(fort_deps STATIC ${FORTRAN_DEPS_SRC} ${EXTRA_FORT_TAMM})
target_include_directories(fort_deps SYSTEM PUBLIC ${tamm_includes} ${PROJECT_SOURCE_DIR}/src/tce_fortran)
target_compile_options(fort_deps PRIVATE ${tamm_fortran_link_flags})


add_library(fort_deps_tammx STATIC ${FORTRAN_DEPS_SRC} ${EXTRA_FORT})
target_include_directories(fort_deps_tammx SYSTEM PUBLIC ${tamm_includes} ${PROJECT_SOURCE_DIR}/src/tce_fortran)
target_compile_options(fort_deps_tammx PRIVATE ${tamm_fortran_link_flags})

add_library(fort_deps_c STATIC ${PROJECT_SOURCE_DIR}/src/nwtest/util_gnxtval.c)
target_include_directories(fort_deps_c SYSTEM PUBLIC ${tamm_includes})

if(OLD_TAMM)
    add_executable(nwtest EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/src/nwtest/test_tamm.cc
            ${PROJECT_SOURCE_DIR}/src/nwtest/test_fortran.cc ${PROJECT_SOURCE_DIR}/src/nwtest/test_tammx.cc ${TESTING_DEPS_SRC})
    add_dependencies(nwtest fort_deps fort_deps_c tammx)
    target_include_directories(nwtest SYSTEM PUBLIC ${tamm_includes} ${BLAS_LIBRARY_PATH}
            ${ANTLR_CPPRUNTIME_PATH}/include/antlr4-runtime ${GTEST_INSTALL_PATH}/include)
    target_compile_options(nwtest PUBLIC ${tamm_cflags})
    target_link_libraries(nwtest fort_deps fort_deps_c tammx tce_fortran ${tamm_cflags} ${CBLAS_LIBRARIES} ${OLD_TAMM_LIB}
            ${tamm_ldflags} ${BLAS_LIBRARIES}  ${tamm_libs} ${BLAS_LIBRARIES} -L${GTEST_INSTALL_PATH}/lib -lgtest ${TAMM_LINK_EXTRA} ${TAMM_OPENMP_FLAGS})
endif()

add_executable(eigen_tests EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/src/nwtest/test_eigen.cc
         ${PROJECT_SOURCE_DIR}/src/nwtest/test_tammx.cc)
add_dependencies(eigen_tests fort_deps fort_deps_c tammx)
target_include_directories(eigen_tests SYSTEM PUBLIC ${tamm_includes}
            ${GTEST_INSTALL_PATH}/include)
target_compile_options(eigen_tests PUBLIC ${tamm_cflags})
target_link_libraries(eigen_tests fort_deps fort_deps_c tammx tce_fortran ${tamm_cflags} ${TAMM_LINK_EXTRA}
        ${tamm_ldflags} ${BLAS_LIBRARIES}  ${tamm_libs} ${BLAS_LIBRARIES} fort_deps -L${GTEST_INSTALL_PATH}/lib -lgtest ${TAMM_OPENMP_FLAGS})

                
#BUILD TAMMX
    set(tammxSources
        ${PROJECT_SOURCE_DIR}/src/tammx/tammx.cc
        ${PROJECT_SOURCE_DIR}/src/tammx/index_sort.cc
        ${PROJECT_SOURCE_DIR}/src/tammx/hartree_fock.cc
        ${PROJECT_SOURCE_DIR}/src/tammx/two_four_index.cc)

add_library(tammx STATIC ${tammxSources})
target_include_directories(tammx SYSTEM PUBLIC ${PROJECT_BINARY_DIR} ${LIBINT_INSTALL_PATH}/include
        ${EIGEN3_INSTALL_PATH}/include/eigen3 ${tamm_includes} ${BLAS_INCLUDE_PATH})
target_link_libraries(tammx PRIVATE ${tamm_ldflags} ${BLAS_LIBRARIES}  ${tamm_libs} ${TAMM_OPENMP_FLAGS})

#add_executable(tammx_main ${PROJECT_SOURCE_DIR}/src/tammx/tammx_main.cc)
#target_link_libraries(tammx_main tammx)
#set_property(TARGET tammx_main PROPERTY CXX_STANDARD 14)
#set_property(TARGET tammx_main PROPERTY CXX_STANDARD_REQUIRED ON)



add_library(tce_fortran ${PROJECT_SOURCE_DIR}/src/tammx/ccsd/ccsd_t1.F ${PROJECT_SOURCE_DIR}/src/tce_fortran/errquit.F)
target_include_directories(tce_fortran SYSTEM PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/tce_fortran ${tamm_includes})
target_compile_options(tce_fortran PRIVATE ${tamm_fortran_link_flags})

add_executable(ccsd_driver ${PROJECT_SOURCE_DIR}/src/tammx/ccsd/ccsd_driver.cc)
add_dependencies(ccsd_driver fort_deps_tammx fort_deps_c tammx)
target_include_directories(ccsd_driver SYSTEM PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/tce_fortran
        ${LIBINT_INSTALL_PATH}/include ${EIGEN3_INSTALL_PATH}/include/eigen3 ${BLAS_INCLUDE_PATH})
target_link_libraries(ccsd_driver fort_deps_tammx fort_deps_c tammx tce_fortran ${TAMM_LINK_EXTRA} ${tamm_cflags} ${TAMM_OPENMP_FLAGS}
        ${tamm_ldflags} ${CBLAS_LIBRARIES}  ${tamm_libs} -L${CBLAS_LIBRARY_PATH} ${CBLAS_LIBRARIES} -L${LIBINT_INSTALL_PATH}/lib -lint2 ${TAMM_LINK_EXTRA})

# add_executable(hf nwchem_main/hartree_fock.cc)
# target_include_directories(hf SYSTEM PUBLIC ${LIBINT_INSTALL_PATH}/include ${EIGEN3_INSTALL_PATH}/include/eigen3)
# target_link_libraries(hf -L${LIBINT_INSTALL_PATH}/lib -lint2)

add_executable(tammx_talsh EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/src/tammx/tammx_talsh.cc 
                ${PROJECT_SOURCE_DIR}/src/tammx/tammx.cc ${PROJECT_SOURCE_DIR}/src/tammx/index_sort.cc)
target_include_directories(tammx_talsh SYSTEM PUBLIC  ${PROJECT_BINARY_DIR}
${TALSH_INSTALL_PATH})
target_link_libraries(tammx_talsh -L${TALSH_INSTALL_PATH} ${TAMM_LINK_EXTRA} -ltalsh ${TAMM_OPENMP_FLAGS})

# message(STATUS "ANTLR C++ runtime found at: " ${ANTLR_CPPRUNTIME_PATH})
# ADD_LIBRARY(a4runtime STATIC IMPORTED)
# SET_TARGET_PROPERTIES(a4runtime PROPERTIES IMPORTED_LOCATION ${ANTLR_CPPRUNTIME_PATH}/lib/libantlr4-runtime.a)


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

# Find Doxygen
FIND_PACKAGE(Doxygen)

if(DOXYGEN_FOUND)
    set(DOCS "${PROJECT_BINARY_DIR}/docs")
    message(STATUS "Documentation will be built at: " ${DOCS})
    execute_process(COMMAND mkdir -p ${DOCS}/src)
    set(DOCOUTPUT ${DOCS}/html)
    # Custom Target to build the Doxygen documentation.
    ADD_CUSTOM_COMMAND(OUTPUT ${DOCOUTPUT}
            COMMAND cmake -E copy ${PROJECT_SOURCE_DIR}/Doxyfile.in ${DOCS}/Doxyfile.in
            COMMAND cmake -E copy_directory ${PROJECT_SOURCE_DIR}/src/tammx ${DOCS}/src/tammx
            # Change to docs folder before building the documentation.
            COMMAND cmake -E chdir ${DOCS} doxygen ${DOCS}/Doxyfile.in
            COMMAND cmake -E remove_directory ${DOCS}/src/tammx
            COMMAND cmake -E remove -f ${DOCS}/Doxyfile.in
            DEPENDS ${PROJECT_SOURCE_DIR}/Doxyfile.in
            COMMENT "Building Doxygen documentation."
            VERBATIM)
    ADD_CUSTOM_TARGET(docs DEPENDS ${DOCOUTPUT})
endif()
        

# file(GLOB TAMM_HEADERS RELATIVE ${PROJECT_SOURCE_DIR} "${PROJECT_SOURCE_DIR}/frontend/*.h" "${PROJECT_SOURCE_DIR}/tensor/*.h")
# install(FILES ${TAMM_HEADERS} DESTINATION ${TAMM_INSTALL_PREFIX}/include)

# file(GLOB EQFILES RELATIVE ${PROJECT_SOURCE_DIR}  "${PROJECT_SOURCE_DIR}/tamm_tests/NWChemEqs/*.eq")
# install(FILES ${EQFILES} DESTINATION ${TAMM_INSTALL_PREFIX}/tests/NWChemEqs)


