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
project (TAMM C CXX)

enable_language(Fortran)

include(${PROJECT_SOURCE_DIR}/external/cmake/check_user_env.cmake)

message(STATUS "Building TAMM!")

# The version number.
set (TAMM_VERSION_MAJOR 0)
set (TAMM_VERSION_MINOR 1)

#Cannot set the following since cmake cannot handle clang
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
   OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(TAMM_CODECOV_FLAGS "-fprofile-arcs -ftest-coverage")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(TAMM_OPENMP_FLAGS -fopenmp)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(TAMM_OPENMP_FLAGS -qopenmp)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    #AppleClang does not have OpenMP support
     set(TAMM_OPENMP_FLAGS -fopenmp=libgomp)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    set(TAMM_OPENMP_FLAGS -mp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --c++14 -g  -O3 -mp --gnu --gnu_extensions")
    set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} --c++14 -mp -g  -O0 --gnu --gnu_extensions")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -g ${TAMM_CODECOV_FLAGS} -Ofast ${TAMM_OPENMP_FLAGS}")#  -Wextra  -Wall")
    set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} -std=gnu++14 -g ${TAMM_CODECOV_FLAGS} -Ofast ${TAMM_OPENMP_FLAGS}")# -Wextra  -Wall")
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

set(TAMM_SRC_DIR ${PROJECT_SOURCE_DIR}/../src)
set(TAMM_TEST_DIR ${PROJECT_SOURCE_DIR}/../tests)


include_directories( ${PROJECT_SOURCE_DIR} )
include_directories( ${TAMM_SRC_DIR} )

    set(GA_CONFIG "${GA_INSTALL_PATH}/bin")
    set(GA_LIB_PATH "${GA_INSTALL_PATH}/lib")



# Install
if(NOT TAMM_INSTALL_PREFIX)
    set(TAMM_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/tamm-install)
endif()

file(GLOB tamm_inputs "${PROJECT_SOURCE_DIR}/tamm_inputs/*.xyz")

install(FILES ${tamm_inputs} DESTINATION ${PROJECT_BINARY_DIR}/tamm_inputs)

find_path (BLAS_INCLUDE_PATH NAMES cblas.h mkl_cblas.h PATHS ${BLAS_INCLUDE_PATH})
message(STATUS "BLAS headers found at: ${BLAS_INCLUDE_PATH}")

set(FINDCBLAS_HEADER cblas.h)

find_path (MKL_INCLUDE_PATH NAMES mkl_cblas.h PATHS ${BLAS_INCLUDE_PATH})

if(MKL_INCLUDE_PATH)
    set(FINDCBLAS_HEADER mkl.h)
endif()

#list(APPEND CBLAS_DEFINITIONS "-DCBLAS_HEADER=\<${FINDCBLAS_HEADER}\>")
add_definitions(-DCBLAS_HEADER=\<${FINDCBLAS_HEADER}\>)

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

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXT_INT")


execute_process(COMMAND ${GA_CONFIG} --cppflags OUTPUT_VARIABLE tamm_includes OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${GA_CONFIG} --cflags OUTPUT_VARIABLE tamm_cflags OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${GA_CONFIG} --ldflags OUTPUT_VARIABLE tamm_ldflags OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${GA_CONFIG} --libs OUTPUT_VARIABLE tamm_libs OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REPLACE "-I" "" tamm_cmake_inc ${tamm_includes})
#To be recognized as a list, the strings must be seperated by ;
string(REPLACE " " ";" tamm_includes ${tamm_cmake_inc})

#execute_process(COMMAND mpicxx --showme:link OUTPUT_VARIABLE mpi_link OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
find_package(MPI REQUIRED)
foreach(_nwx_mpi_lib ${MPI_CXX_LIBRARIES})
    get_filename_component(_nwx_mpi_lib_we ${_nwx_mpi_lib} NAME_WE) #get lib name
    string(SUBSTRING ${_nwx_mpi_lib_we} 3 -1 _nwx_mpi_lib_we) #Strip lib prefix from lib name
    set(mpi_link "${mpi_link} -l${_nwx_mpi_lib_we}")
    string(STRIP ${mpi_link} mpi_link)
endforeach()

if(${CMAKE_Fortran_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_Fortran_COMPILER_ID} STREQUAL "PGI")
    set(STANDARDFORTRAN_LIBS gfortran)
elseif(${CMAKE_Fortran_COMPILER_ID} STREQUAL "Intel")
    set(STANDARDFORTRAN_LIBS ifcore)
else()
    message(FATAL_ERROR "${CMAKE_Fortran_COMPILER_ID} not supported yet.")
endif()

set(tamm_libs ${tamm_libs} ${mpi_link} -l${STANDARDFORTRAN_LIBS})

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

add_executable(eigen_tests EXCLUDE_FROM_ALL ${TAMM_TEST_DIR}/tammx/test_eigen.cc
        ${TAMM_TEST_DIR}/tammx/test_tammx.cc)
add_dependencies(eigen_tests tammx)
target_include_directories(eigen_tests SYSTEM PUBLIC ${tamm_includes}
            ${GTEST_INSTALL_PATH}/include)
target_compile_options(eigen_tests PUBLIC ${tamm_cflags})
target_link_libraries(eigen_tests tammx ${tamm_cflags} ${TAMM_LINK_EXTRA}
        ${tamm_ldflags} ${BLAS_LIBRARIES}  ${tamm_libs} ${BLAS_LIBRARIES}
         -L${GTEST_INSTALL_PATH}/lib -lgtest ${TAMM_OPENMP_FLAGS})

                
#BUILD TAMMX
    set(tammxSources
        ${TAMM_SRC_DIR}/tammx/tammx.cc
        ${TAMM_SRC_DIR}/tammx/index_sort.cc
        ${TAMM_SRC_DIR}/tammx/hartree_fock.cc
        ${TAMM_SRC_DIR}/tammx/two_four_index.cc)

add_library(tammx STATIC ${tammxSources})
target_include_directories(tammx SYSTEM PUBLIC ${PROJECT_BINARY_DIR} ${LIBINT_INSTALL_PATH}/include
        ${EIGEN3_INSTALL_PATH}/include/eigen3 ${tamm_includes} ${BLAS_INCLUDE_PATH})
target_link_libraries(tammx PRIVATE ${tamm_ldflags} ${BLAS_LIBRARIES}  ${tamm_libs} ${TAMM_OPENMP_FLAGS})

add_executable(ccsd_driver ${TAMM_SRC_DIR}/ccsd/ccsd_driver.cc)
add_dependencies(ccsd_driver tammx)
target_include_directories(ccsd_driver SYSTEM PUBLIC ${PROJECT_SOURCE_DIR} 
        ${LIBINT_INSTALL_PATH}/include ${EIGEN3_INSTALL_PATH}/include/eigen3 ${BLAS_INCLUDE_PATH})
target_link_libraries(ccsd_driver tammx ${TAMM_LINK_EXTRA} ${tamm_cflags} ${TAMM_OPENMP_FLAGS}
        ${tamm_ldflags} ${CBLAS_LIBRARIES}  ${tamm_libs} -L${CBLAS_LIBRARY_PATH} ${CBLAS_LIBRARIES} 
        -L${LIBINT_INSTALL_PATH}/lib -lint2 ${TAMM_LINK_EXTRA})

# add_executable(hf nwchem_main/hartree_fock.cc)
# target_include_directories(hf SYSTEM PUBLIC ${LIBINT_INSTALL_PATH}/include ${EIGEN3_INSTALL_PATH}/include/eigen3)
# target_link_libraries(hf -L${LIBINT_INSTALL_PATH}/lib -lint2)

add_executable(tammx_talsh EXCLUDE_FROM_ALL ${TAMM_SRC_DIR}/tammx/tammx_talsh.cc 
                ${TAMM_SRC_DIR}/tammx/tammx.cc ${TAMM_SRC_DIR}/tammx/index_sort.cc)
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

message(STATUS "Installing TAMMX library at: " ${TAMM_INSTALL_PREFIX})

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
            COMMAND cmake -E copy_directory ${TAMM_SRC_DIR}/tammx ${DOCS}/src/tammx
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


