cmake_minimum_required(VERSION 2.8)
project (CTCE_LIBRARY C CXX)

# The version number.
set (CTCE_VERSION_MAJOR 1)
set (CTCE_VERSION_MINOR 0)

set (CMAKE_BUILD_TYPE Debug)

INCLUDE(CTest)
# Should a test fail, display the test program output
set (CTEST_OUTPUT_ON_FAILURE TRUE)

#Uncomment if we use additional cmake modules later
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/" "${CMAKE_CURRENT_BINARY_DIR}")

find_package(FLEX REQUIRED)
#find_package(BLAS REQUIRED)
#find_package(MPI)
#message(STATUS "mpi link flags: " ${MPI_CXX_LINK_FLAGS})
#message(STATUS "mpi compile flags: " ${MPI_CXX_COMPILE_FLAGS})
#message(STATUS "mpi libs: " ${MPI_CXX_LIBRARIES})


find_path (BLAS_INC_PATH NAMES cblas.h mkl_cblas.h PATHS ${CTCE_BLAS_INC})
find_library (BLAS_LIB_PATH blas ${CTCE_BLAS_LIB})

message(STATUS "BLAS headers found at: " "${BLAS_INC_PATH}")
message(STATUS "BLAS libs found: " "${BLAS_LIB_PATH}")


#Load the GA_CONFIG and NWCHEM_TOP env variables
if(NOT DEFINED ENV{NWCHEM_TOP})
    message(FATAL_ERROR "CTCE ERROR: Environment variable NWCHEM_TOP not set")
else()
    set(NWCHEM_TOP $ENV{NWCHEM_TOP})
endif()

set(GA_CONFIG ${NWCHEM_TOP}/src/tools/install/bin/ga-config)

execute_process(COMMAND ${GA_CONFIG} --cc OUTPUT_VARIABLE ctce_c_compiler OUTPUT_STRIP_TRAILING_WHITESPACE)
#execute_process(COMMAND ${GA_CONFIG} --f77 OUTPUT_VARIABLE ctce_f77_compiler OUTPUT_STRIP_TRAILING_WHITESPACE)

SET(CMAKE_C_COMPILER ${ctce_c_compiler})

#Assume Cmake picks the right c++ compiler for now
SET(CMAKE_CXX_COMPILER mpicxx)

execute_process(COMMAND ${GA_CONFIG} --cppflags --cflags OUTPUT_VARIABLE ctce_cflags OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${GA_CONFIG} --ldflags OUTPUT_VARIABLE ctce_ldflags OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${GA_CONFIG} --libs OUTPUT_VARIABLE ctce_libs OUTPUT_STRIP_TRAILING_WHITESPACE)


#Build lemon sources
set(ParserSrcDir ${PROJECT_SOURCE_DIR}/ctce_parser)
add_executable(lemon ${ParserSrcDir}/lemon.c)


# Collect all the parser sources.
set(ctceParserSources
    ctce_parser/absyn.c
    ctce_parser/ctce_parser.c
    ctce_parser/ctce_vector.c
    ctce_parser/intermediate.c
    ctce_parser/semant.c
    ctce_parser/visitor.c
    ctce_parser/symtab.c
    ctce_parser/util.c
    )

set(ctceLibSources
  tensor/capi.cc
  tensor/cc2_t1_ref.cc
  tensor/cc2_t2_ref.cc
  tensor/ccsd_t12.cc
  tensor/icsd_t1_ref.cc
  tensor/ccsd_e_ref.cc
  tensor/ccsd_t1_ref.cc
  tensor/ccsd_t2_ref.cc
  tensor/icsd_t2_ref.cc
  tensor/ipccsd_x1_ref.cc
  tensor/corf.cc
  tensor/dummy.cc
  tensor/equations.cc
  tensor/expression.cc
  tensor/func.cc
  tensor/ga_abstract.cc
  tensor/index_sort.cc
  tensor/preprocess.cc
  tensor/schedulers.cc
  tensor/stats.cc
  tensor/t_assign.cc
  tensor/t_mult.cc
  tensor/tensor.cc
  tensor/tensors_and_ops.cc
  tensor/variables.cc
  )

set(ParserBuildDir "${PROJECT_BINARY_DIR}")

# The generated lexer, parser code would be in ParserBuildDir.
set(ParserSources
    ${ParserBuildDir}/parser.h
    ${ParserBuildDir}/parser.c
    ${ParserBuildDir}/scanner.c
    ${ParserBuildDir}/scanner.h
    )

# Custom Target to build the lemon grammar.
ADD_CUSTOM_COMMAND(OUTPUT ${ParserSources}
    # The parser and scanner spec files are copied over to the ParserBuildDir before building it .
    COMMAND cmake -E copy ${ParserSrcDir}/parser.y ${ParserBuildDir}/parser.y
    COMMAND cmake -E copy ${ParserSrcDir}/scanner.l ${ParserBuildDir}/scanner.l
    COMMAND cmake -E copy ${ParserSrcDir}/lempar.c ${ParserBuildDir}/lempar.c

    #Generate the scanner code
    COMMAND cmake -E chdir ${ParserBuildDir} flex ${ParserBuildDir}/scanner.l

    # Generate the parser code
    COMMAND cmake -E chdir ${ParserBuildDir} ./lemon -s ${ParserBuildDir}/parser.y
    DEPENDS ${ParserSrcDir}/parser.y lemon
    COMMENT "Building the ctce lemon parser."
    VERBATIM
    )

#Build the generated scanner and parser sources
#add_library(ctce_sp STATIC ${ParserSources})
#target_include_directories(ctce_sp SYSTEM PUBLIC ${ParserSrcDir})

string(REPLACE "-I" "" ctce_cmake_cflags ${ctce_cflags})
#To be recognized as a list, the strings must be seperated by ;
string(REPLACE " " ";" ctce_cflags ${ctce_cmake_cflags})

SET(CMAKE_CXX_FLAGS "-DEXT_INT -DGFORTRAN")
add_library(ctce STATIC ${ctceParserSources} ${ParserSources} ${ctceLibSources})
target_include_directories(ctce SYSTEM PUBLIC  ${PROJECT_BINARY_DIR} ${ParserSrcDir} ${ctce_cflags} ${BLAS_INC_PATH})

# Build the ctce_parser_test code
add_executable(ctce_parser_test ${ParserSrcDir}/test.c)
add_dependencies(ctce_parser_test ctce)
# Set target libraries to be linked.
target_link_libraries(ctce_parser_test ctce fl)

string(REPLACE "-L" "" ctce_cmake_ldflags ${ctce_ldflags})
string(REPLACE " " ";" ctce_ldflags ${ctce_cmake_ldflags})

string(REPLACE "-l" "" ctce_cmake_libs ${ctce_libs})
#string(REPLACE "-" "" ctce_cmake_libs ${ctce_cmake_libs1})
string(REPLACE " " ";" ctce_libs ${ctce_cmake_libs})


find_path(cblas_header_name "mkl_cblas.h" ${BLAS_INC_PATH})
if (NOT cblas_header_name)
    set(cblas_header_name "cblas.h")
else()
    set(cblas_header_name "mkl_cblas.h")
endif()
#message(STATUS "cblas header: " ${cblas_header_name})

SET(lib_search "")
set(other_flags "")

foreach(tcelibname ${ctce_libs})
    SET(EACH_CTCE_LIB "EACH_CTCE_LIB-NOTFOUND")
    find_library(EACH_CTCE_LIB ${tcelibname} ${ctce_ldflags})
    if(EACH_CTCE_LIB)
      #message(STATUS "lib ${tcelibname} found at: " "${EACH_CTCE_LIB}")
      LIST(APPEND lib_search ${EACH_CTCE_LIB})
    else()
        LIST(APPEND other_flags ${tcelibname})
    endif()
endforeach(tcelibname)


#message(STATUS "check libs = " "${ctce_libs}")
#message(STATUS "check lib path = " "${ctce_ldflags}")
#message(STATUS "target libs = " "${lib_search}")
#message(STATUS "target flags = " "${other_flags}")


add_executable(ctce_main tensor/main.cc tensor/print.cc) #EXCLUDE_FROM_ALL
add_dependencies(ctce_main ctce)
target_include_directories(ctce_main SYSTEM PUBLIC  ${PROJECT_BINARY_DIR} ${ParserSrcDir} ${ctce_cflags} ${BLAS_INC_PATH})
# Set target libraries to be linked.
target_link_libraries(ctce_main ctce ${lib_search} blas)

# Install
if(NOT CTCE_INSTALL_PREFIX)
    set(CTCE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/ctce-install)
endif()

message(STATUS "Installing CTCE library at: " ${CTCE_INSTALL_PREFIX})

install(TARGETS ctce
        DESTINATION ${CTCE_INSTALL_PREFIX}/lib)

    file(GLOB CTCE_HEADERS RELATIVE ${PROJECT_SOURCE_DIR}  ${PROJECT_SOURCE_DIR}/ctce_parser/*.h ${PROJECT_SOURCE_DIR}/tensor/*.h)
install(FILES ${CTCE_HEADERS} DESTINATION ${CTCE_INSTALL_PREFIX}/include)

file(GLOB EQFILES RELATIVE ${PROJECT_SOURCE_DIR}  "${PROJECT_SOURCE_DIR}/ctce_parser/transform_input/generated/*.eq")
install(FILES ${EQFILES} DESTINATION ${CTCE_INSTALL_PREFIX}/eqs)

SET(CTCE_EQ_PATH ${CTCE_INSTALL_PREFIX}/eqs)

file(WRITE "${PROJECT_BINARY_DIR}/cmake_ctce_headers.h" "#define CBLAS_HEADER_FILE \"${cblas_header_name}\"")
file(APPEND "${PROJECT_BINARY_DIR}/cmake_ctce_headers.h" "\n#define CTCE_EQ_PATH \"${CTCE_INSTALL_PREFIX}/eqs\"")

set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
        "parser.y" "scanner.l" "parser.out" "output.txt" "lempar.c")


# Create directory to store test results
execute_process(COMMAND mkdir -p "TestResults/ParserTests")

# Run each test from the parser testsuite
foreach (test ${EQFILES})
    get_filename_component(Testname ${test} NAME_WE)
    add_test(${test} python ${PROJECT_SOURCE_DIR}/scripts/testing/ParserTest.py "${Testname}"
            "${PROJECT_BINARY_DIR}/ctce_parser_test" "${PROJECT_SOURCE_DIR}/ctce_parser/transform_input/generated/"
            "${PROJECT_BINARY_DIR}/TestResults/ParserTests")
endforeach ()


# Try finding Doxygen
FIND_PACKAGE(Doxygen)

if(NOT DOXYGEN_FOUND)
    SET(FOUND_DOXYGEN "0")
    if(NOT $ENV{DOXYGEN_HOME})
        SET(FOUND_DOXYGEN $ENV{DOXYGEN_HOME})
    endif()
else()
    SET(FOUND_DOXYGEN "1")
    set(DOCS "${PROJECT_BINARY_DIR}/docs")
    message(STATUS "Documentation will be built at: " ${DOCS})
    execute_process(COMMAND mkdir -p ${DOCS})
    set(DOCOUTPUT
            ${DOCS}/html
            ${DOCS}/latex
            )
    # Custom Target to build the Doxygen documentation.
    ADD_CUSTOM_COMMAND(OUTPUT ${DOCOUTPUT}
            COMMAND cmake -E copy ${PROJECT_SOURCE_DIR}/Doxyfile.in ${DOCS}/Doxyfile.in
            COMMAND cmake -E copy_directory ${PROJECT_SOURCE_DIR}/ctce_parser ${DOCS}/ctce_parser
            COMMAND cmake -E copy_directory ${PROJECT_SOURCE_DIR}/tensor ${DOCS}/tensor
            # Change to docs folder before building the documentation.
            COMMAND cmake -E chdir ${DOCS} doxygen ${DOCS}/Doxyfile.in
            COMMAND cmake -E remove_directory ${DOCS}/ctce_parser
            COMMAND cmake -E remove_directory ${DOCS}/tensor
            COMMAND cmake -E remove -f ${DOCS}/Doxyfile.in
            DEPENDS ${PROJECT_SOURCE_DIR}/Doxyfile.in
            COMMENT "Building Doxygen documentation."
            VERBATIM)

    ADD_CUSTOM_TARGET(docs DEPENDS ${DOCOUTPUT})
endif()
