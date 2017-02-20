# ------------------------------------------------------------------------------
#  Copyright (C) 2016, Pacific Northwest National Laboratory
#  This software is subject to copyright protection under the laws of the
#  United States and other countries
# 
#  All rights in this computer software are reserved by the
#  Pacific Northwest National Laboratory (PNNL)
#  Operated by Battelle for the U.S. Department of Energy
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 2.8)
project (TAMM_LIBRARY C CXX)
enable_language (Fortran)

# The version number.
set (TAMM_VERSION_MAJOR 1)
set (TAMM_VERSION_MINOR 0)

set (CMAKE_BUILD_TYPE Debug)

INCLUDE(CTest)
# Should a test fail, display the test program output
set (CTEST_OUTPUT_ON_FAILURE TRUE)


#find_package(FLEX REQUIRED)
#find_package(BLAS REQUIRED)
#find_package(MPI)
#message(STATUS "mpi link flags: " ${MPI_CXX_LINK_FLAGS})
#message(STATUS "mpi compile flags: " ${MPI_CXX_COMPILE_FLAGS})
#message(STATUS "mpi libs: " ${MPI_CXX_LIBRARIES})

include_directories( ${PROJECT_SOURCE_DIR} )

add_definitions(-DMPICH_IGNORE_CXX_SEEK)
SET(tamm_blaslibs "")

if(NOT DEFINED LINUX_BLAS)
    find_path (BLAS_INC_PATH NAMES cblas.h mkl_cblas.h PATHS ${TAMM_BLAS_INC})
    find_library (BLAS_LIB_PATH blas ${TAMM_BLAS_LIB})
    find_library (BLAS_LIB_PATH1 mkl_core ${TAMM_BLAS_LIB})
    find_library (BLAS_LIB_PATH2 mkl_sequential ${TAMM_BLAS_LIB})
    if(BLAS_LIB_PATH)
        LIST(APPEND tamm_blaslibs ${BLAS_LIB_PATH})
    endif()
    if(BLAS_LIB_PATH1)
        SET(tamm_blaslibs "")
        LIST(APPEND tamm_blaslibs ${BLAS_LIB_PATH1})
    endif()
    if(BLAS_LIB_PATH2)
        LIST(APPEND tamm_blaslibs ${BLAS_LIB_PATH2})
    endif()
    message(STATUS "BLAS headers found at: " "${BLAS_INC_PATH}")
    message(STATUS "BLAS libs found: " "${tamm_blaslibs}")
else()
    add_definitions(-DLINUX_BLAS)
endif()

#Load the GA_CONFIG and NWCHEM_TOP env variables
if(NOT DEFINED ENV{NWCHEM_TOP})
    message(FATAL_ERROR "TAMM ERROR: Environment variable NWCHEM_TOP not set")
else()
    set(NWCHEM_TOP $ENV{NWCHEM_TOP})
endif()

set(GA_CONFIG ${NWCHEM_TOP}/src/tools/install/bin/ga-config)

execute_process(COMMAND ${GA_CONFIG} --cc OUTPUT_VARIABLE tamm_c_compiler OUTPUT_STRIP_TRAILING_WHITESPACE)
#execute_process(COMMAND ${GA_CONFIG} --f77 OUTPUT_VARIABLE tamm_f77_compiler OUTPUT_STRIP_TRAILING_WHITESPACE)

SET(CMAKE_C_COMPILER ${tamm_c_compiler})

#Assume Cmake picks the right c++ compiler for now
SET(CMAKE_CXX_COMPILER mpicxx)

execute_process(COMMAND ${GA_CONFIG} --cppflags --cflags OUTPUT_VARIABLE tamm_cflags OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${GA_CONFIG} --ldflags OUTPUT_VARIABLE tamm_ldflags OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${GA_CONFIG} --libs OUTPUT_VARIABLE tamm_libs OUTPUT_STRIP_TRAILING_WHITESPACE)

#Uncomment if we use additional cmake modules later
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")
# Setup ANTLR
include(SetupANTLR)

message("Antlr Binary = " ${AntlrBinary})

set(ParserSrcDir ${PROJECT_SOURCE_DIR}/antlr_parser)
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/parser")

ADD_CUSTOM_TARGET(BuildANTLR ALL DEPENDS ANTLR)


#set(ParserBuildDir "${PROJECT_BINARY_DIR}/parser")

# The generated lexer, parser sources would be in ParserBuildDir.
# set(ParserSources
#        ${ParserBuildDir}/TAMMLexer.cpp
#        ${ParserBuildDir}/TAMMParser.cpp
#     )   
    
  
# Custom Target to build the TAMM grammar.
# ADD_CUSTOM_COMMAND(OUTPUT ${ParserSources}
#   # The grammar is copied over to the ParserBuildDir before building it .
#   COMMAND cmake -E copy ${ParserSrcDir}/TAMM.g4 ${ParserBuildDir}/TAMM.g4
#   # Change to ParserBuildDir before building the grammar.
#   COMMAND cmake -E chdir ${ParserBuildDir} java -jar ${AntlrBinary} -Dlanguage=Cpp -no-listener -visitor ${ParserBuildDir}/TAMM.g4
#   DEPENDS ${ParserSrcDir}/TAMM.g4
#   COMMENT "Building the TAMM grammar."
#   VERBATIM)


    # Compile the lexer and parser.
# add_library(AnltrTammParser STATIC ${ParserSources})
# target_include_directories(AnltrTammParser SYSTEM PUBLIC ${ParserBuildDir} ${ANTLR_CPPRUNTIME}/include/antlr4-runtime)
# # Build the ANTLR CPP runtime library. 
# add_dependencies(AnltrTammParser ANTLR)
# target_link_libraries(AnltrTammParser antlr4-runtime)

# add_library(ParserCall STATIC "${PROJECT_SOURCE_DIR}/antlr_parser/tamm_parser.cc")
# add_dependencies(ParserCall AnltrTammParser ANTLR)
# target_include_directories(ParserCall SYSTEM PUBLIC ${ANTLR_CPPRUNTIME}/include/antlr4-runtime)
# link_directories(${ANTLR_CPPRUNTIME}/lib)
# target_link_libraries(ParserCall AnltrTammParser antlr4-runtime)


# Collect all the parser sources.
set(tammParserSources
    antlr_parser/parse.cc
    antlr_parser/TAMMLexer.cc
    antlr_parser/TAMMParser.cc
    antlr_parser/ASTBuilder.cc
    antlr_parser/TAMMVisitor.cc
    #antlr_parser/intermediate.cc
    #antlr_parser/semant.cc
    #antlr_parser/util.cc
    )

set(tammLibSources
  tensor/capi.cc
  tensor/cisd_e_ref.cc
  tensor/cisd_c1_ref.cc
  tensor/cisd_c2_ref.cc
  tensor/cc2_t1_ref.cc
  tensor/cc2_t2_ref.cc
  tensor/ccsd_t12.cc
  tensor/ccsd_lambda1_ref.cc
  tensor/ccsd_lambda1_mod_ref.cc
  tensor/ccsd_lambda2_ref.cc
  #tensor/cisd_c12.cc
  tensor/icsd_t1_ref.cc
  tensor/ccsd_e_ref.cc
  tensor/ccsd_t1_ref.cc
  tensor/ccsd_t2_ref.cc
  tensor/icsd_t2_ref.cc
  tensor/ipccsd_x1_ref.cc
  tensor/ipccsd_x2_ref.cc
  tensor/eaccsd_x1_ref.cc
  tensor/eaccsd_x2_ref.cc
  tensor/ccsd_1prdm_hh_ref.cc
  tensor/ccsd_1prdm_pp_ref.cc
  tensor/ccsd_1prdm_hp_ref.cc
  tensor/ccsd_1prdm_ph_ref.cc
  tensor/corf.cc
  tensor/dummy.cc
  tensor/equations.cc
  tensor/expression.cc
  tensor/func.cc
  tensor/gmem.cc
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

# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

string(REPLACE "-I" "" tamm_cmake_cflags ${tamm_cflags})
#To be recognized as a list, the strings must be seperated by ;
string(REPLACE " " ";" tamm_cflags ${tamm_cmake_cflags})

message("ANTLR CPPRUNTIME=" ${ANTLR_CPPRUNTIME}/include)
SET(CMAKE_CXX_FLAGS "-DEXT_INT -DGFORTRAN -std=c++14")
#SET(CMAKE_CXX_FLAGS "-DEXT_INT -DGFORTRAN")
add_library(tamm STATIC ${tammParserSources}) # ${tammLibSources})
add_dependencies(tamm BuildANTLR)
link_directories(${ANTLR_CPPRUNTIME}/lib)
target_include_directories(tamm SYSTEM PUBLIC  ${PROJECT_BINARY_DIR} ${ParserSrcDir} ${tamm_cflags} 
                        ${BLAS_INC_PATH} ${ANTLR_CPPRUNTIME}/include/antlr4-runtime/)
target_link_libraries(tamm antlr4-runtime)
#set_property(TARGET tamm PROPERTY CXX_STANDARD 14)
#set_property(TARGET tamm PROPERTY CXX_STANDARD_REQUIRED ON)



# Build the tamm_parser_test code
add_executable(tamm_parser_test ${ParserSrcDir}/test.cc)
add_dependencies(tamm_parser_test tamm)
target_include_directories(tamm_parser_test SYSTEM PUBLIC  ${ParserSrcDir}) 
# Set target libraries to be linked.
target_link_libraries(tamm_parser_test tamm antlr4-runtime)



# string(REPLACE "-L" "" tamm_cmake_ldflags ${tamm_ldflags})
# string(REPLACE " " ";" tamm_ldflags ${tamm_cmake_ldflags})

# string(REPLACE "-l" "" tamm_cmake_libs ${tamm_libs})
# #string(REPLACE "-" "" tamm_cmake_libs ${tamm_cmake_libs1})
# string(REPLACE " " ";" tamm_libs ${tamm_cmake_libs})


# find_path(cblas_header_name "mkl_cblas.h" ${BLAS_INC_PATH})
# if (NOT cblas_header_name)
#     set(cblas_header_name "cblas.h")
# else()
#     set(cblas_header_name "mkl_cblas.h")
# endif()
# #message(STATUS "cblas header: " ${cblas_header_name})

# SET(lib_search "")
# set(other_flags "")

# foreach(tcelibname ${tamm_libs})
#     SET(EACH_TAMM_LIB "EACH_TAMM_LIB-NOTFOUND")
#     find_library(EACH_TAMM_LIB ${tcelibname} ${tamm_ldflags})
#     if(EACH_TAMM_LIB)
#       #message(STATUS "lib ${tcelibname} found at: " "${EACH_TAMM_LIB}")
#       LIST(APPEND lib_search ${EACH_TAMM_LIB})
#     else()
#         LIST(APPEND other_flags ${tcelibname})
#     endif()
# endforeach(tcelibname)


# #message(STATUS "check libs = " "${tamm_libs}")
# #message(STATUS "check lib path = " "${tamm_ldflags}")
# #message(STATUS "target libs = " "${lib_search}")
# #message(STATUS "target flags = " "${other_flags}")


# add_executable(tamm_main tensor/main.cc tensor/print.cc) #EXCLUDE_FROM_ALL
# add_dependencies(tamm_main tamm)
# target_include_directories(tamm_main SYSTEM PUBLIC  ${PROJECT_BINARY_DIR} ${ParserSrcDir} ${tamm_cflags} ${BLAS_INC_PATH})
# # Set target libraries to be linked.
# target_link_libraries(tamm_main tamm ${lib_search} ${tamm_blaslibs})

# # Install
# if(NOT TAMM_INSTALL_PREFIX)
#     set(TAMM_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/tamm-install)
# endif()

# message(STATUS "Installing TAMM library at: " ${TAMM_INSTALL_PREFIX})

# install(TARGETS tamm
#         DESTINATION ${TAMM_INSTALL_PREFIX}/lib)

# file(GLOB TAMM_HEADERS RELATIVE ${PROJECT_SOURCE_DIR}  ${PROJECT_SOURCE_DIR}/tamm_parser/*.h ${PROJECT_SOURCE_DIR}/tensor/*.h)
# install(FILES ${TAMM_HEADERS} DESTINATION ${TAMM_INSTALL_PREFIX}/include)

# file(GLOB EQFILES RELATIVE ${PROJECT_SOURCE_DIR}  "${PROJECT_SOURCE_DIR}/tamm_tests/NWChemEqs/*.eq")
# install(FILES ${EQFILES} DESTINATION ${TAMM_INSTALL_PREFIX}/tests/NWChemEqs)
# file(GLOB EQFILES RELATIVE ${PROJECT_SOURCE_DIR}  "${PROJECT_SOURCE_DIR}/tamm_tests/SoEqs/*.eq")
# install(FILES ${EQFILES} DESTINATION ${TAMM_INSTALL_PREFIX}/tests/SoEqs)

# SET(TAMM_EQ_PATH ${TAMM_INSTALL_PREFIX}/tests/NWChemEqs)

# if(NOT DEFINED LINUX_BLAS)
#   add_definitions(-DCBLAS_HEADER_FILE="${BLAS_INC_PATH}/${cblas_header_name}")
# endif()

# add_definitions(-DTAMM_EQ_PATH=\"${TAMM_INSTALL_PREFIX}/tests/NWChemEqs\")

# set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
#         "parser.y" "scanner.l" "parser.out" "output.txt" "lempar.c")


# # Create directory to store test results
# execute_process(COMMAND mkdir -p "TestResults/ParserTests")

# # Run each test from the parser testsuite
# foreach (test ${EQFILES})
#     get_filename_component(Testname ${test} NAME_WE)
#     add_test(${test} python ${PROJECT_SOURCE_DIR}/scripts/testing/ParserTest.py "${Testname}"
#             "${PROJECT_BINARY_DIR}/tamm_parser_test" "${PROJECT_SOURCE_DIR}/tamm_tests/NWChemEqs/"
#             "${PROJECT_BINARY_DIR}/TestResults/ParserTests")
# endforeach ()


# # Try finding Doxygen
# FIND_PACKAGE(Doxygen)

# if(NOT DOXYGEN_FOUND)
#     SET(FOUND_DOXYGEN "0")
#     if(NOT $ENV{DOXYGEN_HOME})
#         SET(FOUND_DOXYGEN $ENV{DOXYGEN_HOME})
#     endif()
# else()
#     SET(FOUND_DOXYGEN "1")
#     set(DOCS "${PROJECT_BINARY_DIR}/docs")
#     message(STATUS "Documentation will be built at: " ${DOCS})
#     execute_process(COMMAND mkdir -p ${DOCS})
#     set(DOCOUTPUT
#             ${DOCS}/html
#             ${DOCS}/latex
#             )
#     # Custom Target to build the Doxygen documentation.
#     ADD_CUSTOM_COMMAND(OUTPUT ${DOCOUTPUT}
#             COMMAND cmake -E copy ${PROJECT_SOURCE_DIR}/Doxyfile.in ${DOCS}/Doxyfile.in
#             COMMAND cmake -E copy_directory ${PROJECT_SOURCE_DIR}/tamm_parser ${DOCS}/tamm_parser
#             COMMAND cmake -E copy_directory ${PROJECT_SOURCE_DIR}/tensor ${DOCS}/tensor
#             # Change to docs folder before building the documentation.
#             COMMAND cmake -E chdir ${DOCS} doxygen ${DOCS}/Doxyfile.in
#             COMMAND cmake -E remove_directory ${DOCS}/tamm_parser
#             COMMAND cmake -E remove_directory ${DOCS}/tensor
#             COMMAND cmake -E remove -f ${DOCS}/Doxyfile.in
#             DEPENDS ${PROJECT_SOURCE_DIR}/Doxyfile.in
#             COMMENT "Building Doxygen documentation."
#             VERBATIM)

#     ADD_CUSTOM_TARGET(docs DEPENDS ${DOCOUTPUT})
# endif()

# add_custom_target("style")
# add_custom_command(TARGET "style" PRE_BUILD 
#   COMMAND ${CMAKE_COMMAND} ARGS -E chdir ${CMAKE_SOURCE_DIR}/scripts ./cpplint.bash ${CMAKE_SOURCE_DIR}/tensor/*
#   COMMENT "=================== Checking style..." )

# add_custom_target("patch")
# add_custom_command(TARGET "patch" POST_BUILD 
#     COMMAND ${CMAKE_COMMAND} ARGS -E chdir ${CMAKE_SOURCE_DIR}/nwchem-mods-6.6 ./nwchem-fix.bash ${TAMM_INSTALL_PREFIX}/lib
#     COMMENT "=================== Patching nwchem..." )

# add_custom_target("patch-only")
# add_custom_command(TARGET "patch-only" POST_BUILD 
#     COMMAND ${CMAKE_COMMAND} ARGS -E chdir ${CMAKE_SOURCE_DIR}/nwchem-mods-6.6 ./nwchem-fix.bash ${TAMM_INSTALL_PREFIX}/lib -l
#     COMMENT "=================== Patching nwchem..." )

# add_custom_target("unpatch")
# add_custom_command(TARGET "unpatch" POST_BUILD 
#     COMMAND ${CMAKE_COMMAND} ARGS -E chdir ${CMAKE_SOURCE_DIR}/nwchem-mods ./nwchem-unpatch.bash
#     COMMENT "=================== Unpatching nwchem gnumakefile..." )

# add_custom_target(link COMMAND "${CMAKE_COMMAND}" --build . --target install)

# get_filename_component(NW_CC ${CMAKE_C_COMPILER} NAME )
# get_filename_component(NW_FC ${CMAKE_Fortran_COMPILER} NAME )
# get_filename_component(NW_AR ${CMAKE_AR} NAME )

# add_custom_command(TARGET "link" POST_BUILD
#     COMMAND ${CMAKE_COMMAND} ARGS -E chdir ${NWCHEM_TOP}/src make CC=${NW_CC} FC=${NW_FC} AR=${NW_AR} link
#     COMMENT "=================== Relinking nwchem..." )

# if(NOT DEFINED $ENV{TAMM_TEST_PROC_COUNT})
#     set(TAMM_TEST_PROC_COUNT 8)
# else()
#     set(TAMM_TEST_PROC_COUNT $ENV{TAMM_TEST_PROC_COUNT})
# endif()
# set (TEST_COMMAND "mpirun -n ${TAMM_TEST_PROC_COUNT} ${NWCHEM_TOP}/bin/LINUX64/nwchem")

# add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --verbose)

# file(GLOB INTEG_TESTS RELATIVE ${PROJECT_SOURCE_DIR} "${PROJECT_SOURCE_DIR}/test/integration/inputs/*.nw")
# foreach (test ${INTEG_TESTS})
#     get_filename_component(Testname ${test} NAME_WE)
#     string(REGEX REPLACE "\\.[^.]*$" "" Testname "${Testname}")
#     add_test(${test} ${PROJECT_SOURCE_DIR}/scripts/testing/check_output.sh "${TEST_COMMAND} ${CMAKE_SOURCE_DIR}/${test}"
#         ${PROJECT_SOURCE_DIR}/test/integration/correct_outputs/${Testname}.output
#         ${PROJECT_SOURCE_DIR}/test/integration/tests/${Testname}.tests
#         )
# endforeach()

