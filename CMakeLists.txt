cmake_minimum_required(VERSION 2.8)
project (CTCE_LIBRARY C CXX)

# The version number.
set (CTCE_VERSION_MAJOR 1)
set (CTCE_VERSION_MINOR 0)

set (CMAKE_BUILD_TYPE Debug)

#INCLUDE(CTest)
# Should a test fail, display the test program output
#set (CTEST_OUTPUT_ON_FAILURE TRUE)

#Uncomment if we use additional cmake modules later
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/" "${CMAKE_CURRENT_BINARY_DIR}")


find_package(FLEX REQUIRED)
find_package(BLAS REQUIRED)

#Load the GA_CONFIG and NWCHEM_TOP env variables
SET(NWCHEM_TOP $ENV{NWCHEM_TOP})
SET(GA_CONFIG ${NWCHEM_TOP}/src/tools/install/bin/ga-config)

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
        tensor/variables.cc tensor/dummy.cc tensor/func.cc tensor/capi.cc tensor/preprocess.cc tensor/expression.cc
        tensor/expression.cc tensor/t_mult.cc tensor/t_assign.cc tensor/tensor.cc tensor/equations.cc
        tensor/ccsd_t1_eqs.cc tensor/ccsd_t2_eqs.cc tensor/corf.cc tensor/ccsd_t1_ref.cc tensor/ccsd_t2_ref.cc
        tensor/schedulers.cc tensor/ccsd_t12.cc tensor/ccsd_e_eqs.cc tensor/tensors_and_ops.cc tensor/stats.cc
        tensor/index_sort.cc)

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
        VERBATIM)

#Build the generated scanner and parser sources
add_library(ctce_sp STATIC ${ParserSources})
target_include_directories(ctce_sp SYSTEM PUBLIC ${ParserSrcDir})

#Build the ctce parser as a static library
add_library(ctceparser STATIC ${ctceParserSources} ${ParserSources})
target_include_directories(ctceparser SYSTEM PUBLIC ${PROJECT_BINARY_DIR} ${ParserSrcDir})
add_dependencies(ctceparser ctce_sp)

# Build the ctce_test code
add_executable(ctce_test ${ParserSrcDir}/test.c)
add_dependencies(ctce_test ctceparser)
# Set target libraries to be linked.
target_link_libraries(ctce_test ctceparser fl)


string(REPLACE "-I" "" ctce_cmake_cflags ${ctce_cflags})

#To be recognized as a list, the strings must be seperated by ;
string(REPLACE " " ";" ctce_cflags ${ctce_cmake_cflags})

#foreach(path ${ctce_cmake_cflags})
#    message("Path: " ${path})
#endforeach(path)
#message(STATUS "cmake flags = " ${ctce_cmake_cflags})


SET(CMAKE_CXX_FLAGS "-DEXT_INT -DGFORTRAN")
add_library(ctce STATIC ${ctceLibSources})
target_include_directories(ctce SYSTEM PUBLIC  ${PROJECT_BINARY_DIR} ${ParserSrcDir} ${ctce_cflags})
add_dependencies(ctce ctceparser)

#set_target_properties(ctce_test PROPERTIES
#        INSTALL_RPATH_USE_LINK_PATH TRUE)

