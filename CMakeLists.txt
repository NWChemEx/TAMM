cmake_minimum_required(VERSION 2.8)
project (CTCE C)

# The version number.
set (CTCE_VERSION_MAJOR 1)
set (CTCE_VERSION_MINOR 0)

set (CMAKE_BUILD_TYPE Debug)

#INCLUDE(CTest)
# Should a test fail, display the test program output
#set (CTEST_OUTPUT_ON_FAILURE TRUE)

#Uncomment if we use additional cmake modules later
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/" "${CMAKE_CURRENT_BINARY_DIR}")


find_package(FLEX)

include_directories("${PROJECT_SOURCE_DIR}/ctce_parser")

#Build lemon sources
set(ParserSrcDir ${PROJECT_SOURCE_DIR}/ctce_parser)
add_executable(lemon ${ParserSrcDir}/lemon.c)


# Collect all the parser sources.
file(GLOB ctceParserSources RELATIVE ${PROJECT_SOURCE_DIR}  "${PROJECT_SOURCE_DIR}/ctce_parser/*.c") # select all .c files

list(REMOVE_ITEM ctceParserSources "ctce_parser/test.c")
list(REMOVE_ITEM ctceParserSources "ctce_parser/lemon.c")
list(REMOVE_ITEM ctceParserSources "ctce_parser/lempar.c")

file(GLOB ctceLibSources RELATIVE ${PROJECT_SOURCE_DIR}  "${PROJECT_SOURCE_DIR}/tensor/*.cc") # select all .cc files

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

ADD_CUSTOM_TARGET(GrammarBuild ALL
        DEPENDS ${ParserSources})

#Set additional include paths
set(Includes
        ${PROJECT_BINARY_DIR}
   )
include_directories(${Includes})

#Build the generated scanner and parser sources
add_library(ctce_sp STATIC ${ParserSources})

#Build the ctce parser as a static library
add_library(ctce_parser STATIC ${ctceParserSources} ${ParserSources})
add_dependencies(ctce_parser ctce_sp)

# Build the ctce_test code
add_executable(ctce_test ${ParserSrcDir}/test.c)
add_dependencies(ctce_test ctce_parser)
# Set target libraries to be linked.
target_link_libraries(ctce_test ctce_parser fl)

#set_target_properties(ctce_test PROPERTIES
#        INSTALL_RPATH_USE_LINK_PATH TRUE)

