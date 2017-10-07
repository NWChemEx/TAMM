#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Syntax: BasicSetup.sh <Name of Library>"
    exit 1
fi

LIBRARY_NAME=$1
SRC_DIR=${LIBRARY_NAME}
TEST_DIR=${LIBRARY_NAME}_Test
CMAKE_FILE=CMakeLists.txt
SRC_LIST=${SRC_DIR}/${CMAKE_FILE}
TEST_LIST=${TEST_DIR}/${CMAKE_FILE}

#Make required folders if they don't exist
for __dir in ${SRC_DIR} ${TEST_DIR}; do
    if [ ! -d ${__dir} ];then
        mkdir ${__dir}
    fi
done

function check_for_file () {
    if [ -f ${1} ];then
        echo "Error ${1} already exists"
        #exit 1
    fi
}

#Make top-level CMakeLists.txt
check_for_file ${CMAKE_FILE}
echo "cmake_minimum_required(VERSION 3.1)">${CMAKE_FILE}
echo "project(${LIBRARY_NAME} VERSION 0.0.0 LANGUAGES CXX)">>${CMAKE_FILE}
echo "set(NWCHEMEX_LIBRARY_DEPENDS )">>${CMAKE_FILE}
echo "add_subdirectory(NWChemExBase)">>${CMAKE_FILE}

#Make source-dir CMakeLists.txt
check_for_file ${SRC_LIST}
echo "cmake_minimum_required(VERSION 3.1)">${SRC_LIST}
echo "project(${LIBRARY_NAME}-SRC VERSION 0.0.0 LANGUAGES CXX)">>${SRC_LIST}
echo "include(TargetMacros)">>${SRC_LIST}
echo "set(${LIBRARY_NAME}_SRCS )">>${SRC_LIST}
echo "set(${LIBRARY_NAME}_INCLUDES )">>${SRC_LIST}
echo "set(${LIBRARY_NAME}_DEFINITIONS )">>${SRC_LIST}
echo "nwchemex_add_library(${LIBRARY_NAME} ${LIBRARY_NAME}_SRCS">>${SRC_LIST}
echo "                                 ${LIBRARY_NAME}_INCLUDES">>${SRC_LIST}
echo "                                 ${LIBRARY_NAME}_DEFINTIONS">>${SRC_LIST}
echo ")">>${SRC_LIST}

#Make test-dir CMakeLists.txt
check_for_file ${TEST_LIST}
echo "cmake_minimum_required(VERSION 3.1)">${TEST_LIST}
echo "project(${LIBRARY_NAME}-Test VERSION 0.0.0 LANGUAGES CXX)">>${TEST_LIST}
echo "find_package(${LIBRARY_NAME} REQUIRED)">>${TEST_LIST}
echo "include(TargetMacros)">>${TEST_LIST}

#Make a .gitignore
check_for_file .gitignore
echo "#These are configuration files for QtCreator"
echo "${LIBRARY_NAME}.config">>.gitignore
echo "${LIBRARY_NAME}.files">>.gitignore
echo "${LIBRARY_NAME}.includes">>.gitignore
echo "${LIBRARY_NAME}.creator">>.gitignore
echo "${LIBRARY_NAME}.creator.user">>.gitignore
echo "*.autosave">>.gitignore
echo "">>.gitignore
echo "#These are configuration files for CLion">>.gitignore
echo ".idea/">>.gitignore
echo "">>.gitignore
echo "#This is the documentation build directory">>.gitignore
echo "docs/">>.gitignore
echo "">>.gitignore
echo "#These are common build directory names">>.gitignore
echo "Debug/">>.gitignore
echo "Release/">>.gitignore
echo "cmake-build-debug/">>.gitignore
echo "cmake-build-release/">>.gitignore


#Make a .codedocs file
check_for_file .codedocs
echo "DOXYFILE = dox/Doxyfile"
