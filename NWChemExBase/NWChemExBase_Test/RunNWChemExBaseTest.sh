#!/bin/bash
set -e

####
#
# This script is designed to setup a test's directory structure and source
# files.
#
#   ./RunNWChemExBaseTest.sh <test_name> <path_to_folder_with BasicSetup.sh>
#     <path_to_NWXBase> <path_to_source_file> <cmake_command>
#
#   In the build directory, this script will perform the following steps:
#   1. make a directory called <test_name>_repo_test and change to it
#   2. run BasicSetup.sh <test_name>
#   3. copy source files to <test_name>/<test_name>
#   4. Generate a header file and a test source file
#

test_name=${1}
basic_setup=${2}/bin/BasicSetup.sh
nwx_base_path=${2}
path_to_source=${3}
cmake_command=${4}

if [ $# -lt 4 ];then
  echo "Recieved $# args: ${1}, ${2}, ${3}"
  echo "Usage: RunNWChemExBase.sh <name> <path_to_nwxbase> <path_to_source> " \
       "<path_to_cmake_command"
  exit 1
fi

header_file=${test_name}.hpp
source_file=${test_name}.cpp
test_dir=${test_name}_repo_test

if [ -e ${test_dir} ];then
   rm -rf ${test_dir}
fi
mkdir ${test_dir}

#Change to and setup the build directory
cd ${test_dir}
ln -s ${nwx_base_path} NWChemExBase #Pretend NWChemExBase is a sub-folder
${basic_setup} ${test_name}

#Make the header file
echo "#pragma once">${test_name}/${header_file}
echo "struct ${test_name} { void run_test();};">>${test_name}/${header_file}

echo "${path_to_source}"
cp ${path_to_source}/${source_file} ${test_name} #Copy source

#Add the sources and headers to the CMakeLists.txt
srcs_line="set(${test_name}_SRCS"
srcs_replace="${srcs_line} ${source_file}"
sed -i "s/${srcs_line}/${srcs_replace}/g" ${test_name}/CMakeLists.txt
incs_line="set(${test_name}_INCLUDES"
incs_replace="${incs_line} ${header_file}"
sed -i "s/${incs_line}/${incs_replace}/g" ${test_name}/CMakeLists.txt

#Add the tests
test_cmake=${test_name}_Test/CMakeLists.txt
test_src=${test_name}_Test/Test${source_file}
echo "add_cxx_unit_test(Test${test_name} ${test_name})" >> ${test_cmake}
echo "#include<${test_name}/${header_file}>" > ${test_src}
echo "#include \"catch/catch.hpp\"">>${test_src}
echo "int main(){ ${test_name} temp; temp.run_test(); return 0;}">> ${test_src}

cmake_vars=${path_to_source}/CMakeVars.txt

${cmake_command} -C ${cmake_vars} -H. -Bbuild
cd build
VERBOSE=1 make
