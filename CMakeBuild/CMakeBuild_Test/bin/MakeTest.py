"""
This script will set up the skeleton of a test case for the CMakeBuild repo.
It is meant to be run from CMakeBuild_Test/bin and will place the result in
a directory with the name of the test.  That directory will reside up one
level from this script (CMakeBuild_Test).

Syntax: python3 MakeTest.py NameForTest


"""

import os
import sys
curr_dir =  os.path.dirname(os.path.realpath(__file__))
test_name = sys.argv[1]
root_dir = os.path.join(os.path.dirname(curr_dir),test_name)
src_dir = os.path.join(root_dir, test_name)
test_dir = os.path.join(root_dir, test_name +"_Test")

if os.path.isdir(root_dir):
    raise Exception("Directory already exists")

os.mkdir(root_dir)
os.mkdir(src_dir)
os.mkdir(test_dir)

with open(os.path.join(root_dir, "CMakeLists.txt"), 'w') as f:
    f.write("cmake_minimum_required(VERSION 3.6)\n")
    f.write("project({:s} VERSION 1.0.0 LANGUAGES CXX)\n".format(test_name))
    f.write("find_package(CMakeBuild)\n")
    f.write("set({:s}_DEPENDENCIES )\n".format(test_name))
    f.write("build_nwchemex_module(${CMAKE_CURRENT_LIST_DIR})\n")

with open(os.path.join(src_dir, "CMakeLists.txt"), 'w') as f:
    f.write("cmake_minimum_required(VERSION ${CMAKE_VERSION})\n")
    f.write("project({:s} ".format(test_name))
    f.write("VERSION ${PROJECT_VERSION} LANGUAGES CXX)\n")
    f.write("include(TargetMacros)\n")
    f.write("set({0:s}SRCS {0:s}.cpp)\n".format(test_name))
    f.write("set({0:s}HEADERS {0:s}.hpp)\n".format(test_name))
    f.write("nwchemex_add_library({0:s} {0:s}SRCS {0:s}HEADERS \"\" "
            "\"\")\n".format(test_name))
with open(os.path.join(src_dir, "{:s}.hpp".format(test_name)), 'w') as f:
    f.write("struct {:s} ".format(test_name))
    f.write("{\n    bool passed();\n")
    f.write("};\n")

with open(os.path.join(src_dir, "{:s}.cpp".format(test_name)), 'w') as f:
    f.write("#include \"{0:s}/{0:s}.hpp\"\n\n".format(test_name))
    f.write("bool {:s}::passed()".format(test_name))
    f.write("{\n    return true;\n}\n")

with open(os.path.join(test_dir, "CMakeLists.txt"), 'w') as f:
    f.write("cmake_minimum_required(VERSION ${CMAKE_VERSION})\n")
    f.write("project({:s}-Test ".format(test_name))
    f.write("VERSION ${PROJECT_VERSION} LANGUAGES CXX)\n")
    f.write("include(TargetMacros)\n")
    f.write("add_cxx_unit_test({:s})\n".format(test_name))

with open(os.path.join(test_dir, "{:s}.cpp".format(test_name)), 'w') as f:
    f.write("#include <{0:s}/{0:s}.hpp>\n".format(test_name))
    f.write("#include <catch/catch.hpp>\n\n")
    f.write("TEST_CASE(\"{:s}\")\n".format(test_name))
    f.write("{\n")
    f.write("    {:s} test;\n".format(test_name))
    f.write("    REQUIRE(test.passed());\n")
    f.write("}\n")
