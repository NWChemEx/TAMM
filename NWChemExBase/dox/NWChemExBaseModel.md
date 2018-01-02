NWChemExBase Model
==================

The purpose of this page is to provide background on how this repository works
and to provide guidance on how to extend it.

Contents
--------

1. [NWChemExBase Model](#nwchemexbase-model)  
   a. [Superbuild Settings](#superbuild-settings)  
   b. [Declaring Your Library](#declaring-your-library)  
   c. [Declaring Your Tests](#delcaring-your-tests)  
   d. [Known Limitations](#known-limitations)
            


NWChemExBase Model
------------------

Drawing on years of experience writing `CMakeLists.txt` files there is a lot of
boiler-plate to them.  The primary goal of NWChemExBase is to take care of this
boiler-plate for you in a customizable and robust manner.  To that end, it is
far easier to accomplish this mission if we make some basic assumptions.

Given the details in the Superbuild section we assume you have modeled your 
directory layout after the flow of the Superbuild.  Ultimately, we assume your 
project source tree is setup like:

~~~
ProjectRoot/
├──CMakeLists.txt
├──NWChemExBase/
├──ProjectName/
|  └──CMakeLists.txt
└──ProjectName_Test/
    └──CMakeLists.txt
~~~
 
It's important to understand that directories establish scope in CMake, hence it
is possible to have a directory structure different than this; however, doing so
will make it difficult to use NWChemExBase (and CMake in general). Quickly going
through the directory tree.  `NWChemExBase` is a clone of the `NWChemExBase` 
repo (preferably as a git subrepo so that it can be updated as the need occurs).  
Assuming you are using git, `ProjectRoot` would be the root directory of your
repo and the directory users of your library clone.  So far as the outside 
world is concerned it is your project.  Directly inside this folder are the 
configuration instructions for the superbuild (the top-level `CMakeLists.txt`
 and much of `NWChemExBase/`) as well as the recipes for the dependencies (in 
`NWChemExBase/`).  The folder `ProjectName` is, from the perspective of the 
superbuild, your project (this is why we don't name it `src/`; *i.e.* typically 
the more canonical folders like `include/`, `src/`, `share/`, *etc.* will live
inside this folder).  Realizing that tests can be thought of as a set of 
libraries or executables that depend on you library they are elevated to 
the same rank as your project (so as to also be included via the 
`ExternalProject_Add` mechanism).

NWChemExBase allows for your repo to contain multiple libraries and/or 
executables.  Each of these targets must have a separate CMake build.  To tell
NWChemExBase about the targets simply set the `NWX_PROJECTS` variable to be a
list of your project names (by default this list is populated with the name you
provided the top-level `CMakeLists.txt`)


### Superbuild Settings

The file `ProjectRoot/CMakeLists.txt` is the root `CMakeLists.txt` file and is
used as the entry point for CMake into your project and to tell NWChemExBase the
details of the superbuild.  It should be quite minimal, likely only including:
~~~cmake
#What version of CMake includes all your used features? Try to stick to 3.6
cmake_minimum_required(VERSION 3.6)
  
#Details about your project including:
# ProjectName : the name of your project used throughout the build.  It is case
#               sensitive
# a.b.c       : The major, minor, and patch versions of your project
project(ProjectName VERSION a.b.c LANGUAGES CXX) #Change a.b.c

#This line is only required if your repo has multiple sub-projects to be built
#By default it will be set to whatever name you provided for ProjectName
set(NWX_PROJECTS Project1 Project2) 

#The following variables can be set per project you specified in `NWX_PROJECTS`
#For example assuming the above value we would have `Project1_SRC_DIR` and 
#`Project2_SRC_DIR` for the next variable

#This tells NWChemExBase which directory is the source directory for your
#Default is RootOfYourRepo/ProjectName
set(ProjectName_SRC_DIR path/to/src/dir)

#Similar to `_SRC_DIR` except for that project's tests.  Default is 
#RootOfYourRepo/ProjectName_Test
set(ProjectName_TEST_DIR path/to/test/dir)
  
#This line tells us the dependencies that your project depends on.  The name of
#The names assigned to this variable must be valid find_package identifiers 
set(ProjectName_DEPENDENCIES ...)
  
# This line signals that you're done with options and want to turn control over
# to NWChemExBase
add_subdirectory(NWChemExBase)
~~~

#### Declaring Your Library

The actual declaration of your library goes in 
`ProjectRoot/ProjectName/CMakeLists.txt`.  It is here you will specify the
 source files that need compiled, the headers that need installed, and any flags
 required to compile the source files.  Your library will automatically be
 linked to whatever dependencies you requested.

~~~cmake
#Set version off of top-level variable
cmake_minimum_required(VERSION ${CMAKE_VERSION})
  
#Should be same as root `CMakeLists.txt` aside from needing the "-SRC" postfix
project(ProjectName-SRC VERSION 0.0.0 LANGUAGES CXX)
  
#This will allow us to use the nwchemex_add_library command  
include(TargetMacros)

#Strictly speaking the following three variables can have whatever name you want
#as they will be passed to the nwchemex_add_library macro
  
#We create a list of all the source files (paths relative to this file)
set(ProjectName_SRCS ...)
  
#...a list of all header files that are part of the public API (*i.e.* need 
#to be installed with the library)
set(ProjectName_INCLUDES ...)
  
#...and a list of any compile flags/definitions to provide the library
set(ProjectName_DEFINITIONS ...)
  
#Finally we tell NWChemExBase to make a library ProjectName (the end name will
#be postfix-ed properly according to library type) using the specified sources,
#flags, and public API
nwchemex_add_library(ProjectName ProjectName_SRCS 
                                 ProjectName_INCLUDES
                                 ProjectName_DEFINITIONS
                                 ProjectName_LINK_FLAGS  
                                 )
~~~

### Declaring Your Tests


The file `ProjectRoot/ProjectName_Test/CMakeLists.txt` will control the tests 
for your library.  By default the `Catch` C++ testing library will be visible to
your tests.  Simply include `#include catch/catch.hpp` in your test's source
file to use it.

~~~cmake
#Set version based off top-level variable
cmake_minimum_required(VERSION ${CMAKE_VERSION})
  
#Should be same as root `CMakeLists.txt` aside from needing the "-Test" prefix
project(ProjectName-Test VERSION 0.0.0 LANGUAGES CXX)
  
#This will find your staged library (a ProjectNameConfig.cmake file was
#automatically generated for you during the build)  
find_package(ProjectName REQUIRED)
  
#Pull our testing macros into scope
include(TargetMacros)
  
#Add a test that lives in a file Test1.cpp and depends on the target ProjectName
add_cxx_unit_tests(Test1 ProjectName)
  
#Add additional tests...
~~~

At the moment we currently only have macros for adding C++ unit tests.  Other
languages and test types will be added as needed.

### Known Limitations

As you can imagine distilling a complex thing like a build down to a few
customizable options incurs some limitations.  At the moment these are:

- No support for restricting the version of a found library
  - Needs fixed, will happen before a 1.0 release
