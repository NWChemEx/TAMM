Extending NWChemEx Base
=======================

The purpose of this page is to provide background on how this repository works
and to provide guidance on how to extend it.

Contents
--------
1. [Preliminaries](#preliminaries)
2. [Superbuild](#superbuild)
3. [NWChemExBase Model](#nwchemexbase-model)  
   a. [Superbuild Settings](#superbuild-settings)  
   b. [Declaring Your Library](#declaring-your-library)  
   c. [Declaring Your Tests](#delcaring-your-tests)  
   d. [Known Limitations](#known-limitations)
4. [Finding Dependencies](#finding-dependencies)  
5. [Enabling Additional Dependencies](#enabling-additional-dependencies)  
   a. [Writing a FindXXX.cmake File](#writing-a-findxxxcmake-file)  
   b. [Supported Dependencies](#supported-dependencies)
  

Preliminaries
-------------

For its "build" system NWChemExBase uses CMake.  The main assets of CMake is it
provides a nicer syntax than make, provides robust cross-platform builds, and it
is widely supported.  Technically speaking CMake is a build system generator,
which has important ramifications discussed below.  For now, realize that there 
really are two phases to a CMake build: the configuration phase and the build 
phase.  The configuration phase is when the build files are generated and the 
build phase is when the source files are compiled using said build files.  This 
is important because data generated during the build phase can not be acted on 
in the configuration phase as it is already over.

CMake itself uses a series of files named `CMakeLists.txt` to aid in the
configuration.  Typically, there is one of these per directory of your source
tree.  These files are written using the CMake language, the syntax of which, is
reminiscent of a Linux shell script. You can define and print variables like:

~~~cmake
#Comments start with #'s
#This sets a variable name use_library to true
set(use_library TRUE)
  
message(STATUS "The value of use_library is: ${use_library}")
  
#Note that CMake's variables are case-sensitive, so this
message(STATUS "The value of use_library is: ${USE_LIBRARY}")
#will print "The value of use_library is: "
~~~

Admittedly CMake's variables are more complex then they first seem.
~~~cmake
#Sets the variable a_value to /some/path
set(a_value /some/path)
  
#Sets the variable a_value to the string /some/path
set(a_value "/some/path")

#Sets the variable many_values to the list [/some/path,/another/path]
set(many_values /some/path /another/path)

#Sets the variable many_values to the list[/some/path,/another/path]
set(many_values "/some/path;/another/path")

#Sets the variable many_values the string "/some/path /another/path"
set(many_values "/some/path /another/path")
~~~
So why does the string vs. list thing matter?
~~~cmake
set(my_list arg1 arg2)
set(my_string "arg1 arg2")
some_fxn(${my_list}) # Same as some_fxn(arg1 arg2) i.e. passing two args
some_fxn(${my_string}) #Same as some_fxn("arg1 arg2") i.e. passing a single arg
~~~
Thus the difference is important for grouping.  Consequently, one common mistake
is to forward a list to a function when you really want to pass all the 
arguments as one argument (particularly relevant when passing compiler flags and
paths).

CMake supports basic control flow options like loops:
~~~cmake
foreach(x ${SOME_LIST})
   message(STATUS "Current element of list is: ${x}")
endforeach()
~~~
and if-statements:
~~~cmake
if(use_library)
   #Do something that requires the library
elseif(NOT use_library)
   #Do something in the event we aren't using that library
else()
   #Not sure how we get here...
endif()
~~~
It should be noted that the rules of if statements are weird in that it will
automatically dereference the value.  This is easier to grasp by example:
~~~cmake  
set(value1 FALSE)
set(value2 "value1")
if(value2)
  #Auto deref value2 gives string  "value1" which is not false
  message(STATUS "This will be printed")
elseif(${value2})
  #Deref of value2 happens first, if then derefs "value1" obtaining FALSE
  message(STATUS "This will not be printed")
endif()
~~~
Related, to check if a variable is defined it is: `if(DEFINED variable_name)`.

You can look for a particular dependency with:
~~~cmake
#CMake will crash if it doesn't find blas
find_package(BLAS REQUIRED)
  
#This is an error because find_package is case-sensitive
find_package(blas REQUIRED)
  
#This will find LAPACK, but not note that it did
find_package(LAPACK QUIET)
~~~

There are many more CMake commands, options, (and pitfalls) but for the most 
part you'll be interacting with them via the NWChemExBase API, which modifies 
those commands.  The rest of this page will get you acquainted with the 
NWChemExBase workflow.

Superbuild
----------

Unfortunately the two phase build leads to problems for more complex builds.
For example, assume you are building library B, which depends on some external
library A.  You start off by looking for A.  As mentioned, CMake provides a 
function `find_package` specifically for this reason.  Let's say `find_package` 
finds A. Great, you simply compile B against the A found by CMake.  What if it 
doesn't find it? You can crash and tell the user to go build it or you can 
attempt to build it yourself.  Wanting to be user friendly, you add an optional 
target that will build A if it's not found.  CMake provides a command 
`ExternalProject_Add` specifically for this purpose.  The problem is that A 
 will not be built until the build phase, thus all `find_package` calls in B 
 will fail as `find_package` happens during the configuration phase (and A 
 hasn't been built yet).
 
 This problem is common enough that a CMake pattern has emerged for it.  It is
 called the superbuild pattern. In this pattern all dependencies, as well as the
 main project itself, are included in the project as external projects.  This is
 because CMake runs the contents of external projects at build time, even if 
 there is CMake commands inside them.
 
 What does this mean to us?  Well it means the general control flow, at the top
 level of our project is:
 - Find dependencies using `find_package`
   - If dependency is found, great
   - Else if we are capable of building it, add its external project file
   - Crash
 - Build our main project(s) as external projects
   - Internally, find dependencies via `find_package`
   
Although there are other mechanisms for accomplishing the same result few are as
clean and uniform as the superbuild.  Furthermore, the superbuild has the
additional advantage of encapsulating the builds of each target.  This means it
is straightforward within any of the external projects to tweak paths, compiler 
flags, *etc.* without impacting other targets.

NWChemExBase Model
------------------

Drawing on years of experience writing `CMakeLists.txt` files there is a lot of
boiler-plate to them.  The primary goal of NWChemExBase is to take care of this
boiler-plate for you in a customizable and robust manner.  To that end, it is
far easier to accomplish this mission if we make some basic assumptions.

The first assumption is that your project source tree is setup like:

~~~
ProjectRoot/
├──CMakeLists.txt
├──NWChemExBase/
├──ProjectName/
|  └──CMakeLists.txt
└──ProjectName_Test/
    └──CMakeLists.txt
~~~
 
You are free to have additional folders and files, but they are not required for
the purposes of using NWChemExBase.  It should be noted that the folder 
`NWChemExBase` is a clone of the `NWChemExBase` repo (preferably as a git 
subrepo so that it can be updated as the need occurs).  The following 
subsections describe the files in a bit more detail.

### Superbuild Settings

The file `ProjectRoot/CMakeLists.txt` is the root `CMakeLists.txt` file and is
used as the entry point for CMake into your project and to tell NWChemExBase the
details of the superbuild.  It should be quite minimal, likely only including:
~~~cmake
#What version of CMake includes all your used features? Try to stick to 3.1
cmake_minimum_required(VERSION 3.1)
  
#Details about your project including:
# ProjectName : the name of your project used throughout the build.  It is case
#               sensitive
# a.b.c       : The major, minor, and patch versions of your project
project(ProjectName VERSION a.b.c LANGUAGES CXX) #Change a.b.c
  
#This line is considered input to NWChemExBase and tells it a list of external
#dependencies that your project depends on.  The name of the variable must be
#the case-sensitive project name supplied above followed by "_LIBRARY_DEPENDS" 
#in that exact case.  The ellipses would then be replaced by a list of external
#dependencies that your project needs.  The names of which must be valid
#find_package identifiers (a list is below)
set(ProjectName_LIBRARY_DEPENDS ...)
  
# Turn control over to NWChemExBase
add_subdirectory(NWChemExBase)
~~~

#### Declaring Your Library

The actual declaration of your library goes in 
`ProjectRoot/ProjectName/CMakeLists.txt`.  It is here you will specify the
 source files that need compiled, the headers that need installed, and any flags
 required to compile the source files.  Your library will automatically be
 linked to whatever dependencies you requested.

~~~cmake
#Should be same as root `CMakeLists.txt`
cmake_minimum_required(VERSION 3.1)
  
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
set(ProjectName_FLAGS ...)
  
#Finally we tell NWChemExBase to make a library ProjectName (the end name will
#be postfix-ed properly according to library type) using the specified sources,
#flags, and public API
nwchemex_add_library(ProjectName ProjectName_SRCS 
                                 ProjectName_INCLUDES
                                 ProjectName_FLAGS
                                 )
~~~

### Declaring Your Tests


The file `ProjectRoot/ProjectName_Test/CMakeLists.txt` will control the tests 
for your library.  By default the `Catch` C++ testing library will be visible to
your tests.  Simply include `#include catch/catch.hpp` in your test's source
file to use it.

~~~cmake
#Should be same as root `CMakeLists.txt`
cmake_minimum_required(VERSION 3.1)
  
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

- Can only specify one library.
  - Could add another variable `ProjectName_LIBRARIES` (or the like) that
    contains a list of the libraries.  Then NWChemExBase simply loops over them.
    - Each library would get to set its own dependencies, sources, *etc.*
- No support for restricting the version of a found library
  - Needs fixed, will happen before a 1.0 release

Finding Dependencies
--------------------

*N.B.* in this section `<Name>` is the name of a package as passed to 
`find_package` and `<NAME>` is the name of that package in all uppercase.  

CMake provides the `find_package` function for finding dependencies.  
Unfortunately, much of how this function works relies on naming conventions.  By
convention `find_package(<Name>)` is supposed to set minimally three variables:

1. `<NAME>_FOUND`        : Set to true if the package `<Name>` was found.
                           Unfortunately CMake does not specify the state of
                           this variable in the event it is not found.
2. `<NAME>_INCLUDE_DIRS` : All paths that a user of `<Name>` will need to 
                           include (`<Name>`'s headers and its dependencies)
3. `<NAME>_LIBRARIES`    : Same as `<NAME>_INCLUDE_DIRS1 except for libraries to 
                           link against                      
Optionally a package may set:

4. `<NAME>_DEFINITIONS`  : List of definitions and flags to use while compiling
5. `<Name>_FOUND`        : `find_package` expects a variable of the same case
                           back.  Setting this is needed for it to properly use
                           the REQUIRED keyword.                                        

Of course, many packages do not adhere to these standards complicating
automation.  Currently our solution is to write `Find<Name>Ex.cmake` files for
packages not adhering to them and to prefer projects us the *Ex* versions
instead of the normal ones.

Enabling Additional Dependencies
--------------------------------

It is likely inevitable that additional dependencies will occur.  When this
happens the primary responsibility of maintainers is to ensure that dependency
can be found by `find_package`.  This can happen in two ways:

1. If the dependency uses CMake (correctly) already it will generate a 
   `XXXConfig.cmake` file (typically in install/root/share/cmake) which 
   `find_package` can use to pull the dependency in.
2. You will have to write a `FindXXX.cmake` file for it.

The first scenario is ideal and means you don't have to do any work to ensure
CMake can find the dependency (if you want your dependency to be installable in
an automated fashion you'll still have work to do though).  The second scenario
requires work on our end.

### Writing a FindXXX.cmake File

First off let's discuss capitalization as it plays a key role here.  By default
when you call `find_package(aBc)` it will look for a file `FindaBc.cmake` that 
is the case is preserved.  Barring finding that it will look for 
`aBcConfig.cmake` or `abc-config.cmake`; however, 
we're assuming the config files do not exist.  Anyways, after calling 
`FindaBc.cmake`, `find_package` will determine if `XXX` was found
by considering the results of the variable `aBC_FOUND` (note the case
always matches the case given to `find_package`).  Lastly, making matters worse,
it is convention to always return variables (aside from the `XXX_FOUND` 
variable) in all uppercase letters (it's a good idea to return `XXX_FOUND` both
in the native case and in all caps).

Case caveats aside, let's say we want to do this in a textbook manner, then the 
resulting `FindXXX.cmake` file should look something like:

~~~cmake
#File FindXXX.cmake
#
# By convention this file will set the following:
#
# XXX_FOUND        to true if all parts of the XXX package are found
# XXX_INCLUDE_DIR  to the path for includes part of XXX's public API
# XXX_LIBRARY      to the libraries included with XXX
# XXX_INCLUDE_DIRS will be the includes of XXX as well as any dependency it
#                  needs
# XXX_LIBRARIES    will be the libraries of XXX and all its dependencies
# XXX_DEFINITIONS  will be any compile-time flags to use in your project
                   
  
#Call find_package for each dependency
find_package(Depend1)
  
#Try to piggy-back of package-config 
find_package(PkgConfig)
pkg_check_modules(PC_XXX <libname_without_suffix>)
  
#For each header file in the public API try to find it
find_path(XXX_INCLUDE_DIR <path/you/put/in/cxx/source/file>
          HINTS ${PC_XXX_INCLUDEDIR} ${PC_XXX_INCLUDE_DIRS}
)
  
#For each library try to find it
find_path(XXX_LIBRARY <library/name/including/the/lib/and/the/extension>)
          HINTS ${PC_XXX_LIBDIR} ${PC_XXX_LIBRARY_DIRS}
)
  
#Let CMake see see if the found values are sufficient
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XXX DEFAULT_MSG XXX_INCLUDE_DIR XXX_LIBRARY)
  
#In examples you'll see a marked_as_advanced line here a lot, but it's pretty
#useless as barely anyone runs the cmake GUI...
  
#Add dependencies and XXX's includes to XXX_INCLUDE_DIRS
set(XXX_INCLUDE_DIRS ${XXX_INCLUDE_DIR} ...)
  
#Same for libraries to link to
set(XXX_LIBRARIES ${XXX_LIBRARY} ...)
  
#Set the flags needed to compile against XXX
set(XXX_DEFINITIONS ...)
~~~
Once written your file goes in `NWChemBase/cmake/external/FindXXX.cmake`

### Enabling NWChemExBase to Build a Dependency

In an effort to make the build process more user-friendly it is common to want
to build dependencies for the user.  That is, if we are unable to locate a
required dependency on the system, we instead build it.  CMake doesn't have a
particular convention for how this done so we have taken the liberty of defining
a process for you.  We assume the following is in 
`NWChemExBase/cmake/external/BuildXXX.cmake`.

~~~cmake
#You may assume include(ExternalProject) was called

ExternalProject_Add(
    XXX_External #Target name. Must match file name and have _External suffix

)
~~~


### Supported Dependencies

These are dependencies that NWChemExBase currently knows how to find:

| Name            | Brief Description                                          |  
| :-------------: | :--------------------------------------------------------- |  
| MPI             | MPI compilers, includes, and libraries                     | 
| OpenMP          | Determines the flags for compiling/linking to OpenMP       |  
| AntlrCppRuntime | The ANTLR grammar parsing library                          |
| Eigen3          | The Eigen C++ matrix library                               |
| GTest           | Google's testing framework                                 |
| CatchEx         | Catch testing framework installed our way                  |
| GlobalArrays    | The Global Arrays distributed matrix library               |
