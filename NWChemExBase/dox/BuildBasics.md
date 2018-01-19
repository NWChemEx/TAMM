CMake Build Basics
------------------

This page is designed to detail the basic steps required to build a C++ project
using CMake.  It is targeted at users who want to understand how CMake works in
order to contribute to NWChemExBase (or simply because there's a lack of good
tutorials on the internets).

Contents
--------

1. [Building a C++ Project](#building-a-c++-project)
2. [CMake Workflow](#cmake-workflow)
3. [Setting Up a CMake Project](#setting-up-a-cmake-project)
4. [CMake Syntax](#cmake-syntax)

Building a C++ Project
----------------------

C++ is a compiled language.  What this means is a compiler turns the C++ source
code into some form of binary object such as an executable (a program that can
actually be run) or a library (a reusable collection of binary routines).  Many
tutorials make compiling seem simple because the tutorial is a single file.  
When you start making a package you quickly amass a multitude of source and 
header files.  Furthermore, you likely will want to link against other 
people's libraries.  Maybe your source tree is so big and parts change so rarely
that you want to break your source into multiple libraries.  Then you start
caring about performance, so now certain files are compiled with certain options
and others with other options.  Manually compiling the package (*i.e.* calling
the compiler for each and every source file with the appropriate commands) 
becomes error-prone and tedious.  Historically this is where compiling 
languages like `make` came in.

Make provides a "simple" mechanism for expressing rules for making a particular
target and for expressing dependencies among targets.  It is not however 
easy for make to locate the dependencies, nor is it easy for make to adapt the
build process to the current hardware platform.  This is where `autotools` and
`CMake` come in.  Generally speaking they both attempt to generate a set of 
build files that are knowledgeable about dependency locations and details of 
the current platform.  For the most part `autotools` is only used by GNU 
projects and although supposedly cross-platform, really is targeted at Linux.
For this reason many C++ projects prefer CMake for their build system.

To some extent this means we've developed an entire software stack around 
calling a single command a bunch of times with different arguments.  Whether 
there is a better way to do this, in a manner that is cross platform, is 
somewhat irrelevant at the moment.  This is because potential users of your 
package often want to treat compilation as a black box and consequentially 
can be easily frightened by build systems that are different than what they are
used to.  Anyways now that we've motivated the problem CMake intends to fix, 
let's discuss how it goes about doing this.
 
CMake Workflow
--------------

Before discussing how to use CMake let us discuss the workflow CMake is 
designed for.  "User" in this section refers to the person attempting to 
compile your code.

1. Obtain source.  Although it may seem silly to list this step it'll behoove us
   later.  As you can imagine this step is the literal process of obtaining a
   source tree.
2. Configure source.  If the source is designed to use CMake, this is where the
   user will invoke the `cmake` command.  This step generates the files 
   necessary to build the source.  Included in this step is system introspection
   as well as finding the dependencies.
3. Build source. After configuration, the files necessary for actually 
   building the source exist and the user is then responsible for running the
   build (typically by calling `make`, but not strictly necessary).
4. Test project.  After the project is built the user may test to ensure 
   everything built correctly.
5. Install project.  With the knowledge that the resulting project works right,
   the user installs it to a place will it will reside.

Note CMake is only directly called in step 2.  Despite this, we expect it to 
largely set-up everything from that point forward for us.  It does this via the 
build files it generates for the remaining 3 steps.  This is worth noting 
because it means you can't use CMake commands from steps 3 forward as those 
commands are being powered off build files and not CMake.

Setting Up A CMake Project
--------------------------

Talk about `CMakeLists.txt` here.
    
    
CMake Syntax
------------

This section is designed to get you to understand how to write `CMakeLists.txt` 
files.

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

Note `if(value2 AND ${value3})` will result in a cmake error if no variable
value3 is defined or if it is defined as empty/empty string.  Deref of value2 
gives a non-empty value as mentioned above, while deref of value3 gives empty
if not defined.


To check if a variable is defined, use: `if(DEFINED variable_name)`.

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
those commands.

