Target Macros
=============

These are macros for creating new targets (libraries, executables, or tests).

nwchemex_set_up_target
----------------------

This function primarily serves as code factorization by being the boilerplate
CMake required to set up a target.  It is the function that actually adds all
of the dependency include and library directories to the flags.

### Syntax:

```cmake
nwchemex_set_up_target(NAME FLAGS LFLAGS INCLUDES INSTALL)
```

Arguments:
- `NAME` the name of the target we are setting up (must already exist)
- `FLAGS` any additional compile-time flags beyond those for the dependencies
  - Assumed to be the actual string, *e.g.* `-Wall -O3`
- `LFLAGS` additional linking flags beyond those for the dependencies.
  - Assumed to be the actual string, *e.g.* `-lblas -llapack`
- `INSTALL` path relative to `CMAKE_INSTALL_PREFIX` to put the target

CMake Cache Variables:
- `NWX_INCLUDE_DIR` the path to the root of a module's source tree.
  - Set by `build_nwxbase_module`
- `NWX_DEPENDENCIES` a list of `find_package` exposed dependencies  

### Example

This example sets up a target `AProgram` so that it is compiled with the flag
`-Wall` and installed into `CMAKE_INSTALL_PREFIX/bin`

```cmake
add_executable(AProgram aprogram.cpp)
nwchemex_set_up_target(AProgram "-Wall" "" "bin")
```

nwchemex_add_executable
-----------------------

This macro wraps `nwchemex_set_up_target` so that the resulting target will 
be an executable.

### Syntax

```cmake
nwchemex_add_executable(NAME SOURCES FLAGS LFLAGS)
```

Arguments:
- `NAME` the name for the resulting target (and executable)
- `SOURCES` a list of source files for the executable.
  - Should be passed as the name of a variable containing the list
- `FLAGS` a list of the compiler flags to use
  - Should be passed as the name of a variable containing the list  
- `LFLAGS` a list of the flags for the linker
  - Should be passed as the name of a variable containing the list

### Example

This example makes an executable called `AProject` which has one source file 
`src1.cpp`.  It will be compiled with `-Wall -O3` and linked against BLAS and
 LAPACK.

```cmake
set(AProjectSRCs src1.cpp)
set(AProjectFLAGS "-Wall" "-O3")
set(AProjectLFLAGS "-lblas" "-llapack")
nwchemex_add_executable(AProject AProjectSRCs AProjectFLAGS AProjectLFLAGS) 
```

nwchemex_add_library
-----------------------

This macro wraps `nwchemex_set_up_target` so that the resulting target will 
be a library.  Whether the library is static or shared is controlled by the 
CMake variable `BUILD_SHARED_LIBS`

### Syntax

```cmake
nwchemex_add_library(NAME SOURCES HEADERS FLAGS LFLAGS)
```

Arguments:
- `NAME` the name for the resulting target (and library)
- `SOURCES` a list of source files for the library.
  - Should be passed as the name of a variable containing the list
- `HEADERS` a list of the header files that need installed to use the library.
  - Should be passed as the name of a variable containing the list  
- `FLAGS` a list of the compiler flags to use
  - Should be passed as the name of a variable containing the list  
- `LFLAGS` a list of the flags for the linker
  - Should be passed as the name of a variable containing the list

CMake Cache Variables:
- `BUILD_SHARED_LIBS` if true a shared library will be built, otherwise the 
result will be static

### Example

This example makes a library called `AProject` which has one source file 
`src1.cpp` and one header file `src1.hpp`.  It will be compiled with `-Wall -O3` 
and linked against BLAS and LAPACK.

```cmake
set(AProjectSRCS src1.cpp)
set(AProjectHEADERS src1.hpp)
set(AProjectFLAGS "-Wall" "-O3")
set(AProjectLFLAGS "-lblas" "-llapack")
nwchemex_add_library(AProject AProjectSRCS 
                              AProjectHEADERS 
                              AProjectFLAGS 
                              AProjectLFLAGS) 
```

nwchemex_add_test
-----------------

This is the guts of what it takes to define a test (which requires 
compilation of a file).  It shouldn't be called directly and exists primarily as
code factorization.

### Syntax

```cmake
nwchemex_add_test(NAME TEST_FILE)
```

Arguments:
- `NAME` the name of test to create
- `TEST_FILE` the name of the source file containing the test

### Example

This example makes a test called "ATest" whose contents are within a source 
file "ATest.cpp".

```cmake
nwchemex_add_test(ATest ATest.cpp)
```

add_cxx_unit_test
-----------------

Specializes `nwchemex_add_test` to unit tests formed from a single source 
file.  The resulting test will have the label "UnitTest".

### Syntax

```cmake
add_cxx_unit_test(NAME)
```

Arguments:
- `NAME` the name of the resulting test and the name of the source file
  - the source file is assumed to have extension `.cpp`
  
### Example

This example makes a test `ATest` whose contents are contained in a source 
file `ATest.cpp`.  It will have the label "UnitTest".

```cmake
add_cxx_unit_test(ATest)
```  

add_cmake_macro_test
--------------------

Used by the CMakeBuild project to test its macros.  This function takes the
name of a CMake script (without the `.cmake` suffix) and will run that script as
a test.

### Syntax

```cmake
add_cmake_macro_test(NAME)
```

Arguments:
- `NAME` the name of the resulting test and the name of the script to run
  - the script is assumed to have extension `.cmake`
  
### Example

This example makes a test `ATest` whose contents are contained in a script 
file `ATest.cmake`.

```cmake
add_cmake_macro_test(ATest)
``` 
add_nwxbase_test
--------------------

This function is meant to be used by the CMakeBuild project to simulate runs 
of CMakeBuild in order to test functionality. This means it's a bit meta.  In
an attempt to be clear let's call the CMakeBuild distribution that we just 
built and are now testing the "root CMakeBuild".  Since tests are run from 
the build directory all invocations will use the staged version of the root 
CMakeBuild to build the test.

Each test is expected to be a mock CMake project, which can be built with 
CMakeBuild.  Ultimately, that mock project must produce a test with the 
same name in the usual way (*i.e.* make a call to `add_cxx_unit_test`).  The 
root CMakeBuild distribution will then run that test (in a very nested 
folder since tests aren't installed).

There's ultimately a lot of assumptions here so it is recommended that you 
use the file `CMakeBuild_Test/bin/MakeTest.py` to make a skeleton setup for 
your new test.  After running the script you should only need to fill in the 
details of the `passed` method in the resulting `.cpp` file and (depending on
the test) set CMake variables in the `CMakeLists.txt` just inside the 
resulting directory tree.  

### Syntax

```cmake
add_nwxbase_test(NAME)
```

Arguments:
- `NAME` the name of the resulting test and the name of a directory that is set
up correctly for use with CMakeBuild
  
### Example

This example makes a test `ATest` whose contents are contained in a 
subdirectory `ATest`.  The subdirectory `ATest` includes a CMakeLists.txt and
source files to build a library and an executable that will be used as a test.

```cmake
add_nwxbase_test(ATest)
``` 
