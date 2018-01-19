Configuring and Building the Project
====================================

The current project uses
:link:[NWChemExBase](https://github.com/NWChemEx-Project/NWChemExBase) for
its build infrastructure.  NWChemExBase is a collection of infrastructure meant 
to provide a uniform and reproducible build environment for C++ projects.  It
also is meant to greatly simplify the process of writing build files for 
those C++ projects.  Hence the build instructions here are generic; however, 
they should work with any project that uses NWChemExBase.  These instructions
are aimed at people trying to build an NWChemExBase project; users hoping to 
leverage NWChemExBase in their C++ project are encouraged to follow the above
link to the NWChemExBase GitHub repository for more information.


Contents
--------

1. [Building in a Sane, Linux-Like Environment](#building-in-a-sane,-linux-like-environment)
2. [CMake Basics](#cmake-basics)
3. [CMake Supplied Options](#cmake-supplied-options)
4. [NWChemExBase Specific Options](#nwchemexbase-specific-options)
5. [Troubleshooting](#troubleshooting)
   

Building in a Sane, Linux-Like Environment
------------------------------------------

In an idealized Linux environment where all build tools are in their 
expected places.  Building the current project should be as simple as 
(commands issued from the root of the current project):

```bash
cmake -H. -B<name of build directory> 
cd <name of build directory>
make
ctest #optional, command runs test suite
make install
```

If these commands don't work, or you simply want more control over the build 
the remainder of this page is designed to aid you in those endeavors.

CMake Basics
------------

This section is a (very) short tutorial on using CMake to configure a project.
For more information consult the man page for your `cmake` command.  If you are
familiar with using CMake you can skip this section.

For users familiar with `autoconf`, CMake is simply an alternative 
program for configuring the build system.  Like `autoconf`, arguments 
designed to modify the build are passed to the `cmake` command via flags.  
For CMake the syntax is `-D<OPTION_NAME>=<VALUE>` (it's `D` for define, if 
the mnemonic helps you) to set a particular option to a particular value.  
Tables of recognized options (and their allowed values can be found in the 
following sections).  For some options you may wish to proivde multiple 
values (*e.g.* setting dependency search paths).  To do this the syntax is 
similar to above except instead of a single value you pass a semi-colon 
separated list of the values, in quotes, *i.e.* 
`-D<OPTION_WITH_TWO_VALUES>="<VALUE1>;<VALUE2>"`.

Our build requires that the source tree be kept pristine and that objects be
built in a "build directory" (if you look inside that directory after the build
you'll see why we require it; CMake generates a lot of files).  There's a few
ways of setting the path to the build directory, but we prefer it be done 
with the `-B<build_dir>` (`B` for build directory) argument to the `cmake` 
command.  Here `build_dir` is a path to a folder, relative to the directory 
in which you are running the `cmake` command, to place files.  If the folder 
doesn't exist CMake will create it.

Finally, CMake needs to know the root of the project so that it can traverse the
source tree to create the build files.  This is specified via the `-H.` flag 
(`H` for home) if running CMake in the root of the project.

Hence while following these instructions your invocation of CMake ought to look
something like:

```bash
cmake -H. -B<build_dir> -DOPTION1=value -DOPTION2="value1;value2" #...other options
```

CMake Supplied Options
----------------------

CMake by default already includes a whole host of options for configuring common
build settings.  We have striven to honor these variables throughout the build
infrastructure.  A list of the more commonly used CMake variables is included in
the following table along with brief descriptions.  Obviously not every variable
is used by every project using NWChemExBase (if you set a variable and it is 
not used by that project CMake will warn you in the configuration logs). Note
 `Lang` is a placeholder for language, valid choices are `CXX` (for C++),
`C`, or  `Fortran`.

--------------------------------------------------------------------------------
| CMake Variable | Description                                                 |
| :------------:      | :----------------------------------------------------- |
| CMAKE_Lang_COMPILER | The `Lang` compiler that will be used                  |
| CMAKE_Lang_FLAGS    | Flags that will be passed to `Lang` compiler           |
| MPI_Lang_COMPILER   | MPI wrapper compiler (should wrap CMAKE_Lang_COMPILER) |
| CMAKE_BUILD_TYPE    | Debug, Release, or RelWithDebInfo                      |
| CMAKE_PREFIX_PATH   | A list of places CMake will look for dependencies      |
| CMAKE_INSTALL_PREFIX | The install directory                                 |
| BUILD_SHARED_LIBRARIES | If false static libraries will be built             |
--------------------------------------------------------------------------------

:memo: It will greatly behoove you to always pass full paths.

NWChemExBase Specific Options
-----------------------------

In addition to the above standard CMake options NWChemExBase's CMake 
infrastructure also recognizes the following options (Note `XXX` in the 
following table can be any dependency in [this list](SupportedDependencies.md)):

--------------------------------------------------------------------------------
| Variable Name  | Description                                                 |
| :------------: | :---------------------------------------------------------- |
| BUILD_TESTS    | If true tests for the project will be built (default: true) |
| BUILD_XXX      | TRUE forces building XXX, FALSE errs if XXX not found       |
| NWX_CMAKE_DEBUG | TRUE turns on (lots) of extra CMake printing               | 
--------------------------------------------------------------------------------

Troubleshooting
---------------

Nothing will ever go wrong so there's no need to write this section (right?).
Okay...call this section a TODO.
