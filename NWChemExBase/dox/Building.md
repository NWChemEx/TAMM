Configuring and Building the Project
====================================

The current project uses
:link:[NWChemExBase](https://github.com/NWChemEx-Project/NWChemExBase) for
its build infrastructure.  This means in the ideal case it can be built using
the following three commands:

```bash
cmake -H. -B<name of build directory>
cd <name of build directory> && make
make install
```

If that doesn't work, or you simply want more control over the build, read on.

CMake Basics
------------

Before discussing the more specific configure/build options/problems it makes
sense to list some basics of CMake.  First, options are passed to
CMake using the syntax `-D<OPTION_NAME>=<VALUE>`.  If you need to set an option
so that it stores a list of values (*e.g.* including multiple search paths) you
will need to put the list in double quotes and seperate the values with
semicolons, *i.e.* `-D<OPTION_WITH_TWO_VALUES>="<VALUE1>;<VALUE2>"`.

CMake by default already includes a whole host of options for configuring common
build settings.  We have strived to honor these variables throughout the build
infrastructure.  A list of the more commonly used CMake variables is included in
the following table along with brief descriptions.  Obviously not every variable
is used by every project (if you set a variable and it is not used by that
project CMake will warn you).

--------------------------------------------------------------------------------
| CMake Variable | Description                                                 |
| :------------: | :-----------------------------------------------------------|
| CMAKE_CXX_COMPILER | The C++ compiler that will be used                      |
| CMAKE_CXX_FLAGS | Flags that will be passed to C++ compiler                  |
| MPI_CXX_COMPILER | MPI C++ wrapper compiler (should wrap CMAKE_CXX_COMPILER) |
| CMAKE_C_COMPILER | The C compiler that will be used                          |
| CMAKE_C_FLAGS | Flags that will be passed to C compiler                      |
| MPI_C_COMPILER | MPI C wrapper compiler (should wrap CMAKE_C_COMPILER)       |
| CMAKE_Fortran_COMPILER | The Fortran compiler that will be used              |
| CMAKE_Fortran_FLAGS | Flags that will be passed to the Fortran compiler      |
| MPI_Fortran_COMPILER | MPI Fortran wrapper compiler (should wrap CMAKE_Fortran_COMPILER) |
| CMAKE_BUILD_TYPE | Debug, Release, or RelWithDebInfo                         |
| CMAKE_PREFIX_PATH | A list of places CMake will look for dependencies        |
| CMAKE_INSTALL_PREFIX | The install directory                                 |
--------------------------------------------------------------------------------

:memo: It will greatly behoove you to always pass full paths.
