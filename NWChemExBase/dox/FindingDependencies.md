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
automation.  Currently our solution is to write `FindNWX_<Name>.cmake` files
for packages not adhering to them and to prefer our projects us the *NWX* 
versions instead of the normal ones.

Enabling Additional Dependencies
--------------------------------

It is likely inevitable that additional dependencies will occur.  When this
happens the primary responsibility of maintainers is to ensure that a dependency
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
`aBcConfig.cmake` or `abc-config.cmake`; however, we're assuming the config 
files do not exist.  Anyways, after calling `FindaBc.cmake`, `find_package` will
determine if `aBC` was found by considering the results of the variable 
`aBC_FOUND` (note the case always matches the case given to `find_package`). 
Lastly, making matters worse, it is convention to always return variables 
(aside from the `aBc_FOUND` variable) in all uppercase letters (it's a good idea
to return `aBc_FOUND` both in the native case and in all caps).

Case caveats aside, let's say we want to do this in a textbook manner, then the 
resulting `FindaBc.cmake` file should look something like:

~~~cmake
#File FindaBc.cmake
#
# By convention this file will set the following:
#
# aBc_FOUND        to true if all parts of the aBc package are found
# ABC_INCLUDE_DIR  to the path for includes part of aBc's public API
# ABC_LIBRARY      to the libraries included with aBc
# ABC_INCLUDE_DIRS will be the includes of aBc as well as any dependency it
#                  needs
# ABC_LIBRARIES    will be the libraries of aBc and all its dependencies
# ABC_DEFINITIONS  will be any compile-time flags to use in your project
                   
  
#Call find_package for each dependency
find_package(Depend1)
  
#Try to piggy-back of package-config 
find_package(PkgConfig)
pkg_check_modules(PC_ABC <libname_without_suffix>)
  
#For each header file in the public API try to find it
find_path(ABC_INCLUDE_DIR <path/you/put/in/cxx/source/file>
          HINTS ${PC_ABC_INCLUDEDIR} ${PC_ABC_INCLUDE_DIRS}
)
  
#For each library try to find it
find_path(ABC_LIBRARY <library/name/including/the/lib/and/the/extension>)
          HINTS ${PC_ABC_LIBDIR} ${PC_ABC_LIBRARY_DIRS}
)
  
#Let CMake see see if the found values are sufficient
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(aBc DEFAULT_MSG ABC_INCLUDE_DIR ABC_LIBRARY)
  
#In examples you'll see a marked_as_advanced line here a lot, but it's pretty
#useless as barely anyone runs the cmake GUI...
  
#Add dependencies and aBc's includes to ABC_INCLUDE_DIRS
set(ABC_INCLUDE_DIRS ${ABC_INCLUDE_DIR} ...)
  
#Same for libraries to link to
set(ABC_LIBRARIES ${ABC_LIBRARY} ...)
  
#Set the flags needed to compile against aBc
set(ABC_DEFINITIONS ...)
~~~
Once written your file goes in `NWChemBase/cmake/external_find/FindaBc.cmake`

### Enabling NWChemExBase to Build a Dependency

In an effort to make the build process more user-friendly it is common to want
to build dependencies for the user.  That is, if we are unable to locate a
required dependency on the system, we instead build it.  CMake doesn't have a
particular convention for how this done so we have taken the liberty of defining
a process for you.  We assume the following is in 
`NWChemExBase/cmake/external_build/BuildXXX.cmake`.

~~~cmake
#include(ExternalProject) and include(DependancyMacros) are already sourced

ExternalProject_Add(XXX_External #Target name = file name plus _External suffix
  <Rest of settings> 
)

#For each dependency flag it as one
foreach(__depend <list of dependencies>)
    find_or_build_dependency(${__depend})
    #The external target was made for you by find_or_build_dependency
    set_dependencies(XXX_External ${__depend}_External)
endforeach()    
~~~


