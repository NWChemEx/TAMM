Dependency Macros
=================

### package_dependency

This macro will take a dependency and zip-up the include and library paths 
into the appropriately named variables.  Those variables will then further be
zipped-up so that they can be passed to an external project via the CMake cache.

#### Syntax

```cmake
package_dependency(name output)
```

Arguments:

- `name`: The name of the dependency to package (without "_External" suffix)
- `output`: The variable to assign the packaged target to.

### are_we_building

This macro will examine a target and determine if we are building it or not.  We
assume we are building a target if that target has no includes, libraries, or
flags set.

#### Syntax

```cmake
are_we_building(depend value)
```

Arguments:

- `depend`: The name of the dependency we are interested in (without 
"_External" suffix)
- `value`: Will be set to true if we are building the dependency and false 
otherwise.

### print_dependency

This macro will print out a dependency in a pretty syntax.

#### Syntax

```cmake
print_dependency(name)
```

Arguments:

- `name`: The name of the dependency to print (without the "_External" suffix)

### find_dependency

This macro attempts to find a dependency with a given name.  If it finds that
dependency it creates a new target `<name>_External` where `<name>` is the 
value passed into the function.  The target will be set-up in a manner suitable
for linking to via the usual CMake mechanisms. 

#### Syntax

```cmake
find_dependency(depend_name)
```
Arguments:

- `depend_name` : The name of the dependency as CMake's `find_package` would
  need it *e.g.* MPI not mpi, Eigen3 not eigen3 or EIGEN3.
- `found` : This variable will be set to true if we found the dependency and 
  false otherwise.         

#### Example

```cmake
find_dependency(Eigen3 Eigen3_FOUND)
message(STATUS "Eigen3 found: ${Eigen3_FOUND}")           
```

Output (assuming it was found):

```cmake
-- Eigen3 found: True
```

### find_or_build_dependency

This macro will attempt to find a dependency, if it can not, it will instead
look for a script in `build_external` with the same name as the dependency and
run it.  The script should make an external project that builds the dependency.
Regardless of whether or not this macro finds the dependency it will create a
dummy target with the same name as the dependency and the suffix `_External`.
All targets that depend on the dependency should use the resulting target 
to link to that dependency.

#### Syntax

```cmake
find_or_build_dependency(name found)
```

Arguments:
- `name` : The name of the dependency to find.  Must be the name `find_package`
           expects.
- `found` : A variable that will be set to true if the dependency was found.   
        
#### Example

```cmake
find_or_build_dependency(Eigen3 Eigen3_FOUND)
if(Eigen3_FOUND)
    if(TARGET Eigen3_External)
        message(STATUS "Found Eigen3!!!")
    endif()    
endif()
```

Output (depends on system):
```
-- "Found Eigen3!!!"
```

### makify_dependency

This macro will take a dependency and strip the properties off of the target in
a manner such that the result can be passed to a makefile project.  That is 
to say includes will be save to a string like:
 `-I/path/to/include1 -I/path/to/include2` and libraries to a string like:
 `-L/path/to/library/folder -lname_without_lib_prefix_or_suffix` 
 
#### Syntax

```cmake
makeify_dependency(name includes libs)
```

Arguments:

- `name`: The name of the target to make-ify (without "_External" suffix)
- `includes`: The variable to save the include string to
- `libs`: The variable to save the library string to
