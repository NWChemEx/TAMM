BuildNWChemExModule
----------------------

### build_nwchemex_module

This is the main entry point into the CMakeBuild CMake infrastructure.

#### Syntax

```cmake
build_nwchemex_module(PATH_TO_FILE)
```

Arguments:
- `PATH_TO_FILE` this is the path to the directory calling 
  `build_nwchemex_module`.  99.9% of the time it should be the value of 
  `CMAKE_CURRENT_LIST_DIR`.

CMake Environment Variables:
- `NWX_PROJECTS` a list of modules to build
  - The value of this defaults to `PROJECT_NAME` and doesn't have to be set
- For each item, `PROJECT`, in `NWX_PROJECTS` the following are used:
  - `(PROJECT)_SRC_DIR` the directory containing the main `CMakeLists.txt` 
  for `PROJECT`
    - Defaults to the value of `PROJECT`
    - Path should be relative to the file invoking this macro
  - `(PROJECT)_TEST_DIR` the directory containing tests for `PROJECT`
    - Defaults to `(PROJECT)_Test`
    - Path should be relative to the file invoking this macro  
  - `(PROJECT)_DEPENDENCIES` a list of the dependencies for `PROJECT`
    - Should be a name recognized by `find_package`

### Example

This example shows typical usage with dependencies, but otherwise default 
values:

```cmake
find_package(CMakeBuild)
set(${PROJECT_NAME}_DEPENDENCIES NWX_Catch)
build_nwchemex_module(${CMAKE_CURRENT_LIST_DIR})
```      

This example shows how to build two projects:

```cmake
find_package(CMakeBuild)
set(NWX_PROJECTS Project1 Project2)
set(Project1_DEPENDENCIES NWX_Catch)
set(Project2_DEPENDENCIES Eigen3)
build_nwchemex_module(${CMAKE_CURRENT_LIST_DIR})
```      

This example show how to build a project with a non-default named source 
directory, `unique/dir`:

```cmake
find_package(CMakeBuild)
set(${PROJECT_NAME}_DEPENDENCIES unique/dir)
build_nwchemex_module(${CMAKE_CURRENT_LIST_DIR})
```      
