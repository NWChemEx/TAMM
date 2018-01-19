QuickStart
==========

The purpose of this page is to get you up and using NWChemExBase as quickly 
as possible.  For the purposes of this tutorial we assume you haven't started a
repo yet. If you have, you'll want to follow these steps until step 3, at which
point you'll migrate your files over.

Step 0 : Preliminaries
---------------------------

We'll need a directory for the project.  We'll call this directory `workspace`.
Although not strictly necessary, you may want to look into git subrepo and use
that instead of normal git in the commands below.

Step 1 : Obtain NWChemExBase
----------------------------

The most up to date version of NWChemExBase is always given by the `master` 
branch of the GitHub repository.  To obtain it, run the following command 
inside `workspace`:

```git
git clone https://github.com/NWChemEx-Project/NWChemExBase.git
```

Step 2 : Set-up Directory Structure
-----------------------------------

NWChemExBase is easiest to use if you organize your source the way we do.  
You can get a skeleton set-up by running (still inside `workspace`):

```bash
NWChemExBase/BasicSetup.sh NameOfProject
```

where `NameOfProject` should be replaced by a single word, descriptive name for 
your project; this tutorial will continue to use `NameOfProject`.

Step 3 : Add Sources and Headers
--------------------------------

The directory `workspace/NameOfProject` is your source directory.  
Let's say your project involves one source file, `SourceFile.cpp` and one header
file `HeaderFile.hpp`.  Their paths would be 
`workspace/NameOfProject/SourceFile.cpp` and 
`workspace/NameOfProject/HeaderFile.hpp` respectively.  You now need to register
them with the build, this is done inside the file 
`workspace/NameOfProject/CMakeLists.txt`.  Specifically add them to the lists
entitled `NameOfProject_SRCS` and `NameOfProject_INCLUDES` by writing:

```cmake
set(NameOfProject_SRCS SourceFile.cpp)
set(NameOfProject_INCLUDES HeaderFile.hpp) 
```

Step 4 : Add Tests
------------------

Your main testing directory is `workspace/NameOfProject_Tests`.  Let's say you
want to make a test called `Test1` and the source for this test lives in 
`workspace/NameOfProject_Tests/Test1.cpp` then all you have to do is add the
following line to `workspace/NameOfProject_Tests`:

```cmake
add_cxx_unit_test(Test1 NameOfProject)
```

this line tells us that you want to add a test called `Test1` (NWChemExBase 
assumes the test lives in `Test1.cpp`) and that the test depends on 
`NameOfProject`.

Step 5 : Add Dependencies
-------------------------

The last major part of a C++ project is handling the dependencies (*i.e.* the
other libraries your project depends on).  As long as your project uses 
dependencies on [this list](SupportedDependencies.md), then adding 
dependencies is as simple as modifying the following line in 
`workspace/CMakeLists.txt`:

```cmake
set(ProjectName_DEPENDENCIES BLAS)
``` 

would for example make your project depend on BLAS.

Further Reading
---------------

The above 5 steps are all it takes to get a repo up and running with 
NWChemExBase.  NWChemExBase actually allows you to do quite a bit of 
customization on top of what's outlined here, to learn more check out 
[this link](AdvancedUsage.md).

