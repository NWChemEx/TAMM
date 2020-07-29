CMakeBuild Repository
======================

Building C++ projects is a laborious process that is somewhat simplified with
CMake.  Nonetheless CMake still requires quite a bit of hand tuning and has
quite a learning curve.  CMakeBuild is designed to provide a reusable CMake 
setup for C++ projects that is easy to setup, flexible enough to handle 
the dynamic nature of most builds, and robust across multiple platforms.

Topics
------

### User-centric

The following are topics that may be of interest to users building projects 
that currently rely on CMakeBuild or developers of projects wanting to 
leverage CMakeBuild for their projects.  

- [Building a Project that Uses CMakeBuild](dox/Building.md)
- [Using CMakeBuild in Your Project](dox/QuickStart.md)
  - [Current List of Supported Dependencies](dox/SupportedDependencies.md)
  - [BLAS and LAPACK](dox/SoYouWannaFindBLAS.md)

### Developer-centric

The following topics are aimed at people wanting to extend or 
contribute to CMakeBuild:

- [Building Basic CMake Projects](dox/BuildBasics.md)
- [Understanding CMake Superbuilds](dox/CMakeSuperBuild.md)
- [Finding Dependencies](dox/FindingDependencies.md)
- [Finding BLAS/LAPACK](dox/SoYouWannaFindBLAS.md)
- [CMakeBuild Model](dox/CMakeBuildModel.md)
- [CMakeBuild Macros and Functions](dox/MacroDocumentation.md)


