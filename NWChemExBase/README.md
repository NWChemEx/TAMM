NWChemEx Base Repository
==============================

Building C++ projects is a laborious process that is somewhat simplified with
CMake.  Nonetheless CMake still requires quite a bit of hand tuning and has
quite a learning curve.  NWChemExBase is designed to provide a reusable CMake 
setup for C++ projects that is easy to setup, flexible enough to handle 
the dynamic nature of most builds, and robust across multiple platforms.

Topics
------

### User-centric

The following are topics that may be of interest to users building projects 
that currently rely on NWChemExBase or developers of projects wanting to 
leverage NWChemExBase for their projects.  

- [Building a Project that Uses NWChemExBase](dox/Building.md)
- [Using NWChemExBase in Your Project](dox/QuickStart.md)
  - [Current List of Supported Dependencies](dox/SupportedDependencies.md)
  - [BLAS and LAPACK](dox/SoYouWannaFindBLAS.md)

### Developer-centric

The following topics are aimed at people wanting to extend or 
contribute to NWChemExBase:

- [Building Basic CMake Projects](dox/BuildBasics.md)
- [Understanding CMake Superbuilds](dox/CMakeSuperBuild.md)
- [Finding Dependencies](dox/FindingDependencies.md)
- [Finding BLAS/LAPACK](dox/SoYouWannaFindBLAS.md)
- [NWChemExBase Model](dox/NWChemExBaseModel.md)
- [NWChemExBase Macros and Functions](dox/MacroDocumentation.md)

Contributing
------------

Contributions are welcome and at some point I'll come up with an official 
statement regarding what that all entails...

Licensing
---------

This entire repository will be licensed at some point.

Acknowledgements
----------------

At the moment there's nothing to cite to use this repo, doubtful it'll be a 
publication ever...
