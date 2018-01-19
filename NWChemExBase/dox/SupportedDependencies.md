Supported Dependencies
======================

General note, dependencies prefixed with `NWX` are special to `NWChemEx` in some
manner.  The reasons why are given after all listings.  In general you should
prefer the `NWX_` variants.

NWChemExBase relies on CMake's `find_package` module for locating dependencies
and setting paths up correctly.  NWChemExBase takes it a step further by 
attempting to build a dependency for the user if it can't find that 
dependency.  Nonetheless, aside from compilers, the following is a list of 
dependencies that NWChemExBase can find, but won't build:

--------------------------------------------------------------------------------
| Name            | Brief Description                                          |  
| :-------------: | :--------------------------------------------------------- |   
| OpenMP          | Determines the flags for compiling/linking to OpenMP       |
| NWX_MPI         | Wrapper around non-conforming standard CMake FindMPI.cmake |
--------------------------------------------------------------------------------
  

NWChemExBase can find and build the following:

--------------------------------------------------------------------------------
| Name            | Brief Description                                          |  
| :-------------: | :--------------------------------------------------------- |  
| BLAS            | Basic Linear Algebra Subprograms (can build Netlib version)|
| CBLAS           | A wrapper around BLAS to make it C-friendly                |
| LAPACKE         | Linear Algebra PACKage (can build Netlib version)          |
| ScaLAPCK        | Scalable LAPACK (can build Netlib version)
| Eigen3          | The Eigen C++ matrix library                               |
| GTest           | Google's testing framework                                 |
| NWX_Catch       | Catch testing framework installed our way                  |
| LibInt          | Computes Gaussian integrals for quantum mechanics          |
| GlobalArrays    | The Global Arrays distributed matrix library               |
| AntlrCppRuntime | The ANTLR grammar parsing library                          |
--------------------------------------------------------------------------------

NWChemEx Exclusive Dependencies
-------------------------------

The following provides more details about the `NWX_` prefixed dependencies.

- NWX_MPI : CMake comes stock with a `FindMPI.cmake` module.  It's pretty good
  at finding MPI, but the problem is it won't return the variables in a manner
  that follows usual CMake conventions.  Thus we wrap it to make it adhere to
  those conventions
- NWX_Catch : The Catch testing library for C++ is a single header-only testing 
  library.  Given it's simple form, Catch doesn't provide a means of 
  installing itself.  We have thus made up an installation convention and 
  that's what this dependency will look for.  Furthermore, Catch recommends 
  compiling its main function into one file and linking against it.  NWX_Catch
  requires that a library exist that does this.
 
    
