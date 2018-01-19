So You Wanna Find BLAS/LAPACK
=============================

BLAS/LAPACK are hallmarks of high-performance computing.  At first glance CMake
appears to make it easy to find BLAS/LAPACK, just call `find_package(BLAS)` or
`find_package(LAPACK)` and the `FindBLAS.cmake` and `FindLAPACK.cmake` module
files supplied with CMake will give you BLAS/LAPACK.  If was this easy we
wouldn't need an entire page dedicated to the complexities of finding 
BLAS/LAPACK...

Contents
--------

1. [TL;DR User Version](#tl;dr-user-version)
2. [Understanding the Problem](#understanding-the-problem)
3. [What the BLAS/LAPACK Detection Should Do](#what-the-findblas/findlapack-and-findcblas/findlapacke-modules-should-do)
4. [What the BLAS/LAPACK Detection Actually Does](#current-status-of-findblas/findlapack-and-findcblas/findlapacke)

TL;DR User Version
------------------

If you care what version of BLAS/LAPACK and CBLAS/LAPACKE are utilized set the
following variables:

| Variable             | Value                                              |
| :------------------: | :--------------------------------------------------|
| BLAS_LIBRARIES       | path/to/blas/library.a  (path includes library)    |
| CBLAS_LIBRARIES      | path/to/cblas/lbirary.a (path includes library)    |
| CBLAS_INCLUDE_DIRS   | path/to/cblas/include/dir (path excludes header)   |
| LAPACK_LIBRARIES     | path/to/lapack/library.a (path includes library)   |
| LAPACKE_LIBRARIES    | path/to/lapacke/library.a (path includes library)  |
| LAPACKE_INCLUDE_DIRS | path/to/lapacke/include/dir (path excludes header) |

If you only set BLAS/LAPACK libraries we will build the wrapper CBLAS/LAPACKE
libraries around your specified BLAS/LAPACK library.  If you are using MKL it 
suffices to set `MKL_LIBRARIES` and `MKL_INCLUDE_DIRS` and we will set all of 
the above for you.  If you set no variables, we'll build them all for you.

Understanding the Problem
-------------------------

First things first, BLAS/LAPACK are standard APIs, not libraries.  Various 
research groups/vendors have implemented these standards into libraries 
(which they often simply call BLAS/LAPACK adding to the confusion).  
Unfortunately, either there is no standard file structure to a BLAS/LAPACK 
installation or the various groups/vendors all decided to ignore it.  What this
means is locating a given group's/vendor's implementation tends to be a 
procedure that is specific to one particular implementation.  Adding to the fun, 
the standards were designed in an era where people liked Fortran and the ABIs 
all have Fortran linkage.  What this means is once you've figured out what 
libraries you need to link against, it's easy to call any BLAS/LAPACK 
implementation from Fortran and a royal pain to call it from any other language.

Now the next fun part.  Just because the libraries have Fortran linkage doesn't
mean they're written in Fortran.  Particularly if we're building them 
BLAS/LAPACK will be written in Fortran.  The result is we also need to 
link against the standard Fortran libraries (`libgfortran.a`, `libm.a`,...).  
Normally these libraries are taken care of for us by the compiler, but 
because we're now mixing languages they're not.  CMake will take care of BLAS
for us, but LAPACK's usage of CMake (and arguably a poor use) means that 
LAPACK will have a dependency on the standard Fortran libraries that isn't taken
care of for us.

Calling BLAS/LAPACK from languages derived from C is typically facilitated by
going through CBLAS/LAPACKE.  CBLAS/LAPACKE are thin wrapper libraries over a 
standard BLAS/LAPACK implementation designed to give it C-like linkage.  
Unfortunately, their adoption is not as widespread as one would like with most
codes still relying on BLAS/LAPACK directly.  Given that NWChemExBase's target
software stack is written primarily in C/C++ we will insist on using 
CBLAS/LAPACKE.  Finally, note that for most CBLAS/LAPACKE implementations one 
still needs to link against the underlying BLAS/LAPACK implementation so we 
don't avoid the problem of finding them.

What the FindBLAS/FindLAPACK and FindCBLAS/FindLAPACKE Modules Should Do
------------------------------------------------------------------------

At the heart of this problem we need to find two sets of headers (one for 
CBLAS and one for LAPACKE) and four libraries.  Additionally, the BLAS library
must be compatible (same ABI) with the CBLAS library and the LAPACK library. The
LAPACK library must be compatible with the LAPACKE library.  That's for 
"standard" distributions.  Then there's non-standard distributions.  For MKL, 
it's a lot more complicated with the actual set of libraries being given by 
Intel's link line advisor 
([link](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor)).
There's also the Accelerate framework on Macs that needs to be handled and 
there's (probably) other vendor specific implementations to worry about (Cray?).

The above assumes we have to find all the necessary libraries/includes.  In 
truth, there are 16 possibilities for user inputs (assuming if a user provides 
us CBLAS/LAPACKE we get the library and the header file).  These inputs range
from we're given nothing, to we're given a single library, to we're given 
everything.  Doing this in a robust manner is difficult.

Making matters more fun, just because we found the libraries/headers doesn't
mean our dependencies can.  Hence it will fall upon NWChemExBase to manually set
the paths for all dependencies wanting BLAS/LAPACK support.  This becomes 
increasingly difficult for the non-standard distributions like MKL as many 
dependencies may not recognize that it is actually BLAS/LAPACK and 
CBLAS/LAPACKE.

Current Status of FindBLAS/FindLAPACK and FindCBLAS/FindLAPACKE
---------------------------------------------------------------

Given the statement of what the various find modules should do, let's discuss
what the ones included in NWChemExBase actually do.  

### FindBLAS/FindLAPACK

The first decision is whether we want to brand these modules with the `NWX_` 
prefix.  One argument for this is that CMake includes default versions of these
modules; however, the version we are writing will find a traditionally installed
version and all codes using NWChemExBase will be capable of using that version 
without modification.  The last point suggests that we're just writing a 
"better" version of the modules included with CMake and underlies our decision 
to forgo the `NWX_` prefix.  This is facilitated by the fact that (within 
NWChemExBase) CMake will look for `FindBLAS.cmake`/`FindLAPACK.cmake` in our 
`cmake/find_external` directory (and find it) before looking for the one 
included with CMake itself. Hence CMake guarantees we are able to override the 
system version (again only within NWChemExBase, it is possible for dependencies 
to clear `CMAKE_MODULE_PATH` and go back to finding the system version).  

Our version of the module is very simple at the moment.  We'll look for a 
library called `libblas.a`/`liblapack.a` if we find it great, otherwise 
BLAS/LAPACK is set to not found.  By looking for BLAS/LAPACK via the standard 
`find_library` mechanism we guarantee the user is able to provide us a different
version of BLAS/LAPACK via the variable `BLAS_LIBRARIES`/`LAPACK_LIBRARIES` and 
that version will be used.


### FindCBLAS/FindLAPACKE

Again we consider should our find modules be prefaced with `NWX_`?  Again, we
have decided they should not as the current find modules ought to be able to
find any "standard" CBLAS/LAPACKE installation, not just ones we've built.

Again we are at the moment keeping it simple.  What this means is we will look
for a library called `libcblas.a` via the standard `find_library` mechanism. 
And again the user is free to override our choice by defining 
`CBLAS_LIBRARIES` and setting it appropriately.  Unlike the BLAS/LAPACK 
scenario we also need to find the header file include path.  Making matters 
worse, the name of this header file differs depending on the distribution 
(`mkl.h` for MKL or `cblas.h` for basically everyone else).  Our current 
implementation will look for `cblas.h` using `find_path` which means setting 
`CBLAS_INCLUDE_DIRS` to your CBLAS implementation's preferred include directory
will result in overriding our efforts to find a header file (if the path you 
provide is bad then you'll likely get a compile error when building the main 
project).  

In an effort to aid project developers `FindCBLAS`/`FindLAPACKE` will
define a variable `CBLAS_HEADER`/`LAPACKE_HEADER` which is the literal name of
the header to include (angle brackets already included).  This in turn means we
need to determine the name of the header file.  Presently we assume it is 
`cblas.h`/`lapacke.h` unless the BLAS library is MKL.  To determine if the BLAS
library is MKL we assume that the user has not changed the name of the 
libraries Intel installs and thus the string `mkl` appears in their name.  Not a
great solution, but it ought to work 99% of the time.

We assume that whatever BLAS/LAPACK we find is the one paired to our 
CBLAS/LAPACKE implementation. 
