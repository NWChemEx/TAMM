CMake Superbuilds
-----------------

This page contains details regarding the CMake pattern known as a "superbuild".

Contents
--------

1. [Why Do We Need a Superbuild?](#why-do-we-need-a-superbuild?)  
2. [Anatomy of a Superbuild](#anatomy-of-a-superbuild)  
3. [Superbuild Technical Details](#superbuild-technical-details)
   a. [Staging the Build](#staging-the-build)
   b. [RPATHs](#rpaths)
   c. [Target Names](#target-names)

Why Do We Need a Superbuild?
----------------------------

As detailed in [CMake Build Basics](BuildBasics.md) CMake assumes a very 
specific workflow.  Unfortunately for most modern C++ projects that workflow is
too simplistic, particularly when it comes to handling the dependencies.  To 
that end, let us assume that our main project has dependencies.  Furthermore,
let us also assume that we are good software scientists and do not simply 
always build said dependencies without giving the user a chance to provide 
them to us. It then follows that we need a mechanism for finding, and then 
possibly building these dependencies.

To better understand why we need a superbuild consider a simple example.  Assume
you are building library B, which depends on some external library A.  You 
start off by looking for A.  CMake provides a function `find_package` 
specifically for this reason.  Let's say `find_package` finds A. Great, you
simply compile B against the A found by CMake.  What if CMake doesn't find it? 
There's two options:
1. Crash, tell the user to build A, have user rerun CMake.
   - Basically the same as the prior scenario at this point
2. You can attempt to build A yourself.  
Wanting to be user friendly, you add an optional target that will build A if 
it's not found.  CMake provides a command `ExternalProject_Add` specifically for
this purpose.  `ExternalProject_Add` will add some steps to the build phase that
will build A.  The problem is that this means A won't be built (and therefore 
findable) until the build phase.  Thus all `find_package` calls looking for A
during the configuration phase will still fail (A hasn't been built yet).  
 
### Anatomy of a Superbuild

Ultimately a superbuild requires each target to be built via the 
`ExternalProject_Add` command.  This is because `ExternalProject_Add` 
establishes a new set of "phases" per target.  Within a superbuild our 
overall process looks like:

1. The "configure" phase so far as CMake knows  
   a. Establish dependencies among projects  
   b. Determine dependencies that need built.  
2. The "build" phase according to CMake  
   1.  Build dependencies
       1. Configure phase for dependency
       2. Build phase for dependency
       3. Install dependency to staging area
   2.  Build main project
       1. Configure phase for main project
       2. Build phase for main project
       3. Install main project to staging area
   3.  Build tests for main project
       1. Configure phase for tests  
       2. Build phase for tests
       3. Install tests to test staging area
3. Test the main project
4. Install the main project to real install location

Looking only at the outermost bullets you can see to the outside world the
build looks normal.  This is important as it allows CMake projects relying on a
superbuild to be subprojects of other CMake projects (which themselves may
possibly be superbuilds).

Superbuild Technical Details
----------------------------

### Staging the Build

During the build phase each dependency has the whole gamut of phases run on it
including the install phase.  When invoking CMake on the entire project the 
user set the variable `CMAKE_INSTALL_PREFIX` (or if they didn't it defaulted 
to something like `/usr/local`).  This is the root of the path into 
which all products are to be installed (*e.g.* libraries would be installed 
to `${CMAKE_INSTALL_PREFIX}/lib` and headers to 
`${CMAKE_INSTALL_PREFIX}/include`).  If we simply let each dependency install
 itself there's the possibility (particularly if the user didn't specify 
 `CMAKE_INSTALL_PREFIX`) that the build phase would then try to install into 
 a place like `/usr/local`, which requires administrator privileges.  It is 
 generally considered a bad idea to run any part of the build process, with the
 exception of the install phase, with elevated privileges (poorly written or 
 maliciously written builds would then have the ability to wreck hell on your
 system; install just copies files so if they built ok it's more likely that 
 they won't screw up your system).  The solution is simple, during the build
 phase we create a directory `STAGE_DIR` that will focus as the effective root
 of the file system for the duration of the build and testing phases.  All 
 dependencies and projects are then installed to paths like 
 `${STAGE_DIR}${CMAKE_INSTALL_PREFIX}/lib`.  Tests are then run on the staged
 version of the project.  Finally, during the install phase we just copy the
 contents of `${STAGE_DIR}` to `${CMAKE_INSTALL_PREFIX}`.
 
 ### RPATHs
 
 When using a superbuild RPATHs become significantly more complicated.  This is
 because of the staging step.  Basically we need two RPATHs: one for the 
 staging directory that is used during the testing phase, and one for the actual
 install phase.
 
 *Expand on this section when RPATHs are more stable*

### Target Names

If you look at `NWChemExBase/CMakeLists.txt` you'll notice that we put a suffix
on all target names.  CMake (and the underlying build programs) prohibit two 
targets from having the same name.  In particular it is quite natural for say
library A to name its target A.  `ExternalProject_Add` command 
introduces a new namespace (targets defined within the `ExternalProject_Add` 
are not visible to the caller of the `ExternalProject_Add` and vice versa) so if
while building library A, it uses CMake and declares a target A, it will not 
affect the superbuild.  The problem comes in when we call `find_package` (say
in preparing another dependency).  In this scenario, particularly if 
`find_package` finds the package via its `XXXConfig.cmake` file, it is likely
that another target will be produced (to aid in you in linking).  This 
additional target will not be namespace protected and may collided with our 
targets.  To avoid this we append a suffix that we expect to be unique. 
Note that our convention is to append `_External` to each target.
