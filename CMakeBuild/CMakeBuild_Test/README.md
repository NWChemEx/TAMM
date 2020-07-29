CMakeBuild_Test Directory
===========================

This directory contains a series of unit tests to ensure that CMakeBuild is
working correctly.  See the documentation for the various `add_X_test`s 
contained in `CMakeLists.txt` for information and syntax regarding adding new
tests.  

The basic tests are:

1. TestShareLibrary :  Ensures a pretty minimal CMakeBuild project works
2. TestDebugHonored : Ensures `CMAKE_BUILD_TYPE=Debug` propagates to project
3. TestReleaseHonored : Ensures `CMAKE_BUILD_TYPE=Release` propagates to project

The following tests ensure we can build dependencies and that the project is 
provided the proper includes/libraries for compiling.  The names of the tests
should make the feature they test self explanatory...

1. TestBuildCatch
2. TestBuildCBLAS
3. TestBuildEigen3
4. TestBuildLAPACKE
5. TestBuildLibInt

Finally the following tests strive to ensure that our CMake macros work 
correctly:

1. TestAssertMacros
2. TestUtilityMacros
