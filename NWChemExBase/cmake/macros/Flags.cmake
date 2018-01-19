################################################################################
#                                                                              #
# This file contains the machinery for setting the overall compile-time flags  #
# of the project.  Including things like C++11, etc.                           #
#                                                                              #
################################################################################

# CMake doesn't support Intel CXX standard until cmake 3.6
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
  if("${CMAKE_VERSION}" VERSION_LESS "3.6")
      list(APPEND CMAKE_CXX_FLAGS "-std=c++${CMAKE_CXX_STANDARD}")
  endif()
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
