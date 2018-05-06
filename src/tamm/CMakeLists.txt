cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(tamm VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) #For nwchemex_add_library

#Add the current directory's sources to the list
set(tamm_SRCS tamm.cpp index_space.cpp)

#Add the current directory's header files to the list
set(tamm_INCLUDES  
    tamm.hpp
    types.hpp
    errors.hpp 
    strong_num.hpp
    boundvec.hpp
    index_space.hpp
    index_space_interface.hpp
    attribute.hpp
    range.hpp
   )

set(tamm_DEFINITIONS )
set(tamm_LFLAGS )

nwchemex_add_library(tamm tamm_SRCS tamm_INCLUDES tamm_DEFINTIONS tamm_LFLAGS)


