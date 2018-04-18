cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(tamm VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) #For nwchemex_add_library

#Add the current directory's sources to the list
set(tamm_SRCS tamm.cc)

#Add the current directory's header files to the list
set(tamm_INCLUDES  
    tamm.h
    types.h
    errors.h 
    strong_num.h
    boundvec.h
    index_space.h
   )

set(tamm_DEFINITIONS )
set(tamm_LFLAGS )

nwchemex_add_library(tamm tamm_SRCS tamm_INCLUDES tamm_DEFINTIONS tamm_LFLAGS)


