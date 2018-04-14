cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TAMM VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) #For nwchemex_add_library

#Add the current directory's sources to the list
set(TAMM_SRCS tamm.cc)

#Add the current directory's header files to the list
set(TAMM_INCLUDES  
    tamm.h
    ops.h
    types.h
    scheduler.h
    errors.h 
    loops.h
    strong_num.h
    boundvec.h
    index_space.h
    tensor_impl.h
    labeled_tensor.h)

set(TAMM_DEFINITIONS )
set(TAMM_LFLAGS )

nwchemex_add_library(TAMM TAMM_SRCS TAMM_INCLUDES TAMM_DEFINTIONS TAMM_LFLAGS)

