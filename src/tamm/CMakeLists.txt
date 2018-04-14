cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TAMM VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) #For nwchemex_add_library

#Add the current directory's sources to the list
#set(TAMM_SRCS  dummy.cc)

#Add the current directory's header files to the list
set(TAMM_INCLUDES  ao.h            errors.h          loops.h           strong_num.h      boundvec.h
      index_space.h       tensor_impl.h   labeled_tensor.h      ops.h                   scheduler.h       types.h)

set(TAMM_DEFINITIONS )
set(TAMM_LFLAGS )
#nwchemex_add_executable(samples TAMM_SRCS TAMM_DEFINTIONS TAMM_LFLAGS)


