cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TAMMY VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) #For nwchemex_add_library

#Add the current directory's sources to the list
set(TAMMY_SRCS  dummy.cc)

# not working? noga_fb_main.cc tammx_tests.cc tammx_talsh.cc

#Add the current directory's header files to the list
set(TAMMY_INCLUDES  ao.h            errors.h          loops.h              memory_manager_local.h  perm_symmetry.h  strong_num.h                 tensor_base.h
boundvec.h      index_space.h     memory_manager_ga.h  mso.h                   proc_group.h     strong_num_indexed_vector.h  tensor.h
distribution.h  labeled_tensor.h  memory_manager.h     ops.h                   scheduler.h      sub_ao_mso.h                 types.h)

set(TAMMY_DEFINITIONS )
set(TAMMY_LFLAGS )
nwchemex_add_executable(samples TAMMY_SRCS TAMMY_DEFINTIONS TAMMY_LFLAGS)


