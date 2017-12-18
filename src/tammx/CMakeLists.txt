cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TAMM VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(UtilityMacros) #For prefix_paths
include(TargetMacros) #For nwchemex_add_library

#Get files under us (they're returned in TAMM_SRCS and TAMM_INCLUDES)
add_subdirectory(ccsd)

#Add the current directory's sources to the list
set(TAMM_SRCS ${TAMM_SRCS}
               hartree_fock.cc index_sort.cc tammx.cc
               tammx_main.cc  two_four_index.cc)
# not working? noga_fb_main.cc tammx_tests.cc tammx_talsh.cc

#Add the current directory's header files to the list
set(TAMM_INCLUDES ${TAMM_INCLUDES}
                   atomic_counter.h block.h boundvec.h combination.h diis.h
                   distribution.h errors.h execution_context.h fortran.h
                   hartree_fock.h index_sort.h labeled_tensor.h memory_manager.h
                   memory_manager_ga.h memory_manager_local.h mpi_checks.h ops.h
                   perm.h proc_group.h product_iterator.h scheduler.h
                   strong_num.h tammx.h tce.h tensor.h tensor_base.h
                   tensor_dims.h tensor_labels.h triangle_loop.h types.h util.h
                   work.h)

set(TAMM_DEFINITIONS )
set(TAMM_LFLAGS )
nwchemex_add_library(TAMM TAMM_SRCS TAMM_INCLUDES TAMM_DEFINTIONS TAMM_LFLAGS)
