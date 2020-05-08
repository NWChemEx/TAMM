cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(tamm VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) #For nwchemex_add_library

if(USE_CUDA)
    set(tamm_cuda_SRCS cuda_memory_allocator.cu cudamemset.cu)
    set(tamm_cuda_INCS cuda_memory_allocator.hpp cudamemset.hpp talsh_tamm.hpp)
endif()

#Add the current directory's sources to the list
set(tamm_SRCS tamm.cpp index_space.cpp execution_context.cpp "${tamm_cuda_SRCS}")

#Add the current directory's header files to the list
set(tamm_INCLUDES
    atomic_counter.hpp
    work.hpp
    tamm.hpp
    types.hpp
    errors.hpp 
    strong_num.hpp
    boundvec.hpp
    index_space.hpp
    tiled_index_space.hpp
    index_space_interface.hpp
    attribute.hpp
    range.hpp
    ops.hpp
    scheduler.hpp
    tensor.hpp
    tensor_impl.hpp
    tensor_base.hpp
    distribution.hpp
    labeled_tensor.hpp
    execution_context.hpp
    dag_impl.hpp
    proc_group.hpp
    memory_manager.hpp
    memory_manager_ga.hpp
    memory_manager_local.hpp
    index_loop_nest.hpp
    utils.hpp
    tamm_utils.hpp
    eigen_utils.hpp
    runtime_engine.hpp
    block_buffer.hpp
    lru_cache.hpp
    kernels/assign.hpp
    kernels/multiply.hpp
    "${tamm_cuda_INCS}"
    # spin_tensor.hpp
   )

set(tamm_DEFINITIONS )
set(tamm_LFLAGS )

nwchemex_add_library(tamm tamm_SRCS tamm_INCLUDES tamm_DEFINTIONS tamm_LFLAGS)


