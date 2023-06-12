cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TAMM VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) #For cmsb_add_library

if(USE_CUDA)
    set(tamm_gpu_INCS gpu_streams.hpp gpu_memory_pool.hpp)
elseif(USE_HIP)
    set(tamm_gpu_INCS gpu_streams.hpp gpu_memory_pool.hpp)
elseif(USE_DPCPP)
    set(tamm_gpu_INCS sycl_device.hpp gpu_streams.hpp gpu_memory_pool.hpp)
endif()

#Add the current directory's sources to the list
set(TAMM_SRCS tamm.cpp index_space.cpp execution_context.cpp tensor_base.cpp)

#Add the current directory's header files to the list
set(TAMM_INCLUDES
    atomic_counter.hpp
    work.hpp
    tamm.hpp
    types.hpp
    errors.hpp 
    strong_num.hpp
    proc_grid.hpp
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
    ga_over_upcxx.hpp
    eigen_utils.hpp
    eigen_includes.hpp
    runtime_engine.hpp
    block_buffer.hpp
    lru_cache.hpp
    kernels/assign.hpp
    kernels/multiply.hpp
    op_dag.hpp
    interfaces.hpp
    op_profiler.hpp
    mem_profiler.hpp
    op_attributes.hpp
    op_visitors.hpp
    scalar.hpp
    block_plan.hpp
    block_set_plan.hpp
    block_lambda_plan.hpp
    block_assign_plan.hpp
    block_mult_plan.hpp
    block_span.hpp
    blockops_cpu.hpp
    perm.hpp
    iteration.hpp
    ip_hptt.hpp
    blockops_blas.hpp
    tensor_variant.hpp
    op_executor.hpp
    symbol.hpp
    op_cost.hpp
    block_operations.hpp
    setop.hpp
    scanop.hpp
    mapop.hpp
    allocop.hpp
    deallocop.hpp
    addop.hpp
    multop.hpp
    op_base.hpp
    label_translator.hpp
    opmin.hpp
    "${tamm_gpu_INCS}"
    # spin_tensor.hpp
   )

set(TAMM_CFLAGS )
set(TAMM_LFLAGS )

cmsb_add_library(tamm TAMM_SRCS TAMM_INCLUDES TAMM_CFLAGS TAMM_LFLAGS)

