cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(tamm VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) #For nwchemex_add_library

#Add the current directory's sources to the list
set(tamm_SRCS tamm.cpp index_space.cpp

#Chao Eigensolver
    eigensolver/src/cholQR.cpp 
    eigensolver/src/sisubit.cpp 
    eigensolver/src/RayleighRitz.cpp 
    eigensolver/src/GetEigvals.cpp 
    eigensolver/src/GetResiduals.cpp 
    eigensolver/src/itereig.cpp 
    eigensolver/src/evalidate.cpp 
    eigensolver/src/HSLanczos.cpp 
    eigensolver/src/getevbnd.cpp 
    eigensolver/src/parseinput.cpp 
    eigensolver/src/utilities.cpp 
    eigensolver/src/getshifts.cpp 
    eigensolver/src/slicing.cpp 
    eigensolver/src/collectevs.cpp 
    eigensolver/src/sync_slices.cpp 
    eigensolver/src/hsdiag.cpp 
    eigensolver/src/kmeans.cpp
)

#Add the current directory's header files to the list
set(tamm_INCLUDES
    atomic_counter.h
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
    eigen_utils.hpp
    runtime_engine.hpp
    block_buffer.hpp
    kernels/assign.hpp
    kernels/multiply.hpp
    # spin_tensor.hpp

    #Chao Eigensolver
    eigensolver/include/collectevs.hpp
    eigensolver/include/enforcecounts.hpp
    eigensolver/include/evalidate.hpp
    eigensolver/include/getshifts.hpp
    eigensolver/include/itereig.hpp
    eigensolver/include/parseinput.hpp
    eigensolver/include/partev.hpp
    eigensolver/include/sisubit.hpp
    eigensolver/include/slicing.hpp
    eigensolver/include/sync_slices.hpp
    eigensolver/include/utilities.hpp
   )

set(tamm_DEFINITIONS )
set(tamm_LFLAGS )

nwchemex_add_library(tamm tamm_SRCS tamm_INCLUDES tamm_DEFINTIONS tamm_LFLAGS)
target_include_directories(tamm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/eigensolver/include)


