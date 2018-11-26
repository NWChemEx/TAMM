#set(chao_SRCS  #Chao Eigensolver
#    eigensolver/src/cholQR.cpp
#    eigensolver/src/sisubit.cpp
#    eigensolver/src/RayleighRitz.cpp
#    eigensolver/src/GetEigvals.cpp
#    eigensolver/src/GetResiduals.cpp
#    eigensolver/src/itereig.cpp
#    eigensolver/src/evalidate.cpp
#    eigensolver/src/HSLanczos.cpp
#    eigensolver/src/getevbnd.cpp
#    eigensolver/src/parseinput.cpp
#    eigensolver/src/utilities.cpp
#    eigensolver/src/getshifts.cpp
#    eigensolver/src/slicing.cpp
#    eigensolver/src/collectevs.cpp
#    eigensolver/src/sync_slices.cpp
#    eigensolver/src/hsdiag.cpp
#    eigensolver/src/kmeans.cpp
#)
#
#set(chao_INCS
#    #Chao Eigensolver
#    eigensolver/include/collectevs.hpp
#    eigensolver/include/enforcecounts.hpp
#    eigensolver/include/evalidate.hpp
#    eigensolver/include/getshifts.hpp
#    eigensolver/include/itereig.hpp
#    eigensolver/include/parseinput.hpp
#    eigensolver/include/partev.hpp
#    eigensolver/include/sisubit.hpp
#    eigensolver/include/slicing.hpp
#    eigensolver/include/sync_slices.hpp
#    eigensolver/include/utilities.hpp
#    )
#cpp_add_library(
#    eigensolver
#    SOURCES ${chao_SRCS}
#    INCLUDES ${chao_INCS}
#    DEPENDS LAPACKE
#)
#target_include_directories(
#    eigensolver PUBLIC ${PROJECT_SOURCE_DIR}/src/tamm/eigensolver/include
#)

#Add the current directory's sources to the list
set(tamm_SRCS tamm.cpp index_space.cpp)

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
    tamm_utils.hpp
    eigen_utils.hpp
    runtime_engine.hpp
    task_engine.hpp
    kernels/assign.hpp
    kernels/multiply.hpp
    # spin_tensor.hpp
   )

cpp_add_library(
    tamm
    SOURCES ${tamm_SRCS}
    INCLUDES ${tamm_INCLUDES}
    DEPENDS HPTT GlobalArrays MSGSL LAPACKE Eigen3
)
target_include_directories(tamm PRIVATE ${PROJECT_SOURCE_DIR}/src)
cpp_install(TARGETS tamm PREFIX_TO_STRIP ${PROJECT_SOURCE_DIR}/src)

