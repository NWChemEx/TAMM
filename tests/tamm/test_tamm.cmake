
include(TargetMacros)
add_cxx_unit_test(Test_IndexSpace)
add_cxx_unit_test(Test_IndexLoopNest)
# add_mpi_unit_test(Test_Tensors 2 "")
add_mpi_unit_test(Test_Ops 2 "")
# add_mpi_unit_test(Test_OpsExpr 2 "")
add_mpi_unit_test(Test_DependentSpace 2 "")
# add_mpi_unit_test(Test_Eigen 2 "")
# add_mpi_unit_test(Test_PG 2 "")
add_mpi_unit_test(Test_Utils 2 "")
add_mpi_unit_test(Test_Mult_Ops 2 "50 20" )
add_mpi_unit_test(Test_DLPNO_Ops 2 "10 2" )
# add_mpi_unit_test(Test_OpDAG 2 "")
add_mpi_unit_test(Test_Opmin 2 "")
add_mpi_unit_test(Test_IO 2 "10 10" )
add_mpi_unit_test(Test_EVP 2 "10 10" )
add_mpi_unit_test(Test_Unit_Tiled_View_Tensor 2 "")
add_mpi_unit_test(Test_Mem_Profiler 2 "")
add_mpi_unit_test(Test_CCSD 2 "10 40 60 40")
add_mpi_unit_test(Test_CCSD_info 2 "10 40 60 40")
add_mpi_unit_test(Test_LocalTensor 2 "50 20" )

# add_mpi_unit_test(Test_ViewTensor 2 "")
# add_mpi_unit_test(Test_QR 2 "")

if(NOT USE_UPCXX)
  add_mpi_unit_test(Test_DLPNO_CC 2 "${CMAKE_CURRENT_LIST_DIR}/inputs/dlpno_co.json")
endif()

if(NOT "${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Darwin")
  add_cxx_unit_test(Test_LabeledTensor)
  add_cxx_unit_test(Test_TiledIndexSpace)
endif()
