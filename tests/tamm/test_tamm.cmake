
include(TargetMacros)
add_cxx_unit_test(Test_IndexSpace)
#add_cxx_unit_test(Test_IndexLoopNest)
add_mpi_unit_test(Test_Tensors 2 "")
add_mpi_unit_test(Test_Ops 2 "")
add_cxx_unit_test(Test_LabeledTensor)
add_mpi_unit_test(Test_OpsExpr 2 "")
add_cxx_unit_test(Test_TiledIndexSpace)
add_mpi_unit_test(Test_DependentSpace 2 "")
add_mpi_unit_test(Test_Eigen 2 "")

add_mpi_unit_test(Test_Mult_Ops 2 "" )