
include(TargetMacros)
add_mpi_unit_test(Test_CCSD 2 "${CMAKE_SOURCE_DIR}/../tamm_inputs/h2o.xyz")
add_mpi_unit_test(Test_CCSD_CD 2 "${CMAKE_SOURCE_DIR}/../tamm_inputs/h2o.xyz")
#add_mpi_unit_test(Test_CCSD_CD_RM 2 "${CMAKE_SOURCE_DIR}/../tamm_inputs/h2o.xyz")
#add_mpi_unit_test(Test_CCSD_Spin 2 "${CMAKE_SOURCE_DIR}/../tamm_inputs/h2o.xyz")
add_mpi_unit_test(Test_CCSD_Lambda 2 "${CMAKE_SOURCE_DIR}/../tamm_inputs/h2o.xyz")
# add_mpi_unit_test(Test_DAG_CCSD 2)
