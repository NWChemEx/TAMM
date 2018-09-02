
include(TargetMacros)
add_mpi_unit_test_wargs(Test_CCSD 2 "../tamm_inputs/h2o.xyz")
add_mpi_unit_test_wargs(Test_CCSD_CD 2 "../tamm_inputs/h2o.xyz")
add_mpi_unit_test_wargs(Test_CCSD_Lambda 2 "../tamm_inputs/h2o.xyz")
# add_mpi_unit_test(Test_DAG_CCSD 2)
