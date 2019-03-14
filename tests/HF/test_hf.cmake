
include(TargetMacros)
include_directories(${CMAKE_SOURCE_DIR}/../src/tamm)
#add_mpi_unit_test(Test_HartreeFock_Eigen 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.nwx")
add_mpi_unit_test(Test_HartreeFock_TAMM 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.nwx")
# add_mpi_unit_test(Test_HartreeFock_Chao 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.nwx")

