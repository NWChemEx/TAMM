
include(TargetMacros)
include_directories(${CMAKE_SOURCE_DIR}/../src/tamm)
add_mpi_unit_test(HartreeFock_TAMM 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.json")

