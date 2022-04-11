
include(TargetMacros)
add_mpi_unit_test(CD_CCSD 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.json")
add_mpi_unit_test(CholeskyDecomp 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.json")

include(${CMAKE_CURRENT_LIST_DIR}/ccsd_t/ccsd_t.cmake)
