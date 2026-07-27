
include(TargetMacros)

add_mpi_unit_test(Test_CCSD 2 "10 40 60 40")

if(NOT USE_UPCXX)
add_mpi_unit_test(Test_CCSD_V4_BlockSparse 2 "10 40 60 40")
add_mpi_unit_test(Test_CCSD_V4 2 "10 40 60 40")
add_mpi_unit_test(Test_DLPNO_CC 2 "${CMAKE_CURRENT_LIST_DIR}/inputs/dlpno_co.json")
endif()

