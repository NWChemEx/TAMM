
include(TargetMacros)

add_mpi_unit_test(Test_GACreate 2 "10")
add_mpi_unit_test(Test_GA_Alloc 2 "")
add_mpi_unit_test(Test_MPI_Barrier 2 "")
add_mpi_unit_test(Test_MPI_allgather 2 "")
add_mpi_unit_test(Test_MPI_allgather2 2 "")
# add_mpi_unit_test(Test_GACreate_irreg 5 "")

if(NOT "${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Darwin")
  add_mpi_unit_test(Test_Layout 2 "")
endif()
