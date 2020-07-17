
include(TargetMacros)

add_mpi_unit_test(Test_GACreate 2 "")
add_mpi_unit_test(Test_GA_Alloc 2 "")
add_mpi_unit_test(Test_MPI_Barrier 2 "")
add_mpi_unit_test(Test_MPI_allgather 2 "")
add_mpi_unit_test(Test_MPI_allgather2 2 "")

