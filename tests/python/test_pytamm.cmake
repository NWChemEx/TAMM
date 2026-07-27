include(TargetMacros)

# Python tests for tamm. python equivalent of the C++ tests in tests/tamm.
add_python_unit_test(Test_IndexSpace)
add_python_unit_test(Test_IndexLoopNest)

add_mpi_python_unit_test(Test_Tensors 2)
add_mpi_python_unit_test(Test_Ops 2)
add_mpi_python_unit_test(Test_DependentSpace 2)
add_mpi_python_unit_test(Test_Utils 2)
add_mpi_python_unit_test(Test_Mult_Ops 2 "50 20")
add_mpi_python_unit_test(Test_DLPNO_Ops 2 "10 2")
add_mpi_python_unit_test(Test_Opmin 2)
add_mpi_python_unit_test(Test_IO 2 "10 10")
add_mpi_python_unit_test(Test_Unit_Tiled_View_Tensor 2)
add_mpi_python_unit_test(Test_Mem_Profiler 2)
add_mpi_python_unit_test(Test_LocalTensor 2 "50 20")
add_mpi_python_unit_test(Test_Binding_Equivalence 2)

# Skip Test_EVP

if(NOT "${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Darwin")
  add_python_unit_test(Test_LabeledTensor)
  add_mpi_python_unit_test(Test_TiledIndexSpace 2)
endif()

# Coupled-cluster Python tests.
# ccse_tensors.py is a shared helper module (imported), not a standalone test.

add_mpi_python_unit_test(coupledcluster/Test_CCSD 2 "10 40 60 40")

if(NOT USE_UPCXX)
  add_mpi_python_unit_test(coupledcluster/Test_CCSD_V4_BlockSparse 2 "10 40 60 40")
  add_mpi_python_unit_test(coupledcluster/Test_CCSD_V4 2 "10 40 60 40")
  add_mpi_python_unit_test(coupledcluster/Test_DLPNO_CC 2
    "${CMAKE_CURRENT_LIST_DIR}/../tamm/coupledcluster/inputs/dlpno_co.json")
endif()

# Python-only tests with no C++ twin in test_tamm.cmake
add_mpi_python_unit_test(Test_Numpy 2)
add_mpi_python_unit_test(Test_Print_Utils 2)
add_mpi_python_unit_test(Compare_Binding_Equivalence 2)
