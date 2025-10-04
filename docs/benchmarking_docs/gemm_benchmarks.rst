.. _test_gemm_benchmark:

=======================
GEMM Benchmark Test
=======================

Overview
========

Test_GEMM measures the performance of General Matrix Multiply (GEMM) operations using the TAMM framework. It reads test configurations from CSV files and reports timing, GFLOPS, and throughput metrics for both GPU and CPU execution.

Basic Usage
-----------

.. code-block:: bash

   # Run with default settings
   mpirun -n 2 ./Test_GEMM test_cases.csv
   
   # Specify custom iterations
   mpirun -n 2 ./Test_GEMM test_cases.csv 50

Command Line Arguments
======================

.. code-block:: bash

   mpirun -n 2 ./Test_GEMM <csv_file> [iterations]

**Required Arguments:**

* ``csv_file``: Path to CSV file containing GEMM test case definitions

**Optional Arguments:**

* ``iterations``: Number of benchmark runs per test case (default: 100)

**Examples:**

.. code-block:: bash

   # Test with default 100 iterations
   mpirun -n 2 ./Test_GEMM gemm_cases.csv
   
   # Test with 50 iterations
   mpirun -n 2 ./Test_GEMM gemm_cases.csv 50
   
   # Test with 200 iterations
   mpirun -n 2 ./Test_GEMM gemm_cases.csv 200

Input File Format
=================

The program expects a CSV file with GEMM configurations using the following 7-column format:

CSV Header
----------

.. code-block:: text

   contraction_size,output_a_size,output_b_size,total_output_size,batch_size,reduction_a_size,reduction_b_size

Column Definitions
------------------

* **contraction_size**: Size of the contraction dimension (K in GEMM)
* **output_a_size**: Output size for matrix A (M dimension)
* **output_b_size**: Output size for matrix B (N dimension)
* **total_output_size**: Total output matrix size (M×N)
* **batch_size**: Batch size for batched operations
* **reduction_a_size**: Reduction size for matrix A operations
* **reduction_b_size**: Reduction size for matrix B operations

Sample Data
-----------

.. code-block:: text

   contraction_size,output_a_size,output_b_size,total_output_size,batch_size,reduction_a_size,reduction_b_size
   1,2655,526,1396530,1,1,1
   1,2655,1000,2655000,1,1,1
   45,59,45,2655,1,1,1
   45,59,59,3481,1,1,1
   59,45,45,2025,1,1,1
   67,75,67,5025,1,1,1

Understanding the Format
========================

GEMM Dimensions
---------------

* **M (output_a_size)**: Number of rows in matrix A and result matrix C
* **N (output_b_size)**: Number of columns in matrix B and result matrix C
* **K (contraction_size)**: Shared dimension between matrices A and B
* **B (batch_size)**: Number of matrices in batched operations

Matrix Operation
----------------

The GEMM operation computes: **C = α × A × B + β × C**

* Matrix A dimensions: M × K
* Matrix B dimensions: K × N  
* Matrix C dimensions: M × N
* All dimensions must be positive integers

Batch Operations
----------------

* **batch_size**: Defines number of independent GEMM operations
* **reduction_a_size**: Additional reduction factor for matrix A
* **reduction_b_size**: Additional reduction factor for matrix B
* Example: batch_size=4 performs 4 separate GEMM operations

Sample Test Cases
=================

.. code-block:: text

   contraction_size,output_a_size,output_b_size,total_output_size,batch_size,reduction_a_size,reduction_b_size
   1,2655,526,1396530,1,1,1
   1,2655,1000,2655000,1,1,1
   1,5025,526,2643150,1,1,1
   45,59,45,2655,1,1,1
   45,59,59,3481,1,1,1
   59,45,45,2025,1,1,1
   67,75,67,5025,1,1,1
   526,1,1,1,1,1,1
   526,2025,1,2025,1,1,1
   526,2025,2025,4100625,1,1,1

Sample Output
-------------

.. code-block:: text

   Loaded 10 GEMM test cases (double precision)
   
   Testing: GEMM_double_59x45x45_B1_AR1_BR1
     Matrix dimensions: A(59x45) × B(45x45) = C(59x45)
     Batch size: 1
     Reduction dimensions: AR=1, BR=1
     Data type: double (8 bytes per element)
     Buffer sizes: A=2655, B=2025, C=2655
     Allocating buffers...
     Running 100 timing iterations...
     Total FLOPs: 2.385e+05
     Data size: 0.056 MB
     Iterations: 100
     Average time: 0.045123 ms
     Performance: 5.284 GFLOPS
     Throughput: 2.484 GB/s