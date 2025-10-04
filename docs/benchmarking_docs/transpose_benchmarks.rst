.. _test_tensor_transpose:

===============================
Tensor Transpose Benchmark Test
===============================

Overview
========

Test_Tensor_Transpose measures the performance of tensor transpose operations in. It reads test configurations from CSV files and reports timing and throughput metrics for both GPU and CPU execution.

Basic Usage
-----------

.. code-block:: bash

   # Run with default settings
   mpirun -n 2 ./transpose_benchmark test_cases.csv

   # Specify iterations and data type
   mpirun -n 2 ./transpose_benchmark test_cases.csv 50 float

Command Line Arguments
======================

.. code-block:: bash

   mpirun -n <processes> ./transpose_benchmark <csv_file> [iterations] [data_type]

**Required Arguments:**

* ``csv_file``: Path to CSV file containing test case definitions

**Optional Arguments:**

* ``iterations``: Number of benchmark runs per test case (default: 100)
* ``data_type``: Data precision - ``float``, ``double``, or ``all`` (default: ``double``)

**Examples:**

.. code-block:: bash

   # Test double precision with 100 iterations
   mpirun -n 2 ./transpose_benchmark cases.csv

   # Test single precision with 50 iterations  
   mpirun -n 2 ./transpose_benchmark cases.csv 50 float

   # Test both precisions with 200 iterations
   mpirun -n 2 ./transpose_benchmark cases.csv 200 all

Input File Format
=================

The program expects a CSV file with tensor transpose configurations using the following 9-column format:

CSV Header
----------

.. code-block:: text

   block_size_dim_0,block_size_dim_1,block_size_dim_2,block_size_dim_3,permutation_map_idx_0,permutation_map_idx_1,permutation_map_idx_2,permutation_map_idx_3,original_transpose_string

Column Definitions
------------------

* **block_size_dim_0 to block_size_dim_3**: Tensor dimension sizes (use ``1`` for unused dimensions)
* **permutation_map_idx_0 to permutation_map_idx_3**: Target position for each dimension (use ``-1`` for unused positions)
* **original_transpose_string**: Mathematical description of the transpose operation

Sample Data
-----------

.. code-block:: text

   block_size_dim_0,block_size_dim_1,block_size_dim_2,block_size_dim_3,permutation_map_idx_0,permutation_map_idx_1,permutation_map_idx_2,permutation_map_idx_3,original_transpose_string
   45,45,1,1,1,0,-1,-1,C( 0 1 ) <- C'( 1 0 )
   100,45,1,1,1,0,-1,-1,C( 0 1 ) <- C'( 1 0 )
   75,67,2,1,0,2,1,-1,B( 2 1 3 ) -> B'( 2 3 1 )
   75,67,3,1,1,0,2,-1,C( 0 1 2 ) <- C'( 1 0 2 )

Understanding the Format
========================

Dimension Specification
-----------------------

* Valid dimensions must be > 1
* Use ``1`` for unused/singleton dimensions
* Example: ``75,67,4,1`` represents a 3D tensor of size 75×67×4

Permutation Mapping
-------------------

* Each index specifies where the corresponding input dimension goes in the output
* Use ``-1`` for unused positions
* Example: ``1,0,-1,-1`` means dimension 0 → position 1, dimension 1 → position 0

Transpose String Notation
-------------------------

The mathematical notation describes the operation:

* ``C( 0 1 ) <- C'( 1 0 )``: Matrix transpose (swap dimensions 0 and 1)
* ``B( 2 1 3 ) -> B'( 2 3 1 )``: Cyclic permutation of 3D tensor
* ``C( 0 1 2 ) <- C'( 1 0 2 )``: Transpose first two dimensions, keep third unchanged

Sample Test Cases
=================

.. code-block:: text

   block_size_dim_0,block_size_dim_1,block_size_dim_2,block_size_dim_3,permutation_map_idx_0,permutation_map_idx_1,permutation_map_idx_2,permutation_map_idx_3,original_transpose_string
   45,45,1,1,1,0,-1,-1,C( 0 1 ) <- C'( 1 0 )
   100,45,1,1,1,0,-1,-1,C( 0 1 ) <- C'( 1 0 )
   55,45,1,1,1,0,-1,-1,C( 0 1 ) <- C'( 1 0 )
   75,67,2,1,0,2,1,-1,B( 2 1 3 ) -> B'( 2 3 1 )
   75,67,3,1,1,0,2,-1,C( 0 1 2 ) <- C'( 1 0 2 )
   75,67,4,1,1,2,0,-1,B( 0 2 3 ) -> B'( 2 3 0 )
   75,67,5,1,2,0,1,-1,B( 1 3 4 ) -> B'( 4 1 3 )

Sample Output
-------------

.. code-block:: text

   ========== TESTING WITH TYPE: d ==========
   === GPU TESTS ===
   Testing (d): C( 0 1 ) <- C'( 1 0 )
     Dimensions: [45,45] -> [45,45]
     Permutation: [0,1] -> [1,0]
     Elements: 2025
     Data size: 0.015488 MB
     Element size: 8 bytes
     Iterations: 100
     Average time: 0.032145 ms
     Throughput: 0.964 GB/s
     Hardware: GPU
