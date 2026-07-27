Basic Usage Examples
====================

Contraction
-----------

For a given contraction A\ :sub:`ij` = B\ :sub:`ik`\  * C\ :sub:`kj`\ , where A, B, and C have sizes of (10, 10), the following code in TAMM can be written:

.. code:: cpp

   // Initial setup
   ProcGroup        pg = ProcGroup::create_world_coll();
   ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

   IndexSpace is{range(10)}; // Defining an index space equal to the dimensions of the tensors
   TiledIndexSpace tis{is}; // Converting to a tiled index space
   auto [i, j, k] = tis.labels<3>("all"); // Creating labels for the tensors 

   // Defining the tensors
   Tensor A{i, j};
   Tensor B{i, k};
   Tensor C{k, j};

   sch.allocate(A, B, C).execute(); // Allocating the tensors

   sch // Scheduling the following tasks

   (B() = 2.0) // Setting all elements of A to 2.0
   (C() = 21.0) // Setting all elements of C to 21.0
   (A(i, j) = B(i, k) * C(k, j)) // Performing the contraction

   .execute(); // Executing the scheduled tasks

   tamm::write_to_disk(A,"tensor_A"); // Writing the resultant tensor to a file named tensor_A

   sch.deallocate(A, B, C).execute(); // Deallocating the tensors after the contraction has concluded

This code can easily be modifyed for any given contraction or sequence of contractions, as additional index spaces and tensors can be defined and allocated as needed, and additional expressions can be written within the ``sch`` block to perform additional contractions, or to add tensors together.
