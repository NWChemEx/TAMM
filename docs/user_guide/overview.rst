Quick Overview
==============

Index Spaces and Tiled Index Spaces
-----------------------------------
Unlike the NumPy library, which effectively includes index spaces when you define arrays, TAMM requires explicit definition of these spaces. This does allow for some useful behaviors.  
However, this is less important to a general user than the tensor operations discussed later, as TAMM takes care of the finer details of index spaces and tiled index spaces automatically.

.. code:: cpp

    IndexSpace is0{range(10)}; // This defines an index space with values from 0 to 9
    //{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

It is possible to name a subspace of an ``IndexSpace`` which can be referenced later.

.. code:: cpp

    IndexSpace is0{range(10),                   // is0("all") -> {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
                    {{"sub0", {range(0, 5)}},   // is0("sub0") -> {0, 1, 2, 3, 4}
                    {"sub1", {range(5, 10)}}}}; // is0("sub1") -> {5, 6, 7, 8, 9}

    IndexSpace is0_sub0 = is0{"sub0"};

It is also possible to define an ``IndexSpace`` based on existing ``IndexSpaces``.

.. code:: cpp

    IndexSpace is0{range(5)}; // {0, 1, 2, 3, 4}
    IndexSpace is1{range(5, 10)}; // {5, 6, 7, 8, 9}

    IndexSpace is2{{is0, is1}}; // {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}


More extensive examples of defining ``IndexSpaces`` can be found in the Index Spaces page of documentation _note_: link there later

A ``TiledIndexSpace`` is similar to an ``IndexSpace``, although it has useful properties for sparse tensors.

.. code:: cpp

    IndexSpace is0{range(10)}; // {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    // Default tile size, no effect
    TiledIndexSpace tis0{range(10)}; // {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    // Tile size of 5, splits the space into 2 tiles
    TiledIndexSpace tis1{range(10), 5}; // [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]

    // Tile size of 4, splits the space into 3 tiles
    TiledIndexSpace tis2{range(10), 4}; // [{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9}]

These have a slightly different behavior when subspaces have been defined.  
Tiling is applied to subspaces independently.

.. code:: cpp

    IndexSpace is0{range(10),
                    {{"sub0", {range(0, 5)}},
                    {"sub1", {range(5, 10)}}}};
    
    TiledIndexSpace tis0{is0, 3};
    // tis0("sub0") -> [{0, 1, 2}, {3, 4}]
    // tis0('sub1') -> [{5, 6, 7}, {8, 9}]

A useful tool for contructing tensors are labels.

.. code:: cpp

    IndexSpace is0{range(10)};
    TiledIndexSpace tis0{is0};
    auto i, j = tis0.labels<2>("all"); // defines two labels, i and j, which reference the entirety of tis0


    IndexSpace is1{range(10),
                    {{"sub0", {range(0, 5)},
                    {"sub1", {range(5, 10)}}}};
    TiledIndexSpace tis1{is1};

    auto i, j = tis1.labels<2>("sub0"); // defines two labels which reference the sub0 subspace of tis1
    auto k, l = tis1.labels<2>("sub1"); // defines two labels which reference the sub1 subspace of tis1

Constructing Tensors
--------------------

With an understanding of ``IndexSpace``, ``TiledIndexSpace``, and labels, it is now possible to construct a tensor.

.. code:: cpp

    // Creating a 10x10 matrix

    IndexSpace is0{range(10)};
    TiledIndexSpace tis0{is0};

    auto i, j = tis0.labels<2>("all");

    Tensor<double> t0{i, j};



    // Creating a 10x20x5 tensor

    IndexSpace is1{range(20),
                    {{"sub0", {range(0, 5)}},
                    {"sub1", {range(0, 10)}},
                    {"sub2", {range(0, 20)}}}};
    TiledIndexSpace tis1{is1};

    auto i = tis1.labels<1>("sub0"); // Defining a label for a dimension of size 5
    auto j = tis1.labels<1>("sub1"); // Defining a label for a dimension of size 10
    auto k = tis1.labels<1>("sub2"); // Defining a label for a dimension of size 20

    Tensor<double> t1{j, k, i};

Because of TAMM's design, tensors are not allocated automatically. Allocating a tensor has to be performed using a scheduler.

.. code:: cpp

    // Setting up important things
    ProcGroup        pg = ProcGroup::create_world_coll();
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

    IndexSpace is{range(10)};
    TiledIndexSpace tis{is};
    auto [i, j, k] = tis.labels<3>("all");

    // Defining the tensors using labels equivalent to a dimension with a size of 10
    Tensor A{i, j};
    Tensor B{i, k};
    Tensor C{k, j};

    sch.allocate(A, B, C).execute(); // Allocating the tensors

Once tensors are allocated, work can be done, whether that is filling a tensor with a value, multiplying tensors, or saving them to disk. See the next section.  
To deallocate tensors:

.. code:: cpp

    // previous code block

    sch.deallocate(A, B, C).execute();


Tensor Operations
------------------

Similarly to allocation, operations on tensors must be scheduled and executed.

General Operation Structure

.. code:: cpp

    ProcGroup        pg = ProcGroup::create_world_coll();
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

    // The IndexSpaces, TiledIndexSpaces, and labels can be modified as needed
    IndexSpace is0{range(10)};
    TiledIndexSpace tis0{is0};
    auto i = tis0.labels<1>("all");

    Tensor A{i};
    // Additional tensors can be constructed
    
    sch.allocate(A).execute(); // Allocate all constructed tensors

    sch

    // Operations go here

    .execute();

    sch.deallocate(A).execute(); // Deallocate all allocated tensors

Set Operations

.. code:: cpp

    // Setting all elements of A equal to 17.0
    sch

    (A() = 17.0)

    .execute();

    // Setting A equal to B
    sch

    (A() = B())

    .execute();

Add Operations

.. code:: cpp

    // Adding A and B and storing the result in C, assuming that i and j were defined as labels and the tensors were properly constructed
    sch

    (C(i, j) = A(i, j) + B(i, j))

    .execute();

Multiplication Operations

.. code:: cpp

    // Contracting A and B and storing the result in C
    sch

    (C(i, j) = A(i, k) * B(k, j))

    .execute();

    // Contracting A and B and substracting half the result from C
    sch

    (C(i, j) -= 0.5 * A(i, k) * C(k, j))

    .execute();

Now for a full example, the following code performs this contraction, A\ :sub:`ij` = B\ :sub:`ik`\  * C\ :sub:`kj`\ , where A, B, and C have sizes of (10, 10).

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

   sch.deallocate(A, B, C).execute(); // Deallocating the tensors after the contraction has concluded


I/O
----

TAMM provides parallel IO routines for tensors:

.. code:: cpp

    tamm::write_to_disk(A,"tensor_A"); // Writes tensor A to a file called tensor_A

    tamm::read_from_disk(B, "tensor_B"); // Reads tensor B from a file called tensor_B
