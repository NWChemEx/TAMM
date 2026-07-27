Tensor Operations
=================

All tensor operations in TAMM uses labeled tensors (``Tensor`` objects
with labeling within parenthesis) as the main component. By using
different set of labeling users can describe sparse computation over the
dense tensors, or opt out using labels which will result in using
``TiledIndexSpace`` information from the storage/construction.

With the new extensions (``TiledIndexSpace`` transformation operations)
to TAMM syntax on tensor operations, users can use different labels then
the construction which will be intersected with the storage when the
loop nests are being constructed. Below examples will illustrate the
usage and the corresponding loop nests for each corresponding tensor
operation. The only constraint in using different labels than the labels
used in construction is that both labels should have same root
``TiledIndexSpace``. This constraint is mainly coming from the
transformation constraints that we set in relation to the performance
reasons. We assume that ``Tensor`` objects from the previous examples
are constructed and allocated over a ``Scheduler`` object named ``sch``
beforehand.

Tensor Labeling Recap
---------------------

TAMM uses labeling for operating over ``Tensor`` objects, there are two
different ways of labeling with different capabilities.

Using ``strings``
~~~~~~~~~~~~~~~~~~

String based labeling is used mainly for basic operations where the
operations expands over the full tensors:

.. code:: cpp

   // Assigning on full Tensors
   (C("i","j") = B("i","j"));

   // Tensor addition
   (C("i","j") += B("i", "j"));

   // Tensor multiplication
   (C("i","j") = A("i","k") * B("k","j"));

Using ``TiledIndexLabel`` objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more complex operations over the slices of the full tensors or
dependent indices, TAMM implements a ``TiledIndexLabel`` object. These
labels are constructed over ``TiledIndexSpace``\ s and can be used to
accessing portions of the tiled space.

.. code:: cpp

   // TiledIndexSpace used in Tensor construction
   TiledIndexSpace MOs{is_mo};
   TiledIndexSpace depMOs{MOs, dep_relation};

   // Constructing labels
   auto [i, j, k] = MOs.labels<3>("all");
   auto [a, b, c] = MOs.labels<3>("occ");
   auto [p, q] = depMOs.labels<2>("all");

   // Assigning on full Tensors
   (C(i, j) = B(i, j));

   // Assigning on occupied portions of Tensors
   (C(a, b) = B(a, b));

   // Tensor Addition on full Tensors
   (C(i, j) += B(i, j));

   // Tensor Addition on occupied portions of Tensors
   (C(a, b) += B(a, b));

   // Tensor multiplication on full Tensors
   (C(i, j) = A(i, k) * B(k, j));

   // Tensor multiplication on occupied portions of Tensors
   (C(a, b) = A(a, c) * B(c, a));

   // Tensor operations on dependent index spaces
   (D(i, p(i)) = E(i, p(i)));
   (D(i, p(i)) += E(i, p(i)));

Tensor Copy (Shallow vs Deep)
-----------------------------

TAMM uses “single program multiple data
(`SPMD <https://en.wikipedia.org/wiki/SPMD>`__)” model for distributed
computation. In this programming abstraction, all nodes has its own
portion of tensors available locally. So any operation on the whole
tensors results in a message passing to remote portions on the tensor,
with implied communication. More importantly, many/all operations are
implied to be collective. This simplifies management of handles (handles
are not migratable). However, this implies that operations such as
memory allocation and tensor copy need to be done collectively. This
conflicts with supporting deep copy when a tensor is passed by value,
because this can lead to unexpected communication behavior such as
deadlocks.

To avoid these issues, TAMM is designed to:

1. Handle tensors in terms of handles with shallow copy.
2. Require operations on Tensors to be declared explicitly and executed
   using a scheduler.

**NOTE:** This is distinguished from a rooted model in which a single
process/rank can non-collectively perform a ``global`` operation (e.g.,
copy).

In summary, any assignment done on Tensor objects will be a **shallow
copy** (internally it will be copying a shared pointer) as opposed to
**deep copy** that will result in message passing between each node to
do the copy operation:

.. code:: cpp

   Tensor<double> A{AO("occ"), AO("occ")};
   Tensor<double> B{AO("occ"), AO("occ")};

   A = B;               // will be a shallow copy as we will be copying a shared pointer
   Tensor<double> C(B); // this is shallow copy as well as it will copy shared pointer internally
   auto ec = tamm::make_execution_context();

   Scheduler(ec)
     (A("i","k") = B("i","k")) // deep copy using scheduler for informing remote nodes
   .execute();

To make Tensor operations explicit, TAMM is using parenthesis syntax as
follows:

.. code:: cpp

   Tensor<double> A{AO("occ"), AO("occ")};
   Tensor<double> B{AO("occ"), AO("occ")};
   Tensor<double> C{AO("occ"), AO("occ")};

   auto ec = tamm::make_execution_context();

   Scheduler(ec)
     // Tensor assignment 
     (A("i", "k") = B("i","k"))
     // Tensor Addition 
     (A("i", "k") += B("i","k"))
     // Tensor Multiplication
     (C("i","k") = A("i","k") * B("i","k"))
   .execute();

Keep in mind that these operations will not be effective (there will be
no evaluation) until they are scheduled using a scheduler.

.. raw:: html

   <!-- For actual evaluation of these operations, TAMM provides two options: -->

**Scheduling operations directly**

.. code:: cpp

   auto ec = tamm::make_execution_context();

   Scheduler(ec)
   (A("i", "k") = B("i","k"))
   (A("i", "k") += B("i","k"))
   (C("i","k") = A("i","k") * B("i","k"))
   .execute();
     
.. raw:: html

   <!-- 
   **Using a DAG construct**
   ```c++
   Oplist sample_op(Tensor<double> A, Tensor<double> B, Tensor<double> C){
       return {
               A("i", "k") = B("i","k"),
               A("i", "k") += B("i","k"),
               C("i","k") = A("i","k") * B("i","k")
              };
   }
       Tensor<double> A{AO("occ"), AO("occ")};
       Tensor<double> B{AO("occ"), AO("occ")};
       Tensor<double> C{AO("occ"), AO("occ")};
       
       auto sampleDAG = make_dag(sample_op, A, B, C);
       
       Scheduler::execute(sampleDAG);
       
   ```
   -->

Tensor Contraction Operations
-----------------------------

A Tensor operation in TAMM can only be in the single-op expressions of
the form:

``C [+|-]?= [alpha *]? A [* B]?``

Set operations
~~~~~~~~~~~~~~

``C = alpha``

**Examples**:

.. code:: cpp

   (C() = 0.0)

Add operations
~~~~~~~~~~~~~~

``C [+|-]?= [alpha *]? A``

**Examples**:

.. code:: cpp

   (i1("h6", "p5") = f1("h6", "p5"))
   (i0("p2", "h1") -= 0.5 * f1("p2", "h1"))
   (i0("p3", "p4", "h1", "h2") += v2("p3", "p4", "h1", "h2"))

More examples of Set/Add operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examples without using labels:

.. code:: cpp

   // without any labels
   sch
     // Dense Tensor
     (T2() = 42.0)
     // Sparse Tensor
     (T5() = 21.0) 
     // Assignment
     (T2() = T5())
   .execute();

   // Equivalent code written as loops
   // For (T2() = 42.0)
   for(auto i : T2.dim(0))
     for(auto A : T2.dim(1))
       T2[i][A] = 42.0;

   // For (T5() = 21.0)
   for(auto i : T5.dim(0))
     for(auto mu: T5.dim(1)[i])
       T5[i][mu] = 21.0;

   // For (T2() = T5())
   for(auto i : T2.dim(0).intesect(T5.dim(0)))
     for(auto mu : T2.dim(1).intersect(T5.dim(1)[i]))
       T2[i][mu] = T5[i][mu]

Examples using labels:

.. code:: cpp

   // Labeling reference
   // auto [i, j] = MO.labels<2>("all");
   // auto [A, B] = AO.labels<2>("all");
   // auto [mu, nu] = depAO.labels<2>("all");

   // Construct a subset of MO
   TiledIndexSpace subMO{MO, range(1,3)};

   auto i_p = subMO.label("all");

   sch
     // Dense tensor with subset
     (T2(i_p, A) = 13.0)
     // Sparse Tensor
     (T5(i_p, mu(i_p)) = 42.0) 
     // Assignment
     (T2(i_p, mu(i_p)) = T5(i_p, mu(i_p)))
     // Assignment with independent labels
     (T2(i_p, mu) = T5(i_p, mu))
   .execute();


   // Equivalent code written as loops
   // For (T2(i_p, A) = 13.0)
   for(auto i : T2.dim(0).intersect(i_p.tis()))
     for(auto A : T2.dim(1))
       T2[i][A] = 42.0;

   // For (T5(i_p, mu(i_p)) = 42.0) 
   for(auto i : T5.dim(0).intesect(i_p.tis()))
     for(auto mu : T5.dim(1)[i])
       T5[i][mu] = 21.0;

   // For (T2() = T5())
   // Both assignment will result in the same
   // loop nest 
   for(auto i : T2.dim(0).intesect(T5.dim(0)).intesect(i_p))
     for(auto mu : T2.dim(1).intersect(T5.dim(1)[i]))
       T2[i][mu] = T5[i][mu]


Multiplication operations
~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to the set/add operations, users have the option to construct equations
without giving any labels which in return will use the
``TiledIndexSpace``\ s used in the construction. **Note that** in any of
the operations the tensors are assumed to be constructed and allocated
before the using in an equation. Again users may choose to specify
labels (dependent or independent) according to the
``TiledIndexSpace``\ s that are used in tensors’ construction for each
dimension. Similar to set/add operations, the labels used in the
contractions should be compatible with the ``TiledIndexSpace``\ s that
are used in constructing the tensors.

``C [+|-]?= [alpha *]? A * B``

**Examples**:

.. code:: cpp

   (de() += t1("p5", "h6") * i1("h6", "p5"))
   (i1("h6", "p5") -=  0.5  * t1("p3", "h4") * v2("h4", "h6", "p3", "p5"))
   (t2("p1", "p2", "h3", "h4") =  0.5  * t1("p1", "h3") * t1("p2", "h4"))
   (i0("p3", "p4", "h1", "h2") += 2.0 * t2("p5", "p6", "h1", "h2") * v2("p3", "p4", "p5", "p6"))

Examples without using labels:

.. code:: cpp

   // without any labels
   sch
     // Dense Tensor
     (T2() = 2.0)
     // Dense Tensor
     (T4() = 21.0) 
     // Dense Contraction
     (T1() = T2() * T4())
   .execute();

   // Equivalent code written as loops
   // For (T2() = 2.0)
   for(auto i : T2.dim(0))
     for(auto A : T2.dim(1))
       T2[i][A] = 42.0;

   // For (T4() = 21.0)
   for(auto A : T4.dim(0))
     for(auto j: T4.dim(1)[i])
       T4[A][j] = 21.0;

   // For (T1() = T2() * T4())
   for(auto i : T2.dim(0).intesect(T4.dim(0)))
     for(auto j : T2.dim(1).intersect(T4.dim(1)))
       for(auto A : T2.dim(1).intesect(T4.dim(0)))
         T1[i][j] = T2[i][A] * T4[A][j];

Examples using labels:

.. code:: cpp

   // Constructed TiledIndexSpaces
   // TiledIndexSpace AO{AO_is, tile_size};
   // TiledIndexSpace MO{MO_is, tile_sizes};
   // TiledIndexSpace depAO_1{AO, {MO}, dep_rel_2};
   // TiledIndexSpace depAO_2{AO, {MO}, dep_rel_2};

   // Constructed index labels 
   // auto [i, j] = MO.labels<2>("all");
   // auto [A, B] = AO.labels<2>("all");
   // auto [mu, nu] = depAO_1.labels<2>("all");
   // auto [mu_p, nu_p] = depAO_2.labels<2>("all");

   // Dense tensor construction
   tensor_type T1{i, j};     // MO x MO Tensor
   tensor_type T2{i, A};     // MO x AO Tensor
   tensor_type T3{mu, nu};   // AO x AO Tensor
   tensor_type T4{mu, i};    // AO x MO Tensor

   // Sparse tensor construction
   tensor_type T5{i, mu(i)}; // MO x depAO Tensor 
   tensor_type T6{mu_p(j), j}; // depAO x MO Tensor

   auto mu_i = mu.intersect(mu_p);
   sch
     // Sparse Tensor
     (T6(mu_p(j), j) = 2.0)
     // (T6(mu_p, j) = 2.0)
     // Sparse Tensor
     (T5(i, mu(i)) = 21.0)
     // (T5(i, mu) = 21.0) 
     // Sparse Contraction
     (T1(i, j) = T5(i, mu_i(i)) * T6(mu_i(j), j))
     // (T1(i, j) = T5(i, mu_i) * T6(mu_i, j))
   .execute();


   // Equivalent code written as loops
   // For (T6(mu_p(j), j) = 2.0) 
   // &   (T6(mu_p, j) = 2.0)
   for(auto j : T6.dim(1))
     for(auto mu_p : T6.dim(0)[j])
       T2[mu_p][j] = 2.0;

   // For (T5(i, mu(i)) = 21.0)
   // &   (T5(i, mu) = 21.0)
   for(auto i : T5.dim(0))
     for(auto mu: T5.dim(0)[i])
       T5[i][mu] = 21.0;

   // For (T1(i, j) = T5(i, mu_i(i)) * T6(mu_i(j), j))
   // &   (T1(i, j) = T5(i, mu_i) * T6(mu_i, j))
   for(auto i_idx : T1.dim(0).intersect(i).intersect(T5.dim(0)))
     for(auto j_idx : T1.dim(1).intersect(j).intersect(T6.dim(j)))
       for(auto mu_idx : T5.dim(1).intersect(mu_i).intersect(T6.dim(0)))
         T1[i_idx][j_idx] = T5[i_idx][mu_idx] * T6[mu_idx][j_idx];


Multi-operand Tensor Operations (New)
--------------------------------------

TAMM has a new multi-operand tensor operation syntax that allows users
to define complex operations that have more than single tensor operation in it. 
Using this syntax users can define tensor operations as a separate object and 
associate them with output tensors. Once all the updates on the output tensors
are finished, one can directly call an execute on any output tensor to start 
executing each update. 

.. code:: cpp

  // Construct operation using multi operand syntax. Cast to LTOp is required for time being
  auto op_1 = (LTOp) A(i, l) * (LTOp)  B(l, a) * (LTOp) C(j, a) * (LTOp) D(j, b);
  auto op_2 = /* ... */;

  // Associate each operation with an output tensor
  E(i, b).set(op_1);      // assign (=)
  E(i, b).update(op_2);   // accumulate (+=)

  // Construct composite operations that includes tensor contraction and addition
  auto energy_op = 2.0 * (LTOp) F(m, e) * (LTOp) t1(e, m) +
                   2.0 * (LTOp) V(m, n, e, f) * (LTOp) t2(e, f, m, n) +
                   2.0 * (LTOp) V(m, n, e, f) * (LTOp) t1(e, m) * (LTOp) t1(f, n) +
                  -1.0 * (LTOp) V(m, n, f, e) * (LTOp) t2(e, f, m, n) +
                  -1.0 * (LTOp) V(m, n, f, e) * (LTOp) t1(e, m) * (LTOp) t1(f, n);
  
  // Associate energy operation with a scalar tensor
  energy().set(energy_op);

  // Execute on output tensors
  OpExecutor op_executor(/*...*/);
  op_executor.execute(E);
  op_executor.execute(energy);


TAMM employs an operation minimization algorithm that will find the most 
efficient binarization and construct corresponding intermediate tensors automatically. 
TAMM also automatically takes care of allocation/deallocation of these intermediates. 
**Note:** TAMM provides a new execution construct (OpExecutor) for this type of opeation execution.

Tensor utility routines
------------------------

As tensors are the main construct for the computation, TAMM provides a
set of utilities. These are basically tensor-wise update and
access methods as well as point-wise operations over each element in the
tensor. Also included are parallel I/O routines.

Updating using lambda functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TAMM provides two methods for updating the full tensors or a slice of a
tensor using a lambda method: - ``update_tensor(...)`` used for
updating the tensor using a lambda method where the values are not
dependent on the current values.

.. code:: cpp

   // lambda function that assigns zero to non-diagonal elements
   auto lambda = [](const IndexVector& blockid, span<T> buf){
     if(blockid[0] != blockid[1]) {
       for(auto i = 0U; i < buf.size(); i++) 
         buf[i] = 0; 
     }
   };
   // template<typename T, typename Func>
   // update_tensor(LabeledTensor<T> lt, Func lambda)

   // updates a 2-dimensional tensor A using the lambda method
   tamm::update_tensor(A, lambda);

-  ``update_tensor_general(...)``: only difference from
   ``update_tensor(...)`` method is in this case lambda method can use
   the current values from the tensor.

   .. code:: cpp

      std::vector<double> p_evl_sorted(total_orbitals);
      auto lambda_general = [&](Tensor<T> tensor, const IndexVector& blockid, 
                                span<T> buf){
        auto block_dims = tensor.block_dims(blockid);
        auto block_offset = tensor.block_offsets(blockid);

        TAMM_SIZE c = 0;
        for(auto i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
          for(auto j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++, c++) {
            buf[c] = CholVpr(i,j,x);
          }
        }
      };

      // updates each element of the tensor with a computation
      tamm::update_tensor_general(B, lambda_general);

Accessing tensors
~~~~~~~~~~~~~~~~~

TAMM also provides utility routines for specialized accessor for specific
types of tensors:

-  ``get_scalar(...)``: Special accessor for scalar values
   (i.e. zero dimensional tensors).

   .. code:: cpp

      // Get the element value for a zero dimensional tensor A
      auto el_value = tamm::get_scalar(A);

-  ``trace(...)``: utility routine for getting the sum of the
   diagonal in two dimensional square tensors.

   .. code:: cpp

      // get the diagonal sum of the two dimensional square tensor A(N, N)
      auto trace_A = tamm::trace(A);

-  ``diagonal(...)``: utility routine for getting the values at the
   diagonal of a two dimensional square tensor.

   .. code:: cpp

      // get the diagonal values of two dimensional tensor A(N,N)
      auto diagonal_A = tamm::trace(A);

-  ``max_element(...)``\ & ``min_element(...)``: utility routine to
   **collectively** find the maximum/minimum element in a tensor. This
   method returns the maximum/minimum value along with block id the
   value found and the corresponding sizes (for each dimension) of the
   block.

   .. code:: cpp

      // get the max element in a tensor 
      auto [max_el, max_blockid, max_block_sizes] = tamm::max_element(A);
      
      // get the min element in a tensor 
      auto [min_el, min_blockid, min_block_sizes] = tamm::min_element(A);

Point-wise operations
~~~~~~~~~~~~~~~~~~~~~

Different then block-wise and general operations on the tensors, TAMM
provides point-wise operations that can be applied to the whole tensor.
As tensors are distributed over different MPI ranks, these operations
are collective.

-  ``square(...)`` updates each element in an tensor to its square
   value
-  ``log10(...)`` updates each element in a tensor to its
   logarithmic
-  ``inverse(...)`` updates each element in a tensor to its inverse
-  ``pow(...)`` updates each element in a tensor to its ``n``-th
   power
-  ``scale(...)`` updates each element in a tensor by a scale factor
   ``alpha``

Parallel IO operations
~~~~~~~~~~~~~~~~~~~~~~~

- ``tamm::write_to_disk(A,"filename")`` writes a distributed tamm tensor ``A`` to disk in parallel.
- ``tamm::read_from_disk(A,"filename")`` reads a distributed tamm tensor ``A`` from disk in parallel.

- ``read_from_disk_group(ec, tensor_list, filename_list)`` and ``write_to_disk_group(ec, tensor_list, filename_list)`` 
  for reading and writing a batch of distributed tamm tensors concurrently over different process groups.