:orphan:

Utility routines (Work in progress)
===================================

.. code:: cpp

   Tensor<T> result = scale(tensor, alpha);

   //using labeled tensor
   Tensor<T> result = scale(tensor(mu,nu), alpha); 
   //result contains the full tensor with only the portion(slice)
   //indicated by the labels in labeled tensor scaled.

User responsible for destroying ``result``

Routines that update tensors in-place
-----------------------------------------

In-place tensor update routines have ``_ip`` suffix in their name. Only
``conj_ip`` and ``scale_ip`` routines are available currently.

.. code:: cpp

   scale_ip(tensor,alpha);
   scale_ip(tensor(mu,nu),alpha); //using labeled tensor

ScaLAPACK related routines
--------------------------

``NOTE:`` In the following text, a regular TAMM tensor refers to a
tensor allocated using the default TAMM distribution (NW) scheme.

For most uses, prefer the ScaLAPACK grid API (``ScalapackGrid::allocate``, ``to_block_cyclic``,
``from_block_cyclic``), which creates block-cyclic tensors with the right grid and block size; see
:doc:`linalg`. The routines below are the lower-level building blocks.

The following routine copies a regular TAMM tensor into an allocated TAMM tensor with a block
cyclic distribution (created with ``set_block_cyclic`` and allocated by the caller).

.. code:: cpp

   to_block_cyclic_tensor(regular_tensor, block_cyclic_tensor);

The following routine takes a TAMM tensor with block cyclic distribution
and copies the data into a regular TAMM tensor. ``regular_tensor``
should be allocated before calling this routine.

.. code:: cpp

   from_block_cyclic_tensor(block_cyclic_tensor, regular_tensor);

The following routine returns a pointer to the local contiguous block
cyclic buffer owned by the calling mpi process and the buffer size.

.. code:: cpp

   std::tuple<TensorType*,int64_t> access_local_block_cyclic_buffer(Tensor<TensorType> tensor) 

Types of utility routines
-------------------------

-  routines that return a scalar (U_S)
-  routines that return a new tensor (U_NT)
-  routines that update tensor in-place (U_IP)

