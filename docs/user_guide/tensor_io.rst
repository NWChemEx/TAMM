Tensor I/O
==========

TAMM writes and reads distributed tensors to and from disk in parallel using HDF5. Each tensor is
stored in its own file, called a *tensor file*.

Routines
--------

- ``tamm::write_to_disk(ec, tensors, filenames, profile = false)`` writes a list
  of distributed tensors, each to its own tensor file, in parallel.
- ``tamm::read_from_disk(ec, tensors, filenames, profile = false)`` reads a list
  of distributed tensors, each from its own tensor file, in parallel.
- ``tamm::write_to_disk(ec, tensor, filename, ...)`` and
  ``tamm::read_from_disk(ec, tensor, filename, ...)`` do the same for one tensor.

``write_to_disk`` returns ``true`` if every tensor file was written and committed, and ``false``
otherwise (see `Safe writes`_), with the same value on every process. The result may be ignored.
``read_from_disk`` returns nothing, since a failed read terminates the program.

``ec`` is the process group doing the I/O, and every rank of it must call the routine. The
tensors may be allocated on ``ec`` or on a larger process group: for example, a subgroup can write
or read tensors allocated on all ranks. A tensor file can also be read on a different number of
ranks than wrote it.

.. code:: cpp

    tamm::write_to_disk(ec, A, "tensor_A.h5");  // writes tensor A to tensor_A.h5
    tamm::read_from_disk(ec, B, "tensor_B.h5"); // reads tensor B from tensor_B.h5

    std::vector<Tensor<double>> amplitudes{T1, T2};
    std::vector<std::string>    files{"t1.h5", "t2.h5"};
    if(!tamm::write_to_disk(ec, amplitudes, files)) {
      // no file was replaced; the previous versions, where they exist, are kept
    }
    tamm::read_from_disk(ec, amplitudes, files);

With ``profile = true``, the routines print how the work was split among the processes and a
per-file timing breakdown.

Safe writes
-----------

A tensor file is never left partially written. Each write goes to a *staging file*,
``<filename>.tmp``, which is renamed to ``<filename>`` only after every process has written its
part successfully. This rename is the *commit*. As a result, a tensor file at its final name is
always complete: it holds either the previous version or the new one.

This makes tensor files safe to use as checkpoints. If a job is killed while writing a tensor to
disk, the previous version of the tensor file, if there is one, is untouched and can still be read.

A list of tensors is committed **all-or-nothing**: if any of its tensor files fails to write, none
of them is committed and every tensor file keeps its previous version. Tensor files that must stay
consistent with each other (for example, several tensors from the same iteration of a solver)
should therefore be written in one call, rather than in separate calls, each of which commits on
its own.

Safe reads
----------

A read never sees a partially written tensor file. Reads open only the tensor file at its final
name and ignore staging files, and because of the commit that file is always complete. Reads never
modify files on disk.

A read never silently loads bad data. Every process checks each step of the read (opening the
file and reading its data). If any process fails, for example because the file is missing, is not
an HDF5 file, is truncated, or the filesystem returns an error, all processes agree on the failure
and the program terminates. The cause of the failure is reported for each tensor file, followed by
a final message listing every tensor file that could not be read.

A tensor file stores the tensor's data block by block in the order given by its tiling. To make
sure a file is read into a tensor with the same layout, each tensor file also stores a *tensor
description*: the element type, the size of each tile along each mode, and the number of non-zero
blocks with a hash identifying them. Before reading any data, the description is compared with
the tensor being read into. A tensor file without a description, or with a different one, is a
read failure, for example::

    [TAMM ERROR] t.h5: Tiling differs in dimension 1: file has 8 tiles, tensor has 6 tiles

To read a tensor file, use a tensor with the same index spaces and tiling as the tensor that was
written.

Striping on Lustre
------------------

On Lustre, a file's bandwidth grows with the number of storage targets (OSTs) its data is striped
over. Each tensor file is striped over one OST per 4 GiB of tensor data, at least 1 and at most
64, so a 60 GiB tensor is striped over 15 OSTs, and a tensor of 256 GiB or more over 64. The
stripe size is 4 MiB. The runtime parameters ``TAMM_IO_STRIPE_COUNT_MAX`` and
``TAMM_IO_STRIPE_SIZE`` change the maximum number of OSTs and the stripe size (see
:doc:`runtime_parameters`). Other filesystems ignore striping.

How the work is split
---------------------

The nodes of ``ec`` are split into *I/O groups* of whole nodes, each writing or reading one or more
tensor files:

- Each tensor ideally gets one node per 5 GiB of data (the runtime parameter
  ``TAMM_IO_GIB_PER_NODE`` changes this). If every tensor's ideal group fits in ``ec``, every
  tensor gets it and all tensor files are written or read at once.
- Otherwise, if there are no more tensors than nodes, the groups are scaled down in proportion to
  the tensor sizes, at least one node each, so the files take about the same time.
- With more tensors than nodes, every node is an I/O group, and the tensors are handed out to the
  groups as they become free, largest first.

Within an I/O group, only some processes of each node, the *writers*, write or read the file;
the others still open and close it, but do nothing else. The number of writers per node is 4096
divided by the number of nodes in the group, at least 1 and at most all processes of the node, so
a group has at most about 4096 writers, or one per node in groups of more than 4096 nodes (the
runtime parameter ``TAMM_IO_WRITERS_PER_NODE`` sets the number per node instead). The blocks owned by a node's processes are split among its writers
by bytes. A writer writes or reads its own blocks directly from or into its own memory, and copies
the other blocks of its node to or from their owners within the node.

Blocks owned by processes outside the I/O group are moved to or from the group's writers, so the
more of a tensor's owners an I/O group contains, the less data moves. Because an I/O group is
sized from the tensor's data (one node per 5 GiB by default), a tensor holding less than that per
node of ``ec`` is handled by fewer nodes than hold it, and most of its blocks are moved to or from
them.

With the runtime parameter ``TAMM_IO_GROUPS=all``, there is a single I/O group made of all of
``ec``, which handles the tensor files one after another, largest first. Each file is still
written or read in parallel by every node of ``ec``. For tensors allocated on ``ec``, every block
is then on the node writing or reading it and no data leaves its node, at the cost of handling the
files one at a time.

Behavior
--------

.. list-table::
   :header-rows: 1
   :widths: 18 24 24 10 24

   * - Routine
     - On success
     - On failure (files)
     - Fatal?
     - Reported
   * - ``write_to_disk``
     - Writes each ``<f>.tmp``; all processes agree; the files are renamed to ``<f>``; returns
       ``true``
     - All-or-nothing: if any file fails, every ``.tmp`` is deleted and every previous ``<f>`` is
       kept, where one exists; returns ``false``
     - No; the calculation continues
     - ``<call> failed for <f>: <cause>`` for each failed file, then, for one file,
       ``<f> not committed; previous version kept`` (or ``no previous version exists``), or, for
       several, one summary listing the files that failed, those whose previous version was kept,
       and those with no previous version, one file per line
   * - ``read_from_disk``
     - Every tensor filled
     - Files untouched; tensor contents undefined
     - **Yes**
     - ``<call> failed for <f>: <cause>`` for each failed file, then one final message naming
       every tensor file that could not be read

A write fails if any process fails to create, write or close the staging file, or if the final
rename fails. A read fails if any process fails to open or read the tensor file, or if the
tensor file's description is missing or does not match the tensor. A failed read terminates the
program on all processes.

Messages
--------

All messages are printed once, not by every process, and HDF5's own error stack is not printed.
``<cause>`` is the HDF5 error at its origin, for example
``MPI_File_open failed: MPI error string is 'MPI_ERR_BAD_FILE: bad file'`` for a write, or
``file signature not found`` or ``truncated file: ...`` for a read. Write failures are reported
as warnings, since the calculation continues; read failures are reported as errors, since the
program terminates.

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Event
     - Printed by
     - Message
   * - A collective HDF5 call fails while writing (creating, closing)
     - Root of the file's I/O group, once
     - ``[TAMM WARNING] <call> failed for <f>: <cause>``
   * - Writing data fails on one process
     - The process that failed
     - ``[TAMM WARNING] H5Dwrite failed for <f>: <cause>``
   * - The rename fails
     - The process that attempted it
     - ``[TAMM WARNING] rename <f>.tmp -> <f> failed: <reason>``
   * - A collective HDF5 call fails while reading (opening, closing)
     - Root of the file's I/O group, once
     - ``[TAMM ERROR] <call> failed for <f>: <cause>``
   * - The tensor file's description does not match the tensor
     - Root of the file's I/O group, once
     - ``[TAMM ERROR] <f>: <what differs>``
   * - Reading data fails on one process
     - The process that failed
     - ``[TAMM ERROR] H5Dread failed for <f>: <cause>``
   * - Any read fails
     - Rank 0, once, after the messages above
     - The final message naming the tensor file(s), before the program terminates

Interrupted operations
----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Event
     - Result
   * - Job killed while writing
     - ``<f>.tmp`` is left behind, partially written, and ``<f>`` is untouched. Readers ignore
       ``.tmp`` files, and the next write of ``<f>`` overwrites it.
   * - Job killed while reading
     - Files untouched.
   * - Write of several tensors killed, or one rename fails, while the files are being renamed
     - Each tensor file is complete, but the group may mix old and new versions. This window is
       small: it opens only after all processes have agreed that every file was written.
