Tensor I/O
==========

TAMM writes and reads distributed tensors to and from disk in parallel using HDF5. Each tensor is
stored in its own file, called a *tensor file*.

Routines
--------

- ``tamm::write_to_disk(A, "filename")`` writes a distributed tensor ``A`` to a tensor file in
  parallel.
- ``tamm::read_from_disk(A, "filename")`` reads a distributed tensor ``A`` from a tensor file in
  parallel.
- ``tamm::write_to_disk_group(ec, tensor_list, filename_list)`` and
  ``tamm::read_from_disk_group(ec, tensor_list, filename_list)`` write and read a batch of
  distributed tensors concurrently over different process groups.

.. code:: cpp

    tamm::write_to_disk(A, "tensor_A.h5");  // writes tensor A to tensor_A.h5
    tamm::read_from_disk(B, "tensor_B.h5"); // reads tensor B from tensor_B.h5

    tamm::write_to_disk_group(ec, {T1, T2}, {"t1.h5", "t2.h5"});
    tamm::read_from_disk_group(ec, {T1, T2}, {"t1.h5", "t2.h5"});

Safe writes
-----------

A tensor file is never left partially written. Each write goes to a *staging file*,
``<filename>.tmp``, which is renamed to ``<filename>`` only after every process has written its
part successfully. This rename is the *commit*. As a result, a tensor file at its final name is
always complete: it holds either the previous version or the new one.

This makes tensor files safe to use as checkpoints. If a job is killed while writing a tensor to disk, 
the previous tensor file is untouched and can still be read.

``write_to_disk_group`` commits **all-or-nothing**: if any tensor file in the group fails to
write, none of them is committed and every tensor file keeps its previous version. Tensor files
that must stay consistent with each other (for example, several tensors from the same iteration
of a solver) should therefore be written together with ``write_to_disk_group`` rather than with
separate ``write_to_disk`` calls, each of which commits on its own.

Safe reads
----------

A read never sees a partially written tensor file. Reads open only the tensor file at its final
name and ignore staging files, and because of the commit that file is always complete. Reads never
modify files on disk.

A read never silently loads bad data. Every process checks each step of the read (opening the
file and reading its data). If any process fails, for example because the file is missing, is not
an HDF5 file, is truncated, or the filesystem returns an error, all processes agree on the failure
and the program terminates. The cause of the failure is reported for each tensor file, followed by
a final message naming the tensor file; ``read_from_disk_group`` lists every tensor file that
could not be read in that final message.

A tensor file stores the tensor's data block by block in the order given by its tiling. To make
sure a file is read into a tensor with the same layout, each tensor file also stores a *tensor
description*: the element type, the size of each tile along each mode, and the number of non-zero
blocks with a hash identifying them. Before reading any data, the description is compared with
the tensor being read into. A tensor file without a description, or with a different one, is a
read failure, for example::

    [TAMM ERROR] t.h5: Tiling differs in dimension 1: file has 8 tiles, tensor has 6 tiles

To read a tensor file, use a tensor with the same index spaces and tiling as the tensor that was
written.

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
     - Writes ``<f>.tmp``; the writing processes agree; the root renames ``<f>.tmp`` to ``<f>``
     - ``<f>.tmp`` is deleted; the previous ``<f>`` is kept, if one exists
     - No; the calculation continues
     - ``<call> failed for <f>: <cause>``, then ``<f> not committed; previous version kept``
       (or ``no previous version exists``)
   * - ``write_to_disk_group``
     - Writes each ``<fi>.tmp``; all processes agree; each process group root renames its own
       files
     - All-or-nothing: if any file fails, every ``.tmp`` is deleted and every previous ``<fi>`` is
       kept, where one exists
     - No; the calculation continues
     - ``<call> failed for <fi>: <cause>`` for each failed file, then one summary listing the
       files that failed, those whose previous version was kept, and those with no previous
       version, one file per line
   * - ``read_from_disk``
     - Tensor filled
     - Files untouched; tensor contents undefined
     - **Yes**
     - ``<call> failed for <f>: <cause>``, then ``read_from_disk: failed to read tensor file: <f>``
   * - ``read_from_disk_group``
     - All tensors filled
     - Files untouched; tensor contents undefined
     - **Yes**
     - ``<call> failed for <fi>: <cause>`` for each failed file, then one final message listing
       every tensor file that could not be read, one file per line

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
     - Root of the writing process group, once
     - ``[TAMM WARNING] <call> failed for <f>: <cause>``
   * - Writing data fails on one process
     - The process that failed
     - ``[TAMM WARNING] H5Dwrite failed for <f>: <cause>``
   * - The rename fails
     - The process that attempted it
     - ``[TAMM WARNING] rename <f>.tmp -> <f> failed: <reason>``
   * - A collective HDF5 call fails while reading (opening, closing)
     - Root of the reading process group, once
     - ``[TAMM ERROR] <call> failed for <f>: <cause>``
   * - The tensor file's description does not match the tensor
     - Root of the reading process group, once
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
   * - Group write killed, or one rename fails, while the files are being renamed
     - Each tensor file is complete, but the group may mix old and new versions. This window is
       small: it opens only after all processes have agreed that every file was written.
