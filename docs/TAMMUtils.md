# Documentation of available TAMM utility routines (Work in progress)
---------------------------------------------------------------------

```c++
Tensor<T> result = scale(tensor, alpha);

//using labeled tensor
Tensor<T> result = scale(tensor(mu,nu), alpha); 
//result contains the full tensor with only the portion(slice)
//indicated by the labels in labeled tensor scaled.
```
User responsible for destroying `result`


## Routines that update tensors in-place
----------------------------------------
In-place tensor update routines have `_ip` suffix in their name.
Only `conj_ip` and `scale_ip` routines are available currently.

```c++
scale_ip(tensor,alpha);
scale_ip(tensor(mu,nu),alpha); //using labeled tensor
```

## ScaLAPACK related routines
------------------------------
`NOTE:` In the following text, a regular TAMM tensor refers to a tensor allocated using the default TAMM distribution (NW) scheme.

The following routine takes a regular TAMM tensor handle, a processor grid and block sizes for each dimension as 
arguments and returns a TAMM tensor with block cyclic distribution. Caller is responsible for deallocating the new tensor. 

```c++
auto block_cyclic_tamm_tensor = to_block_cyclic_tensor(regular_tensor,{3,2},{8,8}); 
``` 

The following routine takes a TAMM tensor with block cyclic distribution and copies the data into a regular TAMM tensor.
`regular_tensor` should be allocated before calling this routine.
```c++
from_block_cyclic_tensor(block_cyclic_tensor, regular_tensor);
```

The following routine returns a pointer to the local contiguous block cyclic buffer owned by the calling mpi process and the buffer size.
```c++
std::tuple<TensorType*,int64_t> access_local_block_cyclic_buffer(Tensor<TensorType> tensor) 
```

## Types of utility routines
- routines that return a scalar (U_S)
- routines that return a new tensor (U_NT)
- routines that update tensor in-place (U_IP)

## Process groups
 ### Scheduler operations
  - `C+=A*B` is always executed using `EC` of C.
  - `A` or `B` can either be distributed on a PG larger than `C`'s PG or replicated.
  - `C` can be either distributed on a PG smaller than `A` or `B`'s PG or replicated.
  - Entire contraction can be executed on a sub-group when `A,B,C` are allocated on world group.

 ### Utility routines
A call to a utility routine assumes that the routine is executed on the process group (PG) within an ExecutionContext (EC) on which the tensor is created. If not, two possibilties exist: 

- A `tensor` is created on a larger PG and the routine is executed on a smaller PG in which case the execution context representing the smaller PG
  must be passed to the routine. If not passed, this can result in MPI runtime errors and cannot be error-checked within TAMM.
  - Execute routine using (smaller) EC passed. 
  - `U_NT`: If smaller EC is passed, the new tensor returned is created using it.
  - `Todo`: Check if EC passed is actually a subset of larger EC, if not this is case 2 (ignore EC passed).
  - `Possible Soln`: maintain global context that tracks which PG/EC is currently active? 

- A tensor is created on a smaller PG and the routine is executed by a larger PG - new tensor returned always on smaller PG. Ignore optionally passed larger EC.
  - Execute routine on smaller PG extracted from tensor as usual.
   
  - It will work if the tensor is replicated per rank ie smaller PG=`MPI_Comm_SELF`
    - `U_NT`: return new replicated tensor created using smaller PG - already handled
     
  - If not replicated, for all other cases not possible to write such code with `group_incl`? - can write if `comm_split` is used
  ```
      ProcGroup gpg{GA_MPI_Comm()};

      //Create subcomm using group_incl 
      // ranks not in subcomm have MPI_COMM_NULL
      if (subcomm != MPI_COMM_NULL) {
          ProcGroup pg{subcomm};
          ...
          //Create tensor on smaller pg
          call util_name(tensor)
      }

      // if comm_split is used - all ranks have a valid comm, unless color=MPI_UNDEFINED which is eqv. to above code snippet.
        ProcGroup pg{subcomm};
        // create tensor using pg - ranks that dont need the tensor also create it
        call util_name(tensor) - this should already work?
  ```
    `Soln`: Don't worry since `group_incl` case is not possible to write and `comm_split` will always work?


