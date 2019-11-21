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
