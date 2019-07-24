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
