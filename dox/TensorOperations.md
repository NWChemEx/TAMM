## Tensor Operations

### Tensor Labeling

TAMM uses labeling for operating over `Tensor` objects, there are two different ways of labeling with different capabilities. 

#### Using `string` values
String based labeling is used mainly for basic operations where the operations expands over the full tensors:

```c++
// Assigning on full Tensors
(C("i","j") = B("i","j"));

// Tensor addition
(C("i","j") += B("i", "j"));

// Tensor multiplication
(C("i","j") = A("i","k") * B("k","j"));
```

#### Using `TiledIndexLabel` objects

For more complex operations over the slices of the full tensors or dependent indices, TAMM implements a `TiledIndexLabel` object. These labels are constructed over `TiledIndexSpace`s and can be used to accessing portions of the tiled space.

```c++
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
```

### Tensor Copy (Shallow vs Deep)

TAMM uses "single program multiple data ([SPMD](https://en.wikipedia.org/wiki/SPMD))" model for distributed computation. In this programming abstraction, all nodes has its own portion of tensors available locally. So any operation on the whole tensors results in a message passing to remote portions on the tensor, with implied communication. More importantly, many/all operations are implied to be collective. This simplifies management of handles (handles are not migratable). However, this implies that operations such as memory allocation and tensor copy need to be done collectively. This conflicts with supporting deep copy when a tensor is passed by value, because this can lead to unexpected communication behavior such as deadlocks.

To avoid these issues, TAMM is designed to:

1. Handle tensors in terms of handles with shallow copy. 
2. Require operations on Tensors to be declared explicitly and executed using a scheduler. 

**NOTE:** This is distinguished from a rooted model in which a single process/rank can non-collectively perform a "global" operation (e.g., copy).

In summary, any assignment done on Tensor objects will be a **shallow copy** (internally it will be copying a shared pointer) as opposed to **deep copy** that will result in message passing between each node to do the copy operation:
```c++
Tensor<double> A{AO("occ"), AO("occ")};
Tensor<double> B{AO("occ"), AO("occ")};

A = B;               // will be a shallow copy as we will be copying a shared pointer
Tensor<double> C(B); // this is shallow copy as well as it will copy shared pointer internally
auto ec = tamm::make_execution_context();

Scheduler(ec)
  (A("i","k") = B("i","k"))	// deep copy using scheduler for informing remote nodes
.execute();
```

To make Tensor operations explicit, TAMM is using parenthesis syntax as follows: 
```c++
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
```

Keep in mind that these operations will not be effective (there will be no evaluation) until they are scheduled using a scheduler. 

<!-- For actual evaluation of these operations, TAMM provides two options: -->
**Scheduling operations directly**
```c++
int main() {
  auto ec = tamm::make_execution_context();

  Scheduler(ec)
   (A("i", "k") = B("i","k"))
   (A("i", "k") += B("i","k"))
   (C("i","k") = A("i","k") * B("i","k"))
  .execute();
  
	return 1;
}
```
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
int main(){
	Tensor<double> A{AO("occ"), AO("occ")};
	Tensor<double> B{AO("occ"), AO("occ")};
	Tensor<double> C{AO("occ"), AO("occ")};
	
	auto sampleDAG = make_dag(sample_op, A, B, C);
	
	Scheduler::execute(sampleDAG);
	
	return 1;
}	
```
-->

### Tensor Contraction Operations

A Tensor operation in TAMM can only be in the single-op expressions of the form: 


`C [+|-]?= [alpha *]? A [* B]?`


#### Set operations

`C = alpha`


**Examples**:
```c++
(C() = 0.0)
```
  
#### Add operations

`C [+|-]?= [alpha *]? A`

**Examples**:
```c++
(i1("h6", "p5") = f1("h6", "p5"))
(i0("p2", "h1") -= 0.5 * f1("p2", "h1"))
(i0("p3", "p4", "h1", "h2") += v2("p3", "p4", "h1", "h2"))
```
#### Multiplication operations

`C [+|-]?= [alpha *]? A * B` 

**Examples**:
```c++
(de() += t1("p5", "h6") * i1("h6", "p5"))
(i1("h6", "p5") -=  0.5  * t1("p3", "h4") * v2("h4", "h6", "p3", "p5"))
(t2("p1", "p2", "h3", "h4") =  0.5  * t1("p1", "h3") * t1("p2", "h4"))
(i0("p3", "p4", "h1", "h2") += 2.0 * t2("p5", "p6", "h1", "h2") * v2("p3", "p4", "p5", "p6"))
```

### Tensor Utility Methods

As tensors are the main construct for the computation, TAMM provides a set of utility functionality. These are basically tensor-wise update and access methods as well as point-wise operations over each element in the tensor. All these methods 

#### Updating using lambda functions
TAMM provides two methods for updating the full tensors or a slice of a tensor using a lambda method:
- **`update_tensor(...)`:** used for updating the tensor using a lambda method where the values are not dependent on the current values. 
  
  ```c++
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
  tamm::update_tensor(A(), lambda);
  ```

- **`update_tensor_general(...)`:** only difference from `update_tensor(...)` method is in this case lambda method can use the current values from the tensor.

  ```c++
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

  // template<typename T, typename Func>
  // update_tensor_general(LabeledTensor<T> lt, Func lambda)

  // updates each element of the tensor with a computation (CholVpr)
  tamm::update_tensor_general(B(), lambda_general);
  ```

#### Accessing tensors 

TAMM also provides utility methods for specialized accessor for specific types of tensors:

- **`get_scalar(...)`:** Special accessor for scalar values (i.e. zero dimensional tensors).
  ```c++
  // template <typename T> 
  // get_scalar(Tensor<T>& tensor)

  // Get the element value for a zero dimensional tensor A
  auto el_value = tamm::get_scalar(A);
  ```
- **`trace(...)`:** Utility method for getting the sum of the diagonal in two dimensional square tensors.
  ```c++
  // template <typename TensorType>
  // TensorType trace(ExecutionContext &ec, LabeledTensor<TensorType> ltensor)

  // construct a default execution context which will be used
  // to allocate and operate on the tensor
  auto ec = tamm::make_execution_context();
  
  // ...
  
  // get the diagonal sum of the two dimensional square tensor A(N, N)
  auto trace_A = tamm::trace(ec, A());
  ```

- **`diagonal(...)`:** Utility method for getting the values at the diagonal of a two dimensional square tensor.
  ```c++
  // template <typename TensorType>
  // std::vector<TensorType> diagonal(ExecutionContext &ec, LabeledTensor<TensorType> ltensor)

  // construct a default execution context which will be used
  // to allocate and operate on the tensor
  auto ec = tamm::make_execution_context();
  
  // ...

  // get the diagonal values of two dimensional tensor A(N,N)
  auto diagonal_A = tamm::trace(ec, A());
  ```

- **`max_element(...)`& `min_element(...)`:** Utility method to **collectively** find the maximum/minimum element in a tensor. This method returns the maximum/minimum value along with block id the value found and the corresponding sizes (for each dimension) of the block.
  ```c++
  // template<typename TensorType>
  // std::tuple<TensorType, IndexVector, std::vector<size_t>> max_element(ExecutionContext &ec, LabeledTensor<TensorType> ltensor)
  
  // construct a default execution context which will be used
  // to allocate and operate on the tensor
  auto ec = tamm::make_execution_context();

  // ...

  // get the max element in a tensor 
  auto [max_el, max_blockid, max_block_sizes] = tamm::max_element(ec, A());

  // template<typename TensorType>
  // std::tuple<TensorType, IndexVector, std::vector<size_t>> min_element(ExecutionContext &ec, LabeledTensor<TensorType> ltensor)
  auto [min_el, min_blockid, min_block_sizes] = tamm::min_element(ec, A());
  ```

#### Point-wise operations

Different then block-wise and general operations on the tensors, TAMM provides point-wise operations that can be applied to the whole tensor. As tensors are distributed over different MPI ranks, these operations are collective.

- **`square(...)`** updates each element in an tensor to its square value
- **`log10(...)`** updates each element in a tensor to its logarithmic 
- **`inverse(...)`** updates each element in a tensor to its inverse 
- **`pow(...)`** updates each element in a tensor to its `n`-th power
- **`scale(...)`** updates each element in a tensor by a scale factor `alpha`



