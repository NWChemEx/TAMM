## Tensor Operations


### Tensor Copy (Shallow vs Deep)

TAMM uses "single program multiple data ([SPMD](https://en.wikipedia.org/wiki/SPMD))" model for distributed computation. In this programming abstraction, all nodes has its own portion of tensors available locally. So any operation on the whole tensors results in a message passing to remote portions on the tensor, with implied communication. More importantly, many/all operations are implied to be collective. This simplifies management of handles (handles are not migratable). However, this implies that operations such as memory allocation and tensor copy need to be done collectively. This conflicts with supporting deep copy when a tensor is passed by value, because this can lead to unexpected communication behavior such as deadlocks.

To avoid these issues, TAMM is designed to:

1. Handle tensors in terms of handles with shallow copy. 
2. Require operations on Tensors to be declared explicitly and executed using a scheduler. 

**NOTE:** This is distinguished from a rooted model in which a single process/rank can noncollectively perform a "global" operation (e.g., copy).

In summary, any assignment done on Tensor objects will be a **shallow copy** (internally it will be copying a shared pointer) as opposed to **deep copy** that will result in message passing between each node to do the copy operation:
```c++
Tensor<double> A{AO("occ"), AO("occ")};
Tensor<double> B{AO("occ"), AO("occ")};

A = B // will be a shallow copy as we will be copying a shared pointer
Tensor<double> C(B); // this is shallow copy as well as it will copy shared pointer internally

Scheduler()
	(A("i","k") = B("i","k"))	// deep copy using scheduler for informing remote nodes
	.execute();
```

To make Tensor operations explicit, TAMM is using parenthesis syntax as follows: 
```c++
Tensor<double> A{AO("occ"), AO("occ")};
Tensor<double> B{AO("occ"), AO("occ")};
Tensor<double> C{AO("occ"), AO("occ")};

Scheduler()
// Tensor assignment 
(A("i", "k") = B("i","k"))
// Tensor Addition 
(A("i", "k") += B("i","k"))
// Tensor Multiplication
(C("i","k") = A("i","k") * B("i","k"))
.execute();
```

Keep in mind that these operations will not be effective (there will be no evaluation) until they are scheduled using a scheduler. For actual evaluation of these operations, TAMM provides two options:

**Scheduling operations directly**
```c++
int main() {
	Scheduler()
		(A("i", "k") = B("i","k"))
		(A("i", "k") += B("i","k"))
		(C("i","k") = A("i","k") * B("i","k"))
		.execute();

	return 1;
}
```

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

### Tensor contraction operations

A Tensor operation in TAMM can only be in the single-op expressions of the form: 

```
C [+|-]?= [alpha *]? A [* B]?
```

#### Set operations
```
C = alpha
```

**Examples**:
```
C() = 0
```
  
#### Add operations
```
C [+|-]?= [alpha *]? A
```
**Examples**:
```c++
i1("h6", "p5") = f1("h6", "p5")
i0("p2", "h1") -= 0.5 * f1("p2", "h1")
i0("p3", "p4", "h1", "h2") += v2("p3", "p4", "h1", "h2")
```
#### Multiplication operations
```
C [+|-]?= [alpha *]? A * B 
```
**Examples**:
```c++
de() += t1("p5", "h6") * i1("h6", "p5")
i1("h6", "p5") -=  0.5  * t1("p3", "h4") * v2("h4", "h6", "p3", "p5")
t2("p1", "p2", "h3", "h4") =  0.5  * t1("p1", "h3") * t1("p2", "h4")
i0("p3", "p4", "h1", "h2") += 2.0 * t2("p5", "p6", "h1", "h2") * v2("p3", "p4", "p5", "p6")
```