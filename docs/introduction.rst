########################
Introduction to TAMM
########################

Tensor Algebra for Many-body Methods (TAMM) is a framework for productive and performance-portable development of scalable computational chemistry methods.
The TAMM framework decouples the specification of the computation and the execution of these operations on available high-performance computing systems. 
With this design choice, the scientific application developers (domain scientists) can focus on the algorithmic requirements using the tensor algebra interface 
provided by TAMM whereas high-performance computing developers can focus on various optimizations on the underlying constructs such as efficient data distribution, 
optimized scheduling algorithms, efficient use of intra-node resources (e.g., GPUs). The modular structure of TAMM allows it to be extended to support different 
hardware architectures and incorporate new algorithmic advances.
