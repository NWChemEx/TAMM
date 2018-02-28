# IndexSpace and Tiling Specification

## Notation

- **IndexRange** is list of values, $ IR = p_0,p_1...p_n \mid \forall i \in \{0,1...n\}, p_i \in \mathbb{Z}^+ \cup \{0\}$ and $p_i \lt p_{i+1}$. 
- **IndexSpace** is list of **IndexRange** values, $ IS  = IR_0, IR_1,...,IR_n \mid \forall i, j, i \neq j, IR_i \cap IR_j = \emptyset$. 
- **SubIndexSpace** of an **IndexSpace**, $Sub(IS)$, is list of subsets of **IndexRange**s constructing **IS**, $Sub(IS) = Sub(IR_0), Sub(IR_1), ...,Sub(IR_n) \mid \forall i, Sub(IR_i) \subseteq IR_i$. 
- **SubIndexSpace** is an **IndexSpace**. 
- **Tiling** of an **IndexRange**, IR, is a partition of the list of values in IR 
- **Tiling** of an **IndexSpace**, $Tiling(IS)$, is a list of tiled version of **IndexRange**s, $Tiling(IS) = Tiling(IR_0),Tiling(IR_1),...,Tiling(IR_n)$. 
- **Tiling** of an **IndexSpace**, IS, doesn't imply the tiling of a **SubIndexSpace** of IS.
v
----

## TAMM Interface Procedures
1. Building **IndexRange**s from set of values.
    ```c++
    IndexRange ir0 = {1, 2, 3, 4};
    IndexRange ir1 = {5, 6, 7, 8};

    IndexRange s_ir0 = {1, 2};
    IndexRange s_ir1 = {5, 6};
    ```
2. Building **IndexSpace** and **SubIndexSpace** from given index ranges.
    ```c++
    IndexSpace IS0(ir0,ir1);

    SubIndexSpace S_IS0(IS0, s_ir0, s_ir1);
    ```
3. Defining a **Tiling** structure and **TiledIndexSpace**.
    ```c++
    Tile tiling;
    Tile o_tiling;

    TiledIndexSpace T_IS0(IS0, tiling);
    TiledIndexSpace T_SIS0(S_IS0, tiling)

    TiledIndexSpace To_IS0(IS0, o_tiling);
    TiledIndexSpace To_SIS0(S_IS0, o_tiling)
    ```
5. Generating **Tensor**s over different **IndexSpace** constructs
    ```c++
    // Tensor construction with IndexSpace/SubIndexSpace
    Tensor A(IS0);
    Tensor B(S_IS0);

    // Tensor construction with TiledIndexSpaces
    Tensor T_A(T_IS0);
    Tensor T_B(T_SIS0);

    Tensor To_A(To_IS0);
    Tensor To_B(To_SIS0);
    ```
6. Assignments between **Tensor**s.
    - Tensor assignment between compatible **IndexSpace**, the assignment will check if the index ranges are compatible by checking their **IndexSpace**s
        ```c++
        // Valid assignment as B is composed of index ranges that
        // are subset of A's index ranges.
        A(s_ir0) = B(s_ir0);

        // Invalid assignment as B doesn't include corresponding 
        // index range
        B(ir0) = A(ir0);
        ```
    - Tensor assignment between different index ranges, the assignment will validate by checking if the index ranges used in the assignment is equalivalent.
        ```c++
        // Invalid assignment as the index ranges differs 
        A(s_ir0) = B(s_ir1);

        // Users are given the option to translate correponding
        // index ranges to validate the assignment
        A(s_ir0) = translate(s_ir0, s_ir1) * B(s_ir1);
        // Another option with operator overloading
        A(s_ir0) = translate(s_ir0 << s_ir1) * B(s_ir1);
        ```
    - Tensor assignment between tiled index ranges, the assignment validate if the tiling on the index ranges are the same. 
        ```c++
        // Although the index ranges are compatible in both of tensors, 
        // the tiling is different so this will be invalid
        T_A(ir0) = To_A(ir0);

        // Assignment of the tiled sub-index spaces to tiled index spaces 
        // is also invalid, as the tiling applied on different index ranges
        T_A(s_ir0) = T_B(s_ir0);
        ```