# Index Spaces in TAMM

## Terminology

### Index
`Index` is an integral type to index into an array/tensor. Typically, an `int` or a typed version thereof.

### IndexSpace
- An index spaces maps the values (referred to as indices) in an integer interval to a collection of indices.

  ```c++
  // Construct an index space spanning from 0 to N-1 => [0,N)
  // By giving a count - IndexSpace(range r)
  IndexSpace is1{range(10)};                // indices => {0,1,2,3,4,5,6,7,8,9}

  // Range based constructor with named sub-spaces 
  IndexSpace is2{range(10),                 // is2("all)    => {0,1,2,3,4,5,6,7,8,9}
                {{"occ", {range(0,5)}},     // is2("occ")   => {0,1,2,3,4}
                {"virt", {range(5,10)}}}};  // is2("virt")  => {5,6,7,8,9}

  // By specifying the indices it represents - IndexSpace(std::initializer_list<Index> list)
  IndexSpace is3{0,1,2,3,4}               // indices => {0,1,2,3,4}

  // By giving a range - IndexSpace(range r)
  IndexSpace is4{range(0, 10)};           // indices => {0,1,2,3,4,5,6,7,8,9}
  IndexSpace is5{range(5, 10)};           // indices => {5,6,7,8,9}
  ```

- An index space can be queried to get a point at an index, or the index for a point. [Note the discussion below about non-disjoint index spaces and getting an index for a point]
  ```c++
  // Reference IndexSpace objects
  IndexSpace is4{range(0, 10)};           // indices => {0,1,2,3,4,5,6,7,8,9}
  IndexSpace is5{range(5, 10)};           // indices => {5,6,7,8,9}

  // Get the index value from an index space
  // By using point method - Index IndexSpace::index(Index i)
  Index i = is4.index(Index{4}); // index i => 4

  // By using operator[] - Index IndexSpace::operator[](Index i)
  Index j = is5[Index{4}];       // index j => 9
  ```

- An index space can be constructed by aggregating/concatenating other index spaces. In this case, the index spaces maps an interval [0,N-1], where N is the sum of sizes of all aggregated index spaces, to the points in the index spaces aggregated.
  ```c++
  // Reference IndexSpace objects
  IndexSpace is3{0,1,2,3,4}               // indices => {0,1,2,3,4}
  IndexSpace is5{range(5, 10)};           // indices => {5,6,7,8,9}

  // Constructing index spaces from other index spaces 
  // IndexSpace(std::vector<IndexSpace> index_spaces)
  IndexSpace is6{{is3, is5}};   // indices => {0,1,2,3,4,5,6,7,8,9}
  IndexSpace is7{{is5, is3}};   // indices => {5,6,7,8,9,0,1,2,3,4}

  // IndexSpace aggregation with named sub-spaces
  IndexSpace is9{{is3, is5},        // is9("all") => {0,1,2,3,4,5,6,7,8,9}
                  {"occ", "virt"}   // is9("occ") => {0,1,2,3,4}
                };                  // is9("virt")=> {5,6,7,8,9}
  ```

- The aggregation might be disjoint or non-disjoint. The same point might appear multiple times in non-disjoint index space. Some operations might not be defined on such an index space (example get the index for a particular point in the index space).
  ```c++
  // Reference IndexSpace objects
  IndexSpace is3{0,1,2,3,4}      // indices => {0,1,2,3,4}
  IndexSpace is5{range(5, 10)};  // indices => {5,6,7,8,9}

  // Disjoint aggregation
  IndexSpace is9{{is3, is5}};    // indices => {0,1,2,3,4,5,6,7,8,9}
  // Non-disjoint aggregation
  IndexSpace is10{{is3, is3}};   // indices => {0,1,2,3,4,0,1,2,3,4}
  ```

- **Sub-Space:** An index space can be constructed from a permuted sub-list of indices in another index space (referred to as the parent index space). In this case, the domain is [0,N-1], where N is the size of the sub-list, and the points are the points in the parent index space. 
  ```c++
  // Reference IndexSpace object
  IndexSpace is1{range(10)};                  // indices => {0,1,2,3,4,5,6,7,8,9}                  
  
  // Sub-space by permuting the indices of another index space
  // By specifying sub-space with range
  IndexSpace is11{is1, range(0, 4)};           // indices => {0,1,2,3}

  // By specifying range using the reference index space
  IndexSpace is12{is1, range(5, is1.size())}; // indices => {4,5,6,7,8,9} 

  // Constructing from the full index space
  IndexSpace is13{is1};                       // indices => {0,1,2,3,4,5,6,7,8,9}

  // Sub-index space construction with name sub-spaces
  IndexSpace is14{is1, range(0,10,2),         // indices     => {0,2,4,6,8}
                  {{"occ", {range(0,3)}},     // is14("occ") => {0,2,4} 
                  {"virt", {range(3,5)}}}};   // is14("virt")=> {6,8}
  ```
----

**NOTE:** An index space is treated as a read-only object after it is constructed.

### IndexSpace Specialization
- **Attributes:** An index space might partition its indices into groups, each of which is associated with a set of attributes. All indices in a group have the same attribute values. Attribute specification is part of the constructor.
  ```c++
  // Index space constructor with spin specialization
  // Combine index spaces with different spin attributes
  IndexSpace is15{range(100),                                 // is15("all")   => {0,...,99}
                  {{"occ",   {range(0,50)}},                  // is15("occ")   => {0,...,49}
                   {"virt",  {range(50,100)}},                // is15("virt")  => {50,...,99}
                   {"alpha", {range(0,25), range(50,75)}}     // is15("alpha") => {0,...25,50,...,74}
                   {"beta",  {range(25,50), range(75,100)}}}, // is15("beta")  => {25,...,49,75,...,100}
                  {{Spin{1}, {range(0,25), range(50,75)}},
                   {Spin{2}, {range(25,50), range(75,100)}}}};
  ```
- **Aggregation:** An index space might be constructed from other index spaces and can be partitioned using the available partitions in the existing index spaces. 
  ```c++
    // Index space construction (will be used for aggregation)
    IndexSpace is16{range(100,200),                                // is16("all")   => {100,...,199}             
                   {{"occ",   {range(100,140)}},                  // is16("occ")   => {100,...,139}
                    {"virt",  {range(140,200)}},                  // is16("virt")  => {140,...,199}
                    {"alpha", {range(100,125), range(150,175)}},  // is16("alpha") => {100,...,124,150,...,175}
                    {"beta",  {range(125,150), range(175,200)}}}, // is16("beta")  => {125,...,149,175,...,199}
                   {{Spin{1}, {range(100,125), range(150,175)}}, 
                    {Spin{2}, {range(125,150), range(175,200)}}}}};

    // Construction of aggregated index space with subspace names
    IndexSpace is17{{is15, is16},
                    {"first", "second"},
                    {{"occ",  {"first:occ", "second:occ"}},
                     {"virt", {"first:virt", "second:virt"}},
                     {"alpha",{"first:alpha", "second:alpha"}},
                     {"beta", {"first:beta", "second:beta"}}}
                    };
    // is17("occ")   => is1 ~ {0,...,49,100,...,139}
    // is17("virt")  => is2 ~ {50,...,99,140,...,199}
    // is17("alpha") => is15("alpha") + is16("alpha") ~ {25,...,49,75,...,99,125,...,149,175,...,199}
    // is17("beta")  => is15("beta") + is16("beta") ~ {0,...,24,50,...,74,100,...,124,...,150,...,174}
  ```
- **TiledIndexSpace:** A tiled index space segments an index space. Specifically, it maps values (referred to as tile indices) in an integer to a index interval. A valid tiling ensures that all indices in a tile have the same attribute values.
  - **[Default tiling]** A TiledIndexSpace can be constructed from any IndexSpace where all tiles are of size 1.
    ```c++
    // Reference IndexSpace 
    IndexSpace is1{range(10)};    // indices = {0,1,2,3,4,5,6,7,8,9}

    // Constructing tiled index spaces - TiledIndexSpace(IndexSpace& is, size_t tile_size = 1)
    // Construction with default tiling size
    TiledIndexSpace tis1{is1};    // tiles = [0,1,2,3,4,5,6,7,8,9] && tile_size = 1

    // Construction with specific tiling size
    TiledIndexSpace tis2{is1, /*blocked tile size of 5*/ 5}; // indices = [{0,1,2,3,4},{5,6,7,8,9}] && tile_size = 5
    ```
  - **[Specialized tiling]** A TiledIndexSpace can be constructed using a single tile size or a set of tile sizes which tiles the underlying IndexSpace completely (no gaps).
  **NOTE:** User provided set of tile sizes should also consider the named sub-spaces and any attributes related to input IndexSpace
    ```c++
    // TiledIndexSpace construction with single tile size
    TiledIndexSpace tis1{is1, 4};       // tiles = [{0,1,2,3}, {4,5,6,7}, {8,9}] && tile_size = 4

    // TiledIndexSpace construction with a set of tile sizes 
    TiledIndexSpace tis2{is1, {2,5,3}}  // tiles = [{0,1}, {2,3,4,5,6}, {7,8,9}]  && tile_sizes = [2,5,3]
    ```
  - **[Sub-space]** A TiledIndexSpace can be a constructed from another TiledIndexSpace by choosing a permuted sub-list of tiles in the parent TiledIndexSpace.
**NOTE:** Tiling of a sub-index space is not the same the sub-space of a TiledIndexSpace.

    ```c++
    // Constructing tiled sub-spaces from tiled index spaces
    // TiledIndexSpace(TiledIndexSpace& ref, range r)
    TiledIndexSpace tis3{tis1, range(0,5)};  // tiles = [0,1,2,3,4] && tile_size = 1

    // By specifying range 
    TiledIndexSpace tis4{tis2, range(1, tis2.num_tiles())} ;    // indices = [{5,6,7,8,9}] && tile_size = 5
    ```

  - **[Convenience tiled index sub-spaces]** A TiledIndexSpace stores and returns commonly used sub-spaces of that TiledIndexSpace. 
  **NOTE:** An index space can be queried to be an index sub-space or an index range.

    ```c++
    // Reference IndexSpace 
    IndexSpace is2{range(10),                   // is2("all)    => {0,1,2,3,4,5,6,7,8,9}
                   {{"occ", {range(0,5)}},      // is2("occ")   => {0,1,2,3,4}
                    {"virt", {range(5,10)}}}};  // is2("virt")  => {5,6,7,8,9}

    // Apply default tiling
    TiledIndexSpace tis_mo{is2};  

    // Get a specific sub-space by identifier
    TiledIndexSpace& O = tis_mo("occ");   
    TiledIndexSpace& V = tis_mo("virt");
    
    // Identifier "all" will implicitly return itself
    TiledIndexSpace& N = tis_mo("all"); 
    ```

- **Dependent index space** An index space can depend on other tiled index spaces. In this case, the index space becomes a relation that, given a specific value of its dependent index spaces, returns an index space.
  ```c++
  // Creating index spaces MO, AO, and Atom
  IndexSpace MO{range(0, 100),
                {{"occ", range(0, 50)},
                {"virt", range(50, 100)}}};
  IndexSpace AO{range(100,200)};
  IndexSpace Atom{range(0, 5)};

  // Construct dependency relation for Atom indices
  std::map<IndexVector, IndexSpace> dep_relation{
        {IndexVector{0}, MO("occ")},                   
        {IndexVector{1}, MO("virt")},
        {IndexVector{2}, AO},
        {IndexVector{3}, MO("all")},
        {IndexVector{4}, IndexSpace{AO, range(0, 40)}}
  };

  // Tile Atom space with default tiling
  TiledIndexSpace T_Atom{Atom};

  // DependentIndexSpace(const std::vector<TiledIndexSpace>& dep_spaces,
  //                     const std::map<IndexVector, IndexSpace> dep_relation)
  DependentIndexSpace subMO_atom{{T_Atom}, dep_relation};
  ```

- **[Tiling a DependentIndex]** If input IndexSpace to a TiledIndexSpace is a dependent IndexSpace, the tiling spans over the dependency relation. While constructing a sub-TiledIndexSpace from tiled dependent index space, users will have to construct the new dependency out of the tiled dependency
  ```c++
  // Tiling dependent IndexSpaces 
  TiledIndexSpace dep_tis{subMO_atom, 5};

  // Dependency map from TiledIndexSpace
  const std::map<IndexVector, TiledIndexSpace>& t_dep_relation = dep_tis.tiled_dep_map();

  // New sub dependency relation 
  std::map<IndexVector, TiledIndexSpace> sub_relation{ 
    {IndexVector{0}, TiledIndexSpace{t_dep_relation[IndexVector{0}], range(1)}},
    {IndexVector{3}, TiledIndexSpace{t_dep_relation[IndexVector{3}], range(1,2)}}
  };

  // Constructing a sub-TiledIndexSpace
  // Internally the new sub relation will checked for compatibility with the 
  // reference dependency relation in the parent TiledIndexSpace
  TiledIndexSpace sub_dep_tis{dep_tis, sub_relation};

  ```

- **TiledIndexLabel** A TiledIndexLabel pairs a TiledIndexSpace with an integer label. 
  ```c++
      // Generate TiledIndexLabels for a specific sub-space
      TiledIndexLabel i, j, k, l, m, n;
      std::tie(i,j) = tis_mo.labels<2>("occ");
      std::tie(k,l) = tis_mo.labels<2>("virt");
      m = tis_mo.label("all", 9);
      n = tis_mo.label("all", 10);
  ```
-
---

### Tensor Construction

- **[Tensor dimensions]** Tensors are created over a multi-dimensional space, where each dimension is defined using a TiledIndexSpace.
If a tensor dimension over a tiled index space `p` that depends on another tiled index space `q`, some other dimension of the tensor should be defined over the tiled index space `q`.
```c++
// Creating a tensor with different TiledIndexSpaces
TiledIndexSpace t10_MO{subMO_atom, /*tiled with tile size of 10*/ 10};
TiledIndexSpace t20_AO{AO, /*tiled with tile size of 20*/ 20};
Tensor T1{t10_MO, t20_AO}; //2-d MOxAO tensor
TiledIndexLabel q = is_atom.labels("all", 4);
TiledIndexLabel p = is_mo_atom.labels("all", 0);
Tensor T2{p(q), q}; //2-d tensor where the first dimension is the the list of AOs relevant to each atom
```

- **[Labeling a tensor dimension]** A tensor dimension can be labeled by an index label that is over the same space or a sub-space as that dimension. If a label is defined over a dependent (tiled) index space, its use in a operation should bind that label with the label it depends on. 
  ```c++

  ```
---

### Examples from Comments on Documentation
> 1. scalar

```c++
// Construct a scalar value 
Tensor T_1{};
```

> 2. vector of say length 10

```c++
// Create an index space of length 10
IndexSpace is_2{range(10)};
// Apply default tiling
TiledIndexSpace tis_2{is_2};
// Create a vector with index space is_2
Tensor T_2{tis_2};
```

> 3. matrix that is say 10 by 20

```c++
// Create an index space of length 10 and 20
IndexSpace is1_3{range(10)};
IndexSpace is2_3{range(20)};
// Apply default tiling
TiledIndexSpace tis1_3{is1_3}, tis2_3{is2_3};
// Create a matrix on tiled index spaces tis1_3, tis2_3
Tensor T_3{tis1_3, tis2_3};
```

> 4. order 3 tensor that is say 10 by 20 by 30

```c++
// Create an index space of length 10, 20 and 30
IndexSpace is1_4{range(10)};
IndexSpace is2_4{range(20)};
IndexSpace is3_4{range(30)};
// Apply default tiling
TiledIndexSpace tis1_4{is1_4}, tis2_4{is2_4}, tis3_4{is3_4};
// Construct order 3 tensor in tiled index spaces tis1_4, tis2_4 and tis3_4
Tensor T_4{tis1_4, tis2_4, tis3_4};
```

> 5. vector from 2 with subspaces of length 4 and 6

```c++
// Spliting is_2 into two sub-spaces with 4 and 6 elements
IndexSpace is1_5{is_2, range(0, 4)};
IndexSpace is2_5{is_2, range(4, is_2.size())};
// Create index space combining sub-spaces
IndexSpace is3_5{{is1_5, is2_5}};
// Apply default tiling 
TiledIndexSpace tis_5{is3_5};
// Create a vector over combined index space
Tensor T_5{tis1_5};
```

> 6. matrix from 3 whose rows are split into two subspaces of length 4 and 6

- **Do you mean this?**

```c++
// Spliting is1_3 from 3 into two sub-spaces with 4 and 6 elements
IndexSpace is1_6{is1_3, range(0, 4)};  
IndexSpace is2_6{is1_3, range(4, is1_3.size()}; 
// Create index space combining sub-spaces
IndexSpace is3_6{{is1_6, is2_6}};
// Apply default tiling
TiledIndexSpace tis_6{is3_6};
// Create a matrix with rows on combined tiled index space
// columns on tis2_3 from 3
Tensor T_6{tis_6, tis2_3};
```
- **In general, split cannot be post-facto. They need to be specified during construction. Or we end up with a new index space.**

> 7. matrix from 3 whose columns are split into two subspaces of lengths 12 and 8

```c++
// Spliting is2_3 from 3 into two sub-spaces with 12 and 8 elements
IndexSpace is1_7{is2_3, range(0, 12)};  
IndexSpace is2_7{is2_3, range(12, is2_3.size())}; 
// Create index space combining sub-spaces
IndexSpace is3_7{{is1_7, is2_7}};
// Apply default tiling
TiledIndexSpace tis_7{is3_7};
// Create a matrix with rows on tis1_3 from 3
// columns on combined tiled index space
Tensor T_7{tis1_3, tis_7};
```

> 8. matrix from 3 having subspaces of both 6 and 7

```c++
// Create matrix on tis_6 from 6 and tis_7 from 7
Tensor T_8{tis_6, tis_7};
```

> 9. tensor with mode 0 split into subspaces of 4 and 6

```c++
// Create order 3 tensor using split version from 5
// and full spaces from 4
Tensor T_9{tis_5, tis2_4, tis3_4};
```
> 10. tensor with mode 1 split into subspaces of 12 and 8

```c++
// Create order 3 tensor using split version from 7
// and full spaces from 4
Tensor T_10{tis1_4, tis_7, tis3_4};
```
> 11. tensor with mode 2 split into subspaces of 13 and 17

```c++
// Split the index space form 4 into sub-spaces of length 13 and 17
IndexSpace is1_11{is3_4, range(0, 13)};
IndexSpace is2_11{is3_4, range(13, is3_4.size())};
// Combine the sub-spaces into another index space
IndexSpace is3_11{{is1_11, is2_11}};
// Apply default tiling
TiledIndexSpace tis_11{is3_11};
// Create order 3 tensor using new split version
// and full spaces from 4
Tensor T_11{tis1_4, tis2_4, tis_11};
```
> 12. Combine 9 and 10

```c++
// Create order 3 tensor using splits from 9 and 10
// tis_5  --> split length 4 and 6
// tis_7  --> split length 12 and 8
// tis3_4 --> length 30 index space
Tensor T12{tis_5,tis_7,tis3_4};
```
> 13. Combine 9 and 11

```c++
// Create order 3 tensor using splits from 9 and 11
// tis_5  --> split length 4 and 6
// tis2_4 --> length 20 index space
// tis_11 --> split length 13 and 17
Tensor T13{tis_5,tis2_4,tis_11};
```
> 14. Combine 10 and 11

```c++
// Create order 3 tensor using splits from 9 and 11
// tis1_4 --> length 10 index space
// tis_7  --> split length 12 and 8
// tis_11 --> split length 13 and 17
Tensor T14{tis1_4,tis_7,tis_11};
```
> 15. Combine 9, 10, and 11

```c++
// Create order 3 tensor using splits from 9 and 11
// tis_5  --> split length 4 and 6
// tis_7  --> split length 12 and 8
// tis_11 --> split length 13 and 17
Tensor T15{tis_5,tis_7,tis_11};
```
> 16. Vector from 2 with the first subspace split again into a subspaces of size 1 and 3

```c++
// Split the sub-space from 5 into another with size 1 and 3
// is1_5  --> split of size 4
// is2_5  --> split of size 6
IndexSpace is1_16{is1_5, range(0,1)};
IndexSpace is2_16{is1_5, range(1,3)};
// Combine all into a full space
IndexSpace is3_16{{is1_16, is2_16, is2_5}};
// Apply default tiling
TiledIndexSpace tis_16{is3_16};
// Create a vector over new tiled index space
Tensor T16{tis_16};
```
> 17. matrix from 8 with the 4 by 12 subspace split further into a 1 by 12 and a 3 by 12

```c++
// Create a matrix from splits from 16 and 7 
// tis_16 --> split of size 1, 3 and 6
// tis_7  --> split of size 12 and 8
Tensor T17{tis_16, tis_7};
```

> 18. vector from 1 where odd numbered elements are in one space and even numbered elements are in another
  
```c++
// Odd numbered elements from 1 to 9
IndexSpace is1_18{range(1,10,2)};
// Even numbered elements from 0 to 8
IndexSpace is2_18{range(0,10,2)};
// Aggregate odd and even numbered index spaces 
IndexSpace is3_18{{is1_18, is2_18}};
// Apply default tiling
TiledIndexSpace tis3_18{is3_18};
// Create a vector with tiled index space
Tensor T18{tis3_18};
```
> 19. matrix from 2 where odd rows are in one space even in another

```c++
// Odd numbered elements from 1 to 9
IndexSpace is1_19{range(1,10,2)};
// Even numbered elements from 0 to 8
IndexSpace is2_19{range(0,10,2)};
// Aggregate odd and even numbered index spaces 
IndexSpace is3_19{{is1_19, is2_19}};
// Apply default tiling
TiledIndexSpace tis1_19{is3_19};
// Create a matrix using tiled index space with odd and even numbered
// elements as the row and tiled index space from 3 a columns
Tensor T19{tis1_19, tis2_3};
```
> 20. matrix from 6 that also has the odd rows in one space and the even in another

```c++
// Odd numbered elements from 1 to 9
IndexSpace is1_20{range(1,10,2)};
// Even numbered elements from 0 to 8
IndexSpace is2_20{range(0,10,2)};
// Aggregate odd and even numbered index spaces 
IndexSpace is3_20{is1_20, is2_20};
// Spliting is3_20 into two sub-spaces with 4 and 6 elements
IndexSpace is4_20{is3_20, range(0, 4)};  
IndexSpace is5_20{is3_20, range(4, is3_20.size())};
// Aggregate split indexes
IndexSpace is6_20{is4_20, is5_20};
// Apply default tiling
TiledIndexSpace tis1_20{is6_20};
// Create a matrix using tiled index space with odd and even numbered
// elements then splitted as the row and tiled index space 
// from 3 a columns
Tensor T20{tis1_20, tis2_3};
```

### Canonical CCSD E

```c++
// Up-to-date version can be found at ccsd/ccsd_driver.cc
template<typename T>
void ccsd_e(const TiledIndexSpace& MO, 
		    Tensor<T>& de,
		    const Tensor<T>& t1,
		    const Tensor<T>& t2,
		    const Tensor<T>& f1,
		    const Tensor<T>& v2) {
    
	const TiledIndexSpace& O = MO("occ");
	const TiledIndexSpace& V = MO("virt");
	Tensor<T> i1{O, V};

	TiledIndexLabel p1, p2, p3, p4, p5;
	TiledIndexLabel h3, h4, h5, h6;

	std::tie(p1, p2, p3, p4, p5) = MO.labels<5>("virt");
	std::tie(h3, h4, h5, h6) = MO.labels<4>("occ");

	i1(h6,p5) = f1(h6,p5);
	i1(h6,p5) +=  0.5  * t1(p3,h4) * v2(h4,h6,p3,p5);
	de() =  0;
	de() += t1(p5,h6) * i1(h6,p5);
	de() +=  0.25  * t2(p1,p2,h3,h4) * v2(h3,h4,p1,p2);
}

template<typename T>
void driver() {
	// Construction of tiled index space MO from skretch
	IndexSpace MO_IS{range(0,200), {"occ", {range(0,100)}, 
                                  "virt", {range(100,200)}}};
	TiledIndexSpace MO{MO_IS, 10};
	
	const TiledIndexSpace& O = MO("occ");
	const TiledIndexSpace& V = MO("virt");
	const TiledIndexSpace& N = MO("all");
	Tensor<T> de{};
	Tensor<T> t1{V, O};
	Tensor<T> t2{V, V, O, O};
	Tensor<T> f1{N, N};
	Tensor<T> v2{N, N, N, N};
	ccsd_e(MO, de, t1, t2, f1, v2);
}
```
----

### Canonical  T1

```c++
// Up-to-date version can be found at ccsd/ccsd_driver.cc
template<typename T>
void  ccsd_t1(const TiledIndexSpace& MO, 
		      Tensor<T>& i0, 
		      const Tensor<T>& t1, 
		      const Tensor<T>& t2,
          const Tensor<T>& f1, 
          const Tensor<T>& v2) { 

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  Tensor<T> t1_2_1{O, O};
  Tensor<T> t1_2_2_1{O, V};
  Tensor<T> t1_3_1{V, V};
  Tensor<T> t1_5_1{O, V};
  Tensor<T> t1_6_1{O, O, V, V};

  TiledIndexLabel p2, p3, p4, p5, p6, p7;
  TiledIndexLabel h1, h4, h5, h6, h7, h8;

  std::tie(p2, p3, p4, p5, p6, p7) = MO.labels<6>("virt");
  std::tie(h1, h4, h5, h6, h7, h8) = MO.labels<6>("occ");  

  i0(p2,h1)             =   f1(p2,h1);
  t1_2_1(h7,h1)         =   f1(h7,h1);
  t1_2_2_1(h7,p3)       =   f1(h7,p3);
  t1_2_2_1(h7,p3)      +=   -1 * t1(p5,h6) * v2(h6,h7,p3,p5);
  t1_2_1(h7,h1)        +=   t1(p3,h1) * t1_2_2_1(h7,p3);
  t1_2_1(h7,h1)        +=   -1 * t1(p4,h5) * v2(h5,h7,h1,p4);
  t1_2_1(h7,h1)        +=   -0.5 * t2(p3,p4,h1,h5) * v2(h5,h7,p3,p4);
  i0(p2,h1)            +=   -1 * t1(p2,h7) * t1_2_1(h7,h1);
  t1_3_1(p2,p3)         =   f1(p2,p3);
  t1_3_1(p2,p3)        +=   -1 * t1(p4,h5) * v2(h5,p2,p3,p4);
  i0(p2,h1)            +=   t1(p3,h1) * t1_3_1(p2,p3);
  i0(p2,h1)            +=   -1 * t1(p3,h4) * v2(h4,p2,h1,p3);
  t1_5_1(h8,p7)         =   f1(h8,p7);
  t1_5_1(h8,p7)        +=   t1(p5,h6) * v2(h6,h8,p5,p7);
  i0(p2,h1)            +=   t2(p2,p7,h1,h8) * t1_5_1(h8,p7);
  t1_6_1(h4,h5,h1,p3)   =   v2(h4,h5,h1,p3);
  t1_6_1(h4,h5,h1,p3)  +=   -1 * t1(p6,h1) * v2(h4,h5,p3,p6);
  i0(p2,h1)            +=   -0.5 * t2(p2,p3,h4,h5) * t1_6_1(h4,h5,h1,p3);
  i0(p2,h1)            +=   -0.5 * t2(p3,p4,h1,h5) * v2(h5,p2,p3,p4);
}

template<typename T>
void driver() {
	// Construction of tiled index space MO from skretch
	IndexSpace MO_IS{range(0,200), {"occ", {range(0,100)}, 
                                  "virt", {range(100,200)}}};
	TiledIndexSpace MO{MO_IS, 10};
	
	const TiledIndexSpace& O = MO("occ");
	const TiledIndexSpace& V = MO("virt");
	const TiledIndexSpace& N = MO("all");

	Tensor<T> i0{};
	Tensor<T> t1{V, O};
	Tensor<T> t2{V, V, O, O};
	Tensor<T> f1{N, N};
	Tensor<T> v2{N, N, N, N};
	ccsd_t1(MO, i0, t1, t2, f1, v2);
}
```

### Canonical HF (work in progress)

***Note: We do not have an implementation of initial hcore guess (e.g., STO-nG basis assumption in Ed's toy code, etc.). What parts of that can use TAMM***

```c++
void compute_2body_fock(const TiledIndexSpace& AO,
			const std::vector<libint2::Shell> &shells, 
			const Tensor<T> &D, Tensor<T> &F) {
  // auto will correspond to a TiledIndexSpace or
  // a TiledIndexRange depending on the decision
  const auto& N = AO("all");
  TiledIndexLabel s1, s2, s3, s4;
  std::tie(s1,s2, s3, s4) = AO.range_labels<4>("all");
  const auto n = nbasis(shells);
  Tensor<T> G{N,N};
  //TODO: construct D from C
  // construct the 2-electron repulsion integrals engine
  Tensor<T> ERI{N, N, N, N, coulomb_integral_lambda};
  Scheduler()
  .fuse(PermGroup(,,,).iterator(),
	  G(s1, s2) += D(s3, s4) * ERI(s1, s2, s3, s4),
	  G(s3, s4) += D(s1, s2) * ERI(s1, s2, s3, s4),
	  G(s1, s3) -= 0.25*D(s2,s4) * ERI(s1,s2,s3,s4),
	  G(s2, s4) -= 0.25*D(s1,s3) * ERI(s1,s2,s3,s4),
	  G(s1, s4) -= 0.25*D(s2,s3) * ERI(s1,s2,s3,s4),
	  G(s2, s3) -= 0.25*D(s1,s4) * ERI(s1,s2,s3,s4)
	  ).execute();
	  

  // symmetrize the result and return	
  //Tensor<T> Gt{N,N};
  //Gt(a,b) = G(b,a); //G.transpose();
  F(s1,s2) += 0.5 * G(s1,s2);
  F(s1,s2) += 0.5 * G(s2,s1);
}

template<typename T>
void hartree_fock(const TiledIndexSpace& AO, 
		          const Tensor<T>& C,
		          Tensor<T>& F) {
  const TiledIndexSpace& N = AO("all");
  const TiledIndexSpace& O = AO("occ");

  TiledIndexLabel a,b,c;
  TiledIndexLabel ao,bo,co;
  std::tie(a,b,c) = AO.range_labels<3>("all");
  std::tie(ao,bo,co) = AO.range_labels<3>("occ");
	
  // compute overlap integrals
  //Tensor<T> S{N,N};
  //S = compute_1body_ints(shells, Operator::overlap);
  Tensor<T> S{N,N, one_body_overlap_integral_lambda};
  // compute kinetic-energy integrals
  Tensor<T> T{N,N,one_body_kinetic_integral_lambda};
  //T = compute_1body_ints(shells, Operator::kinetic);
  // compute nuclear-attraction integrals
  //Tensor<T> V{N,N};
  //V = compute_1body_ints(shells, Operator::nuclear, atoms);
  Tensor<T> V{N,N, one_body_nuclear_integral_lambda};
  // Core Hamiltonian = T + V
  Tensor<T> H{N, N};
  H(a,b) = T(a,b);
  H(a,b) += V(a,b);
  
  Tensor<T> D{N, N};
  compute_soad(atoms, D); 
    
  const auto maxiter = 100;
  const auto conv = 1e-12;
  auto iter = 0;
  Tensor<T> ehf{},ediff{},rmsd{};
  Tensor<T> eps{N,N};

  do {
    ++iter;
    // Save a copy of the energy and the density
    Tensor<T> ehf_last{};
    Tensor<T> D_last{N,N};
    
    ehf_last() = ehf();
    D_last(a,b) = D(a,b);

    // build a new Fock matrix
    F(a,b) = H(a,b);
    compute_2body_fock(shells, D, F); //accumulate into F
    
    // solve F C = e S C
    //Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(F, S);
    //eps = gen_eig_solver(F,S).eigenvalues();
    //C = gen_eig_solver(F,S).eigenvectors();
    std::tie(C, eps) = eigen_solve(F, S);

    // compute density, D = C(occ) . C(occ)T
    //C_occ(ao,bo) = C(ao,bo); //C.leftCols(ndocc);
    //C_occ_transpose(ao,bo) = C_occ(bo,ao);
    D(ao, bo) = C(ao, xo) * C(xo, bo);

    Tensor<T> tmp1{a, b}, tmp2{a, b};
    // compute HF energy
    //ehf += D(i, j) * (H(i, j) + F(i, j));
    ehf() = 0.0;
    tmp1(a,b) = H(a, b);
    tmp1(a,b) += F(a, b);
    ehf() = D(a,b) * tmp1(a,b);

    // compute difference with last iteration
    ediff() = ehf();
    ediff() = -1.0 * ehf_last();
    tmp2(a,b) = D(a,b);
    tmp2(a,b) += -1.0 * D_last(a,b);
    norm(tmp2,rmsd); //rmsd() = tmp2(a,b).norm();
	rmsd() = tmp2(a,b) * tmp2(a,b);

	//e.g.:Tensor<T> rmsd_local{AllocationModel::replicated};
	//e.g.:rmsd_local(a) = rmsd(a);
	//e.g.: rmsd(a) +=  rmsd_local(a);
	//TODO: only put rmsd_local in process 0 to rmsd
  } while (((fabs(get_scalar(ediff)) > conv) || (fabs(get_scalar(rmsd)) > conv)) && (iter < maxiter));
}

template<typename T>
void driver() {
	// Construction of tiled index space MO from skretch
	IndexSpace AO_IS{range(0,200), {"occ", {range(0,100)}, 
                                  "virt", {range(100,200)}}};
	TiledIndexSpace AO{AO_IS, 10};

	const TiledIndexSpace& N = AO("all");
	
	Tensor<T> C{N, N};
	Tensor<T> F{N, N};
	hartree_fock(AO, C, F);
}
```


### DLPNO CCSD (work in progress)
```c++
double dlpno_ccsd(const TiledIndexSpace& AO, const TiledIndexSpace& MO, 
                  const TiledIndexSpace& AtomSpace, 
                  const TiledIndexSpace& SubMO, const TiledIndexSpace& SubPAO,
                  const Tensor<T>& S,
                  const Tensor<T>& C, const Tensor<T>& Ct,
                  const Tensor<T>& dmat, const Tensor<T>& F,
                  const Tensor<T>& I,
                  const Tensor<T>& Cvirtt){
	  
	
    const TiledIndexSpace& N_ao = AO("all"); 
    const TiledIndexSpace& N_atom = AtomSpace("all");
    const TiledIndexSpace& N_pao = SubPAO("all");
    const TiledIndexSpace& O_mo = MO("occ");
    const TiledIndexSpace& O_submo = SubMO("occ");
    
    TiledIndexLabel mu, nu, mu_p, nu_p;
    TiledIndexLabel i, j, i_p, j_p;
    TiledIndexLabel A, B;

    std::tie(mu, nu) = AO.range_labels<2>("all");
    std::tie(mu_p, nu_p) = SubPAO.range_labels<2>("all");
    std::tie(i,j) = MO.range_labels<2>("occ");
    std::tie(i_p,j_p) = SubMO.range_labels<2>("occ");
    std::tie(A, B) = AtomSpace.range_labels<2>("all");
    
    Tensor<T> F{N_ao, N_ao};
    //Tensor<T> C{A, mu, i_p(A)};
    Tensor<T> C{N_ao, O_mo};
    Tensor<T> S{N_ao, N_ao};
    Tensor<T> TC_2e{N_ao, N_ao, N_aux, lambda};
    
    //Step 31
    Tensor<T> SC{N_ao, O_mo};
    Tensor<T> P{i, A};
    SC(mu, i) = S(mu, nu) * C(nu, i);
    P(i, A) = C(mu(A), i) * SC(mu(A), i);
	
	//middle step
	//make a map from the number of occupied MO to vector of atoms it is on. Now we know which atoms are associated with which occupied MO.
	
	//we now have SubMO dependent on AtomSpace
	TiledIndexLabel i_p; //..define
    Tensor<T> SCCS{N_ao, N_ao};
    //auto SC(A, mu, i_p) = S(mu, nu) * C(A, nu, i);    auto SCCS(mu, nu) = SC(mu, i) * SC(nu, i_p);
    
    Tensor<T> L{N_ao, N_ao};
    L(mu, nu) = S(mu, nu);
    L(mu, nu) += -1 * SCCS(mu,nu)

	//now we interpret L to construct the mu_tilde (mu_p) index space 
	
    //Step 2
    Tensor<T> l{N_atom, N_atom}; 
    //TODO: support for absolute sum is need
    l(A,B) = L(mu_subatom(A), mu_subatom(B));
    //l(i,j) = outer(fn(i,j)) * A(i, j);
    //e.g.: T1(mu_pp(m,n)) = A(n,m);
    //e.g.: Tensor<T> T5{i, a(i)};
    //e.g.: T5(x, y) = 0;
    
    //here we do a map from PAO to set of atoms around the PAO
    
	//Step 4 - ??
	// auto pairdoms =  form\_pair\_domains(occ2atom, atom2PAO);

//Now we have the pair index space in terms of pairs of MOs: mu_p(i,j), nu_p(i,j), ... where (i,j) pairs are defined here.

	//Step 5: skip for now
    Tensor<T> Fpao{}, Focc{}, Tmp1{}, Tmp2{};

    Tmp1(mu_p, nu) = L(mu_subatom(A), mnu) * F(mu, nu);
    Fpao(mu_p, nu_p) = Tmp1(mu_p, nu) * L(nu, nu_p);
    Tmp(i_p, j_pnu) = C(i_, u) * F(mu, nu);
    Focc(i_p, j_p) = Tmp(i_p, nu) * C(nu, j_p);
    
    //EV: somehere here or above we need a canonicalization step
    
    //Step 6
	// auto I_imuP =  transform_tensor(I, {Ct, Lt}, {0, 1});
	TiledIndexLabel i_t{N_olocalmo};
	TiledIndexLabel P_i{..};
	Tensor<T> Integral{mu, nu, N_aux, lambda};
	//Tensor<T> I_imuP{i, mu_p(i), P_i(i)};
    TMP(it, nu_p(it), P_i(it)) = C(it, mu_p(it)) * Integral(mu_p{it}, nu_p(it), P_i(it));
    I_imup(it, mupao_p(it), P_i(it)) = TMP(it, nu_p(it), P_i(it)) * L(mupao_p(it), nu_p(it));

	//Step 7
	// auto D_ii =  diagonal_mp2_densities(I_imuP, pairdoms, Focc, Fpao);

	Tensor<T> t{mupao_i(it), mupao_i(it), it};
	t(mupao_i(it), nupao_i(it), it) = 
		I_imup(it, mupao_i(it), P_i(it)) * 
		I_imup(it, nupao_i(it), P_i(it));

	//this will be a lambda
	//TODO: F will need to be computed above (Step 5), but in it-specific form. PAO Fock matrix needs to be diagonal.
	t(mupao_i(it), nupao_i(it), it) /= 
		F(mupao_i(it),mupao_i(it)) +
		F(nupao_i(it), nupao_i(it)) -
		2 * F_occ_mo(it, it);

	//EV: Different PAO spaces are disjoint. mupao_i(0) and mupao_i(1) are completely different. Union of these is not well-defined.
    Tensor<T> D{it, mupao_i(it), nupao_i(it)};
	D(it,mupao_i(it), nupao_i(it)) = 
		s(mupao_i(it), nupao_i(it), it) *
		t(mupao_i(it), nupao_i(it), it);
	//EV: here, because t is symmetric, we can diagonalize it directly. so we skip the D computation.
	
	//Step 8
	//auto LOSVs =  make_losvs(D_ii);
	//this step is diagonalization. 

	//Step 9
	//std::vector<tensor_type<2>> Faa;
	//for(auto i=0; i<LOSVs.size(); ++i) {
	//	auto LOSVi = LOSVs\[i\].shuffle({1, 0});
	//	Faa.push_back(transform_tensor(F, {LOSVs\[i\], LOSVi}, {0, 1}));
	}
	//auto ea = canonicalize(Faa);
	TMP(it, it) = d(mupao_i(it), mupao_i(it)) * 
				F(mupao_i(it),mupao_i(it));
	F(it, it) = TMP(it, it) * ..;
	
	//Step 

	//Step 11

	//Skipped for now b/c they don't do anything for test system
	//Step 12
	//auto EscOSV =  sc\_osv\_mp2(I_iaP, Focc, ea);
	//auto I_ijP =  transform_tensor(I, {Ct, Ct}, {0, 1});

}
```
### DLPNO MP2 (from Ed)


------

### TAMM Code Sketch from DLPNO Google Docs

```c++
TiledIndexLabel i{N_ao}, k{N_ao};
TiledIndexLabel j_atom; 
TiledIndexLabel A{N_atom};

Tensor<T> tC{N_ao, N_ao, N_atom};
Tensor<T> tA{N_ao,N_ao};
Tensor<T> tB{N_ao,N_ao};
tC(i, k, A) = tA(i, j_atom(A)) * tB(j_atom(A), k);

for i, k in AO {
  for A in Atom {
    Alloc tC(i, k, A) as Cbuf(i0, k0, A0)
    Cbuf(i0,k0,A0) = 0
    for j_atom in DepAO_Atom(A) { 
      Get tA(i, j_atom) to Abuf(i0, j0)
      Get tB(j_atom, k, k) to Bbuf(j0, k0)
      Cbuf(i0,k0,A0) += Abuf(i0, j0) * Bbuf(j0, k0)
    } //for j_atom in DepAO_Atom(A)
    Put Cbuf(i0,k0,A0) to tC(i,k,A)
    Dealloc Abuf, Bbuf, Cbuf
  } // for A
} //for i, k

```