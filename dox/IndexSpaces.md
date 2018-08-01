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

- **TiledIndexLabel** A TiledIndexLabel pairs a TiledIndexSpace with an integer label. These labels can be created using TiledIndexSpace methods: `labels<N>(...)` and `label(...)`. 
  ```c++
      // Generate TiledIndexLabels for a specific sub-space
      TiledIndexLabel i, j, k, l, m, n;
      std::tie(i,j) = tis_mo.labels<2>("occ");
      std::tie(k,l) = tis_mo.labels<2>("virt");1
      m = tis_mo.label("all", 9);
      n = tis_mo.label("all", 10);
  ```
- **[Dependent TiledIndexLabel]** To construct a dependent index label, TiledIndexLabel provides an `operator()` overload.

  ```c++
    // Construction of dependent TiledIndexSpace is same as below
    std::map<IndexVector, IndexSpace> dep_relation{};
    // Tile Atom space with default tiling
    TiledIndexSpace T_Atom{Atom};
    // Construct dependent TiledIndexSpace
    DependentIndexSpace subMO_atom{{T_Atom}, dep_relation};

    // Construct TiledIndexLabels from TiledIndexSpaces 
    TiledIndexLabel i, a;
    i = T_Atom.label("all");
    a = subMO_atom.label("all")

    // Use labels for Tensor construction or Tensor Operations
    Tensor<double> T1{a(i), i};  // a(i) will construct a depedent TiledIndexLabel which internally is validated
  ```

