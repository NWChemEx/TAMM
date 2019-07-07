### TAMM Tensor Construction and Usage
In this documentation, we describe the new syntax and semantics for the tensor notation in TAMM. Most of the usage will stay the same but there are some new syntax that will ease the construction and usage of the tensors. This document aims to give some intuition about how the sparse tensor construction and usage is done in TAMM through dependent `TiledIndexSpace` objects. 

The rest of this documentation will give a brief description of `TiledIndexSpace`s (independent and dependent) as they will be used in `Tensor` construction and usage for details you can check [Index Space documentation](./IndexSpaces.md). After describing the `TiledIndexSpace`s, we will detail the tensor construction and how they will behave under arithmetic operations.

#### TiledIndexSpace Description

Given an `IndexSpace` and a tiling size (this can be single tile or custom list of sizes with full coverage on the indices), `TiledIndexSpace`, is the tiled version of the index space where each tile has multiple indices. Theoretically, an `IndexSpace` is a single tiled `TiledIndexSpace`. By default independent `TiledIndexSpace`s (as well as `TiledIndexLabel`s) are used to construct *dense* tensors. 

```c++
IndexSpace AUX_is{/*...*/}
IndexSpace AO_is{/*...*/};
IndexSpace MO_is{/*...*/};

size_t tile_size = /*some positive value*/;
std::vector<size_t> tile_sizes = {/*multiple positive values*/}; 

TiledIndexSpace AUX{AUX_is, tile_size};
TiledIndexSpace AO{AO_is, tile_size};
TiledIndexSpace MO{MO_is, tile_sizes};
```

Constructing sparse tensors needs extra information to represent the sparsity as a dependency map between indices on different dimensions of the tensors. For this purpose, TAMM has *dependent* `TiledIndexSpace` constructors, that will construct relation between different `TiledIndexSpace`s. The main constructor requires a reference `TiledIndexSpace` which will be the root/parent for the constructed relation. In other words this will be the ***domain*** of the dependency relation, for each indices in the dependency relation the domain will be a subset of this `TiledIndexSpace`. Second argument for the constructor is a set of `TiledIndexSpace`s where the dependencies are defined on, in other words this will be the ***range*** of the dependency relation. And as the final argument for constructing the dependent `TiledIndexSpace` is the dependency map description (of type `std::map<IndexVector, TiledIndexSpace>`). **Note that** the dependency map is defined over the tile indices, not actual indices in the `IndexSpace` definition.

```c++
// Assuming TiledIndexSpace MO has 4 tiles one can describe
// the dependency for each tilex index (0-3), the key point
// is the dependency is always on a subspace of AO which
// are constructed by TiledIndexSpace construction within 
// the dependency pairs. (range for continious tiles,
// IndexVector for selected tiles.)
auto dep_rel = {
  {{0}, TiledIndexSpace{AO, range(0,3)}},
  {{1}, TiledIndexSpace{AO, range(2,5)}},
  {{2}, TiledIndexSpace{AO, IndexVector{0, 2, 4}}}
};

// Dependent TiledIndexSpace where range of the dependencies
// are defined over MO indices, and the domain is subset of AO
TiledIndexSpace depAO{AO, {MO}, dep_rel};
```

#### Using Labels

`TiledIndexLabel` pairs a TiledIndexSpace with an integer label. These labels can be created using TiledIndexSpace methods: `labels<N>(...)` and `label(...)`. (Note that `labels<N>(...)` method starts label value `0` by default if no value is provided,( this might end up problematic label creation if used on the same tiled index space multiple times.). These objects are the main components for describing a Tensor storage and computation over these tensors. 

Labeling comes handy in the use of dependent `TiledIndexSpace` for constructing sparse tensors or describing sparse computation over the tensors. A label for a dependent `TiledIndexLabel` uses secondary labels within parenthesis to describe the dependency in the tensor description. For dense cases, `TiledIndexSpace`s can used for constructing the tensors but in case of sparse tensor construction the only option is to use labeling for describing the dependencies between dimensions. 

```c++
using tensor_type = Tensor<double>;
auto [i, j] = MO.labels<2>("all");
auto mu = depAO.label("all");

// Dense tensor construction
tensor_type T1{i, j};
tensor_type T2{MO, MO};

// Sparse tensor construction
tensor_type T3{i, mu(i)}
```

#### Tensor Construction

Tensor is the main computation and storage data structure in TAMM. The main constructs for creating a `Tensor` object is using `TiledIndexSpace`s or `TiledIndexLabel`s for each dimension. For dense case, the construction uses independent `TiledIndexSpace`s or labels related to these spaces. As an ease of usage, if the users give a dependent label without secondary labels the tensor will be constructed over the reference `TiledIndexSpace` of the given dependent `TiledIndexSpace`.

```c++
using tensor_type = Tensor<double>;
auto [i, j] = MO.labels<2>("all");
auto [A, B] = AO.labels<2>("all");
auto [mu, nu] = depAO.labels<2>("all");

// Dense tensor construction
tensor_type T1{i, j};     // MO x MO Tensor
tensor_type T2{i, A};     // MO x AO Tensor
tensor_type T3{mu, nu};   // AO x AO Tensor
tensor_type T4{mu, i};    // AO x MO Tensor

// Sparse tensor construction
tensor_type T5{i, mu(i)}; // MO x depAO Tensor 
```

#### Tensor Operation Usage

All tensor operations in TAMM uses labeled tensors (`Tensor` objects with labeling within parenthesis) as the main component. By using different set of labeling users can describe sparse computation over the dense tensors, or opt out using labels which will result in using `TiledIndexSpace` information from the storage/construction. 

With the new extensions (`TiledIndexSpace` transformation operations) to TAMM syntax on tensor operations, users can use different labels then the construction which will be intersected with the storage when the loop nests are being constructed. Below examples will illustrate the usage and the corresponding loop nests for each corresponding tensor operation. The only constraint in using different labels than the labels used in construction is that both labels should have same root `TiledIndexSpace`. This constraint is mainly coming from the transformation constraints that we set in relation to the performance reasons. We assume that `Tensor` objects from the previous examples are constructed and allocated over a `Scheduler` object named `sch` beforehand.

##### Set/Add operations:

Examples without using labels:
```c++
// without any labels
sch
  // Dense Tensor
  (T2() = 42.0)
  // Sparse Tensor
  (T5() = 21.0) 
  // Assignment
  (T2() = T5())
.execute();

// Loop Nests constructed
// For (T2() = 42.0)
for(auto i : T2.dim(0))
  for(auto A : T2.dim(1))
    T2[i][A] = 42.0;

// For (T5() = 21.0)
for(auto i : T5.dim(0))
  for(auto mu: T5.dim(1)[i])
    T5[i][mu] = 21.0;

// For (T2() = T5())
for(auto i : T2.dim(0).intesect(T5.dim(0)))
  for(auto mu : T2.dim(1).intersect(T5.dim(1)[i]))
    T2[i][mu] = T5[i][mu]
```

Examples using labels:
```c++
// Labeling reference
// auto [i, j] = MO.labels<2>("all");
// auto [A, B] = AO.labels<2>("all");
// auto [mu, nu] = depAO.labels<2>("all");

// Construct a subset of MO
TiledIndexSpace subMO{MO, range(1,3)};

auto i_p = subMO.label("all");

sch
  // Dense tensor with subset
  (T2(i_p, A) = 13.0)
  // Sparse Tensor
  (T5(i_p, mu(i_p)) = 42.0) 
  // Assignment
  (T2(i_p, mu(i_p)) = T5(i_p, mu(i_p)))
  // Assignment with independent labels
  (T2(i_p, mu) = T5(i_p, mu))
.execute();


// Loop Nests constructed
// For (T2(i_p, A) = 13.0)
for(auto i : T2.dim(0).intersect(i_p.tis())
  for(auto A : T2.dim(1))
    T2[i][A] = 42.0;

// For (T5(i_p, mu(i_p)) = 42.0) 
for(auto i : T5.dim(0).intesect(i_p.tis()))
  for(auto mu : T5.dim(1)[i])
    T5[i][mu] = 21.0;

// For (T2() = T5())
// Both assignment will result in the same
// loop nest 
for(auto i : T2.dim(0).intesect(T5.dim(0)).intesect(i_p))
  for(auto mu : T2.dim(1).intersect(T5.dim(1)[i]))
    T2[i][mu] = T5[i][mu]
```

##### Multiplication Operations

Similar the set operations, users have the option to construct equations without giving any labels which in return will use the `TiledIndexSpace`s used in the construction. **Note that** in any of the operations the tensors are assumed to be constructed and allocated before the using in an equation. Again users may choose to specify labels (dependent or independent) according to the `TiledIndexSpace`s that are used in tensors' construction for each dimension. Similar to set operations, the labels used in the contractions should be compatible with the `TiledIndexSpace`s that are used in constructing the tensors. 

Examples without using labels:
```c++
// without any labels
sch
  // Dense Tensor
  (T2() = 2.0)
  // Dense Tensor
  (T4() = 21.0) 
  // Dense Contraction
  (T1() = T2() * T4())
.execute();

// Loop Nests constructed
// For (T2() = 2.0)
for(auto i : T2.dim(0))
  for(auto A : T2.dim(1))
    T2[i][A] = 42.0;

// For (T4() = 21.0)
for(auto A : T4.dim(0))
  for(auto j: T4.dim(1)[i])
    T4[A][j] = 21.0;

// For (T1() = T2() * T4())
for(auto i : T2.dim(0).intesect(T4.dim(0)))
  for(auto j : T2.dim(1).intersect(T4.dim(1)))
    for(auto A : T2.dim(1).intesect(T4.dim(0)))
      T1[i][j] = T2[i][A] * T4[A][j];
```

Examples using labels:

```c++
// Constructed TiledIndexSpaces
// TiledIndexSpace AO{AO_is, tile_size};
// TiledIndexSpace MO{MO_is, tile_sizes};
// TiledIndexSpace depAO_1{AO, {MO}, dep_rel_2};
// TiledIndexSpace depAO_2{AO, {MO}, dep_rel_2};

// Constructed index labels 
// auto [i, j] = MO.labels<2>("all");
// auto [A, B] = AO.labels<2>("all");
// auto [mu, nu] = depAO_1.labels<2>("all");
// auto [mu_p, nu_p] = depAO_2.labels<2>("all");

// Dense tensor construction
tensor_type T1{i, j};     // MO x MO Tensor
tensor_type T2{i, A};     // MO x AO Tensor
tensor_type T3{mu, nu};   // AO x AO Tensor
tensor_type T4{mu, i};    // AO x MO Tensor

// Sparse tensor construction
tensor_type T5{i, mu(i)}; // MO x depAO Tensor 
tensor_type T6{mu_p(j), j}; // depAO x MO Tensor

auto mu_i = mu.intersect(mu_p);
sch
  // Sparse Tensor
  (T6(mu_p(j), j) = 2.0)
  // (T6(mu_p, j) = 2.0)
  // Sparse Tensor
  (T5(i, mu(i)) = 21.0)
  // (T5(i, mu) = 21.0) 
  // Sparse Contraction
  (T1(i, j) = T5(i, mu_i(i)) * T6(mu_i(j), j))
  // (T1(i, j) = T5(i, mu_i) * T6(mu_i, j))
.execute();


// Loop Nests constructed
// For (T6(mu_p(j), j) = 2.0) 
// &   (T6(mu_p, j) = 2.0)
for(auto j : T6.dim(1))
  for(auto mu_p : T6.dim(0)[j])
    T2[mu_p][j] = 2.0;

// For (T5(i, mu(i)) = 21.0)
// &   (T5(i, mu) = 21.0)
for(auto i : T5.dim(0))
  for(auto mu: T5.dim(0)[i])
    T5[i][mu] = 21.0;

// For (T1(i, j) = T5(i, mu_i(i)) * T6(mu_i(j), j))
// &   (T1(i, j) = T5(i, mu_i) * T6(mu_i, j))
for(auto i_idx : T1.dim(0).intersect(i).intersect(T5.dim(0)))
  for(auto j_idx : T1.dim(1).intersect(j).intersect(T6.dim(j)))
    for(auto mu_idx : T5.dim(1).intersect(mu_i).intersect(T6.dim(0)))
      T1[i_idx][j_idx] = T5[i_idx][mu_idx] * T6[mu_idx][j_idx];
```


##### Loop Nest Order and Construction

The default loop nest ordering is from left-hand side (LHS) to right-hand side (RHS) labels. For example the ordering for a simple assignment with sum over operation on the  `(T1(i, j) = T6(j, i, k)` will end up ordering of "$i \to j \to k$" where k is the summed over index. This ordering becomes more important when the operations are over dependent index spaces, as there will be conflicting orders with the dependency order described in the operation and the storage of the tensors. In case of the dependencies are not satisfied with the default ordering the `TiledIndexSpace` transformations will be used to eliminate the dependencies. 


```c++
// Given an order
std::vector<Label> order = get_order(EXPR);

std::vector<TIS> tis_vec{};
// Intersect for each lbl with reference to storage and usage
for(auto i : order){
  std::vector<IndexLabel> lbls = EXPR.find(i);

  auto tmp_tis;
  for(auto lbl : lbls){
    if(!is_dependency_resolved(lbl, tis_vec)){
      lbl.project(lbl.seconary_labels());
    }
    tmp_tis = lbl.tis().intersect(lbl.tensor().tis());
  }
  
}

LoopNest loop_nest{tis_vec};
```