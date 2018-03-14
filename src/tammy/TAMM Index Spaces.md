# Index Spaces in TAMM

## Terminology

### Index
`Index` is an integral type to index into an array/tensor. Typically, an `int` or a typed version thereof.

### IndexSpaceFragment

`IndexSpaceFragment` is an ordered list of `Index` values. Mathematically, a `IndexSpaceFragment` maps the interval [0, N-1] to an ordered list of indices. Any `IndexSpaceFragment`, isf, satisfies the following properties:

- isf.operator() : [0,N-1] $\rightarrow$ `Index`$^N$
- (Bijective function) N = $|$isf$|$, $\forall$ i, j $\in$ [0,N-1], 
   (i $\neq$ j) $\equiv$ (isf.point(i) $\neq$ isf.point(j))

An additional property would be useful in checking and optimizing operations:

- (Ordered): N = $|$isf$|$, $\forall$ i, j $\in$ [0,N-1], 
   (i < j) $\equiv$ (isf.point(i) < isf.point(j))

Note: The ordered constraint might not hold in some cases. Think about a specific order of paired index spaces.


A sub-`IndexSpaceFragment` is a sublist of elements in another `IndexSpaceFragment`.

Note: In the common case, a `IndexSpaceFragment` can be a interval `[lo,hi]`. This will enable compact representation and manipulation. Subspaces need more general lists.

- `IndexSpaceFragment` has `spin` and `spatial` attributes which are carried into `IndexSpace`s. 
- `Index` values can be accessed via `iterator`s. 
- `IndexSpaceFragment` has `is_subset` method  for checking if another fragment is a subset which will be used for validating sub-spaces. 
- It also has a `is_disjoint` method for checking the disjointness of the each fragment in an `IndexSpace`.

The most common use case for `IndexSpaceFragment`s will be constructing blocks for different ranges (i.e. `"occ"`, `"virt"`) of the `IndexSpace`s.

***Code Examples***
```c++
IndexSpaceFragment isf1(0, 4, {10,20,30,40,50});
IndexSpaceFragment isf2(5, 10, {60, 70, 80, 90, 100}, Spin{1});

Index i_0 = isf1.point(0);
Index i_1 = isf1(1);

if(isf1.is_disjoint(isf2)){
    ...
}

if(isf1.is_subset(isf2)){
    ...
}

for(auto itr = isf1.begin(); itr != isf1.end()){
    ...
}
```

### IndexSpace

`IndexSpace` is an ordered list of `IndexSpaceFragment` objects. Any `IndexSpace`, is, satisfies the following properties:

- is.operator() : [0,N-1] $\rightarrow$ `Index`$^N$
- is.fragment() : [0,N$_F$-1] $\rightarrow$ `IndexSpaceFragment`$^{N_F}$
- is.operator(i) = isf.operator(m) where isf = is.fragment(j), 
      k = $\sum_{n=0}^{j-1}$ $|$is.fragment(n)$|$, i = m + k

An `IndexSpace` may or may not be a list of disjoint fragments. 

- (optional: disjointedness): $\forall$ i, j $\in$ [0,N$_F$-1], 
   is.fragment(i).range() $\cap$ is.fragment(j).range() = $\emptyset$

When a `IndexSpace` satisfies the disjointedness condition, we refer to it as a disjointed index space fragment list.

A sub-`IndexSpace` is ordered list of sub-`IndexSpaceFragment`s in another `IndexSpace`.  **(Q: should they be in the same order?)**

- `IndexSpace` has direct access to the fragments and indices.
- `IndexSpace` as it is doesn't include any `tiling` information, instead it will be used as a building block for `TiledIndexSpace`
- Disjointness property is checked using `is_disjoint` method from `IndexSpaceFragments`.
- It also has method, `is_subset`, for checking if two `IndexSpace` is a subspace or not. 

The common use case for `IndexSpace` is for dynamically constructing new index spaces which will be a reference for `TiledIndexSpace`s. 

***Code Examples***
```c++
IndexSpaceFragment occ_a, occ_b, virt_a, virt_b;
// Initialize IndexSpaceFragment's

IndexSpace MSO({occ_a, occ_b, virt_a, virt_b});
// IndexSpace MSO = {occ_a, occ_b, virt_a, virt_b};

IndexSpaceFragment temp_isf = MSO.fragment(0);
Index temp_i = MSO(0);
    
```

### TiledIndexSpace

An `TiledIndexSpace` associates a tiling with an `IndexSpace`. A valid tiling of an `IndexSpace` ensures that no tile spans multiple index space fragments. An `TiledIndexSpace`'s size N is the number of tiles in it. A `TiledIndexSpace` provides an iterator to iterate over the tiles. Each tile, in turn, can be queried for its size and provides an iterator to iterate over the points it aggregates.

- A default `TiledIndexSpace` can be constructed from a `IndexSpace`, where each tile is of size 1.
- It includes methods for accessing different ranges of the index space
    - `range("occ")` returns a range value, `TiledIndexRange`, for the occupied index space fragment
    - `range_labels("occ", 1, 2)` returns a tuple of `TiledIndexLabel` which is associeated with the specific range and integer label.
    - (**Q: how is the range map is introduced to `TiledIndexSpace`? Is it provided at construction or is it defined over `IndexSpaceFragment`s?**)
- It also includes methods for validating the applicability of the operations over the index spaces 
    - `is_identical_to(TiledIndexSpace)` checks if the index space and the tiling is identical
    - `is_compatible_with(TiledIndexSpace)` checks if the index space and tiling is compatible (compatibility is described in details under ;'Implementation Considerations')


***Code Examples***
```c++
// TiledIndexSpace defined over MSO with tile size of 20
TiledIndexSpace MSO_t20(MSO, 20);

TiledIndexRange mso_O = MSO_t20.range("occ");
TiledIndexRange mso_V = MSO_t20.range("virt");
TiledIndexRange mso_N = MSO_t20.range("all");

TiledIndexLabel i,j,k,l;

std::tie(i,j) = MSO_t20.range_labels("occ", 1, 2);
std::tie(k,l) = MSO_t20.range_labels("virt", 1, 2);


```

### TiledIndexRange

An `TiledIndexRange` is a list of tiles from an `TiledIndexSpace`. If the `TiledIndexSpace` is not disjointed, the index range cannot include tiles from more than one index space fragments in the underlying index space. 

A tensor's dimension is defined in terms of `TiledIndexRange` objects.

NOTE: An `TiledIndexSpace` might provide routines to obtain specific index ranges as a map, with key being a string ("occ," "virt," etc.).

Main use case of `TiledIndexRange` is constructing `Tensor`s using a range value from `TiledIndexSpace`s. 

***Code Examples***
```c++
const TiledIndexRange& O = MSO_t20.range("occ");
const TiledIndexRange& V = MSO_t20.range("virt");

Tensor<T> i1{O, V};
```

### TiledIndexLabel

An `TiledIndexLabel` combines an `TiledIndexRange` and an integer label.

A `TiledIndexLabel` for a dependent index space needs to be bound to tiled index labels corresponding to the spaces it depends on.

`TiledIndexLabel` is the main construct used for `Tensor` operations. 

***Code Examples***
```c++
const TiledIndexRange& N = AO.range("all");
const TiledIndexRange& O = AO.range("occ");

TiledIndexLabel a,b,c;
TiledIndexLabel ao,bo,co;

std::tie(a,b,c) = AO.range_labels("all", 1, 2, 3);
std::tie(ao,bo,co) = AO.range_labels("occ", 1, 2, 3);
```


## Implementation Considerations

### Indexing a tensor dimension using an index label

Consider a dimension d of a tensor, `T`, allocated on a tiled index range, `TIR`. When this dimension is indexed using an `TiledIndexLabel`, `lbl`, as illustrated below:
```c++
Tensor T{TIR};
T(lbl) = 0;
```
we define the following predicates:
- predicate $A$ $\equiv$ `lbl.tiled_index_range() == TIR`
- predicate $B$ $\equiv$ `lbl.tiled_index_range().tiled_index_space() == TIR.tiled_index_space()`
- predicate $C$ $\equiv$ `lbl.tiled_index_range().tiled_index_space().is() == IR.tiled_index_space().is()`

We can define the following relationships between `TIR` and `lbl`:
- Case 1: $A$:
`lbl` is over the same tiled index range `TIR`. This is the most efficient case. 

- Case 2: $\neg A \wedge B$
`lbl` is over a different tiled index range, `tir2`, but over the same `TiledIndexSpace` as `TIR`. In this case, either 
	- (1) `tir2` is a contiguous subspace of the tiles in `TIR`, when a simple offset would suffice, or
	- (2) `tir2` is not a contiguous subspace of the tiles in `TIR`.  Each tile of `tir2` to be checked for
membership. What we do with a non-member tile is up to the user (ignore or flag as error).

- Case 3: $\neg A \wedge \neg B \wedge C$
`lbl` is over a different `TiledIndexSpace` but over the same `IndexSpace`. Optimizing this would be difficult. We might just have to do "decay" both indices to run over degenerately tiled (as in tile size 1 for all tiles) and then do the same we did in case 1. One problem is that the storage is not pointwise. Therefore, we will need support to read/write partial tiles. Overall this can get expensive and should be avoided. We will not implement this to start with.

- Case 4: $\neg A \wedge \neg B \wedge \neg C$
lbl is over the a different `IndexSpace` than `TIR`. We will report this an error.

Examples:
```c++
Tensor Tmo{TIR_mo}, Tao{TIR_ao};
TiledIndexLabel imo{TIR_mo}, iao{TIR_ao};
Tmo(imo) = Tao(imo); //runtime error Tao(imo)
Tmo(iao) = Tao(iao); //runtime error Tmo(iao)
Tmo(iao) = 0; //runtime error
Tmo(imo) = translate(imo,iao) * Tao(iao);
```

(***TODO: dependent index labels***)
(***TODO: sub-spaces***)

### Index Translation
Sometimes we are not matching points, but mapping one point in a range to another. This could be within the same or a different `IndexSpace`s. This is done on a pointwise basis. The i-th point in one index range is mapped to the i-th point in the next index range. One possible optimization: if the tile sizes match, then the corresponding tiles can be mapped without translating individual points.

## Comments / clarifications / notes

* Q: Why is `IndexSpaceFragment` not just `std::vector`?
	A: This is to support attributes
* Q: Is there a use case for general `IndexSpaceFragment` objects? Why not just contiguous ranges (as in Python `range`)
	A: Subspaces might be the most common use case. Needs to be described and illustrated.

* In the examples, include how to support partial trace (Step 2 in Ryan’s DLPNO_CCSD example in CoupledCluster repo in github)
* Index spaces in PNO: atom spaces, PNO, PAO, AO, MO, transformed MO space (cf. Step 3)

## Code Examples

In the examples below, we only focus on aspects of TAMM relating to index spaces. Other details (e.g., scheduler, tensor allocation/deallocation, etc.) are ignored. Even Tensor object constructors need to be discussed/refined.

### Canonical HF (work in progress)

***Note: We do not have an implementation of initial hcore guess (e.g., STO-nG basis assumption in Ed's toy code, etc.). What parts of that can use TAMM***

```c++
void compute_2body_fock(const TiledIndexSpace& AO,
			const std::vector<libint2::Shell> &shells, 
			const Tensor<T> &D, Tensor<T> &F) {
  const TiledIndexRange& N = AO.range("all");
  TiledIndexLabel s1, s2, s3, s4;
  std::tie(s1,s2, s3, s4) = AO.range_labels("all", 1, 2, 3, 4);
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
  const TiledIndexRange& N = AO.range("all");
  const TiledIndexRange& O = AO.range("occ");

  TiledIndexLabel a,b,c;
  TiledIndexLabel ao,bo,co;
  std::tie(a,b,c) = AO.range_labels("all", 1, 2, 3);
  std::tie(ao,bo,co) = AO.range_labels("occ", 1, 2, 3);
	
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
```

### Canonical CCSD E

```c++
template<typename T>
void ccsd_e(const TiledIndexSpace& MO, 
		    Tensor<T>& de,
		    const Tensor<T>& t1,
		    const Tensor<T>& t2,
		    const Tensor<T>& f1,
		    const Tensor<T>& v2) {
	const TiledIndexRange& O = MO.range("occ");
	const TiledIndexRange& V = MO.range("virt");
	Tensor<T> i1{O, V};

	TiledIndexLabel p1, p2, p3, p4, p5;
	TiledIndexLabel h3, h4, h5, h6;

	std::tie(p1, p2, p3, p4, p5) = MO.range_labels("virt", 1, 2, 3, 4, 5);
	std::tie(h3, h4, h5, h6) = MO.range_labels("occ", 1, 2, 3, 4);

	i1(h6,p5) = f1(h6,p5);
	i1(h6,p5) +=  0.5  * t1(p3,h4) * v2(h4,h6,p3,p5);
	de() =  0;
	de() += t1(p5,h6) * i1(h6,p5);
	de() +=  0.25  * t2(p1,p2,h3,h4) * v2(h3,h4,p1,p2);
}

template<typename T>
void driver(const TiledIndexSpace& MO) {
	const TiledIndexRange& O = MO.range("occ");
	const TiledIndexRange& V = MO.range("virt");
	const TiledIndexRange& N = MO.range("all");
	Tensor<T> de{};
	Tensor<T> t1{V, O};
	Tensor<T> t2{V, V, O, O};
	Tensor<T> f1{N, N};
	Tensor<T> v2{N, N, N, N};
	ccsd_e(MO, de, t1, t2, f1, v2);
}
```

### Canonical  T1

```c++
template<typename T>
void  ccsd_t1(const TiledIndexSpace& MO, 
		      Tensor<T>& i0, 
		      const Tensor<T>& t1, 
		      const Tensor<T>& t2,
              const Tensor<T>& f1, 
              const Tensor<T>& v2) { 
  const TiledIndexRange& O = MO.range("occ");
  const TiledIndexRange& V = MO.range("virt");
  Tensor<T> t1_2_1{O, O};
  Tensor<T> t1_2_2_1{O, V};
  Tensor<T> t1_3_1{V, V};
  Tensor<T> t1_5_1{O, V};
  Tensor<T> t1_6_1{O, O, V, V};

  TiledIndexLabel p2, p3, p4, p5, p6, p7;
  TiledIndexLabel h1, h4, h5, h6, h7, h8;

  std::tie(p2, p3, p4, p5, p6, p7) = MO.range_labels("virt", 1, 2, 3, 4, 5, 6);
  std::tie(h1, h4, h5, h6, h7, h8) = MO.range_labels("occ", 1, 2, 3, 4, 5, 6);  

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
void driver(const TiledIndexSpace& MO) {
	const TiledIndexRange& O = MO.range("occ");
	const TiledIndexRange& V = MO.range("virt");
	const TiledIndexRange& N = MO.range("all");

	Tensor<T> i0{};
	Tensor<T> t1{V, O};
	Tensor<T> t2{V, V, O, O};
	Tensor<T> f1{N, N};
	Tensor<T> v2{N, N, N, N};
	ccsd_t1(MO, i0, t1, t2, f1, v2);
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
   
    const TiledIndexRange& N_ao = AO.range("all"); 
    const TiledIndexRange& N_atom = AtomSpace.range("all");
    const TiledIndexRange& N_pao = SubPAO.range("all");
    const TiledIndexRange& O_mo = MO.range("occ");
    const TiledIndexRange& O_submo = SubMO.range("occ");
    
    TiledIndexLabel mu, nu, mu_p, nu_p;
    TiledIndexLabel i, j, i_p, j_p;
    TiledIndexLabel A, B;

    std::tie(mu, nu) = AO.range_labels("all", 0, 1);
    std::tie(mu_p, nu_p) = SubPAO.range_labels("all", 0, 1);
    std::tie(i,j) = MO.range_labels("occ", 0, 1);
    std::tie(i_p,j_p) = SubMO.range_labels("occ", 0, 1);
    std::tie(A, B) = AtomSpace.range_labels("all", 0, 1);
    
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
)or<T> SCCS{N_ao, N_ao};
    //auto SC(A, mu, i_p) = S(mu, nu) * C(A, nu, i);    auto SCCS(mu, nu) = SC(mu, i) * SC(nu, i_p);
    
    Tensor<T> L{N_ao, N_ao};
    L(mu, nu) = S(mu, nu);
    L(mu, nu) += -1 * SCCS(mu,nu)

	//now we interpret L to construct the mu_tilde (mu_p) index space  

    //Step 2
    Tens
    //TODO: support for absolute sum is needed
    l(A,B  
    //Step 2
    Tensor<T> l{N_atom, N_atom}; L(mu_subatom(A), u_subatom(B));
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
