
TAMM Input Equations
=============

Declarations:
------------

- **Range declarations (Optional)**  
   > range O = 50;  
    range V = 100;

- **Index Declarations (can only be of types O,V,N)**  
   > index h1,h2,h3 : O;  
   index p1,p2,p3 : V;

 - **Array Declarations**  
  //default = anti-symmetry  
    > array v[O,O][V,V];  (upper indices follower by lower)  
  (OR)  
  array v[O,O][V,V] : irrep_v; (irrep is optional)  
  
scalar s; // A scalar is a zero dimensional tensor - cannot be initialized with a constant  

- **Operations**  
An assignment can only be of the form:  

  > [S1label:] c += alpha * a  
  [S2label:] c += alpha * a * b,   

  -  where c, a, b are tensors and alpha a constant. The statement labels are optional  

  #### Examples:  
    >t1_1:   t1_2_1[h7,p3] += 1 * f[h7,p3];  
    t1_2:   t1_2_1[h7,p3] += -1 * t_vo[p5,h6] * v[h6,h7,p3,p5];  

  (OR)  

    >t1_2_2_1[h7,p3] += 1 * f[h7,p3];  
    t1_2_2_1[h7,p3] += -1 * t_vo[p5,h6] * v[h6,h7,p3,p5];  

  Involving a scalar:  
  >   s += 1/4  * t_vvoo[p1,p2,h3,h4] * v[h3,h4,p1,p2];  

  A scalar 's' declared cannot be used as a constant variable on the RHS of an assignment.  
  If used, it will be treated as a tensor involved in a contraction or addition.  

An example Equation:  
--------------------
> ccsd_e {  
    index h1,h2,h3,h4,h5,h6,h7,h8 = O;
    index p1,p2,p3,p4,p5,p6,p7 = V;      
    scalar i0;  
    array f[N][N]: irrep_f;  
    array v[N,N][N,N]: irrep_v;  
    array t_vo[V][O]: irrep_t;  
    array t_vvoo[V,V][O,O]: irrep_t;  
    array i1[O][V];  
    e_1_1:   i1[h6,p5] += 1 * f[h6,p5];  
    e_1_2:   i1[h6,p5] += 1/2 * t_vo[p3,h4] * v[h4,h6,p3,p5];  
    e_1:     i0 += 1 * t_vo[p5,h6] * i1[h6,p5];  
    e_2:     i0 += 1/4  * t_vvoo[p1,p2,h3,h4] * v[h3,h4,p1,p2];  
  }

