ccsd_e {

index h1,h2,h3,h4,h5,h6,h7,h8 = O;
index p1,p2,p3,p4,p5,p6,p7 = V;

array i0[][];
array f[N][N]: irrep_f;
array v[N,N][N,N]: irrep_v;
array t_vo[V][O]: irrep_t;
array t_vvoo[V,V][O,O]: irrep_t;
array i1[O][V];

e_1_1:   i1[h6,p5] += 1 * f[h6,p5];
e_1_2:   i1[h6,p5] += 1/2 * t_vo[p3,h4] * v[h4,h6,p3,p5];
e_1:     i0[] += 1 * t_vo[p5,h6] * i1[h6,p5];
e_2:     i0[] += 1/4  * t_vvoo[p1,p2,h3,h4] * v[h3,h4,p1,p2];

}

