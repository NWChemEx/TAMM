e {

index h1,h2,h3,h4 = O;
index p1,p2 = V;

array i0[][];
array t_vo[V][O]: irrep_t;
array f[N][N]: irrep_f;
array t_vvoo[V,V][O,O]: irrep_t;
array v[N,N][N,N]: irrep_v;

e_1:       i0[] += 1 * t_vo[p2,h1] * f[h1,p2];
e_2:       i0[] += 1/4 * t_vvoo[p1,p2,h3,h4] * v[h3,h4,p1,p2];

}
