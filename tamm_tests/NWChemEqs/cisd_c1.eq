c1 {

index h1,h2,h3,h4,h5 = O;
index p1,p2,p3,p4 = V;

array i0[V][O];
array f[N][N]: irrep_f;
array t_vo[V][O]: irrep_t;
array v[N,N][N,N]: irrep_v;
array t_vvoo[V,V][O,O]: irrep_t;
array e[][]: irrep_e;

c1_1:       i0[p2,h1] += 1 * f[p2,h1];
c1_2:       i0[p2,h1] += -1 * t_vo[p2,h3] * f[h3,h1];
c1_3:       i0[p2,h1] += 1 * t_vo[p3,h1] * f[p2,p3];
c1_4:       i0[p2,h1] += -1 * t_vo[p3,h4] * v[h4,p2,h1,p3];
c1_5:       i0[p2,h1] += 1 * t_vvoo[p2,p4,h1,h3] * f[h3,p4];
c1_6:       i0[p2,h1] += -1/2 * t_vvoo[p2,p3,h4,h5] * v[h4,h5,h1,p3];
c1_7:       i0[p2,h1] += -1/2 * t_vvoo[p3,p4,h1,h5] * v[h5,p2,p3,p4];
c1_8:       i0[p2,h1] += -1 * e[] * t_vo[p2,h1];

}
