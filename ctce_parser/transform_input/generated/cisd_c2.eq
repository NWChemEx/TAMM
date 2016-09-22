c2 {

index h1,h2,h3,h4,h5,h6 = O;
index p1,p2,p3,p4,p5,p6 = V;

array i0[V,V][O,O];
array v[N,N][N,N]: irrep_v;
array t_vo[V][O]: irrep_t;
array f[N][N]: irrep_f;
array t_vvoo[V,V][O,O]: irrep_t;
array e[][]: irrep_e;

c2_1:       i0[p3,p4,h1,h2] += 1 * v[p3,p4,h1,h2];
c2_2:       i0[p3,p4,h1,h2] += 1 * t_vo[p3,h1] * f[p4,h2];
c2_3:       i0[p3,p4,h1,h2] += -1 * t_vo[p3,h5] * v[h5,p4,h1,h2];
c2_4:       i0[p3,p4,h1,h2] += -1 * t_vo[p5,h1] * v[p3,p4,h2,p5];
c2_5:       i0[p3,p4,h1,h2] += -1 * t_vvoo[p3,p4,h1,h5] * f[h5,h2];
c2_6:       i0[p3,p4,h1,h2] += 1 * t_vvoo[p3,p5,h1,h2] * f[p4,p5];
c2_7:       i0[p3,p4,h1,h2] += 1/2 * t_vvoo[p3,p4,h5,h6] * v[h5,h6,h1,h2];
c2_8:       i0[p3,p4,h1,h2] += -1 * t_vvoo[p3,p5,h1,h6] * v[h6,p4,h2,p5];
c2_9:       i0[p3,p4,h1,h2] += 1/2 * t_vvoo[p5,p6,h1,h2] * v[p3,p4,p5,p6];
c2_10:      i0[p3,p4,h1,h2] += -1 * e * t_vvoo[p3,p4,h1,h2];

}
