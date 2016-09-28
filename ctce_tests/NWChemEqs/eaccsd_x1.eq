x1 {

index h1,h2,h3,h4,h5,h6 = O;
index p1,p2,p3,p4,p5,p6,p7 = V;

array i0[V][O];
array x_vo[V][O]: irrep_x;
array f[N][N]: irrep_f;
array t_vo[V][O]: irrep_t;
array v[N,N][N,N]: irrep_v;
array x_vvoo[V,V][O,O]: irrep_x;
array t_vvoo[V,V][O,O]: irrep_t;
array x1_4_1_1[O][V];
array x1_4_1[O][O];
array x1_2_1[O][V];
array x1_5_1[O,O][O,V];
array x1_1_1[V][V];

x1_1_1:     x1_1_1[p2,p6] += 1 * f[p2,p6];
x1_1_2:     x1_1_1[p2,p6] += 1 * t_vo[p3,h4] * v[h4,p2,p3,p6];
x1_1:       i0[p2,h1] += 1 * x_vo[p6,h1] * x1_1_1[p2,p6];
x1_2_1:     x1_2_1[h6,p7] += 1 * f[h6,p7];
x1_2_2:     x1_2_1[h6,p7] += 1 * t_vo[p3,h4] * v[h4,h6,p3,p7];
x1_2:       i0[p2,h1] += 1 * x_vvoo[p2,p7,h1,h6] * x1_2_1[h6,p7];
x1_3:       i0[p2,h1] += -1/2 * x_vvoo[p4,p5,h1,h3] * v[h3,p2,p4,p5];
x1_4_1_1:   x1_4_1_1[h3,p7] += 1 * f[h3,p7];
x1_4_1_2:   x1_4_1_1[h3,p7] += -1 * t_vo[p4,h5] * v[h3,h5,p4,p7];
x1_4_1:     x1_4_1[h3,h1] += 1 * x_vo[p7,h1] * x1_4_1_1[h3,p7];
x1_4_2:     x1_4_1[h3,h1] += 1/2 * x_vvoo[p5,p6,h1,h4] * v[h3,h4,p5,p6];
x1_4:       i0[p2,h1] += -1 * t_vo[p2,h3] * x1_4_1[h3,h1];
x1_5_1:     x1_5_1[h4,h5,h1,p3] += 1 * x_vo[p6,h1] * v[h4,h5,p3,p6];
x1_5:       i0[p2,h1] += 1/2 * t_vvoo[p2,p3,h4,h5] * x1_5_1[h4,h5,h1,p3];

}
