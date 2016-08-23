t1 {

index h1,h2,h3,h4,h5,h6,h7,h8 = O;
index p1,p2,p3,p4,p5,p6,p7 = V;

array i0[V][O];
array f[N][N]: irrep_f;

array t1_2_1[O][O];
array t1_2_2_1[O][V];

array t_vo[V][O]: irrep_t;
array v[N,N][N,N]: irrep_v;
array t_vvoo[V,V][O,O]: irrep_t;
array t1_3_1[V][V];
array t1_5_1[O][V];
array t1_6_1[O,O][O,V];


t1_1:       i0[p2,h1] += 1 * f[p2,h1];
t1_2_1:     t1_2_1[h7,h1] += 1 * f[h7,h1];
t1_2_2_1:   t1_2_2_1[h7,p3] += 1 * f[h7,p3];
t1_2_2_2:   t1_2_2_1[h7,p3] += -1 * t_vo[p5,h6] * v[h6,h7,p3,p5];
t1_2_2:     t1_2_1[h7,h1] += 1 * t_vo[p3,h1] * t1_2_2_1[h7,p3];
t1_2_3:     t1_2_1[h7,h1] += -1 * t_vo[p4,h5] * v[h5,h7,h1,p4];
t1_2_4:     t1_2_1[h7,h1] += -1/2 * t_vvoo[p3,p4,h1,h5] * v[h5,h7,p3,p4];
t1_2:       i0[p2,h1] += -1 * t_vo[p2,h7] * t1_2_1[h7,h1];
t1_3_1:     t1_3_1[p2,p3] += 1 * f[p2,p3];
t1_3_2:     t1_3_1[p2,p3] += -1 * t_vo[p4,h5] * v[h5,p2,p3,p4];
t1_3:       i0[p2,h1] += 1 * t_vo[p3,h1] * t1_3_1[p2,p3];
t1_4:       i0[p2,h1] += -1 * t_vo[p3,h4] * v[h4,p2,h1,p3];
t1_5_1:     t1_5_1[h8,p7] += 1 * f[h8,p7];
t1_5_2:     t1_5_1[h8,p7] += 1 * t_vo[p5,h6] * v[h6,h8,p5,p7];
t1_5:       i0[p2,h1] += 1 * t_vvoo[p2,p7,h1,h8] * t1_5_1[h8,p7];
t1_6_1:     t1_6_1[h4,h5,h1,p3] += 1 * v[h4,h5,h1,p3];
t1_6_2:     t1_6_1[h4,h5,h1,p3] += -1 * t_vo[p6,h1] * v[h4,h5,p3,p6];
t1_6:       i0[p2,h1] += -1/2 * t_vvoo[p2,p3,h4,h5] * t1_6_1[h4,h5,h1,p3];
t1_7:       i0[p2,h1] += -1/2 * t_vvoo[p3,p4,h1,h5] * v[h5,p2,p3,p4];

}
