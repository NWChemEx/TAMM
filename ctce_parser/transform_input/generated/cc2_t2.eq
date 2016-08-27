t2 {

index h1,h2,h3,h4,h5,h6,h7,h8,h9,h10 = O;
index p1,p2,p3,p4,p5,p6 = V;

array i0[V,V][O,O];
array v[N,N][N,N]: irrep_v;
array t_vo[V][O]: irrep_t;
array t_vvoo[V,V][O,O]: irrep_t;
array f[N][N]: irrep_f;
array t2_3_1[V,V][O,V];
array t2_2_2_1[O,O][O,O];
array t2_2_2_2_1[O,O][O,V];
array t2_2_3_1[O,V][O,V];
array t2_2_1[O,V][O,O];

t2_1:       i0[p3,p4,h1,h2] += 1 * v[p3,p4,h1,h2];
t2_2_1:     t2_2_1[h10,p3,h1,h2] += 1 * v[h10,p3,h1,h2];
t2_2_2_1:   t2_2_2_1[h8,h10,h1,h2] += 1 * v[h8,h10,h1,h2];
t2_2_2_2_1: t2_2_2_2_1[h8,h10,h1,p5] += 1 * v[h8,h10,h1,p5];
t2_2_2_2_2: t2_2_2_2_1[h8,h10,h1,p5] += -1/2 * t_vo[p6,h1] * v[h8,h10,p5,p6];
t2_2_2_2:   t2_2_2_1[h8,h10,h1,h2] += -1 * t_vo[p5,h1] * t2_2_2_2_1[h8,h10,h2,p5];
t2_2_2:     t2_2_1[h10,p3,h1,h2] += 1/2 * t_vo[p3,h8] * t2_2_2_1[h8,h10,h1,h2];
t2_2_3_1:   t2_2_3_1[h10,p3,h1,p5] += 1 * v[h10,p3,h1,p5];
t2_2_3_2:   t2_2_3_1[h10,p3,h1,p5] += -1/2 * t_vo[p6,h1] * v[h10,p3,p5,p6];
t2_2_3:     t2_2_1[h10,p3,h1,h2] += -1 * t_vo[p5,h1] * t2_2_3_1[h10,p3,h2,p5];
t2_2:       i0[p3,p4,h1,h2] += -1 * t_vo[p3,h10] * t2_2_1[h10,p4,h1,h2];
t2_3_1:     t2_3_1[p3,p4,h1,p5] += 1 * v[p3,p4,h1,p5];
t2_3_2:     t2_3_1[p3,p4,h1,p5] += -1/2 * t_vo[p6,h1] * v[p3,p4,p5,p6];
t2_3:       i0[p3,p4,h1,h2] += -1 * t_vo[p5,h1] * t2_3_1[p3,p4,h2,p5];
t2_4:       i0[p3,p4,h1,h2] += -1 * t_vvoo[p3,p4,h1,h5] * f[h5,h2];
t2_5:       i0[p3,p4,h1,h2] += 1 * t_vvoo[p3,p5,h1,h2] * f[p4,p5];

}
