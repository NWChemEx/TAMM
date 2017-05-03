noga {

// index h1,h2,h3,h4,h5,h6 = O;
// index p1,p2,p3,p4,p5,p6 = V;

index p,q,r,s = N;
index i, j, m, n = O;
index a, b, c, d, e, f = V;

//array f[N][N]: irrep_f;
//array v[N,N][N,N]: irrep_v;
//array t_vo[V][O]: irrep_t;


array bT[N][N];
array bDiag[O][O];
array hT[N][N];
array F[N][N];

array X_VV[V][V];
array X_OO[O][O];
array X_OV[O][V];

array t1[N][N];
array t2[][];
array t3[][];
array t4[O][N];
array t5[O][N];
array t6[O][N];
array t7[V][N];

hf_1: F[p,q] += 1.0 * hT[p,q];

hf_2:  F[p,q] += bT[p,q] * bDiag[i][i];

hf_3_1: t1[] = X_OO[i,j] * bT[i,j];
hf_3: F[p,q] += bT[p,q] * t1[];

hf_4_1: t2[] += X_OV[i][a] * b_T[i][a];
hf_4:  F_T[p,q] += 2 * b_T[p][q] * t2[];

hf_5_1: t3[] += X_VV[a][b] * b_T[a][b]
hf_5: F_T[p][q] += b_T[p][q] * t3[];

hf_6: F_T[p][q] += -1.0 * b_T[p][i] * b_T[i][q];

hf_7_1: t4[i,q] += X_OO[i][j] * b_T[j][q];
hf_7:  F_T[p][q] += -1.0 *  b_T[p][i] * t4[i,q];

hf_8_1: t5[i,q] += X_OV[i][a] * b_T[a][q];
hf_8:  F_T[p][q] += -1.0 * b_T[p][i] * t5[i,q];

hf_9_1: t6[i,p] += X_OV[i][a] * b_T[p][a];
hf_9: F_T[p][q] += -1.0 * b_T[i][q] * t6[i,p];

hf_10_1: t7[a,q] += X_VV[a][b] * b_T[b][q];
hf_10:  F_T[p][q] += -1.0 b_T[p][a] * t7[a,q];


}

