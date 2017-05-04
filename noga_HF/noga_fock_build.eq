noga {

// index h1,h2,h3,h4,h5,h6 = O;
// index p1,p2,p3,p4,p5,p6 = V;

index p,q,r,s = N;
index i, j, m, n = O;
index a, b, c, d, e, f = V;

//array f[N][N]: irrep_f;
//array v[N,N][N,N]: irrep_v;
//array t_vo[V][O]: irrep_t;

scalar bDiag;
array bT[N][N]; //{Q};
array hT[N][N];
array FT[N][N];

array X_VV[V][V];
array X_OO[O][O];
array X_OV[O][V];

scalar t1;
scalar t2;
scalar t3;
array t4[O][N];
array t5[O][N];
array t6[O][N];
array t7[V][N];

hf_1: FT[p,q] += 1.0 * hT[p,q];

hf_2:  FT[p,q] += bDiag * bT[p,q];

hf_3_1: t1 += X_OO[i,j] * bT[i,j];
hf_3: FT[p,q] += bT[p,q] * t1;

hf_4_1: t2 += 2.0 * X_OV[i,a] * bT[i,a];
hf_4:  FT[p,q] += t2 * bT[p,q];

hf_5_1: t3 += X_VV[a,b] * bT[a,b];
hf_5: FT[p,q] += bT[p,q] * t3;

hf_6: FT[p,q] += -1.0 * bT[p,i] * bT[i,q];

hf_7_1: t4[i,q] += X_OO[i,j] * bT[j,q];
hf_7:  FT[p,q] += -1.0 *  bT[p,i] * t4[i,q];

hf_8_1: t5[i,q] += X_OV[i,a] * bT[a,q];
hf_8:  FT[p,q] += -1.0 * bT[p,i] * t5[i,q];

hf_9_1: t6[i,p] += X_OV[i,a] * bT[p,a];
hf_9: FT[p,q] += -1.0 * bT[i,q] * t6[i,p];

hf_10_1: t7[a,q] += X_VV[a,b] * bT[b,q];
hf_10:  FT[p,q] += -1.0 * bT[p,a] * t7[a,q];

}

