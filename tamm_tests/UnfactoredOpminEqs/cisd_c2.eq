c2 {

range O,V = 10.0;
index h1,h2,h3,h4,h5 = O;
index h6 = O;
index p1,p2,p3,p4,p5 = V;
index p6 = V;
array i0[V,V][O,O];
array t_vo[V][O];
array t_vvoo[V,V][O,O];
array e[][];
array v[N,N][N,N];
array f[N][N];
array _a7[V,V][O,O];
array _a6[V,V][O,O];
array _a5[V,V][O,O];
array _a4[V,V][O,O];
array _a3[V,V][O,O];
array _a2[V,V][O,O];
array _a1[V,V][O,O];
array _a9[V,V][O,O];
array _a8[V,V][O,O];
 _a7[p3,p4,h1,h2] = t_vvoo[p3,p1,h1,h3] * v[h3,p4,h2,p1] ;
 _a6[p3,p4,h1,h2] = t_vvoo[p3,p4,h3,h4] * v[h3,h4,h1,h2] ;
 _a5[p3,p4,h1,h2] = t_vvoo[p3,p1,h1,h2] * f[p4,p1] ;
 _a4[p3,p4,h1,h2] = t_vvoo[p3,p4,h1,h3] * f[h3,h2] ;
 _a3[p3,p4,h1,h2] = t_vo[p1,h1] * v[p3,p4,h2,p1] ;
 _a2[p3,p4,h1,h2] = t_vo[p3,h3] * v[h3,p4,h1,h2] ;
 _a1[p3,p4,h1,h2] = t_vo[p3,h1] * f[p4,h2] ;
 _a9[p3,p4,h1,h2] = e[] * t_vvoo[p3,p4,h1,h2] ;
 _a8[p3,p4,h1,h2] = t_vvoo[p1,p2,h1,h2] * v[p3,p4,p1,p2] ;
 i0[p3,p4,h1,h2] = v[p3,p4,h1,h2] + _a1[p3,p4,h1,h2] + - _a2[p3,p4,h1,h2] + - _a3[p3,p4,h1,h2] + - _a4[p3,p4,h1,h2] + _a5[p3,p4,h1,h2] + 0.5 * _a6[p3,p4,h1,h2] + - _a7[p3,p4,h1,h2] + 0.5 * _a8[p3,p4,h1,h2] + - _a9[p3,p4,h1,h2] ;
}
