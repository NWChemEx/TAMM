e {

range O,V = 10.0;
index h1,h2,h3,h4,h5 = O;
index h6,h7,h8 = O;
index p1,p2,p3,p4,p5 = V;
index p6,p7 = V;
array i0[][];
array t_vo[V][O];
array t_vvoo[V,V][O,O];
array i1[O][V];
array f[N][N];
array v[N,N][N,N];
array _a4[O][V];
array _a5[][];
array _a1[][];
array _a8[][];
 _a4[h1,p1] = t_vo[p2,h2] * v[h2,h1,p2,p1] ;
 _a5[] = t_vo[p1,h1] * _a4[h1,p1] ;
 _a1[] = t_vo[p1,h1] * f[h1,p1] ;
 _a8[] = t_vvoo[p1,p2,h1,h2] * v[h1,h2,p1,p2] ;
 i0[] = _a1[] + 0.5 * _a5[] + 0.25 * _a8[] ;
}
