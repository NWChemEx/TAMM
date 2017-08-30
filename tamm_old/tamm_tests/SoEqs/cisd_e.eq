e {

range O = 10.0;
range V = 100.0;
range N = 110.0;
index h1,h2,h3,h4 = O;
index p1,p2 = V;
array f[N][N];
array t_vo[V][O];
array t_vvoo[V,V][O,O];
array v[N,N][N,N];
array i0[][];
array _a2[][];
array _a1[][];
 _a2[] = t_vvoo[p1,p2,h1,h2] * v[h1,h2,p1,p2] ;
 _a1[] = f[h1,p1] * t_vo[p1,h1] ;
 i0[] = _a1[] + 0.25 * _a2[] ;
}
