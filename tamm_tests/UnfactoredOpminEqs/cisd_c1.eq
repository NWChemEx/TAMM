c1 {

range O,V = 10.0;
index h1,h2,h3,h4,h5 = O;
index p1,p2,p3,p4 = V;
array i0[V][O];
array t_vo[V][O];
array t_vvoo[V,V][O,O];
scalar e;
array f[N][N];
array v[N,N][N,N];
 i0[p2,h1] += -1.0 * e * t_vo[p2,h1] ;
 i0[p2,h1] += -0.5 * t_vvoo[p1,p3,h1,h2] * v[h2,p2,p1,p3] ;
 i0[p2,h1] += -0.5 * t_vvoo[p2,p1,h2,h3] * v[h2,h3,h1,p1] ;
 i0[p2,h1] += t_vvoo[p2,p1,h1,h2] * f[h2,p1] ;
 i0[p2,h1] += -1.0 * t_vo[p1,h2] * v[h2,p2,h1,p1] ;
 i0[p2,h1] += t_vo[p1,h1] * f[p2,p1] ;
 i0[p2,h1] += -1.0 * t_vo[p2,h2] * f[h2,h1] ;
 i0[p2,h1] += f[p2,h1];
}
