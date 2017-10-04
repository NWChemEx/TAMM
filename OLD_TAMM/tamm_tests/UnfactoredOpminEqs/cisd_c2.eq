c2 {

range O,V = 10.0;
index h1,h2,h3,h4,h5 = O;
index h6 = O;
index p1,p2,p3,p4,p5 = V;
index p6 = V;
array i0[V,V][O,O];
array t_vo[V][O];
array t_vvoo[V,V][O,O];
scalar e;
array v[N,N][N,N];
array f[N][N];
 i0[p3,p4,h1,h2] += -1.0 * t_vvoo[p3,p1,h1,h3] * v[h3,p4,h2,p1] ;
 i0[p3,p4,h1,h2] += 0.5 * t_vvoo[p3,p4,h3,h4] * v[h3,h4,h1,h2] ;
 i0[p3,p4,h1,h2] += t_vvoo[p3,p1,h1,h2] * f[p4,p1] ;
 i0[p3,p4,h1,h2] += -1.0 * t_vvoo[p3,p4,h1,h3] * f[h3,h2] ;
 i0[p3,p4,h1,h2] += -1.0 * t_vo[p1,h1] * v[p3,p4,h2,p1] ;
 i0[p3,p4,h1,h2] += -1.0 * t_vo[p3,h3] * v[h3,p4,h1,h2] ;
 i0[p3,p4,h1,h2] += t_vo[p3,h1] * f[p4,h2] ;
 i0[p3,p4,h1,h2] += -1.0 * e * t_vvoo[p3,p4,h1,h2] ;
 i0[p3,p4,h1,h2] += 0.5 * t_vvoo[p1,p2,h1,h2] * v[p3,p4,p1,p2] ;
 i0[p3,p4,h1,h2] += v[p3,p4,h1,h2];
}
