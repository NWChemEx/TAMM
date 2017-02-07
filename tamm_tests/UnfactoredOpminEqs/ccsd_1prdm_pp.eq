pp {

range O,V = 10.0;
index h1,h2,h3,h4,h5 = O;
index p1,p2,p3 = V;
array i0[V][V];
array t_vo[V][O];
array y_ov[O][V];
array t_vvoo[V,V][O,O];
array y_oovv[O,O][V,V];
array _a2[V][V];
array _a1[V][V];
 _a2[p1,p2] = t_vvoo[p1,p3,h1,h2] * y_oovv[h1,h2,p2,p3] ;
 _a1[p1,p2] = t_vo[p1,h1] * y_ov[h1,p2] ;
 i0[p1,p2] = _a1[p1,p2] + 0.5 * _a2[p1,p2] ;
}
