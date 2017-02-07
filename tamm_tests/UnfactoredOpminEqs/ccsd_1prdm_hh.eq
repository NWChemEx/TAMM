hh {

range O,V = 10.0;
index h1,h2,h3,h4,h5 = O;
index p1,p2,p3,p4 = V;
array i0[O][O];
array t_vo[V][O];
array y_ov[O][V];
array t_vvoo[V,V][O,O];
array y_oovv[O,O][V,V];
array _a2[O][O];
array _a1[O][O];
 _a2[h2,h1] = t_vvoo[p1,p2,h1,h3] * y_oovv[h2,h3,p1,p2] ;
 _a1[h2,h1] = t_vo[p1,h1] * y_ov[h2,p1] ;
 i0[h2,h1] = - _a1[h2,h1] + - 0.5 * _a2[h2,h1] ;
}
