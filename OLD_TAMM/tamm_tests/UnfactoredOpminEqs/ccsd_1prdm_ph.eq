ph {

range O,V = 10.0;
index h1,h2,h3,h4,h5 = O;
index h6,h7 = O;
index p1,p2,p3,p4,p5 = V;
index p6 = V;
array i0[V][O];
array t_vo[V][O];
array t_vvoo[V,V][O,O];
array y_ov[O][V];
array y_oovv[O,O][V,V];
array _a4[O][O];
array _a10[O][O];
array _a15[V][V];
 _a4[h2,h1] = t_vo[p1,h1] * y_ov[h2,p1] ;
 i0[p2,h1] += -1.0 * t_vo[p2,h2] * _a4[h2,h1] ;
 i0[p2,h1] += t_vvoo[p2,p1,h1,h2] * y_ov[h2,p1] ;
 _a10[h2,h1] = t_vvoo[p1,p3,h1,h3] * y_oovv[h3,h2,p1,p3] ;
 i0[p2,h1] += 0.5 * t_vo[p2,h2] * _a10[h2,h1] ;
 _a15[p2,p3] = t_vvoo[p2,p1,h2,h3] * y_oovv[h2,h3,p1,p3] ;
 i0[p2,h1] += 0.5 * t_vo[p3,h1] * _a15[p2,p3] ;
 i0[p2,h1] += t_vo[p2,h1];
}
