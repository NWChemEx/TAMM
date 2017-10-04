pp {

index h1,h2,h3,h4,h5 = O;
index p1,p2,p3 = V;

array i0[V][V];
array t_vo[V][O]: irrep_t;
array y_ov[O][V]: irrep_y;
array t_vvoo[V,V][O,O]: irrep_t;
array y_oovv[O,O][V,V]: irrep_y;

pp_1:       i0[p1,p2] += 1 * t_vo[p1,h3] * y_ov[h3,p2];
pp_2:       i0[p1,p2] += 1/2 * t_vvoo[p1,p3,h4,h5] * y_oovv[h4,h5,p2,p3];

}
