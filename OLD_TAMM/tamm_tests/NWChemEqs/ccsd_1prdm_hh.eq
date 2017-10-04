hh {

index h1,h2,h3,h4,h5 = O;
index p1,p2,p3,p4 = V;

array i0[O][O];
array t_vo[V][O]: irrep_t;
array y_ov[O][V]: irrep_y;
array t_vvoo[V,V][O,O]: irrep_t;
array y_oovv[O,O][V,V]: irrep_y;

hh_1:       i0[h2,h1] += -1 * t_vo[p3,h1] * y_ov[h2,p3];
hh_2:       i0[h2,h1] += -1/2 * t_vvoo[p3,p4,h1,h5] * y_oovv[h2,h5,p3,p4];

}
