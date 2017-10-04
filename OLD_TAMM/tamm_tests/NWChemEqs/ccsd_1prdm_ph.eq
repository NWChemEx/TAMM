ph {

index h1,h2,h3,h4,h5,h6,h7 = O;
index p1,p2,p3,p4,p5,p6 = V;

array i0[V][O];
array t_vo[V][O]: irrep_t;
array t_vvoo[V,V][O,O]: irrep_t;
array y_ov[O][V]: irrep_y;
array y_oovv[O,O][V,V]: irrep_y;
array ph_3_1[O][O];
array ph_4_1[O,O][O,V];

ph_1:       i0[p2,h1] += 1 * t_vo[p2,h1];
ph_2:       i0[p2,h1] += 1 * t_vvoo[p2,p3,h1,h4] * y_ov[h4,p3];
ph_3_1:     ph_3_1[h7,h1] += 1 * t_vo[p3,h1] * y_ov[h7,p3];
ph_3_2:     ph_3_1[h7,h1] += -1/2 * t_vvoo[p3,p4,h1,h5] * y_oovv[h5,h7,p3,p4];
ph_3:       i0[p2,h1] += -1 * t_vo[p2,h7] * ph_3_1[h7,h1];
ph_4_1:     ph_4_1[h4,h5,h1,p3] += -1 * t_vo[p6,h1] * y_oovv[h4,h5,p3,p6];
ph_4:       i0[p2,h1] += -1/2 * t_vvoo[p2,p3,h4,h5] * ph_4_1[h4,h5,h1,p3];

}
