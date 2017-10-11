{

range O = 10;
range V = 100;
range N = 110;

index h1,h2,h3,h4,h5,h6,h7,h8 = O;
index p1,p2,p3,p4,p5,p6,p7 = V;

array f_vo([V][O]);
array t_vo([V][O]);
array f_ov([O][V]);
array v_oovv([O,O][V,V]);
array v_ooov([O,O][O,V]);
array t_vvoo([V,V][O,O]);
array f_oo([O][O]);
array f_vv([V][V]);
array v_ovvv([O,V][V,V]);
array v_ovov([O,V][O,V]);
array i0([V][O]);

i0[p2,h1] = 
 + 1.0 * f_vo[p2,h1]
 - 1.0 * t_vo[p2,h7] * t_vo[p3,h1] * f_ov[h7,p3]
 + 1.0 * t_vo[p2,h7] * t_vo[p3,h1] * t_vo[p5,h6] * v_oovv[h6,h7,p3,p5]
 + 1.0 * t_vo[p2,h7] * t_vo[p4,h5] * v_ooov[h5,h7,h1,p4]
 + 0.5 * t_vo[p2,h7] * t_vvoo[p3,p4,h1,h5] * v_oovv[h5,h7,p3,p4]
 - 1.0 * t_vo[p2,h7] * f_oo[h7,h1]
 + 1.0 * t_vo[p3,h1] * f_vv[p2,p3]
 - 1.0 * t_vo[p3,h1] * t_vo[p4,h5] * v_ovvv[h5,p2,p3,p4]
 - 1.0 * t_vo[p3,h4] * v_ovov[h4,p2,h1,p3]
 + 1.0 * t_vvoo[p2,p7,h1,h8] * f_ov[h8,p7]
 + 1.0 * t_vvoo[p2,p7,h1,h8] * t_vo[p5,h6] * v_oovv[h6,h8,p5,p7]
 - 0.5 * t_vvoo[p2,p3,h4,h5] * v_ooov[h4,h5,h1,p3]
 + 0.5 * t_vvoo[p2,p3,h4,h5] * t_vo[p6,h1] * v_oovv[h4,h5,p3,p6]
 - 0.5 * t_vvoo[p3,p4,h1,h5] * v_ovvv[h5,p2,p3,p4];

}
