{

range O = 10;
range V = 10;

index h1, h2, h3, h4, h5, h6, h7, h8 = O;
index p1, p2, p3, p4, p5, p6, p7 = V;

array i0([][O]);
array x_o([][O]);
array t_vo([V][O]);
array t_vvoo([V,V][O,O]);
array x_voo([V][O,O]);
array x1_2_1([O][V]);
array x1_1_2_1([O][V]);
array x1_3_1([O,O][O,V]);
array x1_1_1([O][O]);
array f_oo([O][O]);
array f_ov([O][V]);
array v_oovv([O,O][V,V]);
array v_ooov([O,O][O,V]);

i0[h1] = -1.0 * x_o[h6] * f_oo[h6,h1]
-1.0 * x_o[h6] * t_vo[p7,h1] * f_ov[h6,p7]
-1.0 * x_o[h6] * t_vo[p7,h1] * t_vo[p4,h5] * v_oovv[h5,h6,p4,p7]
+1.0 * x_o[h6] * t_vo[p3,h4] * v_ooov[h4,h6,h1,p3]
+0.5 * x_o[h6] * t_vvoo[p3,p4,h1,h5] * v_oovv[h5,h6,p3,p4]
-1.0 * x_voo[p7,h1,h6] * f_ov[h6,p7]
-1.0 * x_voo[p7,h1,h6] * t_vo[p3,h4] * v_oovv[h4,h6,p3,p7]
+0.5 * x_voo[p7,h6,h8] * v_ooov[h6,h8,h1,p7]
+0.5 * x_voo[p7,h6,h8] * t_vo[p3,h1] * v_oovv[h6,h8,p3,p7]
;

}
