{

range O = 10;
range V = 10;

index h1, h2, h3, h4, h5, h6, h7, h8, h9, h10 = O;
index p1, p2, p3, p4, p5, p6, p7, p8, p9 = V;

array i0([V][O,O]);
array x_o([][O]);
array t_vo([V][O]);
array t_vvoo([V,V][O,O]);
array x_voo([V][O,O]);
array x2_2_1([O][O]);
array x2_1_2_1([O,V][O,V]);
array x2_6_1([O][O,O]);
array x2_4_2_1([O,O][O,V]);
array x2_3_1([V][V]);
array x2_6_1_1([O,O][O,O]);
array x2_7_1([][V]);
array x2_8_1([O][O,V]);
array x2_5_1([V,O][O,V]);
array x2_6_1_2_1([O,O][O,V]);
array x2_6_2_1([O][V]);
array x2_1_1([V,O][O,O]);
array x2_4_1([O,O][O,O]);
array x2_6_3_1([O,O][O,V]);
array x2_1_3_1([O][V]);
array x2_2_2_1([O][V]);
array x2_1_4_1([O,O][O,V]);
array v_ovoo([O,V][O,O]);
array v_ovov([O,V][O,V]);
array v_ovvv([O,V][V,V]);
array f_ov([O][V]);
array v_oovv([O,O][V,V]);
array v_ooov([O,O][O,V]);
array f_oo([O][O]);
array f_vv([V][V]);
array v_oooo([O,O][O,O]);

i0[p3,h1,h2] = +1.0 * x_o[h9] * v_ovoo[h9,p3,h1,h2]
-1.0 * x_o[h9] * t_vo[p5,h1] * v_ovov[h9,p3,h1,p5]
+0.5 * x_o[h9] * t_vo[p5,h1] * t_vo[p6,h1] * v_ovvv[h9,p3,p5,p6]
-1.0 * x_o[h9] * t_vvoo[p3,p8,h1,h2] * f_ov[h9,p8]
-1.0 * x_o[h9] * t_vvoo[p3,p8,h1,h2] * t_vo[p6,h7] * v_oovv[h7,h9,p6,p8]
+1.0 * x_o[h9] * t_vvoo[p3,p5,h1,h6] * v_ooov[h6,h9,h1,p5]
-1.0 * x_o[h9] * t_vvoo[p3,p5,h1,h6] * t_vo[p7,h1] * v_oovv[h6,h9,p5,p7]
+0.5 * x_o[h9] * t_vvoo[p5,p6,h1,h2] * v_ovvv[h9,p3,p5,p6]
-1.0 * x_voo[p3,h1,h8] * f_oo[h8,h1]
-1.0 * x_voo[p3,h1,h8] * t_vo[p9,h1] * f_ov[h8,p9]
-1.0 * x_voo[p3,h1,h8] * t_vo[p9,h1] * t_vo[p6,h7] * v_oovv[h7,h8,p6,p9]
+1.0 * x_voo[p3,h1,h8] * t_vo[p5,h6] * v_ooov[h6,h8,h1,p5]
+0.5 * x_voo[p3,h1,h8] * t_vvoo[p5,p6,h1,h7] * v_oovv[h7,h8,p5,p6]
+1.0 * x_voo[p8,h1,h2] * f_vv[p3,p8]
+1.0 * x_voo[p8,h1,h2] * t_vo[p5,h6] * v_ovvv[h6,p3,p5,p8]
+0.5 * x_voo[p8,h1,h2] * t_vvoo[p3,p5,h6,h7] * v_oovv[h6,h7,p5,p8]
+0.5 * x_voo[p3,h9,h10] * v_oooo[h9,h10,h1,h2]
-0.5 * x_voo[p3,h9,h10] * t_vo[p5,h1] * v_ooov[h9,h10,h1,p5]
+0.25 * x_voo[p3,h9,h10] * t_vo[p5,h1] * t_vo[p6,h1] * v_oovv[h9,h10,p5,p6]
+0.25 * x_voo[p3,h9,h10] * t_vvoo[p5,p6,h1,h2] * v_oovv[h9,h10,p5,p6]
-1.0 * x_voo[p8,h1,h7] * v_ovov[h7,p3,h1,p8]
-1.0 * x_voo[p8,h1,h7] * t_vo[p5,h1] * v_ovvv[h7,p3,p5,p8]
-1.0 * t_vo[p3,h10] * x_o[h8] * v_oooo[h8,h10,h1,h2]
+1.0 * t_vo[p3,h10] * x_o[h8] * t_vo[p5,h1] * v_ooov[h8,h10,h1,p5]
-0.5 * t_vo[p3,h10] * x_o[h8] * t_vo[p5,h1] * t_vo[p6,h1] * v_oovv[h8,h10,p5,p6]
-0.5 * t_vo[p3,h10] * x_o[h8] * t_vvoo[p5,p6,h1,h2] * v_oovv[h8,h10,p5,p6]
-1.0 * t_vo[p3,h10] * x_voo[p5,h1,h2] * f_ov[h10,p5]
+1.0 * t_vo[p3,h10] * x_voo[p5,h1,h2] * t_vo[p6,h7] * v_oovv[h7,h10,p5,p6]
+1.0 * t_vo[p3,h10] * x_voo[p9,h1,h8] * v_ooov[h8,h10,h1,p9]
+1.0 * t_vo[p3,h10] * x_voo[p9,h1,h8] * t_vo[p5,h1] * v_oovv[h8,h10,p5,p9]
+0.5 * t_vvoo[p3,p5,h1,h2] * x_voo[p8,h6,h7] * v_oovv[h6,h7,p5,p8]
-1.0 * t_vvoo[p3,p5,h1,h6] * x_voo[p8,h1,h7] * v_oovv[h6,h7,p5,p8]
;

}
