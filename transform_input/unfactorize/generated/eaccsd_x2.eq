{

range O = 10;
range V = 10;

index h1, h2, h3, h4, h5, h6, h7, h8, h9, h10 = O;
index p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = V;

array i0([V,V][O]);
array x_v([V][]);
array x_vvo([V,V][O]);
array t_vo([V][O]);
array t_vvoo([V,V][O,O]);
array x2_10_1([V,O][V]);
array x2_2_1([O][O]);
array x2_6_1([O,V][O]);
array x2_6_5_1([O,O][O]);
array x2_3_1([V][V]);
array x2_6_7_1([O,O][V]);
array x2_8_1([O][]);
array x2_7_1([V,V][V]);
array x2_6_2_1([O][V]);
array x2_8_1_1([O][V]);
array x2_9_1([O,O][O]);
array x2_6_6_1([V,O][V]);
array x2_9_3_1([O,O][V]);
array x2_4_1([O,V][O,V]);
array x2_6_5_3_1([O,O][V]);
array x2_6_3_1([O,O][O,V]);
array x2_2_2_1([O][V]);
array v_vvov([V,V][O,V]);
array f_oo([O][O]);
array f_ov([O][V]);
array v_oovv([O,O][V,V]);
array v_ooov([O,O][O,V]);
array f_vv([V][V]);
array v_ovvv([O,V][V,V]);
array v_ovov([O,V][O,V]);
array v_vvvv([V,V][V,V]);

i0[p3,p4,h1] = +1.0 * x_v[p5] * v_vvov[p3,p4,h2,p5]
-1.0 * x_vvo[p3,p4,h8] * f_oo[h8,h1]
-1.0 * x_vvo[p3,p4,h8] * t_vo[p9,h1] * f_ov[h8,p9]
-1.0 * x_vvo[p3,p4,h8] * t_vo[p9,h1] * t_vo[p6,h7] * v_oovv[h7,h8,p6,p9]
+1.0 * x_vvo[p3,p4,h8] * t_vo[p5,h6] * v_ooov[h6,h8,h1,p5]
+0.5 * x_vvo[p3,p4,h8] * t_vvoo[p5,p6,h1,h7] * v_oovv[h7,h8,p5,p6]
+1.0 * x_vvo[p3,p8,h1] * f_vv[p3,p8]
+1.0 * x_vvo[p3,p8,h1] * t_vo[p5,h6] * v_ovvv[h6,p3,p5,p8]
+0.5 * x_vvo[p3,p8,h1] * t_vvoo[p3,p5,h6,h7] * v_oovv[h6,h7,p5,p8]
-1.0 * x_vvo[p3,p8,h7] * v_ovov[h7,p3,h1,p8]
-1.0 * x_vvo[p3,p8,h7] * t_vo[p5,h1] * v_ovvv[h7,p3,p5,p8]
+0.5 * x_vvo[p5,p6,h1] * v_vvvv[p3,p4,p5,p6]
-1.0 * t_vo[p3,h9] * x_v[p6] * v_ovov[h9,p3,h2,p6]
+1.0 * t_vo[p3,h9] * x_vvo[p3,p5,h1] * f_ov[h9,p5]
-1.0 * t_vo[p3,h9] * x_vvo[p3,p5,h1] * t_vo[p6,h7] * v_oovv[h7,h9,p5,p6]
-1.0 * t_vo[p3,h9] * x_vvo[p3,p10,h8] * v_ooov[h8,h9,h1,p10]
-1.0 * t_vo[p3,h9] * x_vvo[p3,p10,h8] * t_vo[p5,h1] * v_oovv[h8,h9,p5,p10]
-0.5 * t_vo[p3,h9] * x_vvo[p6,p7,h1] * v_ovvv[h9,p3,p6,p7]
+0.5 * t_vo[p3,h9] * t_vo[p3,h10] * x_v[p7] * v_ooov[h9,h10,h2,p7]
+0.25 * t_vo[p3,h9] * t_vo[p3,h10] * x_vvo[p7,p8,h1] * v_oovv[h9,h10,p7,p8]
+0.5 * t_vo[p3,h9] * t_vo[p3,h10] * t_vo[p5,h1] * x_v[p8] * v_oovv[h9,h10,p5,p8]
-1.0 * t_vo[p3,h9] * t_vo[p5,h1] * x_v[p7] * v_ovvv[h9,p3,p5,p7]
+1.0 * t_vo[p3,h9] * t_vvoo[p3,p5,h1,h6] * x_v[p8] * v_oovv[h6,h9,p5,p8]
+1.0 * t_vo[p5,h1] * x_v[p6] * v_vvvv[p3,p4,p5,p6]
-1.0 * t_vvoo[p3,p4,h1,h5] * x_v[p9] * f_ov[h5,p9]
+1.0 * t_vvoo[p3,p4,h1,h5] * x_v[p9] * t_vo[p6,h7] * v_oovv[h5,h7,p6,p9]
+0.5 * t_vvoo[p3,p4,h1,h5] * x_vvo[p7,p8,h6] * v_oovv[h5,h6,p7,p8]
+0.5 * t_vvoo[p3,p4,h5,h6] * x_v[p7] * v_ooov[h5,h6,h2,p7]
+0.25 * t_vvoo[p3,p4,h5,h6] * x_vvo[p7,p8,h1] * v_oovv[h5,h6,p7,p8]
+0.5 * t_vvoo[p3,p4,h5,h6] * t_vo[p7,h1] * x_v[p8] * v_oovv[h5,h6,p7,p8]
+1.0 * t_vvoo[p3,p5,h1,h6] * x_v[p7] * v_ovvv[h6,p3,p5,p7]
-1.0 * t_vvoo[p3,p5,h1,h6] * x_vvo[p3,p8,h7] * v_oovv[h6,h7,p5,p8]
;

}
