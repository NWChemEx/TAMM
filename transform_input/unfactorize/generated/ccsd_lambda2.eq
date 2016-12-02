{

range O = 10;
range V = 10;

index h1, h2, h3, h4, h5, h6, h7, h8, h9, h10 = O;
index p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = V;

array i0([O,O][V,V]);
array y_ov([O][V]);
array t_vo([V][O]);
array y_oovv([O,O][V,V]);
array t_vvoo([V,V][O,O]);
array lambda2_7_1([O,O][O,O]);
array lambda2_8_1([O,V][O,V]);
array lambda2_2_1([O][V]);
array lambda2_15_1([O,O][O,O]);
array lambda2_11_1([O,O][O,V]);
array lambda2_15_2_1([O,O][O,V]);
array lambda2_3_1([O,O][O,V]);
array lambda2_10_1([O][O]);
array lambda2_14_1([V][V]);
array lambda2_16_1([O,V][O,V]);
array lambda2_6_4_1([O][V]);
array lambda2_5_2_1([O][V]);
array lambda2_13_1([O,O][O,V]);
array lambda2_5_1([O][O]);
array lambda2_7_2_1([O,O][O,V]);
array lambda2_6_1([V][V]);
array lambda2_12_1([O,O][O,V]);
array lambda2_16_1_1([O,O][O,V]);
array v_oovv([O,O][V,V]);
array f_ov([O][V]);
array v_ooov([O,O][O,V]);
array v_ovvv([O,V][V,V]);
array f_oo([O][O]);
array f_vv([V][V]);
array v_oooo([O,O][O,O]);
array v_ovov([O,V][O,V]);
array v_vvvv([V,V][V,V]);

i0[h3,h4,p1,p2] = +1.0 * v_oovv[h3,h4,p1,p2]
+1.0 * y_ov[h3,p1] * f_ov[h3,p1]
+1.0 * y_ov[h3,p1] * t_vo[p5,h6] * v_oovv[h3,h6,p1,p5]
-1.0 * y_ov[h7,p1] * v_ooov[h3,h4,h7,p1]
+1.0 * y_ov[h7,p1] * t_vo[p5,h7] * v_oovv[h3,h4,p1,p5]
-1.0 * y_ov[h3,p5] * v_ovvv[h4,p5,p1,p2]
-1.0 * y_oovv[h3,h9,p1,p2] * f_oo[h3,h9]
-1.0 * y_oovv[h3,h9,p1,p2] * t_vo[p5,h9] * f_ov[h3,p5]
-1.0 * y_oovv[h3,h9,p1,p2] * t_vo[p5,h9] * t_vo[p7,h8] * v_oovv[h3,h8,p5,p7]
-1.0 * y_oovv[h3,h9,p1,p2] * t_vo[p5,h6] * v_ooov[h3,h6,h9,p5]
+0.5 * y_oovv[h3,h9,p1,p2] * t_vvoo[p5,p6,h8,h9] * v_oovv[h3,h8,p5,p6]
+1.0 * y_oovv[h3,h4,p1,p10] * f_vv[p10,p1]
-1.0 * y_oovv[h3,h4,p1,p10] * t_vo[p5,h6] * v_ovvv[h6,p10,p1,p5]
+0.5 * y_oovv[h3,h4,p1,p10] * t_vvoo[p6,p10,h7,h8] * v_oovv[h7,h8,p1,p6]
-1.0 * y_oovv[h3,h4,p1,p10] * t_vo[p10,h6] * t_vo[p7,h8] * v_oovv[h6,h8,p1,p7]
+0.5 * y_oovv[h9,h10,p1,p2] * v_oooo[h3,h4,h9,h10]
-0.5 * y_oovv[h9,h10,p1,p2] * t_vo[p5,h9] * v_ooov[h3,h4,h10,p5]
+0.25 * y_oovv[h9,h10,p1,p2] * t_vo[p5,h9] * t_vo[p7,h10] * v_oovv[h3,h4,p5,p7]
+0.25 * y_oovv[h9,h10,p1,p2] * t_vvoo[p5,p6,h9,h10] * v_oovv[h3,h4,p5,p6]
-1.0 * y_oovv[h3,h9,p1,p7] * v_ovov[h3,p7,h9,p1]
+1.0 * y_oovv[h3,h9,p1,p7] * t_vo[p5,h9] * v_ovvv[h3,p7,p1,p5]
+1.0 * y_oovv[h3,h9,p1,p7] * t_vvoo[p6,p7,h8,h9] * v_oovv[h3,h8,p1,p6]
+0.5 * y_oovv[h3,h4,p5,p6] * v_vvvv[p5,p6,p1,p2]
+1.0 * t_vo[p5,h9] * y_ov[h3,p5] * v_oovv[h4,h9,p1,p2]
-0.5 * t_vvoo[p5,p6,h7,h9] * y_oovv[h3,h7,p5,p6] * v_oovv[h4,h9,p1,p2]
-1.0 * t_vo[p6,h5] * y_oovv[h3,h4,p1,p6] * f_ov[h5,p2]
+1.0 * t_vo[p5,h6] * y_oovv[h3,h7,p1,p5] * v_ooov[h4,h6,h7,p2]
-1.0 * t_vo[p5,h6] * y_oovv[h3,h4,p5,p7] * v_ovvv[h6,p7,p1,p2]
-0.5 * t_vvoo[p5,p6,h7,h8] * y_oovv[h7,h8,p1,p5] * v_oovv[h3,h4,p2,p6]
+0.25 * t_vvoo[p5,p6,h8,h9] * y_oovv[h3,h4,p5,p6] * v_oovv[h8,h9,p1,p2]
-0.25 * t_vo[p5,h9] * t_vo[p7,h8] * y_oovv[h3,h4,p5,p7] * v_oovv[h8,h9,p1,p2]
-1.0 * t_vo[p5,h6] * t_vo[p7,h8] * y_oovv[h3,h6,p1,p7] * v_oovv[h4,h8,p2,p5]
;

}
