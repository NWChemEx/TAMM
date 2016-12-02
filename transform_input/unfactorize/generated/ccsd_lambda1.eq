{

range O = 10;
range V = 10;

index h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12 = O;
index p1, p2, p3, p4, p5, p6, p7, p8, p9 = V;

array i0([O][V]);
array y_ov([O][V]);
array t_vo([V][O]);
array t_vvoo([V,V][O,O]);
array y_oovv([O,O][V,V]);
array lambda1_15_2_1([O,O][O,V]);
array lambda1_3_3_1([O][V]);
array lambda1_5_5_1([O][V]);
array lambda1_5_3_1([V][V]);
array lambda1_5_2_1([O][O]);
array lambda1_6_4_1([O][V]);
array lambda1_6_2_2_1([O,O][O,V]);
array lambda1_3_1([V][V]);
array lambda1_2_1([O][O]);
array lambda1_6_2_1([O,O][O,O]);
array lambda1_10_1([O][O]);
array lambda1_6_3_1([O,V][O,V]);
array lambda1_8_1([V][O]);
array lambda1_7_1([V,V][O,V]);
array lambda1_5_2_2_1([O][V]);
array lambda1_6_1([O,V][O,O]);
array lambda1_6_5_1([O,O][O,V]);
array lambda1_13_2_2_1([O,O][O,V]);
array lambda1_11_1([V][V]);
array lambda1_5_1([V][O]);
array lambda1_8_4_1([O,O][O,V]);
array lambda1_13_2_1([O,O][O,O]);
array lambda1_14_1([O,O][O,O]);
array lambda1_13_3_1([O,O][O,V]);
array lambda1_15_1([O,V][O,V]);
array lambda1_8_3_1([O][O]);
array lambda1_14_2_1([O,O][O,V]);
array lambda1_12_1([O,O][O,V]);
array lambda1_13_4_1([O][O]);
array lambda1_5_6_1([O,O][O,V]);
array lambda1_9_1([O][O]);
array lambda1_2_2_1([O][V]);
array lambda1_13_1([O,V][O,O]);
array f_ov([O][V]);
array f_oo([O][O]);
array v_oovv([O,O][V,V]);
array v_ooov([O,O][O,V]);
array f_vv([V][V]);
array v_ovvv([O,V][V,V]);
array v_ovov([O,V][O,V]);
array f_vo([V][O]);
array v_ovoo([O,V][O,O]);
array v_oooo([O,O][O,O]);
array v_vvov([V,V][O,V]);
array v_vvvv([V,V][V,V]);

i0[h2,p1] = +1.0 * f_ov[h2,p1]
-1.0 * y_ov[h7,p1] * f_oo[h2,h7]
-1.0 * y_ov[h7,p1] * t_vo[p3,h7] * f_ov[h2,p3]
-1.0 * y_ov[h7,p1] * t_vo[p3,h7] * t_vo[p5,h6] * v_oovv[h2,h6,p3,p5]
-1.0 * y_ov[h7,p1] * t_vo[p3,h4] * v_ooov[h2,h4,h7,p3]
+0.5 * y_ov[h7,p1] * t_vvoo[p3,p4,h6,h7] * v_oovv[h2,h6,p3,p4]
+1.0 * y_ov[h2,p7] * f_vv[p7,p1]
-1.0 * y_ov[h2,p7] * t_vo[p3,h4] * v_ovvv[h4,p7,p1,p3]
-1.0 * y_ov[h2,p7] * t_vo[p7,h4] * t_vo[p5,h6] * v_oovv[h4,h6,p1,p5]
-1.0 * y_ov[h4,p3] * v_ovov[h2,p3,h4,p1]
+1.0 * y_oovv[h2,h11,p1,p9] * f_vo[p9,h11]
-1.0 * y_oovv[h2,h11,p1,p9] * t_vo[p9,h10] * f_oo[h10,h11]
-1.0 * y_oovv[h2,h11,p1,p9] * t_vo[p9,h10] * t_vo[p3,h11] * f_ov[h10,p3]
+1.0 * y_oovv[h2,h11,p1,p9] * t_vo[p9,h10] * t_vo[p3,h11] * t_vo[p7,h8] * v_oovv[h8,h10,p3,p7]
+1.0 * y_oovv[h2,h11,p1,p9] * t_vo[p9,h10] * t_vo[p5,h6] * v_ooov[h6,h10,h11,p5]
-0.5 * y_oovv[h2,h11,p1,p9] * t_vo[p9,h10] * t_vvoo[p3,p4,h6,h11] * v_oovv[h6,h10,p3,p4]
+1.0 * y_oovv[h2,h11,p1,p9] * t_vo[p7,h11] * f_vv[p9,p7]
+1.0 * y_oovv[h2,h11,p1,p9] * t_vo[p7,h11] * t_vo[p5,h6] * v_ovvv[h6,p9,p5,p7]
-1.0 * y_oovv[h2,h11,p1,p9] * t_vo[p3,h4] * v_ovov[h4,p9,h11,p3]
+1.0 * y_oovv[h2,h11,p1,p9] * t_vvoo[p4,p9,h5,h11] * f_ov[h5,p4]
+1.0 * y_oovv[h2,h11,p1,p9] * t_vvoo[p4,p9,h5,h11] * t_vo[p7,h8] * v_oovv[h5,h8,p4,p7]
+0.5 * y_oovv[h2,h11,p1,p9] * t_vvoo[p4,p9,h5,h6] * v_ooov[h5,h6,h11,p4]
-0.5 * y_oovv[h2,h11,p1,p9] * t_vvoo[p4,p9,h5,h6] * t_vo[p7,h11] * v_oovv[h5,h6,p4,p7]
+0.5 * y_oovv[h2,h11,p1,p9] * t_vvoo[p3,p4,h6,h11] * v_ovvv[h6,p9,p3,p4]
-0.5 * y_oovv[h11,h12,p1,p9] * v_ovoo[h2,p9,h11,h12]
+0.5 * y_oovv[h11,h12,p1,p9] * t_vo[p9,h7] * v_oooo[h2,h7,h11,h12]
-1.0 * y_oovv[h11,h12,p1,p9] * t_vo[p9,h7] * t_vo[p3,h11] * v_ooov[h2,h7,h12,p3]
+0.5 * y_oovv[h11,h12,p1,p9] * t_vo[p9,h7] * t_vo[p3,h11] * t_vo[p5,h12] * v_oovv[h2,h7,p3,p5]
+0.25 * y_oovv[h11,h12,p1,p9] * t_vo[p9,h7] * t_vvoo[p3,p4,h11,h12] * v_oovv[h2,h7,p3,p4]
+0.5 * y_oovv[h11,h12,p1,p9] * t_vo[p3,h11] * v_ovov[h2,p9,h12,p3]
-0.25 * y_oovv[h11,h12,p1,p9] * t_vo[p3,h11] * t_vo[p5,h12] * v_ovvv[h2,p9,p3,p5]
-0.5 * y_oovv[h11,h12,p1,p9] * t_vvoo[p5,p9,h11,h12] * f_ov[h2,p5]
-0.5 * y_oovv[h11,h12,p1,p9] * t_vvoo[p5,p9,h11,h12] * t_vo[p7,h8] * v_oovv[h2,h8,p5,p7]
+0.5 * y_oovv[h11,h12,p1,p9] * t_vvoo[p4,p9,h6,h11] * v_ooov[h2,h6,h12,p4]
-0.5 * y_oovv[h11,h12,p1,p9] * t_vvoo[p4,p9,h6,h11] * t_vo[p7,h12] * v_oovv[h2,h6,p4,p7]
-0.25 * y_oovv[h11,h12,p1,p9] * t_vvoo[p3,p4,h11,h12] * v_ovvv[h2,p9,p3,p4]
-0.5 * y_oovv[h2,h7,p5,p8] * v_vvov[p5,p8,h7,p1]
+0.5 * y_oovv[h2,h7,p5,p8] * t_vo[p3,h7] * v_vvvv[p5,p8,p1,p3]
+1.0 * t_vo[p9,h10] * v_oovv[h2,h10,p1,p9]
+1.0 * t_vvoo[p3,p9,h5,h10] * y_ov[h5,p3] * v_oovv[h2,h10,p1,p9]
-1.0 * t_vo[p9,h6] * t_vo[p5,h10] * y_ov[h6,p5] * v_oovv[h2,h10,p1,p9]
-0.5 * t_vo[p9,h6] * t_vvoo[p3,p4,h5,h10] * y_oovv[h5,h6,p3,p4] * v_oovv[h2,h10,p1,p9]
-0.5 * t_vvoo[p3,p9,h5,h6] * t_vo[p7,h10] * y_oovv[h5,h6,p3,p7] * v_oovv[h2,h10,p1,p9]
-1.0 * t_vo[p4,h3] * y_ov[h2,p4] * f_ov[h3,p1]
-0.5 * t_vvoo[p4,p5,h3,h6] * y_oovv[h2,h6,p4,p5] * f_ov[h3,p1]
+1.0 * t_vo[p3,h8] * y_ov[h6,p3] * v_ooov[h2,h8,h6,p1]
+0.5 * t_vvoo[p3,p4,h5,h8] * y_oovv[h5,h6,p3,p4] * v_ooov[h2,h8,h6,p1]
+1.0 * t_vo[p7,h4] * y_ov[h4,p8] * v_ovvv[h2,p8,p1,p7]
+0.5 * t_vvoo[p3,p7,h5,h6] * y_oovv[h5,h6,p3,p8] * v_ovvv[h2,p8,p1,p7]
+1.0 * t_vo[p3,h4] * y_oovv[h2,h6,p3,p5] * v_ovov[h4,p5,h6,p1]
-0.5 * t_vvoo[p3,p9,h6,h12] * y_ov[h2,p3] * v_oovv[h6,h12,p1,p9]
+0.25 * t_vo[p9,h10] * t_vvoo[p3,p4,h6,h12] * y_oovv[h2,h10,p3,p4] * v_oovv[h6,h12,p1,p9]
-0.25 * t_vo[p9,h10] * t_vo[p5,h12] * t_vo[p7,h6] * y_oovv[h2,h10,p5,p7] * v_oovv[h6,h12,p1,p9]
+0.5 * t_vvoo[p3,p9,h5,h12] * t_vo[p7,h6] * y_oovv[h2,h5,p3,p7] * v_oovv[h6,h12,p1,p9]
-0.25 * t_vo[p9,h6] * t_vvoo[p3,p4,h5,h12] * y_oovv[h2,h5,p3,p4] * v_oovv[h6,h12,p1,p9]
-0.25 * t_vvoo[p3,p4,h6,h8] * y_oovv[h2,h7,p3,p4] * v_ooov[h6,h8,h7,p1]
+0.5 * t_vo[p3,h8] * t_vo[p5,h6] * y_oovv[h2,h7,p3,p5] * v_ooov[h6,h8,h7,p1]
+1.0 * t_vvoo[p3,p8,h5,h6] * y_oovv[h2,h5,p3,p7] * v_ovvv[h6,p7,p1,p8]
-1.0 * t_vo[p8,h4] * t_vo[p5,h6] * y_oovv[h2,h4,p5,p7] * v_ovvv[h6,p7,p1,p8]
;

}
