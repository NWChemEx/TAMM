{

range O = 10;
range V = 10;

index h1, h2, h3, h4, h5, h6 = O;
index p1, p2, p3, p4, p5, p6, p7 = V;

array i0([V][]);
array x_v([V][]);
array t_vo([V][O]);
array x_vvo([V,V][O]);
array t_vvoo([V,V][O,O]);
array x1_4_1_1([O][V]);
array x1_4_1([O][]);
array x1_2_1([O][V]);
array x1_5_1([O,O][V]);
array x1_1_1([V][V]);
array f_vv([V][V]);
array v_ovvv([O,V][V,V]);
array f_ov([O][V]);
array v_oovv([O,O][V,V]);

i0[p2] = +1.0 * x_v[p6] * f_vv[p2,p6]
+1.0 * x_v[p6] * t_vo[p3,h4] * v_ovvv[h4,p2,p3,p6]
-1.0 * x_vvo[p2,p7,h6] * f_ov[h6,p7]
-1.0 * x_vvo[p2,p7,h6] * t_vo[p3,h4] * v_oovv[h4,h6,p3,p7]
+0.5 * x_vvo[p4,p5,h3] * v_ovvv[h3,p2,p4,p5]
-1.0 * t_vo[p2,h3] * x_v[p7] * f_ov[h3,p7]
+1.0 * t_vo[p2,h3] * x_v[p7] * t_vo[p4,h5] * v_oovv[h3,h5,p4,p7]
+0.5 * t_vo[p2,h3] * x_vvo[p5,p6,h4] * v_oovv[h3,h4,p5,p6]
+0.5 * t_vvoo[p2,p3,h4,h5] * x_v[p6] * v_oovv[h4,h5,p3,p6]
;

}
