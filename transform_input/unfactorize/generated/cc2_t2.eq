{

range O = 10;
range V = 10;

index h1, h2, h3, h4, h5, h6, h7, h8, h9, h10 = O;
index p1, p2, p3, p4, p5, p6 = V;

array i0([V,V][O,O]);
array t_vo([V][O]);
array t_vvoo([V,V][O,O]);
array t2_3_1([V,V][O,V]);
array t2_2_2_1([O,O][O,O]);
array t2_2_2_2_1([O,O][O,V]);
array t2_2_3_1([O,V][O,V]);
array t2_2_1([O,V][O,O]);
array v_vvoo([V,V][O,O]);
array v_ovoo([O,V][O,O]);
array v_oooo([O,O][O,O]);
array v_ooov([O,O][O,V]);
array v_oovv([O,O][V,V]);
array v_ovov([O,V][O,V]);
array v_ovvv([O,V][V,V]);
array v_vvov([V,V][O,V]);
array v_vvvv([V,V][V,V]);
array f_oo([O][O]);
array f_vv([V][V]);

i0[p3,p4,h1,h2] = +1.0 * v_vvoo[p3,p4,h1,h2]
-1.0 * t_vo[p3,h10] * v_ovoo[h10,p3,h1,h2]
-0.5 * t_vo[p3,h10] * t_vo[p3,h8] * v_oooo[h8,h10,h1,h2]
+0.5 * t_vo[p3,h10] * t_vo[p3,h8] * t_vo[p5,h1] * v_ooov[h8,h10,h1,p5]
-0.25 * t_vo[p3,h10] * t_vo[p3,h8] * t_vo[p5,h1] * t_vo[p6,h1] * v_oovv[h8,h10,p5,p6]
+1.0 * t_vo[p3,h10] * t_vo[p5,h1] * v_ovov[h10,p3,h1,p5]
-0.5 * t_vo[p3,h10] * t_vo[p5,h1] * t_vo[p6,h1] * v_ovvv[h10,p3,p5,p6]
-1.0 * t_vo[p5,h1] * v_vvov[p3,p4,h1,p5]
+0.5 * t_vo[p5,h1] * t_vo[p6,h1] * v_vvvv[p3,p4,p5,p6]
-1.0 * t_vvoo[p3,p4,h1,h5] * f_oo[h5,h2]
+1.0 * t_vvoo[p3,p5,h1,h2] * f_vv[p4,p5]
;

}
