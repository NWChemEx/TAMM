{

range O = 10;
range V = 10;

index h1, h2, h3, h4, h5, h6 = O;
index p1, p2, p3, p4, p5, p6 = V;

array i0([V,V][O,O]);
array t_vo([V][O]);
array t_vvoo([V,V][O,O]);
array e([][]);
array v_vvoo([V,V][O,O]);
array f_vo([V][O]);
array v_ovoo([O,V][O,O]);
array v_vvov([V,V][O,V]);
array f_oo([O][O]);
array f_vv([V][V]);
array v_oooo([O,O][O,O]);
array v_ovov([O,V][O,V]);
array v_vvvv([V,V][V,V]);

i0[p3,p4,h1,h2] = +1.0 * v_vvoo[p3,p4,h1,h2]
+1.0 * t_vo[p3,h1] * f_vo[p4,h2]
-1.0 * t_vo[p3,h5] * v_ovoo[h5,p4,h1,h2]
-1.0 * t_vo[p5,h1] * v_vvov[p3,p4,h2,p5]
-1.0 * t_vvoo[p3,p4,h1,h5] * f_oo[h5,h2]
+1.0 * t_vvoo[p3,p5,h1,h2] * f_vv[p4,p5]
+0.5 * t_vvoo[p3,p4,h5,h6] * v_oooo[h5,h6,h1,h2]
-1.0 * t_vvoo[p3,p5,h1,h6] * v_ovov[h6,p4,h2,p5]
+0.5 * t_vvoo[p5,p6,h1,h2] * v_vvvv[p3,p4,p5,p6]
-1.0 * e * t_vvoo[p3,p4,h1,h2]
;

}
