{

range O = 10;
range V = 10;

index h1, h2, h3, h4, h5 = O;
index p1, p2, p3, p4 = V;

array i0([V][O]);
array t_vo([V][O]);
array t_vvoo([V,V][O,O]);
array e([][]);
array f_vo([V][O]);
array f_oo([O][O]);
array f_vv([V][V]);
array v_ovov([O,V][O,V]);
array f_ov([O][V]);
array v_ooov([O,O][O,V]);
array v_ovvv([O,V][V,V]);

i0[p2,h1] = +1.0 * f_vo[p2,h1]
-1.0 * t_vo[p2,h3] * f_oo[h3,h1]
+1.0 * t_vo[p3,h1] * f_vv[p2,p3]
-1.0 * t_vo[p3,h4] * v_ovov[h4,p2,h1,p3]
+1.0 * t_vvoo[p2,p4,h1,h3] * f_ov[h3,p4]
-0.5 * t_vvoo[p2,p3,h4,h5] * v_ooov[h4,h5,h1,p3]
-0.5 * t_vvoo[p3,p4,h1,h5] * v_ovvv[h5,p2,p3,p4]
-1.0 * e * t_vo[p2,h1]
;

}
