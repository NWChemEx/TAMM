{

range O = 10;
range V = 10;

index h1, h2, h3, h4 = O;
index p1, p2 = V;

array i0([][]);
array t_vo([V][O]);
array t_vvoo([V,V][O,O]);
array f_ov([O][V]);
array v_oovv([O,O][V,V]);

i0 = +1.0 * t_vo[p2,h1] * f_ov[h1,p2]
+0.25 * t_vvoo[p1,p2,h3,h4] * v_oovv[h3,h4,p1,p2]
;

}
