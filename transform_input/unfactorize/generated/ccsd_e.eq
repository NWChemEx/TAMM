{

range O = 10;
range V = 10;

index h1, h2, h3, h4, h5, h6, h7, h8 = O;
index p1, p2, p3, p4, p5, p6, p7 = V;

array i0([][]);
array t_vo([V][O]);
array t_vvoo([V,V][O,O]);
array i1([O][V]);
array f_ov([O][V]);
array v_oovv([O,O][V,V]);

i0 = +1.0 * t_vo[p5,h6] * f_ov[h6,p5]
+0.5 * t_vo[p5,h6] * t_vo[p3,h4] * v_oovv[h4,h6,p3,p5]
+0.25 * t_vvoo[p1,p2,h3,h4] * v_oovv[h3,h4,p1,p2]
;

}
