{

range O = 10;
range V = 10;

index h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11 = O;
index p1, p2, p3, p4, p5, p6, p7, p8, p9 = V;

array i0([V,V][O,O]);
array t1([V][O]);
array t2([V,V][O,O]);
array t2_2_1([O,V][O,O]);
array t2_2_2_1([O,O][O,O]);
array t2_2_2_2_1([O,O][O,V]);
array t2_2_4_1([O][V]);
array t2_2_5_1([O,O][O,V]);
array t2_4_1([O][O]);
array t2_4_2_1([O][V]);
array t2_5_1([V][V]);
array t2_6_1([O,O][O,O]);
array t2_6_2_1([O,O][O,V]);
array t2_7_1([O,V][O,V]);
array vt1t1_1([O,V][O,O]);
array c2([V,V][O,O]);
array v_vvoo([V,V][O,O]);
array v_ovoo([O,V][O,O]);
array v_oooo([O,O][O,O]);
array v_ooov([O,O][O,V]);
array v_oovv([O,O][V,V]);
array f_ov([O][V]);
array v_ovvv([O,V][V,V]);
array v_vvov([V,V][O,V]);
array f_oo([O][O]);
array f_vv([V][V]);
array v_ovov([O,V][O,V]);
array v_vvvv([V,V][V,V]);

i0[p3,p4,h1,h2] = +1.0 * v_vvoo[p3,p4,h1,h2]
-1.0 * t1[p3,h10] * v_ovoo[h10,p3,h1,h2]
+0.5 * t1[p3,h10] * t1[p3,h11] * v_oooo[h10,h11,h1,h2]
-0.5 * t1[p3,h10] * t1[p3,h11] * t1[p5,h1] * v_ooov[h10,h11,h1,p5]
+0.25 * t1[p3,h10] * t1[p3,h11] * t1[p5,h1] * t1[p6,h1] * v_oovv[h10,h11,p5,p6]
+0.25 * t1[p3,h10] * t1[p3,h11] * t2[p7,p8,h1,h2] * v_oovv[h10,h11,p7,p8]
+1.0 * t1[p3,h10] * t2[p3,p5,h1,h2] * f_ov[h10,p5]
-1.0 * t1[p3,h10] * t2[p3,p5,h1,h2] * t1[p6,h7] * v_oovv[h7,h10,p5,p6]
-1.0 * t1[p3,h10] * t2[p3,p9,h1,h7] * v_ooov[h7,h10,h1,p9]
-1.0 * t1[p3,h10] * t2[p3,p9,h1,h7] * t1[p5,h1] * v_oovv[h7,h10,p5,p9]
-0.5 * t1[p3,h10] * c2[p5,p6,h1,h2] * v_ovvv[h10,p3,p5,p6]
-1.0 * t1[p5,h1] * v_vvov[p3,p4,h2,p5]
-1.0 * t2[p3,p4,h1,h9] * f_oo[h9,h1]
-1.0 * t2[p3,p4,h1,h9] * t1[p8,h1] * f_ov[h9,p8]
-1.0 * t2[p3,p4,h1,h9] * t1[p8,h1] * t1[p6,h7] * v_oovv[h7,h9,p6,p8]
+1.0 * t2[p3,p4,h1,h9] * t1[p6,h7] * v_ooov[h7,h9,h1,p6]
+0.5 * t2[p3,p4,h1,h9] * t2[p6,p7,h1,h8] * v_oovv[h8,h9,p6,p7]
+1.0 * t2[p3,p5,h1,h2] * f_vv[p3,p5]
-1.0 * t2[p3,p5,h1,h2] * t1[p6,h7] * v_ovvv[h7,p3,p5,p6]
-0.5 * t2[p3,p5,h1,h2] * t2[p3,p6,h7,h8] * v_oovv[h7,h8,p5,p6]
+0.5 * t2[p3,p4,h9,h11] * v_oooo[h9,h11,h1,h2]
-0.5 * t2[p3,p4,h9,h11] * t1[p8,h1] * v_ooov[h9,h11,h1,p8]
-0.25 * t2[p3,p4,h9,h11] * t1[p8,h1] * t1[p6,h1] * v_oovv[h9,h11,p6,p8]
+0.25 * t2[p3,p4,h9,h11] * t2[p5,p6,h1,h2] * v_oovv[h9,h11,p5,p6]
-1.0 * t2[p3,p5,h1,h6] * v_ovov[h6,p3,h1,p5]
+1.0 * t2[p3,p5,h1,h6] * t1[p7,h1] * v_ovvv[h6,p3,p5,p7]
+0.5 * t2[p3,p5,h1,h6] * t2[p3,p7,h1,h8] * v_oovv[h6,h8,p5,p7]
+1.0 * t1[p3,h5] * t1[p6,h1] * v_ovov[h5,p3,h2,p6]
+0.5 * c2[p5,p6,h1,h2] * v_vvvv[p3,p4,p5,p6]
;

}
