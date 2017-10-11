{
  range O = 10;
  range V = 100;

  index h1, h2, h3 = O;
  index p1, p2, p3 = V;

  array f_oo([O][O]);
  array f_ov([O][V]);
  array f_vo([V][O]);
  array f_vv([V][V]);
  array v_oovo([O,O][V,O]);
  array v_oovv([O,O][V,V]);
  array v_ovvo([O,V][V,O]);
  array v_vovo([V,O][V,O]);
  array v_vovv([V,O][V,V]);
  array t_vo([V][O]);
  array t_vvoo([V,V][O,O]);
  array x_vo([V][O]);
  array x_vvoo([V,V][O,O]);
  array r_vo([V][O]);

  r_vo[p1, h1] =
    1.0 * f_vv[p1,p2] * x_vo[p2,h1]
    - 1.0 * f_oo[h2,h1] * x_vo[p1,h2]
    + 2.0 * f_ov[h2,p2] * x_vvoo[p1,p2,h1,h2]
    - 1.0 * f_ov[h2,p2] * x_vvoo[p2,p1,h1,h2]
    - 1.0 * v_vovv[p1,h2,p2,p3] * x_vvoo[p3,p2,h1,h2]
    + 2.0 * v_vovv[p1,h2,p2,p3] * x_vvoo[p2,p3,h1,h2]
    - 1.0 * v_vovo[p1,h2,p2,h1] * x_vo[p2,h2]
    + 2.0 * v_ovvo[h2,p1,p2,h1] * x_vo[p2,h2]
    + 1.0 * v_oovo[h2,h3,p2,h1] * x_vvoo[p1,p2,h2,h3]
    - 2.0 * v_oovo[h2,h3,p2,h1] * x_vvoo[p1,p2,h3,h2]
    - 1.0 * f_ov[h2,p2] * x_vo[p2,h1] * t_vo[p1,h2]
    + 2.0 * v_vovv[p1,h2,p2,p3] * x_vo[p2,h1] * t_vo[p3,h2]
    - 1.0 * v_vovv[p1,h2,p2,p3] * x_vo[p3,h1] * t_vo[p2,h2]
    + 1.0 * v_oovv[h2,h3,p2,p3] * x_vo[p2,h1] * t_vvoo[p1,p3,h3,h2]
    - 2.0 * v_oovv[h2,h3,p2,p3] * x_vo[p2,h1] * t_vvoo[p1,p3,h2,h3]
    + 1.0 * v_oovv[h2,h3,p2,p3] * x_vo[p1,h2] * t_vvoo[p3,p2,h1,h3]
    - 2.0 * v_oovv[h2,h3,p2,p3] * x_vo[p1,h2] * t_vvoo[p2,p3,h1,h3]
    + 4.0 * v_oovv[h2,h3,p2,p3] * x_vo[p2,h2] * t_vvoo[p1,p3,h1,h3]
    - 2.0 * v_oovv[h2,h3,p2,p3] * x_vo[p2,h2] * t_vvoo[p3,p1,h1,h3]
    - 2.0 * v_oovv[h2,h3,p2,p3] * x_vo[p3,h2] * t_vvoo[p1,p2,h1,h3]
    + 1.0 * v_oovv[h2,h3,p2,p3] * x_vo[p3,h2] * t_vvoo[p2,p1,h1,h3]
    - 2.0 * v_oovo[h2,h3,p2,h1] * x_vo[p2,h2] * t_vo[p1,h3]
    + 1.0 * v_oovo[h2,h3,p2,h1] * x_vo[p1,h2] * t_vo[p2,h3]
    - 1.0 * f_ov[h2,p2] * t_vo[p2,h1] * x_vo[p1,h2]
    + 2.0 * v_vovv[p1,h2,p2,p3] * t_vo[p2,h1] * x_vo[p3,h2]
    - 1.0 * v_vovv[p1,h2,p2,p3] * t_vo[p3,h1] * x_vo[p2,h2]
    + 1.0 * v_oovv[h2,h3,p2,p3] * t_vo[p2,h1] * x_vvoo[p1,p3,h3,h2]
    - 2.0 * v_oovv[h2,h3,p2,p3] * t_vo[p2,h1] * x_vvoo[p1,p3,h2,h3]
    + 1.0 * v_oovv[h2,h3,p2,p3] * t_vo[p1,h2] * x_vvoo[p3,p2,h1,h3]
    - 2.0 * v_oovv[h2,h3,p2,p3] * t_vo[p1,h2] * x_vvoo[p2,p3,h1,h3]
    + 4.0 * v_oovv[h2,h3,p2,p3] * t_vo[p2,h2] * x_vvoo[p1,p3,h1,h3]
    - 2.0 * v_oovv[h2,h3,p2,p3] * t_vo[p2,h2] * x_vvoo[p3,p1,h1,h3]
    - 2.0 * v_oovv[h2,h3,p2,p3] * t_vo[p3,h2] * x_vvoo[p1,p2,h1,h3]
    + 1.0 * v_oovv[h2,h3,p2,p3] * t_vo[p3,h2] * x_vvoo[p2,p1,h1,h3]
    - 2.0 * v_oovo[h2,h3,p2,h1] * t_vo[p2,h2] * x_vo[p1,h3]
    + 1.0 * v_oovo[h2,h3,p2,h1] * t_vo[p1,h2] * x_vo[p2,h3]
    - 2.0 * v_oovv[h2,h3,p2,p3] * x_vo[p2,h1] * t_vo[p1,h2] * t_vo[p3,h3]
    + 1.0 * v_oovv[h2,h3,p2,p3] * x_vo[p2,h1] * t_vo[p3,h2] * t_vo[p1,h3]
    - 2.0 * v_oovv[h2,h3,p2,p3] * t_vo[p2,h1] * x_vo[p1,h2] * t_vo[p3,h3]
    + 1.0 * v_oovv[h2,h3,p2,p3] * t_vo[p2,h1] * x_vo[p3,h2] * t_vo[p1,h3]
    - 2.0 * v_oovv[h2,h3,p2,p3] * t_vo[p2,h1] * t_vo[p1,h2] * x_vo[p3,h3]
    + 1.0 * v_oovv[h2,h3,p2,p3] * t_vo[p2,h1] * t_vo[p3,h2] * x_vo[p1,h3];
}
