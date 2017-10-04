{
  range Oa = 10;
  range Ob = 10;
  range Va = 100;
  range Vb = 100;

  index h1a, h2a, h3a, h4a = Oa;
  index h1b, h2b, h3b, h4b = Ob;
  index p1a, p2a, p3a, p4a = Va;
  index p1b, p2b, p3b, p4b = Vb;

  array fa_oo([Oa][Oa]);
  array fa_ov([Oa][Va]);
  array fa_vo([Va][Oa]);
  array fa_vv([Va][Va]);
  array fb_oo([Ob][Ob]);
  array fb_ov([Ob][Vb]);
  array fb_vo([Vb][Ob]);
  array fb_vv([Vb][Vb]);
  array vaa_oovo([Oa,Oa][Va,Oa]);
  array vaa_oovv([Oa,Oa][Va,Va]);
  array vaa_vovo([Va,Oa][Va,Oa]);
  array vaa_vovv([Va,Oa][Va,Va]);
  array vbb_oovo([Ob,Ob][Vb,Ob]);
  array vbb_oovv([Ob,Ob][Vb,Vb]);
  array vbb_vovo([Vb,Ob][Vb,Ob]);
  array vbb_vovv([Vb,Ob][Vb,Vb]);
  array vab_ooov([Oa,Ob][Oa,Vb]);
  array vab_oovo([Oa,Ob][Va,Ob]);
  array vab_oovv([Oa,Ob][Va,Vb]);
  array vab_ovvo([Oa,Vb][Va,Ob]);
  array vab_ovvv([Oa,Vb][Va,Vb]);
  array vab_voov([Va,Ob][Oa,Vb]);
  array vab_vovv([Va,Ob][Va,Vb]);
  array ta_vo([Va][Oa]);
  array tb_vo([Vb][Ob]);
  array taa_vvoo([Va,Va][Oa,Oa]);
  array tbb_vvoo([Vb,Vb][Ob,Ob]);
  array tab_vvoo([Va,Vb][Oa,Ob]);
  array xa_vo([Va][Oa]);
  array xb_vo([Vb][Ob]);
  array xaa_vvoo([Va,Va][Oa,Oa]);
  array xbb_vvoo([Vb,Vb][Ob,Ob]);
  array xab_vvoo([Va,Vb][Oa,Ob]);
  array ra_vo([Va][Oa]);
  array rb_vo([Vb][Ob]);

  ra_vo[p1a,h1a] =
    1.0 * fb_ov[h2b,p2b] * xab_vvoo[p1a,p2b,h1a,h2b]
    + 1.0 * fa_ov[h2a,p2a] * xaa_vvoo[p1a,p2a,h1a,h2a]
    - 1.0 * vab_ooov[h2a,h3b,h1a,p2b] * xab_vvoo[p1a,p2b,h2a,h3b]
    + 0.5 * vaa_oovo[h2a,h3a,p2a,h1a] * xaa_vvoo[p1a,p2a,h2a,h3a]
    + 1.0 * vab_vovv[p1a,h2b,p2a,p3b] * xab_vvoo[p2a,p3b,h1a,h2b]
    + 0.5 * vaa_vovv[p1a,h2a,p2a,p3a] * xaa_vvoo[p2a,p3a,h1a,h2a]
    + 1.0 * fa_vv[p1a,p2a] * xa_vo[p2a,h1a]
    - 1.0 * fa_oo[h2a,h1a] * xa_vo[p1a,h2a]
    + 1.0 * vab_voov[p1a,h2b,h1a,p2b] * xb_vo[p2b,h2b]
    - 1.0 * vaa_vovo[p1a,h2a,p2a,h1a] * xa_vo[p2a,h2a]
    + 1.0 * vbb_oovv[h2b,h3b,p2b,p3b] * xb_vo[p2b,h2b] * tab_vvoo[p1a,p3b,h1a,h3b]
    + 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * xb_vo[p3b,h3b] * taa_vvoo[p1a,p2a,h1a,h2a]
    + 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * xa_vo[p2a,h2a] * tab_vvoo[p1a,p3b,h1a,h3b]
    + 1.0 * vaa_oovv[h2a,h3a,p2a,p3a] * xa_vo[p2a,h2a] * taa_vvoo[p1a,p3a,h1a,h3a]
    - 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * xa_vo[p2a,h1a] * tab_vvoo[p1a,p3b,h2a,h3b]
    - 0.5 * vaa_oovv[h2a,h3a,p2a,p3a] * xa_vo[p2a,h1a] * taa_vvoo[p1a,p3a,h2a,h3a]
    - 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * xa_vo[p1a,h2a] * tab_vvoo[p2a,p3b,h1a,h3b]
    - 0.5 * vaa_oovv[h2a,h3a,p2a,p3a] * xa_vo[p1a,h2a] * taa_vvoo[p2a,p3a,h1a,h3a]
    - 1.0 * fa_ov[h2a,p2a] * xa_vo[p2a,h1a] * ta_vo[p1a,h2a]
    + 1.0 * vab_vovv[p1a,h2b,p2a,p3b] * xa_vo[p2a,h1a] * tb_vo[p3b,h2b]
    + 1.0 * vaa_vovv[p1a,h2a,p2a,p3a] * xa_vo[p2a,h1a] * ta_vo[p3a,h2a]
    - 1.0 * vab_ooov[h2a,h3b,h1a,p2b] * xa_vo[p1a,h2a] * tb_vo[p2b,h3b]
    + 1.0 * vaa_oovo[h2a,h3a,p2a,h1a] * xa_vo[p1a,h2a] * ta_vo[p2a,h3a]
    + 1.0 * vbb_oovv[h2b,h3b,p2b,p3b] * tb_vo[p2b,h2b] * xab_vvoo[p1a,p3b,h1a,h3b]
    + 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * tb_vo[p3b,h3b] * xaa_vvoo[p1a,p2a,h1a,h2a]
    + 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * ta_vo[p2a,h2a] * xab_vvoo[p1a,p3b,h1a,h3b]
    + 1.0 * vaa_oovv[h2a,h3a,p2a,p3a] * ta_vo[p2a,h2a] * xaa_vvoo[p1a,p3a,h1a,h3a]
    - 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * ta_vo[p2a,h1a] * xab_vvoo[p1a,p3b,h2a,h3b]
    - 0.5 * vaa_oovv[h2a,h3a,p2a,p3a] * ta_vo[p2a,h1a] * xaa_vvoo[p1a,p3a,h2a,h3a]
    - 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * ta_vo[p1a,h2a] * xab_vvoo[p2a,p3b,h1a,h3b]
    - 0.5 * vaa_oovv[h2a,h3a,p2a,p3a] * ta_vo[p1a,h2a] * xaa_vvoo[p2a,p3a,h1a,h3a]
    - 1.0 * fa_ov[h2a,p2a] * ta_vo[p2a,h1a] * xa_vo[p1a,h2a]
    + 1.0 * vab_vovv[p1a,h2b,p2a,p3b] * ta_vo[p2a,h1a] * xb_vo[p3b,h2b]
    + 1.0 * vaa_vovv[p1a,h2a,p2a,p3a] * ta_vo[p2a,h1a] * xa_vo[p3a,h2a]
    - 1.0 * vab_ooov[h2a,h3b,h1a,p2b] * ta_vo[p1a,h2a] * xb_vo[p2b,h3b]
    + 1.0 * vaa_oovo[h2a,h3a,p2a,h1a] * ta_vo[p1a,h2a] * xa_vo[p2a,h3a]
    - 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * xa_vo[p2a,h1a] * ta_vo[p1a,h2a] * tb_vo[p3b,h3b]
    - 1.0 * vaa_oovv[h2a,h3a,p2a,p3a] * xa_vo[p2a,h1a] * ta_vo[p1a,h2a] * ta_vo[p3a,h3a]
    - 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * ta_vo[p2a,h1a] * xa_vo[p1a,h2a] * tb_vo[p3b,h3b]
    - 1.0 * vaa_oovv[h2a,h3a,p2a,p3a] * ta_vo[p2a,h1a] * xa_vo[p1a,h2a] * ta_vo[p3a,h3a]
    - 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * ta_vo[p2a,h1a] * ta_vo[p1a,h2a] * xb_vo[p3b,h3b]
    - 1.0 * vaa_oovv[h2a,h3a,p2a,p3a] * ta_vo[p2a,h1a] * ta_vo[p1a,h2a] * xa_vo[p3a,h3a];

  rb_vo[p1b,h1b] =
    1.0 * fb_ov[h2b,p2b] * xbb_vvoo[p1b,p2b,h1b,h2b]
    + 1.0 * fa_ov[h2a,p2a] * xab_vvoo[p2a,p1b,h2a,h1b]
    + 0.5 * vbb_oovo[h2b,h3b,p2b,h1b] * xbb_vvoo[p1b,p2b,h2b,h3b]
    - 1.0 * vab_oovo[h2a,h3b,p2a,h1b] * xab_vvoo[p2a,p1b,h2a,h3b]
    + 0.5 * vbb_vovv[p1b,h2b,p2b,p3b] * xbb_vvoo[p2b,p3b,h1b,h2b]
    + 1.0 * vab_ovvv[h2a,p1b,p2a,p3b] * xab_vvoo[p2a,p3b,h2a,h1b]
    + 1.0 * fb_vv[p1b,p2b] * xb_vo[p2b,h1b]
    - 1.0 * fb_oo[h2b,h1b] * xb_vo[p1b,h2b]
    - 1.0 * vbb_vovo[p1b,h2b,p2b,h1b] * xb_vo[p2b,h2b]
    + 1.0 * vab_ovvo[h2a,p1b,p2a,h1b] * xa_vo[p2a,h2a]
    + 1.0 * vbb_oovv[h2b,h3b,p2b,p3b] * xb_vo[p2b,h2b] * tbb_vvoo[p1b,p3b,h1b,h3b]
    + 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * xb_vo[p3b,h3b] * tab_vvoo[p2a,p1b,h2a,h1b]
    + 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * xa_vo[p2a,h2a] * tbb_vvoo[p1b,p3b,h1b,h3b]
    + 1.0 * vaa_oovv[h2a,h3a,p2a,p3a] * xa_vo[p2a,h2a] * tab_vvoo[p3a,p1b,h3a,h1b]
    - 0.5 * vbb_oovv[h2b,h3b,p2b,p3b] * xb_vo[p2b,h1b] * tbb_vvoo[p1b,p3b,h2b,h3b]
    - 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * xb_vo[p3b,h1b] * tab_vvoo[p2a,p1b,h2a,h3b]
    - 0.5 * vbb_oovv[h2b,h3b,p2b,p3b] * xb_vo[p1b,h2b] * tbb_vvoo[p2b,p3b,h1b,h3b]
    - 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * xb_vo[p1b,h3b] * tab_vvoo[p2a,p3b,h2a,h1b]
    - 1.0 * fb_ov[h2b,p2b] * xb_vo[p2b,h1b] * tb_vo[p1b,h2b]
    + 1.0 * vbb_vovv[p1b,h2b,p2b,p3b] * xb_vo[p2b,h1b] * tb_vo[p3b,h2b]
    + 1.0 * vab_ovvv[h2a,p1b,p2a,p3b] * xa_vo[p2a,h2a] * tb_vo[p3b,h1b]
    + 1.0 * vbb_oovo[h2b,h3b,p2b,h1b] * xb_vo[p1b,h2b] * tb_vo[p2b,h3b]
    - 1.0 * vab_oovo[h2a,h3b,p2a,h1b] * xa_vo[p2a,h2a] * tb_vo[p1b,h3b]
    + 1.0 * vbb_oovv[h2b,h3b,p2b,p3b] * tb_vo[p2b,h2b] * xbb_vvoo[p1b,p3b,h1b,h3b]
    + 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * tb_vo[p3b,h3b] * xab_vvoo[p2a,p1b,h2a,h1b]
    + 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * ta_vo[p2a,h2a] * xbb_vvoo[p1b,p3b,h1b,h3b]
    + 1.0 * vaa_oovv[h2a,h3a,p2a,p3a] * ta_vo[p2a,h2a] * xab_vvoo[p3a,p1b,h3a,h1b]
    - 0.5 * vbb_oovv[h2b,h3b,p2b,p3b] * tb_vo[p2b,h1b] * xbb_vvoo[p1b,p3b,h2b,h3b]
    - 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * tb_vo[p3b,h1b] * xab_vvoo[p2a,p1b,h2a,h3b]
    - 0.5 * vbb_oovv[h2b,h3b,p2b,p3b] * tb_vo[p1b,h2b] * xbb_vvoo[p2b,p3b,h1b,h3b]
    - 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * tb_vo[p1b,h3b] * xab_vvoo[p2a,p3b,h2a,h1b]
    - 1.0 * fb_ov[h2b,p2b] * tb_vo[p2b,h1b] * xb_vo[p1b,h2b]
    + 1.0 * vbb_vovv[p1b,h2b,p2b,p3b] * tb_vo[p2b,h1b] * xb_vo[p3b,h2b]
    + 1.0 * vab_ovvv[h2a,p1b,p2a,p3b] * ta_vo[p2a,h2a] * xb_vo[p3b,h1b]
    + 1.0 * vbb_oovo[h2b,h3b,p2b,h1b] * tb_vo[p1b,h2b] * xb_vo[p2b,h3b]
    - 1.0 * vab_oovo[h2a,h3b,p2a,h1b] * ta_vo[p2a,h2a] * xb_vo[p1b,h3b]
    - 1.0 * vbb_oovv[h2b,h3b,p2b,p3b] * xb_vo[p2b,h1b] * tb_vo[p1b,h2b] * tb_vo[p3b,h3b]
    - 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * xa_vo[p2a,h2a] * tb_vo[p3b,h1b] * tb_vo[p1b,h3b]
    - 1.0 * vbb_oovv[h2b,h3b,p2b,p3b] * tb_vo[p2b,h1b] * xb_vo[p1b,h2b] * tb_vo[p3b,h3b]
    - 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * ta_vo[p2a,h2a] * xb_vo[p3b,h1b] * tb_vo[p1b,h3b]
    - 1.0 * vbb_oovv[h2b,h3b,p2b,p3b] * tb_vo[p2b,h1b] * tb_vo[p1b,h2b] * xb_vo[p3b,h3b]
    - 1.0 * vab_oovv[h2a,h3b,p2a,p3b] * ta_vo[p2a,h2a] * tb_vo[p3b,h1b] * xb_vo[p1b,h3b];
}
