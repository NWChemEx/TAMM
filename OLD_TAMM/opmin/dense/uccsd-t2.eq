{
  range Oa = 10;
  range Ob = 10;
  range Va = 100;
  range Vb = 100;

  index  h1a, h2a, h3a, h4a = Oa;
  index  h1b, h2b, h3b, h4b = Ob;
  index  p1a, p2a, p3a, p4a = Va;
  index  p1b, p2b, p3b, p4b = Vb;

  array fa_oo([Oa][Oa]);
  array fa_ov([Oa][Va]);
  array fa_vo([Va][Oa]);
  array fa_vv([Va][Va]);
  array fb_oo([Ob][Ob]);
  array fb_ov([Ob][Vb]);
  array fb_vo([Vb][Ob]);
  array fb_vv([Vb][Vb]);
  array vaa_oooo([Oa,Oa][Oa,Oa]);
  array vaa_oovo([Oa,Oa][Va,Oa]);
  array vaa_oovv([Oa,Oa][Va,Va]);
  array vaa_vooo([Va,Oa][Oa,Oa]);
  array vaa_vovo([Va,Oa][Va,Oa]);
  array vaa_vovv([Va,Oa][Va,Va]);
  array vaa_vvoo([Va,Va][Oa,Oa]);
  array vaa_vvvo([Va,Va][Va,Oa]);
  array vaa_vvvv([Va,Va][Va,Va]);
  array vbb_oooo([Ob,Ob][Ob,Ob]);
  array vbb_oovo([Ob,Ob][Vb,Ob]);
  array vbb_oovv([Ob,Ob][Vb,Vb]);
  array vbb_vooo([Vb,Ob][Ob,Ob]);
  array vbb_vovo([Vb,Ob][Vb,Ob]);
  array vbb_vovv([Vb,Ob][Vb,Vb]);
  array vbb_vvoo([Vb,Vb][Ob,Ob]);
  array vbb_vvvo([Vb,Vb][Vb,Ob]);
  array vbb_vvvv([Vb,Vb][Vb,Vb]);
  array vab_oooo([Oa,Ob][Oa,Ob]);
  array vab_ooov([Oa,Ob][Oa,Vb]);
  array vab_oovo([Oa,Ob][Va,Ob]);
  array vab_oovv([Oa,Ob][Va,Vb]);
  array vab_ovoo([Oa,Vb][Oa,Ob]);
  array vab_ovov([Oa,Vb][Oa,Vb]);
  array vab_ovvo([Oa,Vb][Va,Ob]);
  array vab_ovvv([Oa,Vb][Va,Vb]);
  array vab_vooo([Va,Ob][Oa,Ob]);
  array vab_voov([Va,Ob][Oa,Vb]);
  array vab_vovo([Va,Ob][Va,Ob]);
  array vab_vovv([Va,Ob][Va,Vb]);
  array vab_vvoo([Va,Vb][Oa,Ob]);
  array vab_vvov([Va,Vb][Oa,Vb]);
  array vab_vvvo([Va,Vb][Va,Ob]);
  array vab_vvvv([Va,Vb][Va,Vb]);
  array ta_vo([Va][Oa]);
  array tb_vo([Vb][Ob]);
  array taa_vvoo([Va,Va][Oa,Oa]);
  array tbb_vvoo([Vb,Vb][Ob,Ob]);
  array tab_vvoo([Va,Vb][Oa,Ob]);
  array E([][]);
  array ra_vo([Va][Oa]);
  array rb_vo([Vb][Ob]);
  array raa_vvoo([Va,Va][Oa,Oa]);
  array rbb_vvoo([Vb,Vb][Ob,Ob]);
  array rab_vvoo([Va,Vb][Oa,Ob]);

  rab_vvoo[p1a,p2b,h1a,h2b] =
    1.0 * vab_vvoo[p1a,p2b,h1a,h2b]
    + 1.0 * fa_vv[p1a,p3a] * tab_vvoo[p3a,p2b,h1a,h2b]
    + 1.0 * fb_vv[p2b,p3b] * tab_vvoo[p1a,p3b,h1a,h2b]
    - 1.0 * fa_oo[h3a,h1a] * tab_vvoo[p1a,p2b,h3a,h2b]
    - 1.0 * fb_oo[h3b,h2b] * tab_vvoo[p1a,p2b,h1a,h3b]
    + 1.0 * vab_vvvv[p1a,p2b,p3a,p4b] * tab_vvoo[p3a,p4b,h1a,h2b]
    + 1.0 * vab_voov[p1a,h3b,h1a,p3b] * tbb_vvoo[p2b,p3b,h2b,h3b]
    - 1.0 * vaa_vovo[p1a,h3a,p3a,h1a] * tab_vvoo[p3a,p2b,h3a,h2b]
    - 1.0 * vab_vovo[p1a,h3b,p3a,h2b] * tab_vvoo[p3a,p2b,h1a,h3b]
    - 1.0 * vab_ovov[h3a,p2b,h1a,p3b] * tab_vvoo[p1a,p3b,h3a,h2b]
    - 1.0 * vbb_vovo[p2b,h3b,p3b,h2b] * tab_vvoo[p1a,p3b,h1a,h3b]
    + 1.0 * vab_ovvo[h3a,p2b,p3a,h2b] * taa_vvoo[p1a,p3a,h1a,h3a]
    + 1.0 * vab_oooo[h3a,h4b,h1a,h2b] * tab_vvoo[p1a,p2b,h3a,h4b]
    - 0.5 * vbb_oovv[h3b,h4b,p3b,p4b] * tab_vvoo[p1a,p3b,h1a,h2b] * tbb_vvoo[p2b,p4b,h3b,h4b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p1a,p4b,h1a,h2b] * tab_vvoo[p3a,p2b,h3a,h4b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p3a,p2b,h1a,h2b] * tab_vvoo[p1a,p4b,h3a,h4b]
    - 0.5 * vaa_oovv[h3a,h4a,p3a,p4a] * tab_vvoo[p3a,p2b,h1a,h2b] * taa_vvoo[p1a,p4a,h3a,h4a]
    - 0.5 * vbb_oovv[h3b,h4b,p3b,p4b] * tab_vvoo[p1a,p2b,h1a,h3b] * tbb_vvoo[p3b,p4b,h2b,h4b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p1a,p2b,h1a,h4b] * tab_vvoo[p3a,p4b,h3a,h2b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p3a,p4b,h1a,h4b] * tab_vvoo[p1a,p2b,h3a,h2b]
    + 0.5 * vaa_oovv[h3a,h4a,p3a,p4a] * taa_vvoo[p3a,p4a,h1a,h3a] * tab_vvoo[p1a,p2b,h4a,h2b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p3a,p4b,h1a,h2b] * tab_vvoo[p1a,p2b,h3a,h4b]
    + 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tab_vvoo[p1a,p3b,h1a,h3b] * tbb_vvoo[p2b,p4b,h2b,h4b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p1a,p4b,h1a,h4b] * tab_vvoo[p3a,p2b,h3a,h2b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * taa_vvoo[p1a,p3a,h1a,h3a] * tbb_vvoo[p2b,p4b,h2b,h4b]
    + 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * taa_vvoo[p1a,p3a,h1a,h3a] * tab_vvoo[p4a,p2b,h4a,h2b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p3a,p2b,h1a,h4b] * tab_vvoo[p1a,p4b,h3a,h2b]
    + 1.0 * vab_vvov[p1a,p2b,h1a,p3b] * tb_vo[p3b,h2b]
    + 1.0 * vab_vvvo[p1a,p2b,p3a,h2b] * ta_vo[p3a,h1a]
    - 1.0 * vab_vooo[p1a,h3b,h1a,h2b] * tb_vo[p2b,h3b]
    - 1.0 * vab_ovoo[h3a,p2b,h1a,h2b] * ta_vo[p1a,h3a]
    - 1.0 * fa_ov[h3a,p3a] * ta_vo[p1a,h3a] * tab_vvoo[p3a,p2b,h1a,h2b]
    - 1.0 * fb_ov[h3b,p3b] * tb_vo[p2b,h3b] * tab_vvoo[p1a,p3b,h1a,h2b]
    - 1.0 * fa_ov[h3a,p3a] * ta_vo[p3a,h1a] * tab_vvoo[p1a,p2b,h3a,h2b]
    - 1.0 * fb_ov[h3b,p3b] * tb_vo[p3b,h2b] * tab_vvoo[p1a,p2b,h1a,h3b]
    + 1.0 * vab_vovv[p1a,h3b,p3a,p4b] * tb_vo[p4b,h3b] * tab_vvoo[p3a,p2b,h1a,h2b]
    - 1.0 * vaa_vovv[p1a,h3a,p3a,p4a] * ta_vo[p3a,h3a] * tab_vvoo[p4a,p2b,h1a,h2b]
    - 1.0 * vbb_vovv[p2b,h3b,p3b,p4b] * tb_vo[p3b,h3b] * tab_vvoo[p1a,p4b,h1a,h2b]
    + 1.0 * vab_ovvv[h3a,p2b,p3a,p4b] * ta_vo[p3a,h3a] * tab_vvoo[p1a,p4b,h1a,h2b]
    - 1.0 * vab_ooov[h3a,h4b,h1a,p3b] * tb_vo[p3b,h4b] * tab_vvoo[p1a,p2b,h3a,h2b]
    - 1.0 * vaa_oovo[h3a,h4a,p3a,h1a] * ta_vo[p3a,h3a] * tab_vvoo[p1a,p2b,h4a,h2b]
    - 1.0 * vbb_oovo[h3b,h4b,p3b,h2b] * tb_vo[p3b,h3b] * tab_vvoo[p1a,p2b,h1a,h4b]
    - 1.0 * vab_oovo[h3a,h4b,p3a,h2b] * ta_vo[p3a,h3a] * tab_vvoo[p1a,p2b,h1a,h4b]
    - 1.0 * vab_vovv[p1a,h3b,p3a,p4b] * tb_vo[p2b,h3b] * tab_vvoo[p3a,p4b,h1a,h2b]
    - 1.0 * vab_ovvv[h3a,p2b,p3a,p4b] * ta_vo[p1a,h3a] * tab_vvoo[p3a,p4b,h1a,h2b]
    + 1.0 * vab_vovv[p1a,h3b,p3a,p4b] * ta_vo[p3a,h1a] * tbb_vvoo[p2b,p4b,h2b,h3b]
    + 1.0 * vaa_vovv[p1a,h3a,p3a,p4a] * ta_vo[p3a,h1a] * tab_vvoo[p4a,p2b,h3a,h2b]
    - 1.0 * vab_vovv[p1a,h3b,p3a,p4b] * tb_vo[p4b,h2b] * tab_vvoo[p3a,p2b,h1a,h3b]
    - 1.0 * vab_ovvv[h3a,p2b,p3a,p4b] * ta_vo[p3a,h1a] * tab_vvoo[p1a,p4b,h3a,h2b]
    + 1.0 * vbb_vovv[p2b,h3b,p3b,p4b] * tb_vo[p3b,h2b] * tab_vvoo[p1a,p4b,h1a,h3b]
    + 1.0 * vab_ovvv[h3a,p2b,p3a,p4b] * tb_vo[p4b,h2b] * taa_vvoo[p1a,p3a,h1a,h3a]
    - 1.0 * vab_ooov[h3a,h4b,h1a,p3b] * ta_vo[p1a,h3a] * tbb_vvoo[p2b,p3b,h2b,h4b]
    + 1.0 * vaa_oovo[h3a,h4a,p3a,h1a] * ta_vo[p1a,h3a] * tab_vvoo[p3a,p2b,h4a,h2b]
    + 1.0 * vab_oovo[h3a,h4b,p3a,h2b] * ta_vo[p1a,h3a] * tab_vvoo[p3a,p2b,h1a,h4b]
    + 1.0 * vab_ooov[h3a,h4b,h1a,p3b] * tb_vo[p2b,h4b] * tab_vvoo[p1a,p3b,h3a,h2b]
    + 1.0 * vbb_oovo[h3b,h4b,p3b,h2b] * tb_vo[p2b,h3b] * tab_vvoo[p1a,p3b,h1a,h4b]
    - 1.0 * vab_oovo[h3a,h4b,p3a,h2b] * tb_vo[p2b,h4b] * taa_vvoo[p1a,p3a,h1a,h3a]
    + 1.0 * vab_ooov[h3a,h4b,h1a,p3b] * tb_vo[p3b,h2b] * tab_vvoo[p1a,p2b,h3a,h4b]
    + 1.0 * vab_oovo[h3a,h4b,p3a,h2b] * ta_vo[p3a,h1a] * tab_vvoo[p1a,p2b,h3a,h4b]
    + 1.0 * vab_vvvv[p1a,p2b,p3a,p4b] * ta_vo[p3a,h1a] * tb_vo[p4b,h2b]
    - 1.0 * vab_voov[p1a,h3b,h1a,p3b] * tb_vo[p3b,h2b] * tb_vo[p2b,h3b]
    - 1.0 * vab_vovo[p1a,h3b,p3a,h2b] * ta_vo[p3a,h1a] * tb_vo[p2b,h3b]
    - 1.0 * vab_ovov[h3a,p2b,h1a,p3b] * ta_vo[p1a,h3a] * tb_vo[p3b,h2b]
    - 1.0 * vab_ovvo[h3a,p2b,p3a,h2b] * ta_vo[p3a,h1a] * ta_vo[p1a,h3a]
    + 1.0 * vab_oooo[h3a,h4b,h1a,h2b] * ta_vo[p1a,h3a] * tb_vo[p2b,h4b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p1a,h3a] * tb_vo[p4b,h4b] * tab_vvoo[p3a,p2b,h1a,h2b]
    + 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * ta_vo[p1a,h3a] * ta_vo[p3a,h4a] * tab_vvoo[p4a,p2b,h1a,h2b]
    + 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tb_vo[p2b,h3b] * tb_vo[p3b,h4b] * tab_vvoo[p1a,p4b,h1a,h2b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h3a] * tb_vo[p2b,h4b] * tab_vvoo[p1a,p4b,h1a,h2b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h1a] * tb_vo[p4b,h4b] * tab_vvoo[p1a,p2b,h3a,h2b]
    + 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * ta_vo[p3a,h1a] * ta_vo[p4a,h3a] * tab_vvoo[p1a,p2b,h4a,h2b]
    + 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tb_vo[p3b,h2b] * tb_vo[p4b,h3b] * tab_vvoo[p1a,p2b,h1a,h4b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h3a] * tb_vo[p4b,h2b] * tab_vvoo[p1a,p2b,h1a,h4b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p1a,h3a] * tb_vo[p2b,h4b] * tab_vvoo[p3a,p4b,h1a,h2b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h1a] * ta_vo[p1a,h3a] * tbb_vvoo[p2b,p4b,h2b,h4b]
    - 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * ta_vo[p3a,h1a] * ta_vo[p1a,h3a] * tab_vvoo[p4a,p2b,h4a,h2b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p1a,h3a] * tb_vo[p4b,h2b] * tab_vvoo[p3a,p2b,h1a,h4b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h1a] * tb_vo[p2b,h4b] * tab_vvoo[p1a,p4b,h3a,h2b]
    - 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tb_vo[p3b,h2b] * tb_vo[p2b,h3b] * tab_vvoo[p1a,p4b,h1a,h4b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tb_vo[p4b,h2b] * tb_vo[p2b,h4b] * taa_vvoo[p1a,p3a,h1a,h3a]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h1a] * tb_vo[p4b,h2b] * tab_vvoo[p1a,p2b,h3a,h4b]
    - 1.0 * vab_vovv[p1a,h3b,p3a,p4b] * ta_vo[p3a,h1a] * tb_vo[p4b,h2b] * tb_vo[p2b,h3b]
    - 1.0 * vab_ovvv[h3a,p2b,p3a,p4b] * ta_vo[p3a,h1a] * ta_vo[p1a,h3a] * tb_vo[p4b,h2b]
    + 1.0 * vab_ooov[h3a,h4b,h1a,p3b] * ta_vo[p1a,h3a] * tb_vo[p3b,h2b] * tb_vo[p2b,h4b]
    + 1.0 * vab_oovo[h3a,h4b,p3a,h2b] * ta_vo[p3a,h1a] * ta_vo[p1a,h3a] * tb_vo[p2b,h4b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h1a] * ta_vo[p1a,h3a] * tb_vo[p4b,h2b] * tb_vo[p2b,h4b];

  raa_vvoo[p1a, p2a, h1a, h2a] =
    1.0 * vaa_vvoo[p1a,p2a,h1a,h2a]
    - 1.0 * fa_vv[p1a,p3a] * taa_vvoo[p2a,p3a,h1a,h2a]
    + 1.0 * fa_vv[p2a,p3a] * taa_vvoo[p1a,p3a,h1a,h2a]
    + 1.0 * fa_oo[h3a,h1a] * taa_vvoo[p1a,p2a,h2a,h3a]
    - 1.0 * fa_oo[h3a,h2a] * taa_vvoo[p1a,p2a,h1a,h3a]
    + 0.5 * vaa_vvvv[p1a,p2a,p3a,p4a] * taa_vvoo[p3a,p4a,h1a,h2a]
    + 1.0 * vab_voov[p1a,h3b,h1a,p3b] * tab_vvoo[p2a,p3b,h2a,h3b]
    - 1.0 * vaa_vovo[p1a,h3a,p3a,h1a] * taa_vvoo[p2a,p3a,h2a,h3a]
    - 1.0 * vab_voov[p1a,h3b,h2a,p3b] * tab_vvoo[p2a,p3b,h1a,h3b]
    + 1.0 * vaa_vovo[p1a,h3a,p3a,h2a] * taa_vvoo[p2a,p3a,h1a,h3a]
    - 1.0 * vab_voov[p2a,h3b,h1a,p3b] * tab_vvoo[p1a,p3b,h2a,h3b]
    + 1.0 * vaa_vovo[p2a,h3a,p3a,h1a] * taa_vvoo[p1a,p3a,h2a,h3a]
    + 1.0 * vab_voov[p2a,h3b,h2a,p3b] * tab_vvoo[p1a,p3b,h1a,h3b]
    - 1.0 * vaa_vovo[p2a,h3a,p3a,h2a] * taa_vvoo[p1a,p3a,h1a,h3a]
    + 0.5 * vaa_oooo[h3a,h4a,h1a,h2a] * taa_vvoo[p1a,p2a,h3a,h4a]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * taa_vvoo[p1a,p3a,h1a,h2a] * tab_vvoo[p2a,p4b,h3a,h4b]
    - 0.5 * vaa_oovv[h3a,h4a,p3a,p4a] * taa_vvoo[p1a,p3a,h1a,h2a] * taa_vvoo[p2a,p4a,h3a,h4a]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * taa_vvoo[p2a,p3a,h1a,h2a] * tab_vvoo[p1a,p4b,h3a,h4b]
    + 0.5 * vaa_oovv[h3a,h4a,p3a,p4a] * taa_vvoo[p2a,p3a,h1a,h2a] * taa_vvoo[p1a,p4a,h3a,h4a]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * taa_vvoo[p1a,p2a,h1a,h3a] * tab_vvoo[p3a,p4b,h2a,h4b]
    - 0.5 * vaa_oovv[h3a,h4a,p3a,p4a] * taa_vvoo[p1a,p2a,h1a,h3a] * taa_vvoo[p3a,p4a,h2a,h4a]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p3a,p4b,h1a,h4b] * taa_vvoo[p1a,p2a,h2a,h3a]
    - 0.5 * vaa_oovv[h3a,h4a,p3a,p4a] * taa_vvoo[p3a,p4a,h1a,h3a] * taa_vvoo[p1a,p2a,h2a,h4a]
    + 0.25 * vaa_oovv[h3a,h4a,p3a,p4a] * taa_vvoo[p3a,p4a,h1a,h2a] * taa_vvoo[p1a,p2a,h3a,h4a]
    + 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tab_vvoo[p1a,p3b,h1a,h3b] * tab_vvoo[p2a,p4b,h2a,h4b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p1a,p4b,h1a,h4b] * taa_vvoo[p2a,p3a,h2a,h3a]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * taa_vvoo[p1a,p3a,h1a,h3a] * tab_vvoo[p2a,p4b,h2a,h4b]
    + 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * taa_vvoo[p1a,p3a,h1a,h3a] * taa_vvoo[p2a,p4a,h2a,h4a]
    - 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tab_vvoo[p2a,p3b,h1a,h3b] * tab_vvoo[p1a,p4b,h2a,h4b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p2a,p4b,h1a,h4b] * taa_vvoo[p1a,p3a,h2a,h3a]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * taa_vvoo[p2a,p3a,h1a,h3a] * tab_vvoo[p1a,p4b,h2a,h4b]
    - 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * taa_vvoo[p2a,p3a,h1a,h3a] * taa_vvoo[p1a,p4a,h2a,h4a]
    - 1.0 * vaa_vvvo[p1a,p2a,p3a,h1a] * ta_vo[p3a,h2a]
    + 1.0 * vaa_vvvo[p1a,p2a,p3a,h2a] * ta_vo[p3a,h1a]
    - 1.0 * vaa_vooo[p1a,h3a,h1a,h2a] * ta_vo[p2a,h3a]
    + 1.0 * vaa_vooo[p2a,h3a,h1a,h2a] * ta_vo[p1a,h3a]
    + 1.0 * fa_ov[h3a,p3a] * ta_vo[p1a,h3a] * taa_vvoo[p2a,p3a,h1a,h2a]
    - 1.0 * fa_ov[h3a,p3a] * ta_vo[p2a,h3a] * taa_vvoo[p1a,p3a,h1a,h2a]
    + 1.0 * fa_ov[h3a,p3a] * ta_vo[p3a,h1a] * taa_vvoo[p1a,p2a,h2a,h3a]
    - 1.0 * fa_ov[h3a,p3a] * ta_vo[p3a,h2a] * taa_vvoo[p1a,p2a,h1a,h3a]
    - 1.0 * vab_vovv[p1a,h3b,p3a,p4b] * tb_vo[p4b,h3b] * taa_vvoo[p2a,p3a,h1a,h2a]
    + 1.0 * vaa_vovv[p1a,h3a,p3a,p4a] * ta_vo[p3a,h3a] * taa_vvoo[p2a,p4a,h1a,h2a]
    + 1.0 * vab_vovv[p2a,h3b,p3a,p4b] * tb_vo[p4b,h3b] * taa_vvoo[p1a,p3a,h1a,h2a]
    - 1.0 * vaa_vovv[p2a,h3a,p3a,p4a] * ta_vo[p3a,h3a] * taa_vvoo[p1a,p4a,h1a,h2a]
    + 1.0 * vab_ooov[h3a,h4b,h1a,p3b] * tb_vo[p3b,h4b] * taa_vvoo[p1a,p2a,h2a,h3a]
    + 1.0 * vaa_oovo[h3a,h4a,p3a,h1a] * ta_vo[p3a,h3a] * taa_vvoo[p1a,p2a,h2a,h4a]
    - 1.0 * vab_ooov[h3a,h4b,h2a,p3b] * tb_vo[p3b,h4b] * taa_vvoo[p1a,p2a,h1a,h3a]
    - 1.0 * vaa_oovo[h3a,h4a,p3a,h2a] * ta_vo[p3a,h3a] * taa_vvoo[p1a,p2a,h1a,h4a]
    - 0.5 * vaa_vovv[p1a,h3a,p3a,p4a] * ta_vo[p2a,h3a] * taa_vvoo[p3a,p4a,h1a,h2a]
    + 0.5 * vaa_vovv[p2a,h3a,p3a,p4a] * ta_vo[p1a,h3a] * taa_vvoo[p3a,p4a,h1a,h2a]
    + 1.0 * vab_vovv[p1a,h3b,p3a,p4b] * ta_vo[p3a,h1a] * tab_vvoo[p2a,p4b,h2a,h3b]
    + 1.0 * vaa_vovv[p1a,h3a,p3a,p4a] * ta_vo[p3a,h1a] * taa_vvoo[p2a,p4a,h2a,h3a]
    - 1.0 * vab_vovv[p1a,h3b,p3a,p4b] * ta_vo[p3a,h2a] * tab_vvoo[p2a,p4b,h1a,h3b]
    - 1.0 * vaa_vovv[p1a,h3a,p3a,p4a] * ta_vo[p3a,h2a] * taa_vvoo[p2a,p4a,h1a,h3a]
    - 1.0 * vab_vovv[p2a,h3b,p3a,p4b] * ta_vo[p3a,h1a] * tab_vvoo[p1a,p4b,h2a,h3b]
    - 1.0 * vaa_vovv[p2a,h3a,p3a,p4a] * ta_vo[p3a,h1a] * taa_vvoo[p1a,p4a,h2a,h3a]
    + 1.0 * vab_vovv[p2a,h3b,p3a,p4b] * ta_vo[p3a,h2a] * tab_vvoo[p1a,p4b,h1a,h3b]
    + 1.0 * vaa_vovv[p2a,h3a,p3a,p4a] * ta_vo[p3a,h2a] * taa_vvoo[p1a,p4a,h1a,h3a]
    - 1.0 * vab_ooov[h3a,h4b,h1a,p3b] * ta_vo[p1a,h3a] * tab_vvoo[p2a,p3b,h2a,h4b]
    + 1.0 * vaa_oovo[h3a,h4a,p3a,h1a] * ta_vo[p1a,h3a] * taa_vvoo[p2a,p3a,h2a,h4a]
    + 1.0 * vab_ooov[h3a,h4b,h2a,p3b] * ta_vo[p1a,h3a] * tab_vvoo[p2a,p3b,h1a,h4b]
    - 1.0 * vaa_oovo[h3a,h4a,p3a,h2a] * ta_vo[p1a,h3a] * taa_vvoo[p2a,p3a,h1a,h4a]
    + 1.0 * vab_ooov[h3a,h4b,h1a,p3b] * ta_vo[p2a,h3a] * tab_vvoo[p1a,p3b,h2a,h4b]
    - 1.0 * vaa_oovo[h3a,h4a,p3a,h1a] * ta_vo[p2a,h3a] * taa_vvoo[p1a,p3a,h2a,h4a]
    - 1.0 * vab_ooov[h3a,h4b,h2a,p3b] * ta_vo[p2a,h3a] * tab_vvoo[p1a,p3b,h1a,h4b]
    + 1.0 * vaa_oovo[h3a,h4a,p3a,h2a] * ta_vo[p2a,h3a] * taa_vvoo[p1a,p3a,h1a,h4a]
    - 0.5 * vaa_oovo[h3a,h4a,p3a,h1a] * ta_vo[p3a,h2a] * taa_vvoo[p1a,p2a,h3a,h4a]
    + 0.5 * vaa_oovo[h3a,h4a,p3a,h2a] * ta_vo[p3a,h1a] * taa_vvoo[p1a,p2a,h3a,h4a]
    + 1.0 * vaa_vvvv[p1a,p2a,p3a,p4a] * ta_vo[p3a,h1a] * ta_vo[p4a,h2a]
    + 1.0 * vaa_vovo[p1a,h3a,p3a,h1a] * ta_vo[p3a,h2a] * ta_vo[p2a,h3a]
    - 1.0 * vaa_vovo[p1a,h3a,p3a,h2a] * ta_vo[p3a,h1a] * ta_vo[p2a,h3a]
    - 1.0 * vaa_vovo[p2a,h3a,p3a,h1a] * ta_vo[p3a,h2a] * ta_vo[p1a,h3a]
    + 1.0 * vaa_vovo[p2a,h3a,p3a,h2a] * ta_vo[p3a,h1a] * ta_vo[p1a,h3a]
    + 1.0 * vaa_oooo[h3a,h4a,h1a,h2a] * ta_vo[p1a,h3a] * ta_vo[p2a,h4a]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p1a,h3a] * tb_vo[p4b,h4b] * taa_vvoo[p2a,p3a,h1a,h2a]
    - 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * ta_vo[p1a,h3a] * ta_vo[p3a,h4a] * taa_vvoo[p2a,p4a,h1a,h2a]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p2a,h3a] * tb_vo[p4b,h4b] * taa_vvoo[p1a,p3a,h1a,h2a]
    + 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * ta_vo[p2a,h3a] * ta_vo[p3a,h4a] * taa_vvoo[p1a,p4a,h1a,h2a]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h1a] * tb_vo[p4b,h4b] * taa_vvoo[p1a,p2a,h2a,h3a]
    - 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * ta_vo[p3a,h1a] * ta_vo[p4a,h3a] * taa_vvoo[p1a,p2a,h2a,h4a]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h2a] * tb_vo[p4b,h4b] * taa_vvoo[p1a,p2a,h1a,h3a]
    + 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * ta_vo[p3a,h2a] * ta_vo[p4a,h3a] * taa_vvoo[p1a,p2a,h1a,h4a]
    + 0.5 * vaa_oovv[h3a,h4a,p3a,p4a] * ta_vo[p1a,h3a] * ta_vo[p2a,h4a] * taa_vvoo[p3a,p4a,h1a,h2a]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h1a] * ta_vo[p1a,h3a] * tab_vvoo[p2a,p4b,h2a,h4b]
    - 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * ta_vo[p3a,h1a] * ta_vo[p1a,h3a] * taa_vvoo[p2a,p4a,h2a,h4a]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h2a] * ta_vo[p1a,h3a] * tab_vvoo[p2a,p4b,h1a,h4b]
    + 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * ta_vo[p3a,h2a] * ta_vo[p1a,h3a] * taa_vvoo[p2a,p4a,h1a,h4a]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h1a] * ta_vo[p2a,h3a] * tab_vvoo[p1a,p4b,h2a,h4b]
    + 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * ta_vo[p3a,h1a] * ta_vo[p2a,h3a] * taa_vvoo[p1a,p4a,h2a,h4a]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h2a] * ta_vo[p2a,h3a] * tab_vvoo[p1a,p4b,h1a,h4b]
    - 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * ta_vo[p3a,h2a] * ta_vo[p2a,h3a] * taa_vvoo[p1a,p4a,h1a,h4a]
    + 0.5 * vaa_oovv[h3a,h4a,p3a,p4a] * ta_vo[p3a,h1a] * ta_vo[p4a,h2a] * taa_vvoo[p1a,p2a,h3a,h4a]
    - 1.0 * vaa_vovv[p1a,h3a,p3a,p4a] * ta_vo[p3a,h1a] * ta_vo[p4a,h2a] * ta_vo[p2a,h3a]
    + 1.0 * vaa_vovv[p2a,h3a,p3a,p4a] * ta_vo[p3a,h1a] * ta_vo[p4a,h2a] * ta_vo[p1a,h3a]
    - 1.0 * vaa_oovo[h3a,h4a,p3a,h1a] * ta_vo[p3a,h2a] * ta_vo[p1a,h3a] * ta_vo[p2a,h4a]
    + 1.0 * vaa_oovo[h3a,h4a,p3a,h2a] * ta_vo[p3a,h1a] * ta_vo[p1a,h3a] * ta_vo[p2a,h4a]
    + 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * ta_vo[p3a,h1a] * ta_vo[p4a,h2a] * ta_vo[p1a,h3a] * ta_vo[p2a,h4a];

  rbb_vvoo[p1b,p2b,h1b,h2b] =
    1.0 * vbb_vvoo[p1b,p2b,h1b,h2b]
    - 1.0 * fb_vv[p1b,p3b] * tbb_vvoo[p2b,p3b,h1b,h2b]
    + 1.0 * fb_vv[p2b,p3b] * tbb_vvoo[p1b,p3b,h1b,h2b]
    + 1.0 * fb_oo[h3b,h1b] * tbb_vvoo[p1b,p2b,h2b,h3b]
    - 1.0 * fb_oo[h3b,h2b] * tbb_vvoo[p1b,p2b,h1b,h3b]
    + 0.5 * vbb_vvvv[p1b,p2b,p3b,p4b] * tbb_vvoo[p3b,p4b,h1b,h2b]
    - 1.0 * vbb_vovo[p1b,h3b,p3b,h1b] * tbb_vvoo[p2b,p3b,h2b,h3b]
    + 1.0 * vab_ovvo[h3a,p1b,p3a,h1b] * tab_vvoo[p3a,p2b,h3a,h2b]
    + 1.0 * vbb_vovo[p1b,h3b,p3b,h2b] * tbb_vvoo[p2b,p3b,h1b,h3b]
    - 1.0 * vab_ovvo[h3a,p1b,p3a,h2b] * tab_vvoo[p3a,p2b,h3a,h1b]
    + 1.0 * vbb_vovo[p2b,h3b,p3b,h1b] * tbb_vvoo[p1b,p3b,h2b,h3b]
    - 1.0 * vab_ovvo[h3a,p2b,p3a,h1b] * tab_vvoo[p3a,p1b,h3a,h2b]
    - 1.0 * vbb_vovo[p2b,h3b,p3b,h2b] * tbb_vvoo[p1b,p3b,h1b,h3b]
    + 1.0 * vab_ovvo[h3a,p2b,p3a,h2b] * tab_vvoo[p3a,p1b,h3a,h1b]
    + 0.5 * vbb_oooo[h3b,h4b,h1b,h2b] * tbb_vvoo[p1b,p2b,h3b,h4b]
    - 0.5 * vbb_oovv[h3b,h4b,p3b,p4b] * tbb_vvoo[p1b,p3b,h1b,h2b] * tbb_vvoo[p2b,p4b,h3b,h4b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p3a,p2b,h3a,h4b] * tbb_vvoo[p1b,p4b,h1b,h2b]
    + 0.5 * vbb_oovv[h3b,h4b,p3b,p4b] * tbb_vvoo[p2b,p3b,h1b,h2b] * tbb_vvoo[p1b,p4b,h3b,h4b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p3a,p1b,h3a,h4b] * tbb_vvoo[p2b,p4b,h1b,h2b]
    - 0.5 * vbb_oovv[h3b,h4b,p3b,p4b] * tbb_vvoo[p1b,p2b,h1b,h3b] * tbb_vvoo[p3b,p4b,h2b,h4b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p3a,p4b,h3a,h2b] * tbb_vvoo[p1b,p2b,h1b,h4b]
    - 0.5 * vbb_oovv[h3b,h4b,p3b,p4b] * tbb_vvoo[p3b,p4b,h1b,h3b] * tbb_vvoo[p1b,p2b,h2b,h4b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p3a,p4b,h3a,h1b] * tbb_vvoo[p1b,p2b,h2b,h4b]
    + 0.25 * vbb_oovv[h3b,h4b,p3b,p4b] * tbb_vvoo[p3b,p4b,h1b,h2b] * tbb_vvoo[p1b,p2b,h3b,h4b]
    + 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tbb_vvoo[p1b,p3b,h1b,h3b] * tbb_vvoo[p2b,p4b,h2b,h4b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p3a,p2b,h3a,h2b] * tbb_vvoo[p1b,p4b,h1b,h4b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p3a,p1b,h3a,h1b] * tbb_vvoo[p2b,p4b,h2b,h4b]
    + 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * tab_vvoo[p3a,p1b,h3a,h1b] * tab_vvoo[p4a,p2b,h4a,h2b]
    - 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tbb_vvoo[p2b,p3b,h1b,h3b] * tbb_vvoo[p1b,p4b,h2b,h4b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p3a,p1b,h3a,h2b] * tbb_vvoo[p2b,p4b,h1b,h4b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tab_vvoo[p3a,p2b,h3a,h1b] * tbb_vvoo[p1b,p4b,h2b,h4b]
    - 1.0 * vaa_oovv[h3a,h4a,p3a,p4a] * tab_vvoo[p3a,p2b,h3a,h1b] * tab_vvoo[p4a,p1b,h4a,h2b]
    - 1.0 * vbb_vvvo[p1b,p2b,p3b,h1b] * tb_vo[p3b,h2b]
    + 1.0 * vbb_vvvo[p1b,p2b,p3b,h2b] * tb_vo[p3b,h1b]
    - 1.0 * vbb_vooo[p1b,h3b,h1b,h2b] * tb_vo[p2b,h3b]
    + 1.0 * vbb_vooo[p2b,h3b,h1b,h2b] * tb_vo[p1b,h3b]
    + 1.0 * fb_ov[h3b,p3b] * tb_vo[p1b,h3b] * tbb_vvoo[p2b,p3b,h1b,h2b]
    - 1.0 * fb_ov[h3b,p3b] * tb_vo[p2b,h3b] * tbb_vvoo[p1b,p3b,h1b,h2b]
    + 1.0 * fb_ov[h3b,p3b] * tb_vo[p3b,h1b] * tbb_vvoo[p1b,p2b,h2b,h3b]
    - 1.0 * fb_ov[h3b,p3b] * tb_vo[p3b,h2b] * tbb_vvoo[p1b,p2b,h1b,h3b]
    + 1.0 * vbb_vovv[p1b,h3b,p3b,p4b] * tb_vo[p3b,h3b] * tbb_vvoo[p2b,p4b,h1b,h2b]
    - 1.0 * vab_ovvv[h3a,p1b,p3a,p4b] * ta_vo[p3a,h3a] * tbb_vvoo[p2b,p4b,h1b,h2b]
    - 1.0 * vbb_vovv[p2b,h3b,p3b,p4b] * tb_vo[p3b,h3b] * tbb_vvoo[p1b,p4b,h1b,h2b]
    + 1.0 * vab_ovvv[h3a,p2b,p3a,p4b] * ta_vo[p3a,h3a] * tbb_vvoo[p1b,p4b,h1b,h2b]
    + 1.0 * vbb_oovo[h3b,h4b,p3b,h1b] * tb_vo[p3b,h3b] * tbb_vvoo[p1b,p2b,h2b,h4b]
    + 1.0 * vab_oovo[h3a,h4b,p3a,h1b] * ta_vo[p3a,h3a] * tbb_vvoo[p1b,p2b,h2b,h4b]
    - 1.0 * vbb_oovo[h3b,h4b,p3b,h2b] * tb_vo[p3b,h3b] * tbb_vvoo[p1b,p2b,h1b,h4b]
    - 1.0 * vab_oovo[h3a,h4b,p3a,h2b] * ta_vo[p3a,h3a] * tbb_vvoo[p1b,p2b,h1b,h4b]
    - 0.5 * vbb_vovv[p1b,h3b,p3b,p4b] * tb_vo[p2b,h3b] * tbb_vvoo[p3b,p4b,h1b,h2b]
    + 0.5 * vbb_vovv[p2b,h3b,p3b,p4b] * tb_vo[p1b,h3b] * tbb_vvoo[p3b,p4b,h1b,h2b]
    + 1.0 * vbb_vovv[p1b,h3b,p3b,p4b] * tb_vo[p3b,h1b] * tbb_vvoo[p2b,p4b,h2b,h3b]
    + 1.0 * vab_ovvv[h3a,p1b,p3a,p4b] * tb_vo[p4b,h1b] * tab_vvoo[p3a,p2b,h3a,h2b]
    - 1.0 * vbb_vovv[p1b,h3b,p3b,p4b] * tb_vo[p3b,h2b] * tbb_vvoo[p2b,p4b,h1b,h3b]
    - 1.0 * vab_ovvv[h3a,p1b,p3a,p4b] * tb_vo[p4b,h2b] * tab_vvoo[p3a,p2b,h3a,h1b]
    - 1.0 * vbb_vovv[p2b,h3b,p3b,p4b] * tb_vo[p3b,h1b] * tbb_vvoo[p1b,p4b,h2b,h3b]
    - 1.0 * vab_ovvv[h3a,p2b,p3a,p4b] * tb_vo[p4b,h1b] * tab_vvoo[p3a,p1b,h3a,h2b]
    + 1.0 * vbb_vovv[p2b,h3b,p3b,p4b] * tb_vo[p3b,h2b] * tbb_vvoo[p1b,p4b,h1b,h3b]
    + 1.0 * vab_ovvv[h3a,p2b,p3a,p4b] * tb_vo[p4b,h2b] * tab_vvoo[p3a,p1b,h3a,h1b]
    + 1.0 * vbb_oovo[h3b,h4b,p3b,h1b] * tb_vo[p1b,h3b] * tbb_vvoo[p2b,p3b,h2b,h4b]
    - 1.0 * vab_oovo[h3a,h4b,p3a,h1b] * tb_vo[p1b,h4b] * tab_vvoo[p3a,p2b,h3a,h2b]
    - 1.0 * vbb_oovo[h3b,h4b,p3b,h2b] * tb_vo[p1b,h3b] * tbb_vvoo[p2b,p3b,h1b,h4b]
    + 1.0 * vab_oovo[h3a,h4b,p3a,h2b] * tb_vo[p1b,h4b] * tab_vvoo[p3a,p2b,h3a,h1b]
    - 1.0 * vbb_oovo[h3b,h4b,p3b,h1b] * tb_vo[p2b,h3b] * tbb_vvoo[p1b,p3b,h2b,h4b]
    + 1.0 * vab_oovo[h3a,h4b,p3a,h1b] * tb_vo[p2b,h4b] * tab_vvoo[p3a,p1b,h3a,h2b]
    + 1.0 * vbb_oovo[h3b,h4b,p3b,h2b] * tb_vo[p2b,h3b] * tbb_vvoo[p1b,p3b,h1b,h4b]
    - 1.0 * vab_oovo[h3a,h4b,p3a,h2b] * tb_vo[p2b,h4b] * tab_vvoo[p3a,p1b,h3a,h1b]
    - 0.5 * vbb_oovo[h3b,h4b,p3b,h1b] * tb_vo[p3b,h2b] * tbb_vvoo[p1b,p2b,h3b,h4b]
    + 0.5 * vbb_oovo[h3b,h4b,p3b,h2b] * tb_vo[p3b,h1b] * tbb_vvoo[p1b,p2b,h3b,h4b]
    + 1.0 * vbb_vvvv[p1b,p2b,p3b,p4b] * tb_vo[p3b,h1b] * tb_vo[p4b,h2b]
    + 1.0 * vbb_vovo[p1b,h3b,p3b,h1b] * tb_vo[p3b,h2b] * tb_vo[p2b,h3b]
    - 1.0 * vbb_vovo[p1b,h3b,p3b,h2b] * tb_vo[p3b,h1b] * tb_vo[p2b,h3b]
    - 1.0 * vbb_vovo[p2b,h3b,p3b,h1b] * tb_vo[p3b,h2b] * tb_vo[p1b,h3b]
    + 1.0 * vbb_vovo[p2b,h3b,p3b,h2b] * tb_vo[p3b,h1b] * tb_vo[p1b,h3b]
    + 1.0 * vbb_oooo[h3b,h4b,h1b,h2b] * tb_vo[p1b,h3b] * tb_vo[p2b,h4b]
    - 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tb_vo[p1b,h3b] * tb_vo[p3b,h4b] * tbb_vvoo[p2b,p4b,h1b,h2b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h3a] * tb_vo[p1b,h4b] * tbb_vvoo[p2b,p4b,h1b,h2b]
    + 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tb_vo[p2b,h3b] * tb_vo[p3b,h4b] * tbb_vvoo[p1b,p4b,h1b,h2b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h3a] * tb_vo[p2b,h4b] * tbb_vvoo[p1b,p4b,h1b,h2b]
    - 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tb_vo[p3b,h1b] * tb_vo[p4b,h3b] * tbb_vvoo[p1b,p2b,h2b,h4b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h3a] * tb_vo[p4b,h1b] * tbb_vvoo[p1b,p2b,h2b,h4b]
    + 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tb_vo[p3b,h2b] * tb_vo[p4b,h3b] * tbb_vvoo[p1b,p2b,h1b,h4b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * ta_vo[p3a,h3a] * tb_vo[p4b,h2b] * tbb_vvoo[p1b,p2b,h1b,h4b]
    + 0.5 * vbb_oovv[h3b,h4b,p3b,p4b] * tb_vo[p1b,h3b] * tb_vo[p2b,h4b] * tbb_vvoo[p3b,p4b,h1b,h2b]
    - 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tb_vo[p3b,h1b] * tb_vo[p1b,h3b] * tbb_vvoo[p2b,p4b,h2b,h4b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tb_vo[p4b,h1b] * tb_vo[p1b,h4b] * tab_vvoo[p3a,p2b,h3a,h2b]
    + 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tb_vo[p3b,h2b] * tb_vo[p1b,h3b] * tbb_vvoo[p2b,p4b,h1b,h4b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tb_vo[p4b,h2b] * tb_vo[p1b,h4b] * tab_vvoo[p3a,p2b,h3a,h1b]
    + 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tb_vo[p3b,h1b] * tb_vo[p2b,h3b] * tbb_vvoo[p1b,p4b,h2b,h4b]
    + 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tb_vo[p4b,h1b] * tb_vo[p2b,h4b] * tab_vvoo[p3a,p1b,h3a,h2b]
    - 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tb_vo[p3b,h2b] * tb_vo[p2b,h3b] * tbb_vvoo[p1b,p4b,h1b,h4b]
    - 1.0 * vab_oovv[h3a,h4b,p3a,p4b] * tb_vo[p4b,h2b] * tb_vo[p2b,h4b] * tab_vvoo[p3a,p1b,h3a,h1b]
    + 0.5 * vbb_oovv[h3b,h4b,p3b,p4b] * tb_vo[p3b,h1b] * tb_vo[p4b,h2b] * tbb_vvoo[p1b,p2b,h3b,h4b]
    - 1.0 * vbb_vovv[p1b,h3b,p3b,p4b] * tb_vo[p3b,h1b] * tb_vo[p4b,h2b] * tb_vo[p2b,h3b]
    + 1.0 * vbb_vovv[p2b,h3b,p3b,p4b] * tb_vo[p3b,h1b] * tb_vo[p4b,h2b] * tb_vo[p1b,h3b]
    - 1.0 * vbb_oovo[h3b,h4b,p3b,h1b] * tb_vo[p3b,h2b] * tb_vo[p1b,h3b] * tb_vo[p2b,h4b]
    + 1.0 * vbb_oovo[h3b,h4b,p3b,h2b] * tb_vo[p3b,h1b] * tb_vo[p1b,h3b] * tb_vo[p2b,h4b]
    + 1.0 * vbb_oovv[h3b,h4b,p3b,p4b] * tb_vo[p3b,h1b] * tb_vo[p4b,h2b] * tb_vo[p1b,h3b] * tb_vo[p2b,h4b];
}
