
{

range O = 10;
range V = 100;
 index h1, h2, h3 = O;
 index p1, p2, p3 = V;
 array f_oo[O][O], f_ov[O][V], f_vo[V][O], f_vv[V][V], v_oovo[O,O][V,O];
 array v_oovv[O,O][V,V], v_ovvo[O,V][V,O], v_vovo[V,O][V,O], v_vovv[V,O][V,V], t_vo[V][O];
 array t_vvoo[V,V][O,O], r_vo[V][O], _a6376[O][V], _a10024[O,V][V,O], _a10025[V][O];
 array _a271[V,V][O,O], _a10145[O][O], _a171[O,O][V,O], _a143[V][O], _a10093[O][O];
 array _a9129[O][O], _a690[O,O][V,O], _a1038[O,O][V,O], _a1039[V][O], _a10042[O,O][V,O];
 array _a205[O][V], _a6379[O][V], _a8036[O][V], _a8037[V][O], _a9131[O][O];
 array _a9146[O][O], _a10146[O][O], _a1121[V,V][O,O], _a1122[V,V][O,O], _a1170[V,V][O,O];
 array _a1171[V][O], _a10125[O][O], _a10094[O][O], _a10126[O][O], _a10148[O][O];
 array _a10149[V][O];

 _a6376[h3,p3] = v_oovv[h2,h3,p2,p3] * t_vo[p2,h2];
 _a10024[h2,p1,p2,h1] = (2.0 * v_ovvo[h2,p1,p2,h1] * -v_vovo[p1,h2,p2,h1]) + (t_vvoo[p2,p3,h1,h2] * -t_vvoo[p3,p2,h1,h2]) + (t_vvoo[p2,p3,h1,h2] * -t_vvoo[p3,p2,h1,h2]);

 _a271[p2,p3,h1,h2] = 2.0 * t_vvoo[p2,p3,h1,h2] + -t_vvoo[p3,p2,h1,h2];
 _a171[h2,h3,p3,h1] = v_oovv[h2,h3,p2,p3] * t_vo[p2,h1];
 _a143[p1,h1] = f_vv[p1,p2] * t_vo[p2,h1];
 _a10093[h2,h1] = f_ov[h2,p2] * t_vo[p2,h1];
 _a10042[h2,h3,p2,h1] = 2.0 * v_oovo[h2,h3,p2,h1] + -v_oovo[h3,h2,p2,h1];
 _a205[h3,p2] = v_oovv[h2,h3,p2,p3] * t_vo[p3,h2];
 _a8036[h2,p2] = _a6379[h2,p2] + -f_ov[h2,p2];
 _a1121[p3,p2,h1,h2] = t_vo[p3,h1] * t_vo[p2,h2];

 _a10025[p1,h1] = t_vo[p2,h2] * _a10024[h2,p1,p2,h1];
 _a10145[h2,h1] = v_oovv[h2,h3,p2,p3] * _a271[p2,p3,h1,h3];
 _a9129[h2,h1] = t_vo[p3,h3] * _a171[h2,h3,p3,h1];
 _a690[h2,h3,p3,h1] = _a171[h2,h3,p3,h1] + v_oovo[h3,h2,p3,h1];
 _a6379[h3,p3] = _a205[h3,p3] + (-2.0 * _a6376[h3,p3]);
 _a8037[p1,h1] = _a271[p1,p2,h1,h2] * _a8036[h2,p2];
 _a9131[h2,h1] = t_vo[p2,h1] * _a205[h2,p2];
 _a1122[p3,p2,h1,h2] = _a1121[p3,p2,h1,h2] + -_a271[p2,p3,h1,h2];
 _a10125[h3,h1] = t_vo[p2,h2] * _a10042[h2,h3,p2,h1];
 _a10094[h2,h1] = _a10093[h2,h1] + f_oo[h2,h1];

 _a1038[h2,h3,p3,h1] = 2.0 * _a690[h2,h3,p3,h1] + -_a690[h3,h2,p3,h1];
 _a1170[p2,p3,h1,h2] = 2.0 * _a1121[p2,p3,h1,h2] + -_a1122[p3,p2,h1,h2];
 _a10126[h3,h1] = _a10094[h3,h1] + _a10125[h3,h1];
 _a1039[p1,h1] = t_vvoo[p1,p3,h2,h3] * _a1038[h2,h3,p3,h1];
 _a1171[p1,h1] = v_vovv[p1,h2,p2,p3] * _a1170[p2,p3,h1,h2];
 _a9146[h2,h1] = 2.0 * _a9129[h2,h1] + -_a9131[h2,h1];
 _a10146[h2,h1] = _a10145[h2,h1] + _a9146[h2,h1];
 _a10148[h2,h1] = _a10126[h2,h1] + _a10146[h2,h1];
 _a10149[p1,h1] = t_vo[p1,h2] * _a10148[h2,h1];

 r_vo[p1,h1] += f_vo[p1,h1];

 r_vo[p1,h1] += -_a1039[p1,h1];
 r_vo[p1,h1] += _a1171[p1,h1] + -_a8037[p1,h1] + _a10025[p1,h1] + _a143[p1,h1] + -_a10149[p1,h1];

}
