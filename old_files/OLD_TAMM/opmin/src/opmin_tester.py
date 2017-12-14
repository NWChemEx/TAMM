#! /usr/contrib/bin/python
import random
from helper import makeSymArray, makeArray, replicateArray, replicateSymArray
def run(): 

 #---------- Declarations ---------- 

 O = 2
 V = 2
 v_oovo_input = makeArray([O,O,V,O], True) 
 f_vv_input = makeArray([V,V], True) 
 t_vvoo_input = makeArray([V,V,O,O], True) 
 v_vovv_input = makeArray([V,O,V,V], True) 
 f_ov_input = makeArray([O,V], True) 
 v_ovvo_input = makeArray([O,V,V,O], True) 
 f_oo_input = makeArray([O,O], True) 
 t_vo_input = makeArray([V,O], True) 
 f_vo_input = makeArray([V,O], True) 
 v_oovv_input = makeArray([O,O,V,V], True) 
 v_vovo_input = makeArray([V,O,V,O], True) 

 #---------- Original Equation ---------- 

 f_oo = replicateSymArray(f_oo_input,[O,O],[[0], [1]]) 
 f_ov = replicateSymArray(f_ov_input,[O,V],[[0], [1]]) 
 f_vo = replicateSymArray(f_vo_input,[V,O],[[0], [1]]) 
 f_vv = replicateSymArray(f_vv_input,[V,V],[[0], [1]]) 
 v_oovo = replicateSymArray(v_oovo_input,[O,O,V,O],[[0], [1], [2], [3]]) 
 v_oovv = replicateSymArray(v_oovv_input,[O,O,V,V],[[0], [1], [2], [3]]) 
 v_ovvo = replicateSymArray(v_ovvo_input,[O,V,V,O],[[0], [1], [2], [3]]) 
 v_vovo = replicateSymArray(v_vovo_input,[V,O,V,O],[[0], [1], [2], [3]]) 
 v_vovv = replicateSymArray(v_vovv_input,[V,O,V,V],[[0], [1], [2], [3]]) 
 t_vo = replicateSymArray(t_vo_input,[V,O],[[0], [1]]) 
 t_vvoo = replicateSymArray(t_vvoo_input,[V,V,O,O],[[0], [1], [2], [3]]) 
 r_vo = makeArray([V,O]) 
 _temp_2 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for p2 in range(0, V):
    _temp_2[p1][h1] += (f_vv[p1][p2]) * (t_vo[p2][h1])
 _temp_3 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    _temp_3[p1][h1] += -((f_oo[h2][h1]) * (t_vo[p1][h2]))
 _temp_4 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     _temp_4[p1][h1] += (2.0 * (f_ov[h2][p2]) * (t_vvoo[p1][p2][h1][h2]))
 _temp_5 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     _temp_5[p1][h1] += -((f_ov[h2][p2]) * (t_vvoo[p2][p1][h1][h2]))
 _temp_6 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     for p3 in range(0, V):
      _temp_6[p1][h1] += -((v_vovv[p1][h2][p2][p3]) * (t_vvoo[p3][p2][h1][h2]))
 _temp_7 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     for p3 in range(0, V):
      _temp_7[p1][h1] += (2.0 * (v_vovv[p1][h2][p2][p3]) * (t_vvoo[p2][p3][h1][h2]))
 _temp_8 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     _temp_8[p1][h1] += -((v_vovo[p1][h2][p2][h1]) * (t_vo[p2][h2]))
 _temp_9 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     _temp_9[p1][h1] += (2.0 * (v_ovvo[h2][p1][p2][h1]) * (t_vo[p2][h2]))
 _temp_10 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      _temp_10[p1][h1] += (v_oovo[h2][h3][p2][h1]) * (t_vvoo[p1][p2][h2][h3])
 _temp_11 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      _temp_11[p1][h1] += (-2.0 * (v_oovo[h2][h3][p2][h1]) * (t_vvoo[p1][p2][h3][h2]))
 _temp_12 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     _temp_12[p1][h1] += -((f_ov[h2][p2]) * (t_vo[p2][h1]) * (t_vo[p1][h2]))
 _temp_13 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     for p3 in range(0, V):
      _temp_13[p1][h1] += (2.0 * (v_vovv[p1][h2][p2][p3]) * (t_vo[p2][h1]) * (t_vo[p3][h2]))
 _temp_14 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     for p3 in range(0, V):
      _temp_14[p1][h1] += -((v_vovv[p1][h2][p2][p3]) * (t_vo[p3][h1]) * (t_vo[p2][h2]))
 _temp_15 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      for p3 in range(0, V):
       _temp_15[p1][h1] += (v_oovv[h2][h3][p2][p3]) * (t_vo[p2][h1]) * (t_vvoo[p1][p3][h3][h2])
 _temp_16 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      for p3 in range(0, V):
       _temp_16[p1][h1] += (-2.0 * (v_oovv[h2][h3][p2][p3]) * (t_vo[p2][h1]) * (t_vvoo[p1][p3][h2][h3]))
 _temp_17 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      for p3 in range(0, V):
       _temp_17[p1][h1] += (v_oovv[h2][h3][p2][p3]) * (t_vo[p1][h2]) * (t_vvoo[p3][p2][h1][h3])
 _temp_18 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      for p3 in range(0, V):
       _temp_18[p1][h1] += (-2.0 * (v_oovv[h2][h3][p2][p3]) * (t_vo[p1][h2]) * (t_vvoo[p2][p3][h1][h3]))
 _temp_19 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      for p3 in range(0, V):
       _temp_19[p1][h1] += (4.0 * (v_oovv[h2][h3][p2][p3]) * (t_vo[p2][h2]) * (t_vvoo[p1][p3][h1][h3]))
 _temp_20 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      for p3 in range(0, V):
       _temp_20[p1][h1] += (-2.0 * (v_oovv[h2][h3][p2][p3]) * (t_vo[p2][h2]) * (t_vvoo[p3][p1][h1][h3]))
 _temp_21 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      for p3 in range(0, V):
       _temp_21[p1][h1] += (-2.0 * (v_oovv[h2][h3][p2][p3]) * (t_vo[p3][h2]) * (t_vvoo[p1][p2][h1][h3]))
 _temp_22 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      for p3 in range(0, V):
       _temp_22[p1][h1] += (v_oovv[h2][h3][p2][p3]) * (t_vo[p3][h2]) * (t_vvoo[p2][p1][h1][h3])
 _temp_23 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      _temp_23[p1][h1] += (-2.0 * (v_oovo[h2][h3][p2][h1]) * (t_vo[p2][h2]) * (t_vo[p1][h3]))
 _temp_24 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      _temp_24[p1][h1] += (v_oovo[h2][h3][p2][h1]) * (t_vo[p1][h2]) * (t_vo[p2][h3])
 _temp_25 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      for p3 in range(0, V):
       _temp_25[p1][h1] += (-2.0 * (v_oovv[h2][h3][p2][p3]) * (t_vo[p2][h1]) * (t_vo[p1][h2]) * (t_vo[p3][h3]))
 _temp_26 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      for p3 in range(0, V):
       _temp_26[p1][h1] += (v_oovv[h2][h3][p2][p3]) * (t_vo[p2][h1]) * (t_vo[p3][h2]) * (t_vo[p1][h3])
 _temp_1 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   _temp_1[p1][h1] += (f_vo[p1][h1]) + (_temp_2[p1][h1]) + (_temp_3[p1][h1]) + (_temp_4[p1][h1]) + (_temp_5[p1][h1]) + (_temp_6[p1][h1]) + (_temp_7[p1][h1]) + (_temp_8[p1][h1]) + (_temp_9[p1][h1]) + (_temp_10[p1][h1]) + (_temp_11[p1][h1]) + (_temp_12[p1][h1]) + (_temp_13[p1][h1]) + (_temp_14[p1][h1]) + (_temp_15[p1][h1]) + (_temp_16[p1][h1]) + (_temp_17[p1][h1]) + (_temp_18[p1][h1]) + (_temp_19[p1][h1]) + (_temp_20[p1][h1]) + (_temp_21[p1][h1]) + (_temp_22[p1][h1]) + (_temp_23[p1][h1]) + (_temp_24[p1][h1]) + (_temp_25[p1][h1]) + (_temp_26[p1][h1])
 for p1 in range(0, V):
  for h1 in range(0, O):
   r_vo[p1][h1] = _temp_1[p1][h1]
 
 r_vo_original = replicateArray(r_vo) 

 #---------- Optimized Equation ---------- 

 f_oo = replicateSymArray(f_oo_input,[O,O],[[0], [1]]) 
 f_ov = replicateSymArray(f_ov_input,[O,V],[[0], [1]]) 
 f_vo = replicateSymArray(f_vo_input,[V,O],[[0], [1]]) 
 f_vv = replicateSymArray(f_vv_input,[V,V],[[0], [1]]) 
 v_oovo = replicateSymArray(v_oovo_input,[O,O,V,O],[[0], [1], [2], [3]]) 
 v_oovv = replicateSymArray(v_oovv_input,[O,O,V,V],[[0], [1], [2], [3]]) 
 v_ovvo = replicateSymArray(v_ovvo_input,[O,V,V,O],[[0], [1], [2], [3]]) 
 v_vovo = replicateSymArray(v_vovo_input,[V,O,V,O],[[0], [1], [2], [3]]) 
 v_vovv = replicateSymArray(v_vovv_input,[V,O,V,V],[[0], [1], [2], [3]]) 
 t_vo = replicateSymArray(t_vo_input,[V,O],[[0], [1]]) 
 t_vvoo = replicateSymArray(t_vvoo_input,[V,V,O,O],[[0], [1], [2], [3]]) 
 r_vo = makeArray([V,O]) 
 _a7 = makeArray([V,O]) 
 _a6 = makeArray([V,O]) 
 _a5 = makeArray([V,O]) 
 _a4 = makeArray([V,O]) 
 _a3 = makeArray([V,O]) 
 _a2 = makeArray([V,O]) 
 _a1 = makeArray([V,O]) 
 _a65 = makeArray([O,V]) 
 _a9 = makeArray([V,O]) 
 _a8 = makeArray([V,O]) 
 _a84 = makeArray([O,O]) 
 _a87 = makeArray([V,O]) 
 _a77 = makeArray([O,O]) 
 _a82 = makeArray([V,O]) 
 _a24 = makeArray([V,V]) 
 _a27 = makeArray([V,O]) 
 _a35 = makeArray([O,O,V,O]) 
 _a71 = makeArray([O,V]) 
 _a76 = makeArray([V,O]) 
 _a91 = makeArray([O,V]) 
 _a99 = makeArray([O,O]) 
 _a111 = makeArray([V,O]) 
 _a29 = makeArray([O,O,V,O]) 
 _a34 = makeArray([V,O]) 
 _a117 = makeArray([O,V]) 
 _a123 = makeArray([O,O]) 
 _a139 = makeArray([V,O]) 
 _a18 = makeArray([V,V]) 
 _a21 = makeArray([V,O]) 
 _a40 = makeArray([V,O]) 
 _a42 = makeArray([O,O]) 
 _a45 = makeArray([V,O]) 
 _a70 = makeArray([V,O]) 
 _a53 = makeArray([O,V]) 
 _a11 = makeArray([O,O]) 
 _a10 = makeArray([V,O]) 
 _a48 = makeArray([O,O]) 
 _a51 = makeArray([V,O]) 
 _a16 = makeArray([V,O]) 
 _a59 = makeArray([O,V]) 
 _a64 = makeArray([V,O]) 
 _a58 = makeArray([V,O]) 
 _temp_27 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     _temp_27[p1][h1] += (v_vovo[p1][h2][p2][h1]) * (t_vo[p2][h2])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a7[p1][h1] = _temp_27[p1][h1]
 _temp_28 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     for p3 in range(0, V):
      _temp_28[p1][h1] += (v_vovv[p1][h2][p2][p3]) * (t_vvoo[p2][p3][h1][h2])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a6[p1][h1] = _temp_28[p1][h1]
 _temp_29 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     for p3 in range(0, V):
      _temp_29[p1][h1] += (v_vovv[p1][h2][p2][p3]) * (t_vvoo[p3][p2][h1][h2])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a5[p1][h1] = _temp_29[p1][h1]
 _temp_30 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     _temp_30[p1][h1] += (f_ov[h2][p2]) * (t_vvoo[p2][p1][h1][h2])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a4[p1][h1] = _temp_30[p1][h1]
 _temp_31 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     _temp_31[p1][h1] += (f_ov[h2][p2]) * (t_vvoo[p1][p2][h1][h2])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a3[p1][h1] = _temp_31[p1][h1]
 _temp_32 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    _temp_32[p1][h1] += (f_oo[h2][h1]) * (t_vo[p1][h2])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a2[p1][h1] = _temp_32[p1][h1]
 _temp_33 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for p2 in range(0, V):
    _temp_33[p1][h1] += (f_vv[p1][p2]) * (t_vo[p2][h1])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a1[p1][h1] = _temp_33[p1][h1]
 _temp_34 = makeArray([O,V]) 
 for h3 in range(0, O):
  for p2 in range(0, V):
   for h2 in range(0, O):
    for p3 in range(0, V):
     _temp_34[h3][p2] += (v_oovv[h2][h3][p2][p3]) * (t_vo[p3][h2])
 for h3 in range(0, O):
  for p2 in range(0, V):
   _a65[h3][p2] = _temp_34[h3][p2]
 _temp_35 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      _temp_35[p1][h1] += (v_oovo[h2][h3][p2][h1]) * (t_vvoo[p1][p2][h2][h3])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a9[p1][h1] = _temp_35[p1][h1]
 _temp_36 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     _temp_36[p1][h1] += (v_ovvo[h2][p1][p2][h1]) * (t_vo[p2][h2])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a8[p1][h1] = _temp_36[p1][h1]
 _temp_37 = makeArray([O,O]) 
 for h2 in range(0, O):
  for h1 in range(0, O):
   for h3 in range(0, O):
    for p2 in range(0, V):
     _temp_37[h2][h1] += (v_oovo[h2][h3][p2][h1]) * (t_vo[p2][h3])
 for h2 in range(0, O):
  for h1 in range(0, O):
   _a84[h2][h1] = _temp_37[h2][h1]
 _temp_38 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    _temp_38[p1][h1] += (t_vo[p1][h2]) * (_a84[h2][h1])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a87[p1][h1] = _temp_38[p1][h1]
 _temp_39 = makeArray([O,O]) 
 for h3 in range(0, O):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p2 in range(0, V):
     _temp_39[h3][h1] += (v_oovo[h2][h3][p2][h1]) * (t_vo[p2][h2])
 for h3 in range(0, O):
  for h1 in range(0, O):
   _a77[h3][h1] = _temp_39[h3][h1]
 _temp_40 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h3 in range(0, O):
    _temp_40[p1][h1] += (t_vo[p1][h3]) * (_a77[h3][h1])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a82[p1][h1] = _temp_40[p1][h1]
 _temp_41 = makeArray([V,V]) 
 for p1 in range(0, V):
  for p3 in range(0, V):
   for h2 in range(0, O):
    for p2 in range(0, V):
     _temp_41[p1][p3] += (v_vovv[p1][h2][p2][p3]) * (t_vo[p2][h2])
 for p1 in range(0, V):
  for p3 in range(0, V):
   _a24[p1][p3] = _temp_41[p1][p3]
 _temp_42 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for p3 in range(0, V):
    _temp_42[p1][h1] += (t_vo[p3][h1]) * (_a24[p1][p3])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a27[p1][h1] = _temp_42[p1][h1]
 _temp_43 = makeArray([O,O,V,O]) 
 for h2 in range(0, O):
  for h3 in range(0, O):
   for p3 in range(0, V):
    for h1 in range(0, O):
     for p2 in range(0, V):
      _temp_43[h2][h3][p3][h1] += (v_oovv[h2][h3][p2][p3]) * (t_vo[p2][h1])
 for h2 in range(0, O):
  for h3 in range(0, O):
   for p3 in range(0, V):
    for h1 in range(0, O):
     _a35[h2][h3][p3][h1] = _temp_43[h2][h3][p3][h1]
 _temp_44 = makeArray([O,V]) 
 for h3 in range(0, O):
  for p2 in range(0, V):
   for h2 in range(0, O):
    for p3 in range(0, V):
     _temp_44[h3][p2] += (v_oovv[h2][h3][p2][p3]) * (t_vo[p3][h2])
 for h3 in range(0, O):
  for p2 in range(0, V):
   _a71[h3][p2] = _temp_44[h3][p2]
 _temp_45 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for p2 in range(0, V):
    for h3 in range(0, O):
     _temp_45[p1][h1] += (t_vvoo[p2][p1][h1][h3]) * (_a71[h3][p2])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a76[p1][h1] = _temp_45[p1][h1]
 _temp_46 = makeArray([O,V]) 
 for h2 in range(0, O):
  for p2 in range(0, V):
   for h3 in range(0, O):
    for p3 in range(0, V):
     _temp_46[h2][p2] += (v_oovv[h2][h3][p2][p3]) * (t_vo[p3][h3])
 for h2 in range(0, O):
  for p2 in range(0, V):
   _a91[h2][p2] = _temp_46[h2][p2]
 _temp_47 = makeArray([O,O]) 
 for h2 in range(0, O):
  for h1 in range(0, O):
   for p2 in range(0, V):
    _temp_47[h2][h1] += (t_vo[p2][h1]) * (_a91[h2][p2])
 for h2 in range(0, O):
  for h1 in range(0, O):
   _a99[h2][h1] = _temp_47[h2][h1]
 _temp_48 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    _temp_48[p1][h1] += (t_vo[p1][h2]) * (_a99[h2][h1])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a111[p1][h1] = _temp_48[p1][h1]
 _temp_49 = makeArray([O,O,V,O]) 
 for h2 in range(0, O):
  for h3 in range(0, O):
   for p3 in range(0, V):
    for h1 in range(0, O):
     for p2 in range(0, V):
      _temp_49[h2][h3][p3][h1] += (v_oovv[h2][h3][p2][p3]) * (t_vo[p2][h1])
 for h2 in range(0, O):
  for h3 in range(0, O):
   for p3 in range(0, V):
    for h1 in range(0, O):
     _a29[h2][h3][p3][h1] = _temp_49[h2][h3][p3][h1]
 _temp_50 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for p3 in range(0, V):
    for h3 in range(0, O):
     for h2 in range(0, O):
      _temp_50[p1][h1] += (t_vvoo[p1][p3][h3][h2]) * (_a29[h2][h3][p3][h1])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a34[p1][h1] = _temp_50[p1][h1]
 _temp_51 = makeArray([O,V]) 
 for h3 in range(0, O):
  for p2 in range(0, V):
   for h2 in range(0, O):
    for p3 in range(0, V):
     _temp_51[h3][p2] += (v_oovv[h2][h3][p2][p3]) * (t_vo[p3][h2])
 for h3 in range(0, O):
  for p2 in range(0, V):
   _a117[h3][p2] = _temp_51[h3][p2]
 _temp_52 = makeArray([O,O]) 
 for h3 in range(0, O):
  for h1 in range(0, O):
   for p2 in range(0, V):
    _temp_52[h3][h1] += (t_vo[p2][h1]) * (_a117[h3][p2])
 for h3 in range(0, O):
  for h1 in range(0, O):
   _a123[h3][h1] = _temp_52[h3][h1]
 _temp_53 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h3 in range(0, O):
    _temp_53[p1][h1] += (t_vo[p1][h3]) * (_a123[h3][h1])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a139[p1][h1] = _temp_53[p1][h1]
 _temp_54 = makeArray([V,V]) 
 for p1 in range(0, V):
  for p2 in range(0, V):
   for h2 in range(0, O):
    for p3 in range(0, V):
     _temp_54[p1][p2] += (v_vovv[p1][h2][p2][p3]) * (t_vo[p3][h2])
 for p1 in range(0, V):
  for p2 in range(0, V):
   _a18[p1][p2] = _temp_54[p1][p2]
 _temp_55 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for p2 in range(0, V):
    _temp_55[p1][h1] += (t_vo[p2][h1]) * (_a18[p1][p2])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a21[p1][h1] = _temp_55[p1][h1]
 _temp_56 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for p3 in range(0, V):
    for h2 in range(0, O):
     for h3 in range(0, O):
      _temp_56[p1][h1] += (t_vvoo[p1][p3][h2][h3]) * (_a35[h2][h3][p3][h1])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a40[p1][h1] = _temp_56[p1][h1]
 _temp_57 = makeArray([O,O]) 
 for h2 in range(0, O):
  for h1 in range(0, O):
   for h3 in range(0, O):
    for p2 in range(0, V):
     for p3 in range(0, V):
      _temp_57[h2][h1] += (v_oovv[h2][h3][p2][p3]) * (t_vvoo[p3][p2][h1][h3])
 for h2 in range(0, O):
  for h1 in range(0, O):
   _a42[h2][h1] = _temp_57[h2][h1]
 _temp_58 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    _temp_58[p1][h1] += (t_vo[p1][h2]) * (_a42[h2][h1])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a45[p1][h1] = _temp_58[p1][h1]
 _temp_59 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for p2 in range(0, V):
    for h3 in range(0, O):
     _temp_59[p1][h1] += (t_vvoo[p1][p2][h1][h3]) * (_a65[h3][p2])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a70[p1][h1] = _temp_59[p1][h1]
 _temp_60 = makeArray([O,V]) 
 for h3 in range(0, O):
  for p3 in range(0, V):
   for h2 in range(0, O):
    for p2 in range(0, V):
     _temp_60[h3][p3] += (v_oovv[h2][h3][p2][p3]) * (t_vo[p2][h2])
 for h3 in range(0, O):
  for p3 in range(0, V):
   _a53[h3][p3] = _temp_60[h3][p3]
 _temp_61 = makeArray([O,O]) 
 for h2 in range(0, O):
  for h1 in range(0, O):
   for p2 in range(0, V):
    _temp_61[h2][h1] += (f_ov[h2][p2]) * (t_vo[p2][h1])
 for h2 in range(0, O):
  for h1 in range(0, O):
   _a11[h2][h1] = _temp_61[h2][h1]
 _temp_62 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for h3 in range(0, O):
     for p2 in range(0, V):
      _temp_62[p1][h1] += (v_oovo[h2][h3][p2][h1]) * (t_vvoo[p1][p2][h3][h2])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a10[p1][h1] = _temp_62[p1][h1]
 _temp_63 = makeArray([O,O]) 
 for h2 in range(0, O):
  for h1 in range(0, O):
   for h3 in range(0, O):
    for p2 in range(0, V):
     for p3 in range(0, V):
      _temp_63[h2][h1] += (v_oovv[h2][h3][p2][p3]) * (t_vvoo[p2][p3][h1][h3])
 for h2 in range(0, O):
  for h1 in range(0, O):
   _a48[h2][h1] = _temp_63[h2][h1]
 _temp_64 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    _temp_64[p1][h1] += (t_vo[p1][h2]) * (_a48[h2][h1])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a51[p1][h1] = _temp_64[p1][h1]
 _temp_65 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    _temp_65[p1][h1] += (t_vo[p1][h2]) * (_a11[h2][h1])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a16[p1][h1] = _temp_65[p1][h1]
 _temp_66 = makeArray([O,V]) 
 for h3 in range(0, O):
  for p3 in range(0, V):
   for h2 in range(0, O):
    for p2 in range(0, V):
     _temp_66[h3][p3] += (v_oovv[h2][h3][p2][p3]) * (t_vo[p2][h2])
 for h3 in range(0, O):
  for p3 in range(0, V):
   _a59[h3][p3] = _temp_66[h3][p3]
 _temp_67 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for p3 in range(0, V):
    for h3 in range(0, O):
     _temp_67[p1][h1] += (t_vvoo[p3][p1][h1][h3]) * (_a59[h3][p3])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a64[p1][h1] = _temp_67[p1][h1]
 _temp_68 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   for p3 in range(0, V):
    for h3 in range(0, O):
     _temp_68[p1][h1] += (t_vvoo[p1][p3][h1][h3]) * (_a53[h3][p3])
 for p1 in range(0, V):
  for h1 in range(0, O):
   _a58[p1][h1] = _temp_68[p1][h1]
 _temp_69 = makeArray([V,O]) 
 for p1 in range(0, V):
  for h1 in range(0, O):
   _temp_69[p1][h1] += (f_vo[p1][h1]) + (_a1[p1][h1]) + (-_a2[p1][h1]) + ((2.0 * _a3[p1][h1])) + (-_a4[p1][h1]) + (-_a5[p1][h1]) + ((2.0 * _a6[p1][h1])) + (-_a7[p1][h1]) + ((2.0 * _a8[p1][h1])) + (_a9[p1][h1]) + ((-2.0 * _a10[p1][h1])) + (-_a16[p1][h1]) + ((2.0 * _a21[p1][h1])) + (-_a27[p1][h1]) + (_a34[p1][h1]) + ((-2.0 * _a40[p1][h1])) + (_a45[p1][h1]) + ((-2.0 * _a51[p1][h1])) + ((4.0 * _a58[p1][h1])) + ((-2.0 * _a64[p1][h1])) + ((-2.0 * _a70[p1][h1])) + (_a76[p1][h1]) + ((-2.0 * _a82[p1][h1])) + (_a87[p1][h1]) + ((-2.0 * _a111[p1][h1])) + (_a139[p1][h1])
 for p1 in range(0, V):
  for h1 in range(0, O):
   r_vo[p1][h1] = _temp_69[p1][h1]
 
 r_vo_optimum = replicateArray(r_vo) 

 #---------- Results Comparison ---------- 

 print r_vo_original
 print r_vo_optimum
 if (r_vo_original != r_vo_optimum): 
  return False 
 return True 

print run()