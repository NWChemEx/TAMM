#! /usr/contrib/bin/python
import random
from helper import makeSymArray, makeArray, replicateArray, replicateSymArray
def run(): 

 #---------- Declarations ---------- 

 O = 2
 V = 2
 f_vv_input = makeArray([V,V], True) 
 t_vvoo_input = makeArray([V,V,O,O], True) 
 f_ov_input = makeArray([O,V], True) 
 v_ovov_input = makeArray([O,V,O,V], True) 
 f_oo_input = makeArray([O,O], True) 
 v_ovvv_input = makeArray([O,V,V,V], True) 
 t_vo_input = makeArray([V,O], True) 
 f_vo_input = makeArray([V,O], True) 
 v_oovv_input = makeArray([O,O,V,V], True) 
 v_ooov_input = makeArray([O,O,O,V], True) 

 #---------- Original Equation ---------- 

 i0 = makeArray([V,O]) 
 t_vo = replicateSymArray(t_vo_input,[V,O],[[0], [1]]) 
 t_vvoo = replicateSymArray(t_vvoo_input,[V,V,O,O],[[0], [1], [2], [3]]) 
 t1_2_1 = makeArray([O,O]) 
 t1_2_2_1 = makeArray([O,V]) 
 t1_3_1 = makeArray([V,V]) 
 t1_5_1 = makeArray([O,V]) 
 t1_6_1 = makeArray([O,O,O,V]) 
 f_vo = replicateSymArray(f_vo_input,[V,O],[[0], [1]]) 
 f_oo = replicateSymArray(f_oo_input,[O,O],[[0], [1]]) 
 f_ov = replicateSymArray(f_ov_input,[O,V],[[0], [1]]) 
 v_oovv = replicateSymArray(v_oovv_input,[O,O,V,V],[[0], [1], [2], [3]]) 
 v_ooov = replicateSymArray(v_ooov_input,[O,O,O,V],[[0], [1], [2], [3]]) 
 f_vv = replicateSymArray(f_vv_input,[V,V],[[0], [1]]) 
 v_ovvv = replicateSymArray(v_ovvv_input,[O,V,V,V],[[0], [1], [2], [3]]) 
 v_ovov = replicateSymArray(v_ovov_input,[O,V,O,V],[[0], [1], [2], [3]]) 
 _temp_2 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    _temp_2[p2][h1] += -((t_vo[p2][h2]) * (f_oo[h2][h1]))
 _temp_3 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p1 in range(0, V):
     _temp_3[p2][h1] += -((t_vo[p2][h2]) * (t_vo[p1][h1]) * (f_ov[h2][p1]))
 _temp_4 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p1 in range(0, V):
     for p3 in range(0, V):
      for h3 in range(0, O):
       _temp_4[p2][h1] += (t_vo[p2][h2]) * (t_vo[p1][h1]) * (t_vo[p3][h3]) * (v_oovv[h3][h2][p1][p3])
 _temp_5 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p1 in range(0, V):
     for h3 in range(0, O):
      _temp_5[p2][h1] += (t_vo[p2][h2]) * (t_vo[p1][h3]) * (v_ooov[h3][h2][h1][p1])
 _temp_6 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    for p1 in range(0, V):
     for p3 in range(0, V):
      for h3 in range(0, O):
       _temp_6[p2][h1] += (0.5 * (t_vo[p2][h2]) * (t_vvoo[p1][p3][h1][h3]) * (v_oovv[h3][h2][p1][p3]))
 _temp_7 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p1 in range(0, V):
    _temp_7[p2][h1] += (t_vo[p1][h1]) * (f_vv[p2][p1])
 _temp_8 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p1 in range(0, V):
    for p3 in range(0, V):
     for h2 in range(0, O):
      _temp_8[p2][h1] += -((t_vo[p1][h1]) * (t_vo[p3][h2]) * (v_ovvv[h2][p2][p1][p3]))
 _temp_9 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p1 in range(0, V):
    for h2 in range(0, O):
     _temp_9[p2][h1] += -((t_vo[p1][h2]) * (v_ovov[h2][p2][h1][p1]))
 _temp_10 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p1 in range(0, V):
    for h2 in range(0, O):
     _temp_10[p2][h1] += (t_vvoo[p2][p1][h1][h2]) * (f_ov[h2][p1])
 _temp_11 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p1 in range(0, V):
    for h2 in range(0, O):
     for p3 in range(0, V):
      for h3 in range(0, O):
       _temp_11[p2][h1] += (t_vvoo[p2][p1][h1][h2]) * (t_vo[p3][h3]) * (v_oovv[h3][h2][p3][p1])
 _temp_12 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p1 in range(0, V):
    for h2 in range(0, O):
     for h3 in range(0, O):
      _temp_12[p2][h1] += (-0.5 * (t_vvoo[p2][p1][h2][h3]) * (v_ooov[h2][h3][h1][p1]))
 _temp_13 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p1 in range(0, V):
    for h2 in range(0, O):
     for h3 in range(0, O):
      for p3 in range(0, V):
       _temp_13[p2][h1] += (0.5 * (t_vvoo[p2][p1][h2][h3]) * (t_vo[p3][h1]) * (v_oovv[h2][h3][p1][p3]))
 _temp_14 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p1 in range(0, V):
    for p3 in range(0, V):
     for h2 in range(0, O):
      _temp_14[p2][h1] += (-0.5 * (t_vvoo[p1][p3][h1][h2]) * (v_ovvv[h2][p2][p1][p3]))
 _temp_1 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   _temp_1[p2][h1] += (f_vo[p2][h1]) + (_temp_2[p2][h1]) + (_temp_3[p2][h1]) + (_temp_4[p2][h1]) + (_temp_5[p2][h1]) + (_temp_6[p2][h1]) + (_temp_7[p2][h1]) + (_temp_8[p2][h1]) + (_temp_9[p2][h1]) + (_temp_10[p2][h1]) + (_temp_11[p2][h1]) + (_temp_12[p2][h1]) + (_temp_13[p2][h1]) + (_temp_14[p2][h1])
 for p2 in range(0, V):
  for h1 in range(0, O):
   i0[p2][h1] = _temp_1[p2][h1]
 
 i0_original = replicateArray(i0) 

 #---------- Optimized Equation ---------- 

 i0 = makeArray([V,O]) 
 t_vo = replicateSymArray(t_vo_input,[V,O],[[0], [1]]) 
 t_vvoo = replicateSymArray(t_vvoo_input,[V,V,O,O],[[0], [1], [2], [3]]) 
 t1_2_1 = makeArray([O,O]) 
 t1_2_2_1 = makeArray([O,V]) 
 t1_3_1 = makeArray([V,V]) 
 t1_5_1 = makeArray([O,V]) 
 t1_6_1 = makeArray([O,O,O,V]) 
 f_vo = replicateSymArray(f_vo_input,[V,O],[[0], [1]]) 
 f_oo = replicateSymArray(f_oo_input,[O,O],[[0], [1]]) 
 f_ov = replicateSymArray(f_ov_input,[O,V],[[0], [1]]) 
 v_oovv = replicateSymArray(v_oovv_input,[O,O,V,V],[[0], [1], [2], [3]]) 
 v_ooov = replicateSymArray(v_ooov_input,[O,O,O,V],[[0], [1], [2], [3]]) 
 f_vv = replicateSymArray(f_vv_input,[V,V],[[0], [1]]) 
 v_ovvv = replicateSymArray(v_ovvv_input,[O,V,V,V],[[0], [1], [2], [3]]) 
 v_ovov = replicateSymArray(v_ovov_input,[O,V,O,V],[[0], [1], [2], [3]]) 
 _a4 = makeArray([O,O]) 
 _a5 = makeArray([V,O]) 
 _a63 = makeArray([V,O]) 
 _a1 = makeArray([V,O]) 
 _a56 = makeArray([V,O]) 
 _a38 = makeArray([O,O]) 
 _a39 = makeArray([V,O]) 
 _a44 = makeArray([O,O]) 
 _a45 = makeArray([V,O]) 
 _a55 = makeArray([V,O]) 
 _a51 = makeArray([V,V]) 
 _a52 = makeArray([V,O]) 
 _a65 = makeArray([V,V]) 
 _a13 = makeArray([O,V]) 
 _a23 = makeArray([O,O]) 
 _a26 = makeArray([V,O]) 
 _a68 = makeArray([V,O]) 
 _a48 = makeArray([V,O]) 
 _a70 = makeArray([V,O]) 
 _a59 = makeArray([O,V]) 
 _a60 = makeArray([V,O]) 
 _temp_15 = makeArray([O,O]) 
 for h2 in range(0, O):
  for h1 in range(0, O):
   for p1 in range(0, V):
    _temp_15[h2][h1] += (t_vo[p1][h1]) * (f_ov[h2][p1])
 for h2 in range(0, O):
  for h1 in range(0, O):
   _a4[h2][h1] = _temp_15[h2][h1]
 _temp_16 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    _temp_16[p2][h1] += (t_vo[p2][h2]) * (_a4[h2][h1])
 for p2 in range(0, V):
  for h1 in range(0, O):
   _a5[p2][h1] = _temp_16[p2][h1]
 _temp_17 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p1 in range(0, V):
    for h2 in range(0, O):
     for h3 in range(0, O):
      _temp_17[p2][h1] += (t_vvoo[p2][p1][h2][h3]) * (v_ooov[h2][h3][h1][p1])
 for p2 in range(0, V):
  for h1 in range(0, O):
   _a63[p2][h1] = _temp_17[p2][h1]
 _temp_18 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    _temp_18[p2][h1] += (t_vo[p2][h2]) * (f_oo[h2][h1])
 for p2 in range(0, V):
  for h1 in range(0, O):
   _a1[p2][h1] = _temp_18[p2][h1]
 _temp_19 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p1 in range(0, V):
    for h2 in range(0, O):
     _temp_19[p2][h1] += (t_vvoo[p2][p1][h1][h2]) * (f_ov[h2][p1])
 for p2 in range(0, V):
  for h1 in range(0, O):
   _a56[p2][h1] = _temp_19[p2][h1]
 _temp_20 = makeArray([O,O]) 
 for h2 in range(0, O):
  for h1 in range(0, O):
   for p1 in range(0, V):
    for h3 in range(0, O):
     _temp_20[h2][h1] += (t_vo[p1][h3]) * (v_ooov[h3][h2][h1][p1])
 for h2 in range(0, O):
  for h1 in range(0, O):
   _a38[h2][h1] = _temp_20[h2][h1]
 _temp_21 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    _temp_21[p2][h1] += (t_vo[p2][h2]) * (_a38[h2][h1])
 for p2 in range(0, V):
  for h1 in range(0, O):
   _a39[p2][h1] = _temp_21[p2][h1]
 _temp_22 = makeArray([O,O]) 
 for h2 in range(0, O):
  for h1 in range(0, O):
   for p1 in range(0, V):
    for p3 in range(0, V):
     for h3 in range(0, O):
      _temp_22[h2][h1] += (t_vvoo[p1][p3][h1][h3]) * (v_oovv[h3][h2][p1][p3])
 for h2 in range(0, O):
  for h1 in range(0, O):
   _a44[h2][h1] = _temp_22[h2][h1]
 _temp_23 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    _temp_23[p2][h1] += (t_vo[p2][h2]) * (_a44[h2][h1])
 for p2 in range(0, V):
  for h1 in range(0, O):
   _a45[p2][h1] = _temp_23[p2][h1]
 _temp_24 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p1 in range(0, V):
    for h2 in range(0, O):
     _temp_24[p2][h1] += (t_vo[p1][h2]) * (v_ovov[h2][p2][h1][p1])
 for p2 in range(0, V):
  for h1 in range(0, O):
   _a55[p2][h1] = _temp_24[p2][h1]
 _temp_25 = makeArray([V,V]) 
 for p2 in range(0, V):
  for p1 in range(0, V):
   for p3 in range(0, V):
    for h2 in range(0, O):
     _temp_25[p2][p1] += (t_vo[p3][h2]) * (v_ovvv[h2][p2][p1][p3])
 for p2 in range(0, V):
  for p1 in range(0, V):
   _a51[p2][p1] = _temp_25[p2][p1]
 _temp_26 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p1 in range(0, V):
    _temp_26[p2][h1] += (t_vo[p1][h1]) * (_a51[p2][p1])
 for p2 in range(0, V):
  for h1 in range(0, O):
   _a52[p2][h1] = _temp_26[p2][h1]
 _temp_27 = makeArray([V,V]) 
 for p2 in range(0, V):
  for p3 in range(0, V):
   for p1 in range(0, V):
    for h2 in range(0, O):
     for h3 in range(0, O):
      _temp_27[p2][p3] += (t_vvoo[p2][p1][h2][h3]) * (v_oovv[h2][h3][p1][p3])
 for p2 in range(0, V):
  for p3 in range(0, V):
   _a65[p2][p3] = _temp_27[p2][p3]
 _temp_28 = makeArray([O,V]) 
 for h2 in range(0, O):
  for p1 in range(0, V):
   for p3 in range(0, V):
    for h3 in range(0, O):
     _temp_28[h2][p1] += (t_vo[p3][h3]) * (v_oovv[h3][h2][p1][p3])
 for h2 in range(0, O):
  for p1 in range(0, V):
   _a13[h2][p1] = _temp_28[h2][p1]
 _temp_29 = makeArray([O,O]) 
 for h2 in range(0, O):
  for h1 in range(0, O):
   for p1 in range(0, V):
    _temp_29[h2][h1] += (t_vo[p1][h1]) * (_a13[h2][p1])
 for h2 in range(0, O):
  for h1 in range(0, O):
   _a23[h2][h1] = _temp_29[h2][h1]
 _temp_30 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for h2 in range(0, O):
    _temp_30[p2][h1] += (t_vo[p2][h2]) * (_a23[h2][h1])
 for p2 in range(0, V):
  for h1 in range(0, O):
   _a26[p2][h1] = _temp_30[p2][h1]
 _temp_31 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p3 in range(0, V):
    _temp_31[p2][h1] += (t_vo[p3][h1]) * (_a65[p2][p3])
 for p2 in range(0, V):
  for h1 in range(0, O):
   _a68[p2][h1] = _temp_31[p2][h1]
 _temp_32 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p1 in range(0, V):
    _temp_32[p2][h1] += (t_vo[p1][h1]) * (f_vv[p2][p1])
 for p2 in range(0, V):
  for h1 in range(0, O):
   _a48[p2][h1] = _temp_32[p2][h1]
 _temp_33 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p1 in range(0, V):
    for p3 in range(0, V):
     for h2 in range(0, O):
      _temp_33[p2][h1] += (t_vvoo[p1][p3][h1][h2]) * (v_ovvv[h2][p2][p1][p3])
 for p2 in range(0, V):
  for h1 in range(0, O):
   _a70[p2][h1] = _temp_33[p2][h1]
 _temp_34 = makeArray([O,V]) 
 for h2 in range(0, O):
  for p1 in range(0, V):
   for p3 in range(0, V):
    for h3 in range(0, O):
     _temp_34[h2][p1] += (t_vo[p3][h3]) * (v_oovv[h3][h2][p3][p1])
 for h2 in range(0, O):
  for p1 in range(0, V):
   _a59[h2][p1] = _temp_34[h2][p1]
 _temp_35 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   for p1 in range(0, V):
    for h2 in range(0, O):
     _temp_35[p2][h1] += (t_vvoo[p2][p1][h1][h2]) * (_a59[h2][p1])
 for p2 in range(0, V):
  for h1 in range(0, O):
   _a60[p2][h1] = _temp_35[p2][h1]
 _temp_36 = makeArray([V,O]) 
 for p2 in range(0, V):
  for h1 in range(0, O):
   _temp_36[p2][h1] += (f_vo[p2][h1]) + (-_a1[p2][h1]) + (-_a5[p2][h1]) + (_a26[p2][h1]) + (_a39[p2][h1]) + ((0.5 * _a45[p2][h1])) + (_a48[p2][h1]) + (-_a52[p2][h1]) + (-_a55[p2][h1]) + (_a56[p2][h1]) + (_a60[p2][h1]) + ((-0.5 * _a63[p2][h1])) + ((0.5 * _a68[p2][h1])) + ((-0.5 * _a70[p2][h1]))
 for p2 in range(0, V):
  for h1 in range(0, O):
   i0[p2][h1] = _temp_36[p2][h1]
 
 i0_optimum = replicateArray(i0) 

 #---------- Results Comparison ---------- 

 print i0_original
 print i0_optimum
 if (i0_original != i0_optimum): 
  return False 
 return True 

print run()