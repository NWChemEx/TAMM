/*
 *  t2 {
 *  
 *  index h1,h2,h3,h4,h5,h6,h7,h8,h9,h10 = O;
 *  index p1,p2,p3,p4,p5,p6 = V;
 *  
 *  array i0[V,V][O,O];
 *  array v[N,N][N,N]: irrep_v;
 *  array t_vo[V][O]: irrep_t;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array f[N][N]: irrep_f;
 *  array t2_3_1[V,V][O,V];
 *  array t2_2_2_3[O,V][O,V];
 *  array t2_2_2_1[O,O][O,O];
 *  array t2_2_2_2_1[O,O][O,V];
 *  array t2_2_1[O,V][O,O];
 *  
 *  t2_1:       i0[p3,p4,h1,h2] += 1 * v[p3,p4,h1,h2];
 *  t2_2_1:     t2_2_1[h10,p3,h1,h2] += 1 * v[h10,p3,h1,h2];
 *  t2_2_2_1:   t2_2_2_1[h8,h10,h1,h2] += 1 * v[h8,h10,h1,h2];
 *  t2_2_2_2_1: t2_2_2_2_1[h8,h10,h1,p5] += 1 * v[h8,h10,h1,p5];
 *  t2_2_2_2_2: t2_2_2_2_1[h8,h10,h1,p5] += -1/2 * t_vo[p6,h1] * v[h8,h10,p5,p6];
 *  t2_2_2_2:   t2_2_2_1[h8,h10,h1,h2] += -1 * t_vo[p5,h1] * t2_2_2_2_1[h8,h10,h2,p5];
 *  t2_2_2:     t2_2_1[h10,p3,h1,h2] += 1/2 * t_vo[p3,h8] * t2_2_2_1[h8,h10,h1,h2];
 *  t2_2_2_3:   t2_2_2_3[h10,p3,h1,p5] += 1 * v[h10,p3,h1,p5];
 *  t2_2_2_4:   t2_2_2_3[h10,p3,h1,p5] += -1/2 * t_vo[p6,h1] * v[h10,p3,p5,p6];
 *  t2_2_3:     t2_2_1[h10,p3,h1,h2] += -1 * t_vo[p5,h1] * t2_2_2_3[h10,p3,h2,p5];
 *  t2_2:       i0[p3,p4,h1,h2] += -1 * t_vo[p3,h10] * t2_2_1[h10,p4,h1,h2];
 *  t2_3_1:     t2_3_1[p3,p4,h1,p5] += 1 * v[p3,p4,h1,p5];
 *  t2_3_2:     t2_3_1[p3,p4,h1,p5] += -1/2 * t_vo[p6,h1] * v[p3,p4,p5,p6];
 *  t2_3:       i0[p3,p4,h1,h2] += -1 * t_vo[p5,h1] * t2_3_1[p3,p4,h2,p5];
 *  t2_4:       i0[p3,p4,h1,h2] += -1 * t_vvoo[p3,p4,h1,h5] * f[h5,h2];
 *  t2_5:       i0[p3,p4,h1,h2] += 1 * t_vvoo[p3,p5,h1,h2] * f[p4,p5];
 *  
 *  }
*/


extern "C" {
  void cc2_t2_1_(Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);
  void cc2_t2_2_1_(Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_1, Integer *k_t2_2_1_offset);
  void cc2_t2_2_2_1_(Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_2_1, Integer *k_t2_2_2_1_offset);
  void cc2_t2_2_2_2_1_(Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_2_2_1, Integer *k_t2_2_2_2_1_offset);
  void cc2_t2_2_2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_2_2_1, Integer *k_t2_2_2_2_1_offset);
  void cc2_t2_2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t2_2_2_2_1, Integer *k_t2_2_2_2_1_offset,Integer *d_t2_2_2_1, Integer *k_t2_2_2_1_offset);
  void cc2_t2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t2_2_2_1, Integer *k_t2_2_2_1_offset,Integer *d_t2_2_1, Integer *k_t2_2_1_offset);
  void cc2_t2_2_2_3_(Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_2_3, Integer *k_t2_2_2_3_offset);
  void cc2_t2_2_2_4_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_2_3, Integer *k_t2_2_2_3_offset);
  void cc2_t2_2_3_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t2_2_2_3, Integer *k_t2_2_2_3_offset,Integer *d_t2_2_1, Integer *k_t2_2_1_offset);
  void cc2_t2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t2_2_1, Integer *k_t2_2_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void cc2_t2_3_1_(Integer *d_v, Integer *k_v_offset,Integer *d_t2_3_1, Integer *k_t2_3_1_offset);
  void cc2_t2_3_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_3_1, Integer *k_t2_3_1_offset);
  void cc2_t2_3_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t2_3_1, Integer *k_t2_3_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void cc2_t2_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_f, Integer *k_f_offset,Integer *d_i0, Integer *k_i0_offset);
  void cc2_t2_5_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_f, Integer *k_f_offset,Integer *d_i0, Integer *k_i0_offset);

  void offset_cc2_t2_2_1_(Integer *l_t2_2_1_offset, Integer *k_t2_2_1_offset, Integer *size_t2_2_1);
  void offset_cc2_t2_2_2_1_(Integer *l_t2_2_2_1_offset, Integer *k_t2_2_2_1_offset, Integer *size_t2_2_2_1);
  void offset_cc2_t2_2_2_2_1_(Integer *l_t2_2_2_2_1_offset, Integer *k_t2_2_2_2_1_offset, Integer *size_t2_2_2_2_1);
  void offset_cc2_t2_2_2_3_(Integer *l_t2_2_2_3_offset, Integer *k_t2_2_2_3_offset, Integer *size_t2_2_2_3);
  void offset_cc2_t2_3_1_(Integer *l_t2_3_1_offset, Integer *k_t2_3_1_offset, Integer *size_t2_3_1);
}

namespace ctce {
  extern "C" {
    void cc2_t2_cxx(Integer *d_i0,Integer *d_t_vvoo,Integer *d_f,Integer *d_t_vo,Integer *d_v,Integer *k_i0_offset,Integer *k_t_vvoo_offset,Integer *k_f_offset,Integer *k_t_vo_offset,Integer *k_v_offset){
      static bool set_t2 = true;
      
      Assignment op_t2_1;
      Assignment op_t2_2_1;
      Assignment op_t2_2_2_1;
      Assignment op_t2_2_2_2_1;
      Assignment op_t2_2_2_3;
      Assignment op_t2_3_1;
      Multiplication op_t2_2_2_2_2;
      Multiplication op_t2_2_2_2;
      Multiplication op_t2_2_2;
      Multiplication op_t2_2_2_4;
      Multiplication op_t2_2_3;
      Multiplication op_t2_2;
      Multiplication op_t2_3_2;
      Multiplication op_t2_3;
      Multiplication op_t2_4;
      Multiplication op_t2_5;
      
      DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
      static Equations eqs;

      if (set_t2) {
        cc2_t2_equations(eqs);
        set_t2 = false;
      }

      std::vector <Tensor> tensors;
      std::vector <Operation> ops;
      tensors_and_ops(eqs, tensors, ops);

      Tensor *i0 = &tensors[0];
      Tensor *v = &tensors[1];
      Tensor *t2_2_1 = &tensors[2];
      Tensor *t2_2_2_1 = &tensors[3];
      Tensor *t2_2_2_2_1 = &tensors[4];
      Tensor *t_vo = &tensors[5];
      Tensor *t2_2_2_3 = &tensors[6];
      Tensor *t2_3_1 = &tensors[7];
      Tensor *t_vvoo = &tensors[8];
      Tensor *f = &tensors[9];

      op_t2_1 = ops[0].add;
      op_t2_2_1 = ops[1].add;
      op_t2_2_2_1 = ops[2].add;
      op_t2_2_2_2_1 = ops[3].add;
      op_t2_2_2_2_2 = ops[4].mult;
      op_t2_2_2_2 = ops[5].mult;
      op_t2_2_2 = ops[6].mult;
      op_t2_2_2_3 = ops[7].add;
      op_t2_2_2_4 = ops[8].mult;
      op_t2_2_3 = ops[9].mult;
      op_t2_2 = ops[10].mult;
      op_t2_3_1 = ops[11].add;
      op_t2_3_2 = ops[12].mult;
      op_t2_3 = ops[13].mult;
      op_t2_4 = ops[14].mult;
      op_t2_5 = ops[15].mult;
      
/* ----- Insert attach code ------ */

      CorFortran(1, op_t2_1, cc2_t2_1_);
      CorFortran(1, op_t2_2_1, ofsset_cc2_t2_2_1_);
      CorFortran(1, op_t2_2_1, cc2_t2_2_1_);
      CorFortran(1, op_t2_2_2_1, ofsset_cc2_t2_2_2_1_);
      CorFortran(1, op_t2_2_2_1, cc2_t2_2_2_1_);
      CorFortran(1, op_t2_2_2_2_1, ofsset_cc2_t2_2_2_2_1_);
      CorFortran(1, op_t2_2_2_2_1, cc2_t2_2_2_2_1_);
      CorFortran(1, op_t2_2_2_2_2, cc2_t2_2_2_2_2_);
      CorFortran(1, op_t2_2_2_2, cc2_t2_2_2_2_);
      destroy(t2_2_2_2_1);
      CorFortran(1, op_t2_2_2, cc2_t2_2_2_);
      destroy(t2_2_2_1);
      CorFortran(1, op_t2_2_2_3, ofsset_cc2_t2_2_2_3_);
      CorFortran(1, op_t2_2_2_3, cc2_t2_2_2_3_);
      CorFortran(1, op_t2_2_2_4, cc2_t2_2_2_4_);
      CorFortran(1, op_t2_2_3, cc2_t2_2_3_);
      destroy(t2_2_2_3);
      CorFortran(1, op_t2_2, cc2_t2_2_);
      destroy(t2_2_1);
      CorFortran(1, op_t2_3_1, ofsset_cc2_t2_3_1_);
      CorFortran(1, op_t2_3_1, cc2_t2_3_1_);
      CorFortran(1, op_t2_3_2, cc2_t2_3_2_);
      CorFortran(1, op_t2_3, cc2_t2_3_);
      destroy(t2_3_1);
      CorFortran(1, op_t2_4, cc2_t2_4_);
      CorFortran(1, op_t2_5, cc2_t2_5_);
      
/* ----- Insert detach code ------ */

    }
  } // extern C
}; // namespace ctce
