/*
 *  x1 {
 *  
 *  index h1,h2,h3,h4,h5,h6,h7,h8 = O;
 *  index p1,p2,p3,p4,p5,p6,p7 = V;
 *  
 *  array i0[][O];
 *  array x_o[][O]: irrep_x;
 *  array f[N][N]: irrep_f;
 *  array t_vo[V][O]: irrep_t;
 *  array v[N,N][N,N]: irrep_v;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array x_voo[V][O,O]: irrep_x;
 *  array x1_2_1[O][V];
 *  array x1_3_1[O,O][O,V];
 *  array x1_1_1_1[O][V];
 *  array x1_1_1[O][O];
 *  
 *  x1_1_1:     x1_1_1[h6,h1] += 1 * f[h6,h1];
 *  x1_1_1_1:   x1_1_1_1[h6,p7] += 1 * f[h6,p7];
 *  x1_1_1_2:   x1_1_1_1[h6,p7] += 1 * t_vo[p4,h5] * v[h5,h6,p4,p7];
 *  x1_1_2:     x1_1_1[h6,h1] += 1 * t_vo[p7,h1] * x1_1_1_1[h6,p7];
 *  x1_1_3:     x1_1_1[h6,h1] += -1 * t_vo[p3,h4] * v[h4,h6,h1,p3];
 *  x1_1_4:     x1_1_1[h6,h1] += -1/2 * t_vvoo[p3,p4,h1,h5] * v[h5,h6,p3,p4];
 *  x1_1:       i0[h1] += -1 * x_o[h6] * x1_1_1[h6,h1];
 *  x1_2_1:     x1_2_1[h6,p7] += 1 * f[h6,p7];
 *  x1_2_2:     x1_2_1[h6,p7] += 1 * t_vo[p3,h4] * v[h4,h6,p3,p7];
 *  x1_2:       i0[h1] += 1 * x_voo[p7,h1,h6] * x1_2_1[h6,p7];
 *  x1_3_1:     x1_3_1[h6,h8,h1,p7] += 1 * v[h6,h8,h1,p7];
 *  x1_3_2:     x1_3_1[h6,h8,h1,p7] += 1 * t_vo[p3,h1] * v[h6,h8,p3,p7];
 *  
 *  }
*/


extern "C" {
  void ipccsd_x1_1_1_(Integer *d_f, Integer *k_f_offset,Integer *d_x1_1_1, Integer *k_x1_1_1_offset);
  void ipccsd_x1_1_1_1_(Integer *d_f, Integer *k_f_offset,Integer *d_x1_1_1_1, Integer *k_x1_1_1_1_offset);
  void ipccsd_x1_1_1_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_x1_1_1_1, Integer *k_x1_1_1_1_offset);
  void ipccsd_x1_1_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_x1_1_1_1, Integer *k_x1_1_1_1_offset,Integer *d_x1_1_1, Integer *k_x1_1_1_offset);
  void ipccsd_x1_1_3_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_x1_1_1, Integer *k_x1_1_1_offset);
  void ipccsd_x1_1_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_x1_1_1, Integer *k_x1_1_1_offset);
  void ipccsd_x1_1_(Integer *d_x_o, Integer *k_x_o_offset,Integer *d_x1_1_1, Integer *k_x1_1_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void ipccsd_x1_2_1_(Integer *d_f, Integer *k_f_offset,Integer *d_x1_2_1, Integer *k_x1_2_1_offset);
  void ipccsd_x1_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_x1_2_1, Integer *k_x1_2_1_offset);
  void ipccsd_x1_2_(Integer *d_x_voo, Integer *k_x_voo_offset,Integer *d_x1_2_1, Integer *k_x1_2_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void ipccsd_x1_3_1_(Integer *d_v, Integer *k_v_offset,Integer *d_x1_3_1, Integer *k_x1_3_1_offset);
  void ipccsd_x1_3_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_x1_3_1, Integer *k_x1_3_1_offset);

  void offset_ipccsd_x1_1_1_(Integer *l_x1_1_1_offset, Integer *k_x1_1_1_offset, Integer *size_x1_1_1);
  void offset_ipccsd_x1_1_1_1_(Integer *l_x1_1_1_1_offset, Integer *k_x1_1_1_1_offset, Integer *size_x1_1_1_1);
  void offset_ipccsd_x1_2_1_(Integer *l_x1_2_1_offset, Integer *k_x1_2_1_offset, Integer *size_x1_2_1);
  void offset_ipccsd_x1_3_1_(Integer *l_x1_3_1_offset, Integer *k_x1_3_1_offset, Integer *size_x1_3_1);
}

namespace ctce {
  extern "C" {
    void ipccsd_x1_cxx(Integer *d_t_vvoo,Integer *d_f,Integer *d_x_voo,Integer *d_x_o,Integer *d_v,Integer *d_t_vo,Integer *d_i0,Integer *k_t_vvoo_offset,Integer *k_f_offset,Integer *k_x_voo_offset,Integer *k_x_o_offset,Integer *k_v_offset,Integer *k_t_vo_offset,Integer *k_i0_offset){
      static bool set_x1 = true;
      
      Assignment op_x1_1_1;
      Assignment op_x1_1_1_1;
      Assignment op_x1_2_1;
      Assignment op_x1_3_1;
      Multiplication op_x1_1_1_2;
      Multiplication op_x1_1_2;
      Multiplication op_x1_1_3;
      Multiplication op_x1_1_4;
      Multiplication op_x1_1;
      Multiplication op_x1_2_2;
      Multiplication op_x1_2;
      Multiplication op_x1_3_2;
      
      DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
      static Equations eqs;

      if (set_x1) {
        ipccsd_x1_equations(eqs);
        set_x1 = false;
      }

      std::vector <Tensor> tensors;
      std::vector <Operation> ops;
      tensors_and_ops(eqs, tensors, ops);

      Tensor *x1_1_1 = &tensors[0];
      Tensor *f = &tensors[1];
      Tensor *x1_1_1_1 = &tensors[2];
      Tensor *t_vo = &tensors[3];
      Tensor *v = &tensors[4];
      Tensor *t_vvoo = &tensors[5];
      Tensor *i0 = &tensors[6];
      Tensor *x_o = &tensors[7];
      Tensor *x1_2_1 = &tensors[8];
      Tensor *x_voo = &tensors[9];
      Tensor *x1_3_1 = &tensors[10];

      op_x1_1_1 = ops[0].add;
      op_x1_1_1_1 = ops[1].add;
      op_x1_1_1_2 = ops[2].mult;
      op_x1_1_2 = ops[3].mult;
      op_x1_1_3 = ops[4].mult;
      op_x1_1_4 = ops[5].mult;
      op_x1_1 = ops[6].mult;
      op_x1_2_1 = ops[7].add;
      op_x1_2_2 = ops[8].mult;
      op_x1_2 = ops[9].mult;
      op_x1_3_1 = ops[10].add;
      op_x1_3_2 = ops[11].mult;
      
/* ----- Insert attach code ------ */

      CorFortran(1, op_x1_1_1, ofsset_ipccsd_x1_1_1_);
      CorFortran(1, op_x1_1_1, ipccsd_x1_1_1_);
      CorFortran(1, op_x1_1_1_1, ofsset_ipccsd_x1_1_1_1_);
      CorFortran(1, op_x1_1_1_1, ipccsd_x1_1_1_1_);
      CorFortran(1, op_x1_1_1_2, ipccsd_x1_1_1_2_);
      CorFortran(1, op_x1_1_2, ipccsd_x1_1_2_);
      destroy(x1_1_1_1);
      CorFortran(1, op_x1_1_3, ipccsd_x1_1_3_);
      CorFortran(1, op_x1_1_4, ipccsd_x1_1_4_);
      CorFortran(1, op_x1_1, ipccsd_x1_1_);
      destroy(x1_1_1);
      CorFortran(1, op_x1_2_1, ofsset_ipccsd_x1_2_1_);
      CorFortran(1, op_x1_2_1, ipccsd_x1_2_1_);
      CorFortran(1, op_x1_2_2, ipccsd_x1_2_2_);
      CorFortran(1, op_x1_2, ipccsd_x1_2_);
      destroy(x1_2_1);
      CorFortran(1, op_x1_3_1, ofsset_ipccsd_x1_3_1_);
      CorFortran(1, op_x1_3_1, ipccsd_x1_3_1_);
      CorFortran(1, op_x1_3_2, ipccsd_x1_3_2_);
      
/* ----- Insert detach code ------ */

    }
  } // extern C
}; // namespace ctce
