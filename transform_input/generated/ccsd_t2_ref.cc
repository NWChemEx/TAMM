/*
 *  t2 {
 *  
 *  index h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11 = O;
 *  index p1,p2,p3,p4,p5,p6,p7,p8,p9 = V;
 *  
 *  array i0[V,V][O,O];
 *  array v[N,N][N,N]: irrep_v;
 *  array t_vo[V][O]: irrep_t;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array f[N][N]: irrep_f;
 *  array t2_2_5_1[O,O][O,V];
 *  array t2_7_1[O,V][O,V];
 *  array t2_3_1[V,V][O,V];
 *  array t2_2_2_1[O,O][O,O];
 *  array t2_4_2_1[O][V];
 *  array t2_4_1[O][O];
 *  array t2_2_2_2_1[O,O][O,V];
 *  array t2_2_3_1[O,V][O,V];
 *  array t2_5_1[V][V];
 *  array t2_6_2_1[O,O][O,V];
 *  array t2_2_4_1[O][V];
 *  array t2_6_1[O,O][O,O];
 *  array t2_2_1[O,V][O,O];
 *  
 *  t2_1:       i0[p3,p4,h1,h2] += 1 * v[p3,p4,h1,h2];
 *  t2_2_1:     t2_2_1[h10,p3,h1,h2] += 1 * v[h10,p3,h1,h2];
 *  t2_2_2_1:   t2_2_2_1[h10,h11,h1,h2] += -1 * v[h10,h11,h1,h2];
 *  t2_2_2_2_1: t2_2_2_2_1[h10,h11,h1,p5] += 1 * v[h10,h11,h1,p5];
 *  t2_2_2_2_2: t2_2_2_2_1[h10,h11,h1,p5] += -1/2 * t_vo[p6,h1] * v[h10,h11,p5,p6];
 *  t2_2_2_2:   t2_2_2_1[h10,h11,h1,h2] += 1 * t_vo[p5,h1] * t2_2_2_2_1[h10,h11,h2,p5];
 *  t2_2_2_3:   t2_2_2_1[h10,h11,h1,h2] += -1/2 * t_vvoo[p7,p8,h1,h2] * v[h10,h11,p7,p8];
 *  t2_2_2:     t2_2_1[h10,p3,h1,h2] += 1/2 * t_vo[p3,h11] * t2_2_2_1[h10,h11,h1,h2];
 *  t2_2_3_1:   t2_2_3_1[h10,p3,h1,p5] += 1 * v[h10,p3,h1,p5];
 *  t2_2_3_2:   t2_2_3_1[h10,p3,h1,p5] += -1/2 * t_vo[p6,h1] * v[h10,p3,p5,p6];
 *  t2_2_3:     t2_2_1[h10,p3,h1,h2] += -1 * t_vo[p5,h1] * t2_2_3_1[h10,p3,h2,p5];
 *  t2_2_4_1:   t2_2_4_1[h10,p5] += 1 * f[h10,p5];
 *  t2_2_4_2:   t2_2_4_1[h10,p5] += -1 * t_vo[p6,h7] * v[h7,h10,p5,p6];
 *  t2_2_4:     t2_2_1[h10,p3,h1,h2] += -1 * t_vvoo[p3,p5,h1,h2] * t2_2_4_1[h10,p5];
 *  t2_2_5_1:   t2_2_5_1[h7,h10,h1,p9] += 1 * v[h7,h10,h1,p9];
 *  t2_2_5_2:   t2_2_5_1[h7,h10,h1,p9] += 1 * t_vo[p5,h1] * v[h7,h10,p5,p9];
 *  t2_2_5:     t2_2_1[h10,p3,h1,h2] += 1 * t_vvoo[p3,p9,h1,h7] * t2_2_5_1[h7,h10,h2,p9];
 *  t2_2_6:     t2_2_1[h10,p3,h1,h2] += 1/2 * t_vvoo[p5,p6,h1,h2] * v[h10,p3,p5,p6];
 *  t2_2:       i0[p3,p4,h1,h2] += -1 * t_vo[p3,h10] * t2_2_1[h10,p4,h1,h2];
 *  t2_3_1:     t2_3_1[p3,p4,h1,p5] += 1 * v[p3,p4,h1,p5];
 *  t2_3_2:     t2_3_1[p3,p4,h1,p5] += -1/2 * t_vo[p6,h1] * v[p3,p4,p5,p6];
 *  t2_3:       i0[p3,p4,h1,h2] += -1 * t_vo[p5,h1] * t2_3_1[p3,p4,h2,p5];
 *  t2_4_1:     t2_4_1[h9,h1] += 1 * f[h9,h1];
 *  t2_4_2_1:   t2_4_2_1[h9,p8] += 1 * f[h9,p8];
 *  t2_4_2_2:   t2_4_2_1[h9,p8] += 1 * t_vo[p6,h7] * v[h7,h9,p6,p8];
 *  t2_4_2:     t2_4_1[h9,h1] += 1 * t_vo[p8,h1] * t2_4_2_1[h9,p8];
 *  t2_4_3:     t2_4_1[h9,h1] += -1 * t_vo[p6,h7] * v[h7,h9,h1,p6];
 *  t2_4_4:     t2_4_1[h9,h1] += -1/2 * t_vvoo[p6,p7,h1,h8] * v[h8,h9,p6,p7];
 *  t2_4:       i0[p3,p4,h1,h2] += -1 * t_vvoo[p3,p4,h1,h9] * t2_4_1[h9,h2];
 *  t2_5_1:     t2_5_1[p3,p5] += 1 * f[p3,p5];
 *  t2_5_2:     t2_5_1[p3,p5] += -1 * t_vo[p6,h7] * v[h7,p3,p5,p6];
 *  t2_5_3:     t2_5_1[p3,p5] += -1/2 * t_vvoo[p3,p6,h7,h8] * v[h7,h8,p5,p6];
 *  t2_5:       i0[p3,p4,h1,h2] += 1 * t_vvoo[p3,p5,h1,h2] * t2_5_1[p4,p5];
 *  t2_6_1:     t2_6_1[h9,h11,h1,h2] += -1 * v[h9,h11,h1,h2];
 *  t2_6_2_1:   t2_6_2_1[h9,h11,h1,p8] += 1 * v[h9,h11,h1,p8];
 *  t2_6_2_2:   t2_6_2_1[h9,h11,h1,p8] += 1/2 * t_vo[p6,h1] * v[h9,h11,p6,p8];
 *  t2_6_2:     t2_6_1[h9,h11,h1,h2] += 1 * t_vo[p8,h1] * t2_6_2_1[h9,h11,h2,p8];
 *  t2_6_3:     t2_6_1[h9,h11,h1,h2] += -1/2 * t_vvoo[p5,p6,h1,h2] * v[h9,h11,p5,p6];
 *  t2_6:       i0[p3,p4,h1,h2] += -1/2 * t_vvoo[p3,p4,h9,h11] * t2_6_1[h9,h11,h1,h2];
 *  t2_7_1:     t2_7_1[h6,p3,h1,p5] += 1 * v[h6,p3,h1,p5];
 *  t2_7_2:     t2_7_1[h6,p3,h1,p5] += -1 * t_vo[p7,h1] * v[h6,p3,p5,p7];
 *  t2_7_3:     t2_7_1[h6,p3,h1,p5] += -1/2 * t_vvoo[p3,p7,h1,h8] * v[h6,h8,p5,p7];
 *  t2_7:       i0[p3,p4,h1,h2] += -1 * t_vvoo[p3,p5,h1,h6] * t2_7_1[h6,p4,h2,p5];
 *  t2_8:       i0[p3,p4,h1,h2] += 1/2 * t_vvoo[p5,p6,h1,h2] * v[p3,p4,p5,p6];
 *  
 *  }
*/


extern "C" {
  void ccsd_t2_1_(Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t2_2_1_(Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_1, Integer *k_t2_2_1_offset);
  void ccsd_t2_2_2_1_(Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_2_1, Integer *k_t2_2_2_1_offset);
  void ccsd_t2_2_2_2_1_(Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_2_2_1, Integer *k_t2_2_2_2_1_offset);
  void ccsd_t2_2_2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_2_2_1, Integer *k_t2_2_2_2_1_offset);
  void ccsd_t2_2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t2_2_2_2_1, Integer *k_t2_2_2_2_1_offset,Integer *d_t2_2_2_1, Integer *k_t2_2_2_1_offset);
  void ccsd_t2_2_2_3_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_2_1, Integer *k_t2_2_2_1_offset);
  void ccsd_t2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t2_2_2_1, Integer *k_t2_2_2_1_offset,Integer *d_t2_2_1, Integer *k_t2_2_1_offset);
  void ccsd_t2_2_3_1_(Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_3_1, Integer *k_t2_2_3_1_offset);
  void ccsd_t2_2_3_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_3_1, Integer *k_t2_2_3_1_offset);
  void ccsd_t2_2_3_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t2_2_3_1, Integer *k_t2_2_3_1_offset,Integer *d_t2_2_1, Integer *k_t2_2_1_offset);
  void ccsd_t2_2_4_1_(Integer *d_f, Integer *k_f_offset,Integer *d_t2_2_4_1, Integer *k_t2_2_4_1_offset);
  void ccsd_t2_2_4_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_4_1, Integer *k_t2_2_4_1_offset);
  void ccsd_t2_2_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_t2_2_4_1, Integer *k_t2_2_4_1_offset,Integer *d_t2_2_1, Integer *k_t2_2_1_offset);
  void ccsd_t2_2_5_1_(Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_5_1, Integer *k_t2_2_5_1_offset);
  void ccsd_t2_2_5_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_5_1, Integer *k_t2_2_5_1_offset);
  void ccsd_t2_2_5_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_t2_2_5_1, Integer *k_t2_2_5_1_offset,Integer *d_t2_2_1, Integer *k_t2_2_1_offset);
  void ccsd_t2_2_6_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_2_1, Integer *k_t2_2_1_offset);
  void ccsd_t2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t2_2_1, Integer *k_t2_2_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t2_3_1_(Integer *d_v, Integer *k_v_offset,Integer *d_t2_3_1, Integer *k_t2_3_1_offset);
  void ccsd_t2_3_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_3_1, Integer *k_t2_3_1_offset);
  void ccsd_t2_3_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t2_3_1, Integer *k_t2_3_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t2_4_1_(Integer *d_f, Integer *k_f_offset,Integer *d_t2_4_1, Integer *k_t2_4_1_offset);
  void ccsd_t2_4_2_1_(Integer *d_f, Integer *k_f_offset,Integer *d_t2_4_2_1, Integer *k_t2_4_2_1_offset);
  void ccsd_t2_4_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_4_2_1, Integer *k_t2_4_2_1_offset);
  void ccsd_t2_4_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t2_4_2_1, Integer *k_t2_4_2_1_offset,Integer *d_t2_4_1, Integer *k_t2_4_1_offset);
  void ccsd_t2_4_3_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_4_1, Integer *k_t2_4_1_offset);
  void ccsd_t2_4_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_4_1, Integer *k_t2_4_1_offset);
  void ccsd_t2_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_t2_4_1, Integer *k_t2_4_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t2_5_1_(Integer *d_f, Integer *k_f_offset,Integer *d_t2_5_1, Integer *k_t2_5_1_offset);
  void ccsd_t2_5_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_5_1, Integer *k_t2_5_1_offset);
  void ccsd_t2_5_3_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_5_1, Integer *k_t2_5_1_offset);
  void ccsd_t2_5_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_t2_5_1, Integer *k_t2_5_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t2_6_1_(Integer *d_v, Integer *k_v_offset,Integer *d_t2_6_1, Integer *k_t2_6_1_offset);
  void ccsd_t2_6_2_1_(Integer *d_v, Integer *k_v_offset,Integer *d_t2_6_2_1, Integer *k_t2_6_2_1_offset);
  void ccsd_t2_6_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_6_2_1, Integer *k_t2_6_2_1_offset);
  void ccsd_t2_6_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t2_6_2_1, Integer *k_t2_6_2_1_offset,Integer *d_t2_6_1, Integer *k_t2_6_1_offset);
  void ccsd_t2_6_3_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_6_1, Integer *k_t2_6_1_offset);
  void ccsd_t2_6_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_t2_6_1, Integer *k_t2_6_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t2_7_1_(Integer *d_v, Integer *k_v_offset,Integer *d_t2_7_1, Integer *k_t2_7_1_offset);
  void ccsd_t2_7_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_7_1, Integer *k_t2_7_1_offset);
  void ccsd_t2_7_3_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t2_7_1, Integer *k_t2_7_1_offset);
  void ccsd_t2_7_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_t2_7_1, Integer *k_t2_7_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t2_8_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);

  void offset_ccsd_t2_2_1_(Integer *l_t2_2_1_offset, Integer *k_t2_2_1_offset, Integer *size_t2_2_1);
  void offset_ccsd_t2_2_2_1_(Integer *l_t2_2_2_1_offset, Integer *k_t2_2_2_1_offset, Integer *size_t2_2_2_1);
  void offset_ccsd_t2_2_2_2_1_(Integer *l_t2_2_2_2_1_offset, Integer *k_t2_2_2_2_1_offset, Integer *size_t2_2_2_2_1);
  void offset_ccsd_t2_2_3_1_(Integer *l_t2_2_3_1_offset, Integer *k_t2_2_3_1_offset, Integer *size_t2_2_3_1);
  void offset_ccsd_t2_2_4_1_(Integer *l_t2_2_4_1_offset, Integer *k_t2_2_4_1_offset, Integer *size_t2_2_4_1);
  void offset_ccsd_t2_2_5_1_(Integer *l_t2_2_5_1_offset, Integer *k_t2_2_5_1_offset, Integer *size_t2_2_5_1);
  void offset_ccsd_t2_3_1_(Integer *l_t2_3_1_offset, Integer *k_t2_3_1_offset, Integer *size_t2_3_1);
  void offset_ccsd_t2_4_1_(Integer *l_t2_4_1_offset, Integer *k_t2_4_1_offset, Integer *size_t2_4_1);
  void offset_ccsd_t2_4_2_1_(Integer *l_t2_4_2_1_offset, Integer *k_t2_4_2_1_offset, Integer *size_t2_4_2_1);
  void offset_ccsd_t2_5_1_(Integer *l_t2_5_1_offset, Integer *k_t2_5_1_offset, Integer *size_t2_5_1);
  void offset_ccsd_t2_6_1_(Integer *l_t2_6_1_offset, Integer *k_t2_6_1_offset, Integer *size_t2_6_1);
  void offset_ccsd_t2_6_2_1_(Integer *l_t2_6_2_1_offset, Integer *k_t2_6_2_1_offset, Integer *size_t2_6_2_1);
  void offset_ccsd_t2_7_1_(Integer *l_t2_7_1_offset, Integer *k_t2_7_1_offset, Integer *size_t2_7_1);
}

namespace tamm {

void schedule_linear(std::vector<Tensor> &tensors, std::vector<Operation> &ops);
void schedule_linear_lazy(std::vector<Tensor> &tensors, std::vector<Operation> &ops);
void schedule_levels(std::vector<Tensor> &tensors, std::vector<Operation> &ops);

extern "C" {
  void ccsd_t2_cxx_(Integer *d_i0, Integer *d_t_vvoo, Integer *d_f, Integer *d_t_vo, Integer *d_v, 
  Integer *k_i0_offset, Integer *k_t_vvoo_offset, Integer *k_f_offset, Integer *k_t_vo_offset, Integer *k_v_offset) {

  static bool set_t2 = true;
  
  Assignment op_t2_1;
  Assignment op_t2_2_1;
  Assignment op_t2_2_2_1;
  Assignment op_t2_2_2_2_1;
  Assignment op_t2_2_3_1;
  Assignment op_t2_2_4_1;
  Assignment op_t2_2_5_1;
  Assignment op_t2_3_1;
  Assignment op_t2_4_1;
  Assignment op_t2_4_2_1;
  Assignment op_t2_5_1;
  Assignment op_t2_6_1;
  Assignment op_t2_6_2_1;
  Assignment op_t2_7_1;
  Multiplication op_t2_2_2_2_2;
  Multiplication op_t2_2_2_2;
  Multiplication op_t2_2_2_3;
  Multiplication op_t2_2_2;
  Multiplication op_t2_2_3_2;
  Multiplication op_t2_2_3;
  Multiplication op_t2_2_4_2;
  Multiplication op_t2_2_4;
  Multiplication op_t2_2_5_2;
  Multiplication op_t2_2_5;
  Multiplication op_t2_2_6;
  Multiplication op_t2_2;
  Multiplication op_t2_3_2;
  Multiplication op_t2_3;
  Multiplication op_t2_4_2_2;
  Multiplication op_t2_4_2;
  Multiplication op_t2_4_3;
  Multiplication op_t2_4_4;
  Multiplication op_t2_4;
  Multiplication op_t2_5_2;
  Multiplication op_t2_5_3;
  Multiplication op_t2_5;
  Multiplication op_t2_6_2_2;
  Multiplication op_t2_6_2;
  Multiplication op_t2_6_3;
  Multiplication op_t2_6;
  Multiplication op_t2_7_2;
  Multiplication op_t2_7_3;
  Multiplication op_t2_7;
  Multiplication op_t2_8;
  
  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_t2) {
    ccsd_t2_equations(eqs);
    set_t2 = false;
  }

  std::vector <Tensor> tensors;
  std::vector <Operation> ops;
  tensors_and_ops(eqs, tensors, ops);

  Tensor *i0 = &tensors[0];
  Tensor *v = &tensors[1];
  Tensor *t_vo = &tensors[2];
  Tensor *t_vvoo = &tensors[3];
  Tensor *f = &tensors[4];
  Tensor *t2_2_5_1 = &tensors[5];
  Tensor *t2_7_1 = &tensors[6];
  Tensor *t2_3_1 = &tensors[7];
  Tensor *t2_2_2_1 = &tensors[8];
  Tensor *t2_4_2_1 = &tensors[9];
  Tensor *t2_4_1 = &tensors[10];
  Tensor *t2_2_2_2_1 = &tensors[11];
  Tensor *t2_2_3_1 = &tensors[12];
  Tensor *t2_5_1 = &tensors[13];
  Tensor *t2_6_2_1 = &tensors[14];
  Tensor *t2_2_4_1 = &tensors[15];
  Tensor *t2_6_1 = &tensors[16];
  Tensor *t2_2_1 = &tensors[17];

  /* ----- Insert attach code ------ */
  v->set_dist(idist)
  i0->attach(*k_i0_offset, 0, *d_i0);
  v->attach(*k_v_offset, 0, *d_v);

  #if 1
    schedule_levels(tensors, ops);
  #else
    op_t2_1 = ops[0].add;
    op_t2_2_1 = ops[1].add;
    op_t2_2_2_1 = ops[2].add;
    op_t2_2_2_2_1 = ops[3].add;
    op_t2_2_2_2_2 = ops[4].mult;
    op_t2_2_2_2 = ops[5].mult;
    op_t2_2_2_3 = ops[6].mult;
    op_t2_2_2 = ops[7].mult;
    op_t2_2_3_1 = ops[8].add;
    op_t2_2_3_2 = ops[9].mult;
    op_t2_2_3 = ops[10].mult;
    op_t2_2_4_1 = ops[11].add;
    op_t2_2_4_2 = ops[12].mult;
    op_t2_2_4 = ops[13].mult;
    op_t2_2_5_1 = ops[14].add;
    op_t2_2_5_2 = ops[15].mult;
    op_t2_2_5 = ops[16].mult;
    op_t2_2_6 = ops[17].mult;
    op_t2_2 = ops[18].mult;
    op_t2_3_1 = ops[19].add;
    op_t2_3_2 = ops[20].mult;
    op_t2_3 = ops[21].mult;
    op_t2_4_1 = ops[22].add;
    op_t2_4_2_1 = ops[23].add;
    op_t2_4_2_2 = ops[24].mult;
    op_t2_4_2 = ops[25].mult;
    op_t2_4_3 = ops[26].mult;
    op_t2_4_4 = ops[27].mult;
    op_t2_4 = ops[28].mult;
    op_t2_5_1 = ops[29].add;
    op_t2_5_2 = ops[30].mult;
    op_t2_5_3 = ops[31].mult;
    op_t2_5 = ops[32].mult;
    op_t2_6_1 = ops[33].add;
    op_t2_6_2_1 = ops[34].add;
    op_t2_6_2_2 = ops[35].mult;
    op_t2_6_2 = ops[36].mult;
    op_t2_6_3 = ops[37].mult;
    op_t2_6 = ops[38].mult;
    op_t2_7_1 = ops[39].add;
    op_t2_7_2 = ops[40].mult;
    op_t2_7_3 = ops[41].mult;
    op_t2_7 = ops[42].mult;
    op_t2_8 = ops[43].mult;

    CorFortran(1, op_t2_1, ccsd_t2_1_);
    CorFortran(1, op_t2_2_1, ofsset_ccsd_t2_2_1_);
    CorFortran(1, op_t2_2_1, ccsd_t2_2_1_);
    CorFortran(1, op_t2_2_2_1, ofsset_ccsd_t2_2_2_1_);
    CorFortran(1, op_t2_2_2_1, ccsd_t2_2_2_1_);
    CorFortran(1, op_t2_2_2_2_1, ofsset_ccsd_t2_2_2_2_1_);
    CorFortran(1, op_t2_2_2_2_1, ccsd_t2_2_2_2_1_);
    CorFortran(1, op_t2_2_2_2_2, ccsd_t2_2_2_2_2_);
    CorFortran(1, op_t2_2_2_2, ccsd_t2_2_2_2_);
    destroy(t2_2_2_2_1);
    CorFortran(1, op_t2_2_2_3, ccsd_t2_2_2_3_);
    CorFortran(1, op_t2_2_2, ccsd_t2_2_2_);
    destroy(t2_2_2_1);
    CorFortran(1, op_t2_2_3_1, ofsset_ccsd_t2_2_3_1_);
    CorFortran(1, op_t2_2_3_1, ccsd_t2_2_3_1_);
    CorFortran(1, op_t2_2_3_2, ccsd_t2_2_3_2_);
    CorFortran(1, op_t2_2_3, ccsd_t2_2_3_);
    destroy(t2_2_3_1);
    CorFortran(1, op_t2_2_4_1, ofsset_ccsd_t2_2_4_1_);
    CorFortran(1, op_t2_2_4_1, ccsd_t2_2_4_1_);
    CorFortran(1, op_t2_2_4_2, ccsd_t2_2_4_2_);
    CorFortran(1, op_t2_2_4, ccsd_t2_2_4_);
    destroy(t2_2_4_1);
    CorFortran(1, op_t2_2_5_1, ofsset_ccsd_t2_2_5_1_);
    CorFortran(1, op_t2_2_5_1, ccsd_t2_2_5_1_);
    CorFortran(1, op_t2_2_5_2, ccsd_t2_2_5_2_);
    CorFortran(1, op_t2_2_5, ccsd_t2_2_5_);
    destroy(t2_2_5_1);
    CorFortran(1, op_t2_2_6, ccsd_t2_2_6_);
    CorFortran(1, op_t2_2, ccsd_t2_2_);
    destroy(t2_2_1);
    CorFortran(1, op_t2_3_1, ofsset_ccsd_t2_3_1_);
    CorFortran(1, op_t2_3_1, ccsd_t2_3_1_);
    CorFortran(1, op_t2_3_2, ccsd_t2_3_2_);
    CorFortran(1, op_t2_3, ccsd_t2_3_);
    destroy(t2_3_1);
    CorFortran(1, op_t2_4_1, ofsset_ccsd_t2_4_1_);
    CorFortran(1, op_t2_4_1, ccsd_t2_4_1_);
    CorFortran(1, op_t2_4_2_1, ofsset_ccsd_t2_4_2_1_);
    CorFortran(1, op_t2_4_2_1, ccsd_t2_4_2_1_);
    CorFortran(1, op_t2_4_2_2, ccsd_t2_4_2_2_);
    CorFortran(1, op_t2_4_2, ccsd_t2_4_2_);
    destroy(t2_4_2_1);
    CorFortran(1, op_t2_4_3, ccsd_t2_4_3_);
    CorFortran(1, op_t2_4_4, ccsd_t2_4_4_);
    CorFortran(1, op_t2_4, ccsd_t2_4_);
    destroy(t2_4_1);
    CorFortran(1, op_t2_5_1, ofsset_ccsd_t2_5_1_);
    CorFortran(1, op_t2_5_1, ccsd_t2_5_1_);
    CorFortran(1, op_t2_5_2, ccsd_t2_5_2_);
    CorFortran(1, op_t2_5_3, ccsd_t2_5_3_);
    CorFortran(1, op_t2_5, ccsd_t2_5_);
    destroy(t2_5_1);
    CorFortran(1, op_t2_6_1, ofsset_ccsd_t2_6_1_);
    CorFortran(1, op_t2_6_1, ccsd_t2_6_1_);
    CorFortran(1, op_t2_6_2_1, ofsset_ccsd_t2_6_2_1_);
    CorFortran(1, op_t2_6_2_1, ccsd_t2_6_2_1_);
    CorFortran(1, op_t2_6_2_2, ccsd_t2_6_2_2_);
    CorFortran(1, op_t2_6_2, ccsd_t2_6_2_);
    destroy(t2_6_2_1);
    CorFortran(1, op_t2_6_3, ccsd_t2_6_3_);
    CorFortran(1, op_t2_6, ccsd_t2_6_);
    destroy(t2_6_1);
    CorFortran(1, op_t2_7_1, ofsset_ccsd_t2_7_1_);
    CorFortran(1, op_t2_7_1, ccsd_t2_7_1_);
    CorFortran(1, op_t2_7_2, ccsd_t2_7_2_);
    CorFortran(1, op_t2_7_3, ccsd_t2_7_3_);
    CorFortran(1, op_t2_7, ccsd_t2_7_);
    destroy(t2_7_1);
    CorFortran(1, op_t2_8, ccsd_t2_8_);
  #endif

  /* ----- Insert detach code ------ */
  f->detach();
  i0->detach();
  v->detach();
  }
} // extern C
}; // namespace tamm
