/*
 *  c2 {
 *  
 *  index h1,h2,h3,h4,h5,h6 = O;
 *  index p1,p2,p3,p4,p5,p6 = V;
 *  
 *  array i0[V,V][O,O];
 *  array v[N,N][N,N]: irrep_v;
 *  array t_vo[V][O]: irrep_t;
 *  array f[N][N]: irrep_f;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array e[][]: irrep_e;
 *  
 *  c2_1:       i0[p3,p4,h1,h2] += 1 * v[p3,p4,h1,h2];
 *  c2_2:       i0[p3,p4,h1,h2] += 1 * t_vo[p3,h1] * f[p4,h2];
 *  c2_3:       i0[p3,p4,h1,h2] += -1 * t_vo[p3,h5] * v[h5,p4,h1,h2];
 *  c2_4:       i0[p3,p4,h1,h2] += -1 * t_vo[p5,h1] * v[p3,p4,h2,p5];
 *  c2_5:       i0[p3,p4,h1,h2] += -1 * t_vvoo[p3,p4,h1,h5] * f[h5,h2];
 *  c2_6:       i0[p3,p4,h1,h2] += 1 * t_vvoo[p3,p5,h1,h2] * f[p4,p5];
 *  c2_7:       i0[p3,p4,h1,h2] += 1/2 * t_vvoo[p3,p4,h5,h6] * v[h5,h6,h1,h2];
 *  c2_8:       i0[p3,p4,h1,h2] += -1 * t_vvoo[p3,p5,h1,h6] * v[h6,p4,h2,p5];
 *  c2_9:       i0[p3,p4,h1,h2] += 1/2 * t_vvoo[p5,p6,h1,h2] * v[p3,p4,p5,p6];
 *  c2_10:      i0[p3,p4,h1,h2] += -1 * e * t_vvoo[p3,p4,h1,h2];
 *  
 *  }
*/


extern "C" {
  void cisd_c2_1_(Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_f, Integer *k_f_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c2_3_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c2_4_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c2_5_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_f, Integer *k_f_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c2_6_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_f, Integer *k_f_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c2_7_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c2_8_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c2_9_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);
  void cisd_c2_10_(Integer *d_e, Integer *k_e_offset,Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_i0, Integer *k_i0_offset);

}

namespace tamm {

void schedule_linear(std::map<std::string, tamm::Tensor> &tensors, std::vector<Operation> &ops);
void schedule_linear_lazy(std::map<std::string, tamm::Tensor> &tensors, std::vector<Operation> &ops);
void schedule_levels(std::map<std::string, tamm::Tensor> &tensors, std::vector<Operation> &ops);

extern "C" {
  void cisd_c2_cxx_(Integer *d_t_vvoo, Integer *d_e, Integer *d_f, Integer *d_i0, Integer *d_t_vo, Integer *d_v, 
  Integer *k_t_vvoo_offset, Integer *k_e_offset, Integer *k_f_offset, Integer *k_i0_offset, Integer *k_t_vo_offset, Integer *k_v_offset) {

  static bool set_c2 = true;
  
  Assignment op_c2_1;
  Multiplication op_c2_2;
  Multiplication op_c2_3;
  Multiplication op_c2_4;
  Multiplication op_c2_5;
  Multiplication op_c2_6;
  Multiplication op_c2_7;
  Multiplication op_c2_8;
  Multiplication op_c2_9;
  Multiplication op_c2_10;
  
  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_c2) {
    cisd_c2_equations(eqs);
    set_c2 = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector <Operation> ops;
  tensors_and_ops(eqs, tensors, ops);

  Tensor *i0 = &tensors["i0"];
  Tensor *v = &tensors["v"];
  Tensor *t_vo = &tensors["t_vo"];
  Tensor *f = &tensors["f"];
  Tensor *t_vvoo = &tensors["t_vvoo"];
  Tensor *e = &tensors["e"];

  /* ----- Insert attach code ------ */
  v->set_dist(idist);
  i0->attach(*k_i0_offset, 0, *d_i0);
  f->attach(*k_f_offset, 0, *d_f);
  v->attach(*k_v_offset, 0, *d_v);

  #if 1
    schedule_levels(tensors, ops);
  #else
    op_c2_1 = ops[0].add;
    op_c2_2 = ops[1].mult;
    op_c2_3 = ops[2].mult;
    op_c2_4 = ops[3].mult;
    op_c2_5 = ops[4].mult;
    op_c2_6 = ops[5].mult;
    op_c2_7 = ops[6].mult;
    op_c2_8 = ops[7].mult;
    op_c2_9 = ops[8].mult;
    op_c2_10 = ops[9].mult;

    CorFortran(1, op_c2_1, cisd_c2_1_);
    CorFortran(1, op_c2_2, cisd_c2_2_);
    CorFortran(1, op_c2_3, cisd_c2_3_);
    CorFortran(1, op_c2_4, cisd_c2_4_);
    CorFortran(1, op_c2_5, cisd_c2_5_);
    CorFortran(1, op_c2_6, cisd_c2_6_);
    CorFortran(1, op_c2_7, cisd_c2_7_);
    CorFortran(1, op_c2_8, cisd_c2_8_);
    CorFortran(1, op_c2_9, cisd_c2_9_);
    CorFortran(1, op_c2_10, cisd_c2_10_);
  #endif

  /* ----- Insert detach code ------ */
  f->detach();
  i0->detach();
  v->detach();
  }
} // extern C
}; // namespace tamm
