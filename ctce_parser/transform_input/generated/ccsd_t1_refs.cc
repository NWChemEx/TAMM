extern "C" {
  void ccsd_t1_1_(Integer *d_f, Integer *k_f_offset,Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t1_2_1_(Integer *d_f, Integer *k_f_offset,Integer *d_t1_2_1, Integer *k_t1_2_1_offset);
  void ccsd_t1_2_2_1_(Integer *d_f, Integer *k_f_offset,Integer *d_t1_2_2_1, Integer *k_t1_2_2_1_offset);
  void ccsd_t1_2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t1_2_2_1, Integer *k_t1_2_2_1_offset);
  void ccsd_t1_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t1_2_2_1, Integer *k_t1_2_2_1_offset,Integer *d_t1_2_1, Integer *k_t1_2_1_offset);
  void ccsd_t1_2_3_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t1_2_1, Integer *k_t1_2_1_offset);
  void ccsd_t1_2_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t1_2_1, Integer *k_t1_2_1_offset);
  void ccsd_t1_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t1_2_1, Integer *k_t1_2_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t1_3_1_(Integer *d_f, Integer *k_f_offset,Integer *d_t1_3_1, Integer *k_t1_3_1_offset);
  void ccsd_t1_3_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t1_3_1, Integer *k_t1_3_1_offset);
  void ccsd_t1_3_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_t1_3_1, Integer *k_t1_3_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t1_4_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t1_5_1_(Integer *d_f, Integer *k_f_offset,Integer *d_t1_5_1, Integer *k_t1_5_1_offset);
  void ccsd_t1_5_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t1_5_1, Integer *k_t1_5_1_offset);
  void ccsd_t1_5_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_t1_5_1, Integer *k_t1_5_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t1_6_1_(Integer *d_v, Integer *k_v_offset,Integer *d_t1_6_1, Integer *k_t1_6_1_offset);
  void ccsd_t1_6_2_(Integer *d_t_vo, Integer *k_t_vo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_t1_6_1, Integer *k_t1_6_1_offset);
  void ccsd_t1_6_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_t1_6_1, Integer *k_t1_6_1_offset,Integer *d_i0, Integer *k_i0_offset);
  void ccsd_t1_7_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,Integer *d_v, Integer *k_v_offset,Integer *d_i0, Integer *k_i0_offset);

  void offset_ccsd_t1_2_1_(Integer *l_t1_2_1_offset, Integer *k_t1_2_1_offset, Integer *size_t1_2_1);
  void offset_ccsd_t1_2_2_1_(Integer *l_t1_2_2_1_offset, Integer *k_t1_2_2_1_offset, Integer *size_t1_2_2_1);
  void offset_ccsd_t1_3_1_(Integer *l_t1_3_1_offset, Integer *k_t1_3_1_offset, Integer *size_t1_3_1);
  void offset_ccsd_t1_5_1_(Integer *l_t1_5_1_offset, Integer *k_t1_5_1_offset, Integer *size_t1_5_1);
  void offset_ccsd_t1_6_1_(Integer *l_t1_6_1_offset, Integer *k_t1_6_1_offset, Integer *size_t1_6_1);
}

namespace ctce {
  extern "C" {
    void ccsd_t1_cxx(Integer *d_t_vvoo,Integer *d_i0,Integer *d_v,Integer *d_t_vo,Integer *d_f,Integer *k_t_vvoo_offset,Integer *k_i0_offset,Integer *k_v_offset,Integer *k_t_vo_offset,Integer *k_f_offset){
      static bool set_t1 = true;
      
      Assignment op_t1_1;
      Assignment op_t1_2_1;
      Assignment op_t1_2_2_1;
      Assignment op_t1_3_1;
      Assignment op_t1_5_1;
      Assignment op_t1_6_1;
      Multiplication op_t1_2_2_2;
      Multiplication op_t1_2_2;
      Multiplication op_t1_2_3;
      Multiplication op_t1_2_4;
      Multiplication op_t1_2;
      Multiplication op_t1_3_2;
      Multiplication op_t1_3;
      Multiplication op_t1_4;
      Multiplication op_t1_5_2;
      Multiplication op_t1_5;
      Multiplication op_t1_6_2;
      Multiplication op_t1_6;
      Multiplication op_t1_7;
      
      DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
      static Equations eqs;

      if (set_t1) {
        ccsd_t1_equations(eqs);
        set_t1 = false;
      }

      std::vector <Tensor> tensors;
      std::vector <Operation> ops;
      tensors_and_ops(eqs, tensors, ops);

      Tensor *i0 = &tensors[0];
      Tensor *f = &tensors[1];
      Tensor *t1_2_1 = &tensors[2];
      Tensor *t1_2_2_1 = &tensors[3];
      Tensor *t_vo = &tensors[4];
      Tensor *v = &tensors[5];
      Tensor *t_vvoo = &tensors[6];
      Tensor *t1_3_1 = &tensors[7];
      Tensor *t1_5_1 = &tensors[8];
      Tensor *t1_6_1 = &tensors[9];

      op_t1_1 = ops[0].add;
      op_t1_2_1 = ops[1].add;
      op_t1_2_2_1 = ops[2].add;
      op_t1_2_2_2 = ops[3].mult;
      op_t1_2_2 = ops[4].mult;
      op_t1_2_3 = ops[5].mult;
      op_t1_2_4 = ops[6].mult;
      op_t1_2 = ops[7].mult;
      op_t1_3_1 = ops[8].add;
      op_t1_3_2 = ops[9].mult;
      op_t1_3 = ops[10].mult;
      op_t1_4 = ops[11].mult;
      op_t1_5_1 = ops[12].add;
      op_t1_5_2 = ops[13].mult;
      op_t1_5 = ops[14].mult;
      op_t1_6_1 = ops[15].add;
      op_t1_6_2 = ops[16].mult;
      op_t1_6 = ops[17].mult;
      op_t1_7 = ops[18].mult;
      
/* ----- Insert attach code ------ */

      CorFortran(1, op_t1_1, ccsd_t1_1_);
      CorFortran(1, op_t1_2_1, ofsset_ccsd_t1_2_1_);
      CorFortran(1, op_t1_2_1, ccsd_t1_2_1_);
      CorFortran(1, op_t1_2_2_1, ofsset_ccsd_t1_2_2_1_);
      CorFortran(1, op_t1_2_2_1, ccsd_t1_2_2_1_);
      CorFortran(1, op_t1_2_2_2, ccsd_t1_2_2_2_);
      CorFortran(1, op_t1_2_2, ccsd_t1_2_2_);
      destroy(t1_2_2_1);
      CorFortran(1, op_t1_2_3, ccsd_t1_2_3_);
      CorFortran(1, op_t1_2_4, ccsd_t1_2_4_);
      CorFortran(1, op_t1_2, ccsd_t1_2_);
      destroy(t1_2_1);
      CorFortran(1, op_t1_3_1, ofsset_ccsd_t1_3_1_);
      CorFortran(1, op_t1_3_1, ccsd_t1_3_1_);
      CorFortran(1, op_t1_3_2, ccsd_t1_3_2_);
      CorFortran(1, op_t1_3, ccsd_t1_3_);
      destroy(t1_3_1);
      CorFortran(1, op_t1_4, ccsd_t1_4_);
      CorFortran(1, op_t1_5_1, ofsset_ccsd_t1_5_1_);
      CorFortran(1, op_t1_5_1, ccsd_t1_5_1_);
      CorFortran(1, op_t1_5_2, ccsd_t1_5_2_);
      CorFortran(1, op_t1_5, ccsd_t1_5_);
      destroy(t1_5_1);
      CorFortran(1, op_t1_6_1, ofsset_ccsd_t1_6_1_);
      CorFortran(1, op_t1_6_1, ccsd_t1_6_1_);
      CorFortran(1, op_t1_6_2, ccsd_t1_6_2_);
      CorFortran(1, op_t1_6, ccsd_t1_6_);
      destroy(t1_6_1);
      CorFortran(1, op_t1_7, ccsd_t1_7_);
      
/* ----- Insert detach code ------ */

    }
  } // extern C
}; // namespace ctce
