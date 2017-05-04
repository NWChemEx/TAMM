
/*
 *  noga {
 *
 *
 *  index p,q,r,s = N;
 *  index i, j, m, n = O;
 *  index a, b, c, d, e, f = V;
 *
 *  array bDiag[][];
 *  array bT[N][N];
 *  array hT[N][N];
 *  array FT[N][N];
 *
 *  array X_VV[V][V];
 *  array X_OO[O][O];
 *  array X_OV[O][V];
 *
 *  array t1[][];
 *  array t2[][];
 *  array t3[][];
 *  array t4[O][N];
 *  array t5[O][N];
 *  array t6[O][N];
 *  array t7[V][N];
 *
 *  hf_1: FT[p,q] += 1.0 * hT[p,q];
 *
 *  hf_2:  FT[p,q] += bDiag[] * bT[p,q];
 *
 *  hf_3_1: t1[] += X_OO[i,j] * bT[i,j];
 *  hf_3: FT[p,q] += bT[p,q] * t1[];
 *
 *  hf_4_1: t2[] += 2.0 * X_OV[i,a] * bT[i,a];
 *  hf_4:  FT[p,q] += t2[] * bT[p,q];
 *
 *  hf_5_1: t3[] += X_VV[a,b] * bT[a,b];
 *  hf_5: FT[p,q] += bT[p,q] * t3[];
 *
 *  hf_6: FT[p,q] += -1.0 * bT[p,i] * bT[i,q];
 *
 *  hf_7_1: t4[i,q] += X_OO[i,j] * bT[j,q];
 *  hf_7:  FT[p,q] += -1.0 *  bT[p,i] * t4[i,q];
 *
 *  hf_8_1: t5[i,q] += X_OV[i,a] * bT[a,q];
 *  hf_8:  FT[p,q] += -1.0 * bT[p,i] * t5[i,q];
 *
 *  hf_9_1: t6[i,p] += X_OV[i,a] * bT[p,a];
 *  hf_9: FT[p,q] += -1.0 * bT[i,q] * t6[i,p];
 *
 *  hf_10_1: t7[a,q] += X_VV[a,b] * bT[b,q];
 *  hf_10:  FT[p,q] += -1.0 * bT[p,a] * t7[a,q];
 *
 *  }
 *
 */

namespace tamm {

void schedule_linear(std::map<std::string, tamm::Tensor> &tensors,
                     std::vector<Operation> &ops);
void schedule_levels(std::map<std::string, tamm::Tensor> &tensors,
                     std::vector<Operation> &ops);

void noga_fock_build(
    Integer *d_t5, Integer *d_FT, Integer *d_bDiag, Integer *d_t6,
    Integer *d_t2, Integer *d_t4, Integer *d_X_VV, Integer *d_X_OV,
    Integer *d_t3, Integer *d_hT, Integer *d_t1, Integer *d_bT, Integer *d_X_OO,
    Integer *d_t7, Integer *k_t5_offset, Integer *k_FT_offset,
    Integer *k_bDiag_offset, Integer *k_t6_offset, Integer *k_t2_offset,
    Integer *k_t4_offset, Integer *k_X_VV_offset, Integer *k_X_OV_offset,
    Integer *k_t3_offset, Integer *k_hT_offset, Integer *k_t1_offset,
    Integer *k_bT_offset, Integer *k_X_OO_offset, Integer *k_t7_offset) {
  static bool set_build = true;

  Assignment op_hf_1;
  Multiplication op_hf_2;
  Multiplication op_hf_3_1;
  Multiplication op_hf_3;
  Multiplication op_hf_4_1;
  Multiplication op_hf_4;
  Multiplication op_hf_5_1;
  Multiplication op_hf_5;
  Multiplication op_hf_6;
  Multiplication op_hf_7_1;
  Multiplication op_hf_7;
  Multiplication op_hf_8_1;
  Multiplication op_hf_8;
  Multiplication op_hf_9_1;
  Multiplication op_hf_9;
  Multiplication op_hf_10_1;
  Multiplication op_hf_10;

  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_build) {
    noga_fock_build_equations(&eqs);
    set_build = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector<Operation> ops;
  tensors_and_ops(&eqs, &tensors, &ops);

  Tensor *bDiag = &tensors["bDiag"];
  Tensor *bT = &tensors["bT"];
  Tensor *hT = &tensors["hT"];
  Tensor *FT = &tensors["FT"];
  Tensor *X_VV = &tensors["X_VV"];
  Tensor *X_OO = &tensors["X_OO"];
  Tensor *X_OV = &tensors["X_OV"];
  Tensor *t1 = &tensors["t1"];
  Tensor *t2 = &tensors["t2"];
  Tensor *t3 = &tensors["t3"];
  Tensor *t4 = &tensors["t4"];
  Tensor *t5 = &tensors["t5"];
  Tensor *t6 = &tensors["t6"];
  Tensor *t7 = &tensors["t7"];

  /* ----- Attach ------ */
  // v->set_dist(idist);
  FT->attach(*k_FT_offset, 0, *d_FT);
  hT->attach(*k_hT_offset, 0, *d_hT);
  bT->attach(*k_bT_offset, 0, *d_bT);
  X_VV->attach(*k_X_VV_offset, 0, *d_X_VV);
  X_OV->attach(*k_X_OV_offset, 0, *d_X_OV);
  X_OO->attach(*k_X_OO_offset, 0, *d_X_OO);

  schedule_levels(&tensors, &ops);

  /* ----- Detach ------ */
  FT->detach();
  hT->detach();
  bT->detach();
  X_VV->detach();
  X_OV->detach();
  X_OO->detach();
}  // noga_fock_build

};  // namespace tamm
