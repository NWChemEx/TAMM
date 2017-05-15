
/*
 *  noga {
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

#include <iostream>
#include "tensor/corf.h"
#include "tensor/equations.h"
#include "tensor/input.h"
#include "tensor/schedulers.h"
#include "tensor/t_assign.h"
#include "tensor/t_mult.h"
#include "tensor/tensor.h"
#include "tensor/tensors_and_ops.h"
#include "tensor/variables.h"

namespace tamm {

void schedule_linear(std::map<std::string, tamm::Tensor> &tensors,
                     std::vector<Operation> &ops);
void schedule_levels(std::map<std::string, tamm::Tensor> &tensors,
                     std::vector<Operation> &ops);

void noga_fock_build() {

  static bool set_build = true;

  F77Integer x0 = 0, noab = 20, k_spin = 1, nvab=100, noa = 10, nva = 50;
  logical intorb = 0, restricted = 0;
  double dbl_mb = 0.0;

  Dummy::construct();
  Table::construct();
  Variables::set_ov(&noab,&nvab);
  Variables::set_ova(&noa,&nva);
  Variables::set_irrep(&x0,&x0,&x0);
  Variables::set_idmb(&x0,&dbl_mb);
  Variables::set_log(&intorb,&restricted);
  Variables::set_k1(&nvab,&k_spin,&x0);

  DistType idist = dist_nw;
  static Equations eqs;

  if (set_build) {
    noga_fock_build_equations(&eqs);
    set_build = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector<Operation> ops;
  tensors_and_ops(&eqs, &tensors, &ops);

  //Tensor *bDiag = &tensors["bDiag"];
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

//(F77Integer *noab, F77Integer *nvab, F77Integer *int_mb, double *dbl_mb,
//             F77Integer *k_range, F77Integer *k_spin, F77Integer *k_sym,
//            logical *intorb, logical *restricted, F77Integer *irrep_v,
//            F77Integer *irrep_t, F77Integer *irrep_f)



  /* ----- Attach ------ */
    Fint k_FT_offset = 0, d_FT = 0;
    Fint k_hT_offset = 0, d_hT = 0;
    Fint k_bT_offset = 0, d_bT = 0;
    Fint k_X_VV_offset = 0, d_X_VV = 0;
    Fint k_X_OV_offset = 0, d_X_OV = 0;
    Fint k_X_OO_offset = 0, d_X_OO = 0;

  // v->set_dist(idist);
  FT->attach(0, 0, 0);
  hT->attach(0, 0, 0);
  bT->attach(0, 0, 0);
  X_VV->attach(0, 0, 0);
  X_OV->attach(0, 0, 0);
  X_OO->attach(0, 0, 0);

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

