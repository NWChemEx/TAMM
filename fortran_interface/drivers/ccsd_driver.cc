//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------

#include <cassert>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include "tammx/tammx.h"

using namespace std;
using namespace tammx;

void diis_init() {

}
void diis_next();
void diss_tidy();

void compute_residual(Tensor& tensor) {
  Tensor resid;

  resid.allocate();
  resid.init(0);
  resid() += tensor() * tensor();
  Block resblock = resid.get({});
  return *reinterpret_cast<double*>(resblock.buf());
  resid.destruct();
}

void tce_diss() {


}

/**
 * ref, corr
 */
double ccsd_driver(Tensor& d_t1, Tensor& d_t2,
		 Tensor& d_f1, Tensor& d_v2,
		 int maxiter, double thresh) {
  diis_init();

  TensorVec<SymmGroup> indices_vo{SymmGroup{DimType::v}, SymmGroup{DimType::o}};
  TensorVec<SymmGroup> indices_vvoo{SymmGroup{DimType::v, DimType::v}, SymmGroup{DimType::o, DimType::o}};

  Tensor d_e{t_scalar, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor d_r1{indices_vo, Type::double_precision, Distribution::tce_nwma, 2, irrep_t, false};
  Tensor d_r2{indices_vvoo, Type::double_precision, Distribution::tce_nwma, 2, irrep_t, false};

  d_e.allocate();
  d_r1.allocate();
  d_r2.allocate();

  double corr = 0;
  for(int iter=0; iter<maxiter; iter++) {
    d_r1.init(0);
    d_r2.init(0);

    ccsd_e(d_f1, d_e, d_t1, d_t2, d_v2);
    ccsd_t1(d_f1, d_r1, d_t1, d_t2, d_v2);
    ccsd_t2(d_f1, d_r2, d_t1, d_t2, d_v2);

    double r1 = compute_residual(d_r1);
    double r2 = compute_residual(d_r2);
    double residual = std::max(r1, r2);

    Block eblock = d_e.get({});
    corr = *reinterpret_cast<double*>(eblock.buf());
    //nodezero_print();
    if(residual < thresh) {
      //nodezero_print();
      break;
    }
    diis_next();
  }

  d_e.destruct();
  d_r1.destruct();
  d_r2.destruct();
  
  diis_tidy();
  return corr;
}

