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
#include "tammx/diis.h"

using namespace std;
using namespace tammx;
using namespace tammx::tensor_dims;
using namespace tammx::tensor_labels;

template<typename T>
void compute_residual(Scheduler &sch, Tensor<T>& tensor, Tensor<T>& scalar) {
  sch.alloc(scalar)
      (scalar() = tensor() * tensor());
}


template<typename T>
void ccsd_e(Scheduler &sch, Tensor<T>& f1, Tensor<T>& de,
            Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& v2) {
  auto &i1 = sch.tensor<T>(O|V);

  sch.alloc(i1)
      (i1(h6,p5) +=        f1(h6,p5))
      (i1(h6,p5) += 0.5  * t1(p3,h4)       * v2(h4,h6,p3,p5))
      (de()      +=        t1(p5,h6)       * i1(h6,p5))
      (de()      += 0.25 * t2(p1,p2,h3,h4) * v2(h3,h4,p1,p2))
      .dealloc(i1);
}

template<typename T>
void ccsd_t1(Scheduler& sch, Tensor<T>& f1, Tensor<T>& i0,
             Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& v2) {
  auto &t1_2_1 = sch.tensor<T>(O|O);
  auto &t1_2_2_1 = sch.tensor<T>(O|V);
  auto &t1_3_1 = sch.tensor<T>(V|V);
  auto &t1_5_1 = sch.tensor<T>(O|V);
  auto &t1_6_1 = sch.tensor<T>(OO|OV);

  sch.alloc(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1)
      (i0(p2,h1)            =        f1(p2,h1))
      (t1_2_1(h7,h1)        =        f1(h7,h1))
      (t1_2_2_1(h7,p3)      =        f1(h7,p3))
      (t1_2_2_1(h7,p3)     += -1   * t1(p5,h6)       * v2(h6,h7,p3,p5))
      (t1_2_1(h7,h1)       +=        t1(p3,h1)       * t1_2_2_1(h7,p3))
      (t1_2_1(h7,h1)       += -1   * t1(p4,h5)       * v2(h5,h7,h1,p4))
      (t1_2_1(h7,h1)       += -0.5 * t2(p3,p4,h1,h5) * v2(h5,h7,p3,p4))
      (i0(p2,h1)           += -1   * t1(p2,h7)       * t1_2_1(h7,h1))
      (t1_3_1(p2,p3)        =        f1(p2,p3))
      (t1_3_1(p2,p3)       += -1   * t1(p4,h5)       * v2(h5,p2,p3,p4))
      (i0(p2,h1)           +=        t1(p3,h1)       * t1_3_1(p2,p3))
      (i0(p2,h1)           += -1   * t1(p3,h4)       * v2(h4,p2,h1,p3))
      (t1_5_1(h8,p7)        =        f1(h8,p7))
      (t1_5_1(h8,p7)       +=        t1(p5,h6)       * v2(h6,h8,p5,p7))
      (i0(p2,h1)           +=        t2(p2,p7,h1,h8) * t1_5_1(h8,p7))
      (t1_6_1(h4,h5,h1,p3)  =        v2(h4,h5,h1,p3))
      (t1_6_1(h4,h5,h1,p3) += -1   * t1(p6,h1)       * v2(h4,h5,p3,p6))
      (i0(p2,h1)           += -0.5 * t2(p2,p3,h4,h5) * t1_6_1(h4,h5,h1,p3))
      (i0(p2,h1)           += -0.5 * t2(p3,p4,h1,h5) * v2(h5,p2,p3,p4))
      .dealloc(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1);
}

template<typename T>
void ccsd_t2(Scheduler& sch, Tensor<T>& f1, Tensor<T>& i0,
             Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& v2) {
  auto &t2_2_1 = sch.tensor<T>(OV|OO);
  auto &t2_2_2_1 = sch.tensor<T>(OO|OO);
  auto &t2_2_2_2_1 = sch.tensor<T>(OO|OV);
  auto &t2_2_4_1 = sch.tensor<T>(O|V);
  auto &t2_2_5_1 = sch.tensor<T>(OO|OV);
  auto &t2_4_1 = sch.tensor<T>(O|O);
  auto &t2_4_2_1 = sch.tensor<T>(O|V);
  auto &t2_5_1 = sch.tensor<T>(V|V);
  auto &t2_6_1 = sch.tensor<T>(OO|OO);
  auto &t2_6_2_1 = sch.tensor<T>(OO|OV);
  auto &t2_7_1 = sch.tensor<T>(OV|OV);
  auto &vt1t1_1 = sch.tensor<T>(OV|OO);

  sch.alloc(t2_2_1, t2_2_2_1, t2_2_2_2_1, t2_2_4_1, t2_2_5_1, t2_4_1, t2_4_2_1,
            t2_5_1, t2_6_1, t2_6_2_1, t2_7_1, vt1t1_1)
      (i0(p3,p4,h1,h2)            =        v2(p3,p4,h1,h2))
      (t2_2_1(h10,p3,h1,h2)       =        v2(h10,p3,h1,h2))
      (t2_2_2_1(h10,h11,h1,h2)   += -1  *  v2(h10,h11,h1,h2))
      (t2_2_2_2_1(h10,h11,h1,p5) =         v2(h10,h11,h1,p5))
      (t2_2_2_2_1(h10,h11,h1,p5) += -0.5 * t1(p6,h1) * v2(h10,h11,p5,p6))
      (t2_2_2_1(h10,h11,h1,h2)   +=        t1(p5,h1) * t2_2_2_2_1(h10,h11,h2,p5))
      (t2_2_2_1(h10,h11,h1,h2)   += -0.5 * t2(p7,p8,h1,h2) * v2(h10,h11,p7,p8))
      (t2_2_1(h10,p3,h1,h2)      += 0.5  * t1(p3,h11) * t2_2_2_1(h10,h11,h1,h2))
      (t2_2_4_1(h10,p5)           =        f1(h10,p5))
      (t2_2_4_1(h10,p5)          += -1   * t1(p6,h7) * v2(h7,h10,p5,p6))
      (t2_2_1(h10,p3,h1,h2)      += -1   * t2(p3,p5,h1,h2) * t2_2_4_1(h10,p5))
      (t2_2_5_1(h7,h10,h1,p9)     =        v2(h7,h10,h1,p9))

      (t2_2_5_1(h7,h10,h1,p9)    +=        t1(p5,h1) * v2(h7,h10,p5,p9))
      (t2_2_1(h10,p3,h1,h2)      +=        t2(p3,p9,h1,h7) * t2_2_5_1(h7,h10,h2,p9))
      (t2(p1,p2,h3,h4)           += 0.5  * t1(p1,h3) * t1(p2,h4))
      (t2_2_1(h10,p3,h1,h2)      += 0.5  * t2(p5,p6,h1,h2) * v2(h10,p3,p5,p6))
      (t2(p1,p2,h3,h4)           += -0.5 * t1(p1,h3) * t1(p2,h4))
      (i0(p3,p4,h1,h2)           += -1   * t1(p3,h10) * t2_2_1(h10,p4,h1,h2))

      (i0(p3,p4,h1,h2)           += -1   * t1(p5,h1) * v2(p3,p4,h2,p5))
      (t2_4_1(h9,h1)              =        f1(h9,h1))
      (t2_4_2_1(h9,p8)            =        f1(h9,p8))
      (t2_4_2_1(h9,p8)           +=        t1(p6,h7) * v2(h7,h9,p6,p8))
      (t2_4_1(h9,h1)             +=        t1(p8,h1) * t2_4_2_1(h9,p8))
      (t2_4_1(h9,h1)             += -1   * t1(p6,h7) * v2(h7,h9,h1,p6))
      (t2_4_1(h9,h1)             += -0.5 * t2(p6,p7,h1,h8) * v2(h8,h9,p6,p7))
      (i0(p3,p4,h1,h2)           += -1   * t2(p3,p4,h1,h9) * t2_4_1(h9,h2))
      (t2_5_1(p3,p5)              =        f1(p3,p5))
      (t2_5_1(p3,p5)             += -1   * t1(p6,h7) * v2(h7,p3,p5,p6))
      (t2_5_1(p3,p5)             += -0.5 * t2(p3,p6,h7,h8) * v2(h7,h8,p5,p6))
      (i0(p3,p4,h1,h2)           += 1    * t2(p3,p5,h1,h2) * t2_5_1(p4,p5))
      (t2_6_1(h9,h11,h1,h2)      += -1   * v2(h9,h11,h1,h2))
      (t2_6_2_1(h9,h11,h1,p8)     =        v2(h9,h11,h1,p8))
      (t2_6_2_1(h9,h11,h1,p8)    += 0.5  * t1(p6,h1) * v2(h9,h11,p6,p8))
      (t2_6_1(h9,h11,h1,h2)      +=        t1(p8,h1) * t2_6_2_1(h9,h11,h2,p8))
      (t2_6_1(h9,h11,h1,h2)      += -0.5 * t2(p5,p6,h1,h2) * v2(h9,h11,p5,p6))
      (i0(p3,p4,h1,h2)           += -0.5 * t2(p3,p4,h9,h11) * t2_6_1(h9,h11,h1,h2))
      (t2_7_1(h6,p3,h1,p5)       +=        v2(h6,p3,h1,p5))
      (t2_7_1(h6,p3,h1,p5)       += -1   * t1(p7,h1) * v2(h6,p3,p5,p7))
      (t2_7_1(h6,p3,h1,p5)       += -0.5 * t2(p3,p7,h1,h8) * v2(h6,h8,p5,p7))
      (i0(p3,p4,h1,h2)           += -1   * t2(p3,p5,h1,h6) * t2_7_1(h6,p4,h2,p5))
      (vt1t1_1(h5,p3,h1,h2)      += -2   * t1(p6,h1) * v2(h5,p3,h2,p6))
      (i0(p3,p4,h1,h2)           += -0.5 * t1(p3,h5) * vt1t1_1(h5,p4,h1,h2))
      (t2(p1,p2,h3,h4)           += 0.5  * t1(p1,h3) * t1(p2,h4))
      (i0(p3,p4,h1,h2)           += 0.5  * t2(p5,p6,h1,h2) * v2(p3,p4,p5,p6))
      (t2(p1,p2,h3,h4)           += -0.5 * t1(p1,h3) * t1(p2,h4))
      .dealloc(t2_2_1, t2_2_2_1, t2_2_2_2_1, t2_2_4_1, t2_2_5_1, t2_4_1, t2_4_2_1,
               t2_5_1, t2_6_1, t2_6_2_1, t2_7_1, vt1t1_1);
}

/**
 * ref, corr
 */
template<typename T>
double ccsd_driver(Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_f1, Tensor<T>& d_v2,
                   int maxiter, double thresh,
                   double zshiftl,
                   int ndiis,
                   double* p_evl_sorted,
                   ProcGroup pg,
                   Distribution* distribution,
                   MemoryManager* mgr) {
  Irrep irrep{0};
  bool spin_restricted = false;

  std::vector<Tensor<T>*> d_r1s, d_r2s;
      
  Tensor<T> d_e{E|E, irrep, spin_restricted};
  Tensor<T> d_r1_residual{E|E, irrep, spin_restricted};
  Tensor<T> d_r2_residual{E|E, irrep, spin_restricted};
  Tensor<T>::allocate(pg, distribution, mgr, d_e);
  for(int i=0; i<ndiis; i++) {
    d_r1s.push_back(new Tensor<T>{{V|O}, irrep, spin_restricted});
    d_r2s.push_back(new Tensor<T>{{VV|OO}, irrep, spin_restricted});
    Tensor<T>::allocate(pg, distribution, mgr, *d_r1s[i], *d_r2s[i]);
  }

  auto get_scalar = [] (Tensor<T>& tensor) -> T {
    Expects(tensor.rank() == 0);
    Block<T> resblock = tensor.get({});
    return *resblock.buf();    
  };

  double corr = 0;
  for(int titer=0; titer<maxiter; titer+=ndiis) {
    for(int iter = iter; iter < std::min(titer+ndiis,maxiter); iter++) {
      Scheduler sch{pg, distribution, mgr, irrep, spin_restricted};
      int off = iter - titer;
      sch.io(d_t1, d_t2, d_f1, d_v2)
          .output(d_e, *d_r1s[off], *d_r2s[off]);
      ccsd_e(sch, d_f1, d_e, d_t1, d_t2, d_v2);
      ccsd_t1(sch, d_f1, *d_r1s[off], d_t1, d_t2, d_v2);
      ccsd_t2(sch, d_f1, *d_r2s[off], d_t1, d_t2, d_v2);
      compute_residual(sch, *d_r1s[off], d_r1_residual);
      compute_residual(sch, *d_r2s[off], d_r2_residual);
    
      double r1 = get_scalar(d_r1_residual);
      double r2 = get_scalar(d_r2_residual);
      double residual = std::max(r1, r2);
      double energy = get_scalar(d_e);
      if(residual < thresh) {
        //nodezero_print();
        break;
      }
      jacobi(*d_r1s[off], d_t1, -1.0 * zshiftl, false, p_evl_sorted);
      jacobi(*d_r2s[off], d_t2, -2.0 * zshiftl, false, p_evl_sorted);
    }
    Scheduler sch{pg, distribution, mgr, irrep, spin_restricted};
    std::vector<std::vector<Tensor<T>*>*> rs{&d_r1s, &d_r2s};
    std::vector<Tensor<T>*> ts{&d_t1, &d_t2};
    diis<T>(sch, rs, ts);
  }

  for(int i=0; i<ndiis; i++) {
    Tensor<T>::deallocate(*d_r1s[i], *d_r2s[i]);
  }
  d_r1s.clear();
  d_r2s.clear();
  return corr;
}

int main() {
  using T = double;
  Irrep irrep{0};
  bool spin_restricted = false;
  Tensor<T> d_t1{V|O, irrep, spin_restricted};
  Tensor<T> d_t2{VV|OO, irrep, spin_restricted};
  Tensor<T> d_f1{N|N, irrep, spin_restricted};
  Tensor<T> d_v2{NN|NN, irrep, spin_restricted};
  int maxiter = 20;
  double thresh = 1.0e-6;
  double zshiftl = 1.0;
  int ndiis = 6;
  ccsd_driver(d_t1, d_t2, d_f1, d_v2,
              maxiter, thresh, zshiftl,
              ndiis, nullptr, ProcGroup{},
              nullptr, nullptr);
}
