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
#include <iomanip>
#include <map>
#include <vector>
#include <string>

#include "mpi.h"
#include "macdecls.h"
#include "ga.h"

#include "tammx/tammx.h"
#include "tammx/work.h"
#include "tammx/diis.h"
#include "tammx/memory_manager_ga.h"
#include "tammx/hartree_fock.h"

using namespace std;
using namespace tammx;
using namespace tammx::tensor_dims;
using namespace tammx::tensor_labels;

extern "C" {

void set_fort_vars_(Integer *int_mb_f, double *dbl_mb_f)
{
  MA::init(int_mb_f, dbl_mb_f);
  // tammx::int_mb_tammx = int_mb_f - 1;
  // tammx::dbl_mb_tammx = dbl_mb_f - 1;
}

}

#if 0
template<typename T>
void tensor_print(Tensor<T>& t)  {
  auto lambda = [&] (auto& val) {
    std::cout << val << '\n';
  };
  Irrep irrep{0};
  bool spin_restricted = false;
  auto distribution = Distribution_NW();

  auto pg = ProcGroup{MPI_COMM_WORLD};
  auto mgr = MemoryManagerSequential(pg);
  Scheduler sch{pg, &distribution, &mgr, irrep, spin_restricted};
  using LabeledTensorType = LabeledTensor<T>;
  using Func = decltype(lambda);
  sch.io(t)
      // .template sop<Func, LabeledTensorType, 0>(t(), lambda)
    .sop(t(), lambda)
    .execute();
}
#endif

template<typename T>
void compute_residual(Scheduler &sch, Tensor<T>& tensor, Tensor<T>& scalar) {
  sch(scalar() = tensor() * tensor());
}


template<typename T>
void ccsd_e(Scheduler &sch, Tensor<T>& f1, Tensor<T>& de,
            Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& v2) {
  auto &i1 = *sch.tensor<T>(O|V);
  sch.alloc(i1)
      // .io(f1,v2,t1,t2)
      // .output(de)
      (i1(h6,p5) =        f1(h6,p5))
      (i1(h6,p5) += 0.5  * t1(p3,h4)       * v2(h4,h6,p3,p5))
      (de() = 0)
      (de()      +=        t1(p5,h6)       * i1(h6,p5))
      (de()      += 0.25 * t2(p1,p2,h3,h4) * v2(h3,h4,p1,p2))
    .dealloc(i1);

}


extern "C" {
add_fn ccsd_t1_1_, ccsd_t1_2_1_, ccsd_t1_2_2_1_, ccsd_t1_3_1_;  // ccsd_t1
add_fn ccsd_t1_5_1_, ccsd_t1_6_1_;  // ccsd_t1
add_fn ccsd_t2_1_, ccsd_t2_2_1_, ccsd_t2_2_2_1_;  // ccsd_t2
add_fn ccsd_t2_2_2_2_1_, ccsd_t2_2_4_1_, ccsd_t2_2_5_1_;  // ccsd_t2
add_fn ccsd_t2_4_1_, ccsd_t2_4_2_1_, ccsd_t2_5_1_;  // ccsd_t2
add_fn ccsd_t2_6_1_, ccsd_t2_6_2_1_, ccsd_t2_7_1_;  // ccsd_t2

mult_fn ccsd_t1_2_2_2_, ccsd_t1_2_2_, ccsd_t1_2_3_, ccsd_t1_2_4_;
mult_fn ccsd_t1_2_, ccsd_t1_3_2_, ccsd_t1_3_;
mult_fn ccsd_t1_4_, ccsd_t1_5_2_, ccsd_t1_5_;
mult_fn ccsd_t1_6_2_, ccsd_t1_6_, ccsd_t1_7_;

mult_fn ccsd_t2_2_2_2_2_,ccsd_t2_2_2_2_,ccsd_t2_2_2_3_,ccsd_t2_2_2_,
  ccsd_t2_2_4_2_,ccsd_t2_2_4_,ccsd_t2_2_5_2_,ccsd_t2_2_5_,c2f_t2_t12_,
  c2d_t2_t12_,ccsd_t2_2_6_,ccsd_t2_2_,lccsd_t2_3x_,ccsd_t2_4_2_2_,ccsd_t2_4_2_,
  ccsd_t2_4_3_,ccsd_t2_4_4_,ccsd_t2_4_,ccsd_t2_5_2_,ccsd_t2_5_3_,ccsd_t2_5_,
  ccsd_t2_6_2_2_,ccsd_t2_6_2_,ccsd_t2_6_3_,ccsd_t2_6_,ccsd_t2_7_2_,ccsd_t2_7_3_,
  ccsd_t2_7_,vt1t1_1_2_,vt1t1_1_,ccsd_t2_8_;

static const auto sch = ExecutionMode::sch;
static const auto fortran = ExecutionMode::fortran;

}

static const auto e_sch = ExecutionMode::sch;
static const auto e_fortran = ExecutionMode::fortran;

template<typename T>
void ccsd_t1(Scheduler& sch, Tensor<T>& f1, Tensor<T>& i0,
             Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& v2) {
  auto &t1_2_1 = *sch.tensor<T>(O|O);
  auto &t1_2_2_1 = *sch.tensor<T>(O|V);
  auto &t1_3_1 = *sch.tensor<T>(V|V);
  auto &t1_5_1 = *sch.tensor<T>(O|V);
  auto &t1_6_1 = *sch.tensor<T>(OO|OV);

  sch.alloc(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1)
    // .io(t1,t2,f1,v2)
    // .output(i0)
      (ccsd_t1_1_    |= i0(p2,h1)            =        f1(p2,h1))
      (ccsd_t1_2_1_  |= t1_2_1(h7,h1)        =        f1(h7,h1))
      (ccsd_t1_2_2_1_|= t1_2_2_1(h7,p3)      =        f1(h7,p3))
      (ccsd_t1_2_2_2_ |= t1_2_2_1(h7,p3)     += -1   * t1(p5,h6)       * v2(h6,h7,p3,p5))
      (ccsd_t1_2_2_  |= t1_2_1(h7,h1)       +=        t1(p3,h1)       * t1_2_2_1(h7,p3))
      (ccsd_t1_2_3_  |= t1_2_1(h7,h1)       += -1   * t1(p4,h5)       * v2(h5,h7,h1,p4))
      (ccsd_t1_2_4_  |= t1_2_1(h7,h1)       += -0.5 * t2(p3,p4,h1,h5) * v2(h5,h7,p3,p4))
      (ccsd_t1_2_    |= i0(p2,h1)           += -1   * t1(p2,h7)       * t1_2_1(h7,h1))
      (ccsd_t1_3_1_  |= t1_3_1(p2,p3)        =        f1(p2,p3))
      (ccsd_t1_3_2_  |= t1_3_1(p2,p3)       += -1   * t1(p4,h5)       * v2(h5,p2,p3,p4))
      (ccsd_t1_3_    |= i0(p2,h1)           +=        t1(p3,h1)       * t1_3_1(p2,p3))
      (ccsd_t1_4_    |= i0(p2,h1)           += -1   * t1(p3,h4)       * v2(h4,p2,h1,p3))
      (ccsd_t1_5_1_  |= t1_5_1(h8,p7)        =        f1(h8,p7))

      (ccsd_t1_5_2_  |= t1_5_1(h8,p7)       +=        t1(p5,h6)       * v2(h6,h8,p5,p7))
      (ccsd_t1_5_    |= i0(p2,h1)           +=        t2(p2,p7,h1,h8) * t1_5_1(h8,p7))
      (ccsd_t1_6_1_  |= t1_6_1(h4,h5,h1,p3)  =        v2(h4,h5,h1,p3))
      (ccsd_t1_6_2_  |= t1_6_1(h4,h5,h1,p3) += -1   * t1(p6,h1)       * v2(h4,h5,p3,p6))
      (ccsd_t1_6_    |= i0(p2,h1)           += -0.5 * t2(p2,p3,h4,h5) * t1_6_1(h4,h5,h1,p3))
      (ccsd_t1_7_    |= i0(p2,h1)           += -0.5 * t2(p3,p4,h1,h5) * v2(h5,p2,p3,p4))
    .dealloc(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1);
}

template<typename T>
void ccsd_t2(Scheduler& sch, Tensor<T>& f1, Tensor<T>& i0,
             Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& v2) {
  auto &t2_2_1 = *sch.tensor<T>(OV|OO);
  auto &t2_2_2_1 = *sch.tensor<T>(OO|OO);
  auto &t2_2_2_2_1 = *sch.tensor<T>(OO|OV);
  auto &t2_2_4_1 = *sch.tensor<T>(O|V);
  auto &t2_2_5_1 = *sch.tensor<T>(OO|OV);
  auto &t2_4_1 = *sch.tensor<T>(O|O);
  auto &t2_4_2_1 = *sch.tensor<T>(O|V);
  auto &t2_5_1 = *sch.tensor<T>(V|V);
  auto &t2_6_1 = *sch.tensor<T>(OO|OO);
  auto &t2_6_2_1 = *sch.tensor<T>(OO|OV);
  auto &t2_7_1 = *sch.tensor<T>(OV|OV);
  auto &vt1t1_1 = *sch.tensor<T>(OV|OO);

  sch.alloc(t2_2_1, t2_2_2_1, t2_2_2_2_1, t2_2_4_1, t2_2_5_1, t2_4_1, t2_4_2_1,
            t2_5_1, t2_6_1, t2_6_2_1, t2_7_1, vt1t1_1)
            // .io(t1,t2,f1,v2)
            // .output(i0)
      (ccsd_t2_1_     |= i0(p3,p4,h1,h2)            =        v2(p3,p4,h1,h2))
      (ccsd_t2_2_1_   |= t2_2_1(h10,p3,h1,h2)       =        v2(h10,p3,h1,h2))
      (ccsd_t2_2_2_1_ |= t2_2_2_1(h10,h11,h1,h2)    = -1  *  v2(h10,h11,h1,h2))
      (ccsd_t2_2_2_2_1_ |= t2_2_2_2_1(h10,h11,h1,p5)  =         v2(h10,h11,h1,p5))
      (ccsd_t2_2_2_2_2_ |= t2_2_2_2_1(h10,h11,h1,p5) += -0.5 * t1(p6,h1) * v2(h10,h11,p5,p6))
      (ccsd_t2_2_2_2_  |= t2_2_2_1(h10,h11,h1,h2)   +=        t1(p5,h1) * t2_2_2_2_1(h10,h11,h2,p5))
      (ccsd_t2_2_2_3_ |= t2_2_2_1(h10,h11,h1,h2)   += -0.5 * t2(p7,p8,h1,h2) * v2(h10,h11,p7,p8))
      (ccsd_t2_2_2_ |= t2_2_1(h10,p3,h1,h2)      += 0.5  * t1(p3,h11) * t2_2_2_1(h10,h11,h1,h2))
      (ccsd_t2_2_4_1_ |= t2_2_4_1(h10,p5)           =        f1(h10,p5))
      (ccsd_t2_2_4_2_ |= t2_2_4_1(h10,p5)          += -1   * t1(p6,h7) * v2(h7,h10,p5,p6))
      (ccsd_t2_2_4_ |= t2_2_1(h10,p3,h1,h2)      += -1   * t2(p3,p5,h1,h2) * t2_2_4_1(h10,p5))
      (ccsd_t2_2_5_1_ |= t2_2_5_1(h7,h10,h1,p9)     =        v2(h7,h10,h1,p9))
      (ccsd_t2_2_5_2_ |= t2_2_5_1(h7,h10,h1,p9)    +=        t1(p5,h1) * v2(h7,h10,p5,p9))
      (ccsd_t2_2_5_ |= t2_2_1(h10,p3,h1,h2)      +=        t2(p3,p9,h1,h7) * t2_2_5_1(h7,h10,h2,p9))
      (/*c2f_t2_t12_ |= */ t2(p1,p2,h3,h4)           += 0.5  * t1(p1,h3) * t1(p2,h4))
      (ccsd_t2_2_6_ |= t2_2_1(h10,p3,h1,h2)      += 0.5  * t2(p5,p6,h1,h2) * v2(h10,p3,p5,p6))
      (/*c2d_t2_t12_ |= */ t2(p1,p2,h3,h4)           += -0.5 * t1(p1,h3) * t1(p2,h4))
      (ccsd_t2_2_ |= i0(p3,p4,h1,h2)           += -1   * t1(p3,h10) * t2_2_1(h10,p4,h1,h2))
      (lccsd_t2_3x_ |= i0(p3,p4,h1,h2)           += -1   * t1(p5,h1) * v2(p3,p4,h2,p5))
      (ccsd_t2_4_1_ |= t2_4_1(h9,h1)              =        f1(h9,h1))
      (ccsd_t2_4_2_1_ |= t2_4_2_1(h9,p8)            =        f1(h9,p8))
      (ccsd_t2_4_2_2_ |= t2_4_2_1(h9,p8)           +=        t1(p6,h7) * v2(h7,h9,p6,p8))
      (ccsd_t2_4_2_ |= t2_4_1(h9,h1)             +=        t1(p8,h1) * t2_4_2_1(h9,p8))
      (ccsd_t2_4_3_ |= t2_4_1(h9,h1)             += -1   * t1(p6,h7) * v2(h7,h9,h1,p6))
      (ccsd_t2_4_4_ |= t2_4_1(h9,h1)             += -0.5 * t2(p6,p7,h1,h8) * v2(h8,h9,p6,p7))
      (ccsd_t2_4_ |= i0(p3,p4,h1,h2)           += -1   * t2(p3,p4,h1,h9) * t2_4_1(h9,h2))
      (ccsd_t2_5_1_ |= t2_5_1(p3,p5)              =        f1(p3,p5))
      (ccsd_t2_5_2_ |= t2_5_1(p3,p5)             += -1   * t1(p6,h7) * v2(h7,p3,p5,p6))
      (ccsd_t2_5_3_ |= t2_5_1(p3,p5)             += -0.5 * t2(p3,p6,h7,h8) * v2(h7,h8,p5,p6))
      (ccsd_t2_5_ |= i0(p3,p4,h1,h2)           += 1    * t2(p3,p5,h1,h2) * t2_5_1(p4,p5))
      (ccsd_t2_6_1_ |= t2_6_1(h9,h11,h1,h2)       = -1   * v2(h9,h11,h1,h2))
      (ccsd_t2_6_2_1_ |= t2_6_2_1(h9,h11,h1,p8)     =        v2(h9,h11,h1,p8))
      (ccsd_t2_6_2_2_ |= t2_6_2_1(h9,h11,h1,p8)    += 0.5  * t1(p6,h1) * v2(h9,h11,p6,p8))
      (ccsd_t2_6_2_ |= t2_6_1(h9,h11,h1,h2)      +=        t1(p8,h1) * t2_6_2_1(h9,h11,h2,p8))
      (ccsd_t2_6_3_ |= t2_6_1(h9,h11,h1,h2)      += -0.5 * t2(p5,p6,h1,h2) * v2(h9,h11,p5,p6))
      (ccsd_t2_6_ |= i0(p3,p4,h1,h2)           += -0.5 * t2(p3,p4,h9,h11) * t2_6_1(h9,h11,h1,h2))
      (ccsd_t2_7_1_ |= t2_7_1(h6,p3,h1,p5)        =        v2(h6,p3,h1,p5))
      (ccsd_t2_7_2_ |= t2_7_1(h6,p3,h1,p5)       += -1   * t1(p7,h1) * v2(h6,p3,p5,p7))
      (ccsd_t2_7_3_ |= t2_7_1(h6,p3,h1,p5)       += -0.5 * t2(p3,p7,h1,h8) * v2(h6,h8,p5,p7))
      (ccsd_t2_7_ |= i0(p3,p4,h1,h2)           += -1   * t2(p3,p5,h1,h6) * t2_7_1(h6,p4,h2,p5))
      ( vt1t1_1(h5,p3,h1,h2)       = 0)
      (vt1t1_1_2_ |= vt1t1_1(h5,p3,h1,h2)      += -2   * t1(p6,h1) * v2(h5,p3,h2,p6))
      (vt1t1_1_ |= i0(p3,p4,h1,h2)           += -0.5 * t1(p3,h5) * vt1t1_1(h5,p4,h1,h2))
      (/*c2f_t2_t12_ |= */ t2(p1,p2,h3,h4)           += 0.5  * t1(p1,h3) * t1(p2,h4))
      (/*ccsd_t2_8_ |= */ i0(p3,p4,h1,h2)           += 0.5  * t2(p5,p6,h1,h2) * v2(p3,p4,p5,p6))
      (/*c2d_t2_t12_ |= */ t2(p1,p2,h3,h4)           += -0.5 * t1(p1,h3) * t1(p2,h4))
    .dealloc(t2_2_1, t2_2_2_1, t2_2_2_2_1, t2_2_4_1, t2_2_5_1, t2_4_1, t2_4_2_1,
             t2_5_1, t2_6_1, t2_6_2_1, t2_7_1, vt1t1_1);
}

/**
 * ref, corr
 */
template<typename T>
double ccsd_driver(ExecutionContext& ec,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_f1, Tensor<T>& d_v2,
                   int maxiter, double thresh,
                   double zshiftl,
                   int ndiis) {
  std::cout.precision(15);
  // ,
  //   ProcGroup pg,
  //   Distribution* distribution,
  //   MemoryManager* mgr) {
  Irrep irrep{0};
  bool spin_restricted = false;
  std::vector<double> p_evl_sorted;

 long ndim = d_f1.rank();
 long lo_offset[ndim], hi_offset[ndim];
 long int total_orbitals = 0;
 const auto &flindices = d_f1.flindices();

 for (long i = 0; i < ndim; i++) {
  BlockDim blo, bhi;
  std::tie(blo, bhi) = tensor_index_range(flindices[i]);
  lo_offset[i] = TCE::offset(blo);
  hi_offset[i] = TCE::offset(bhi);
  total_orbitals += hi_offset[i] - lo_offset[i];
 }

  //p_evl_sorted.reserve(total_orbitals);
  // ec->sop_execute(d_f1, [&] (auto p, auto q, auto& val) {
  //     if(p == q) {
  //       p_evl_sorted.push_back(val);
  //     }
  //   });
  // {
  //   auto lambda = [&] (auto p, auto q, auto& val) {
  //     if(p == q) {
  //       p_evl_sorted.push_back(val);
  //     }
  //   };
  //   Scheduler sch{pg, distribution, mgr, irrep, spin_restricted};
  //   using LabeledTensorType = LabeledTensor<T>;
  //   using Func = decltype(lambda);
  //   sch.io(d_f1)
  //       // .template sop<Func, LabeledTensorType, 2>(d_f1(), lambda)
  //       .sop(d_f1(), lambda)
  //       .execute();
  // }
{
    p_evl_sorted.resize(total_orbitals);
    auto lambda = [&] (const auto& blockid) {
      if(blockid[0] == blockid[1]) {
        auto block = d_f1.get(blockid);
        auto dim = d_f1.block_dims(blockid)[0].value();
        auto offset = d_f1.block_offset(blockid)[0].value();
        size_t i=0;
        for(auto p = offset; p < offset + dim; p++,i++) {
          p_evl_sorted[p] = block.buf()[i*dim + i];
        }
      }
    };
    block_for(d_f1(), lambda);


std::cout << "p_evl_sorted:" << '\n';
  for(auto p = 0; p < p_evl_sorted.size(); p++)
      std::cout << p_evl_sorted[p] << '\n';
}

  std::vector<Tensor<T>*> d_r1s, d_r2s;

  Tensor<T> d_e{E|E, irrep, spin_restricted};
  Tensor<T> d_r1_residual{E|E, irrep, spin_restricted};
  Tensor<T> d_r2_residual{E|E, irrep, spin_restricted};
  ec.allocate(d_e, d_r1_residual, d_r2_residual);
  for(int i=0; i<ndiis; i++) {
    d_r1s.push_back(new Tensor<T>{{V|O}, irrep, spin_restricted});
    d_r2s.push_back(new Tensor<T>{{VV|OO}, irrep, spin_restricted});
    ec.allocate(*d_r1s[i], *d_r2s[i]);
  }

  //void Tensor<T>::operator = (std::pair<Tensor<T>, Tensor<T>> rhs);

  auto get_scalar = [] (Tensor<T>& tensor) -> T {
    EXPECTS(tensor.rank() == 0);
    Block<T> resblock = tensor.get({});
    return *resblock.buf();
  };

  std::cout << "debug ccsd 1\n";

  double corr = 0;
  double residual = 0.0;
  double energy = 0.0;
  for(int titer=0; titer<maxiter; titer+=ndiis) {
    for(int iter = titer; iter < std::min(titer+ndiis,maxiter); iter++) {
      std::cerr<<"++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
      int off = iter - titer;

      Tensor<T> d_t1_local(d_t1.tindices(), 1, Irrep{0}, ec.is_spin_restricted());
      MemoryManagerSequential mseq{ProcGroup{MPI_COMM_SELF}};
      d_t1_local.alloc(ProcGroup{MPI_COMM_SELF},
                       ec.distribution(),
                       &mseq);

      Scheduler sch = ec.scheduler();//{pg, distribution, mgr, irrep, spin_restricted};
      sch.io(d_t1_local, d_t1, d_t2, d_f1, d_v2, *d_r1s[off], *d_r2s[off])
        .output(d_e, d_r1_residual, d_r2_residual);

      sch(d_t1_local() = d_t1());

      ccsd_e(sch, d_f1, d_e, d_t1_local, d_t2, d_v2);
      ccsd_t1(sch, d_f1, *d_r1s[off], d_t1_local, d_t2, d_v2);
      ccsd_t2(sch, d_f1, *d_r2s[off], d_t1_local, d_t2, d_v2);

      sch(d_r1_residual() = 0)
        (d_r1_residual() += (*d_r1s[off])()  * (*d_r1s[off])())
        (d_r2_residual() = 0)
        (d_r2_residual() += (*d_r2s[off])()  * (*d_r2s[off])())
        ;
      sch.execute();
      d_t1_local.dealloc();

      // std::cout << "------------print d_f1-------------------\n";
      // tensor_print(d_f1);
      // std::cout << "------------print i1-------------------\n";
      // tensor_print(*d_r1s[off]);
      // std::cout << "------------end print i1--------------\n";

      // std::cerr<<"----------------------------------------------"<<std::endl;

      double r1 = 0.5*std::sqrt(get_scalar(d_r1_residual));
      double r2 = 0.5*std::sqrt(get_scalar(d_r2_residual));
      residual = std::max(r1, r2);
      energy = get_scalar(d_e);
      std::cout << "iteration:" << iter << '\n';
      std::cout << "r1=" << r1 <<" r2="<<r2 << '\n';
      // tensor_print(*d_r1s[off]);
      // tensor_print(d_r1_residual);
      std::cout << std::setprecision(15) << "residual:" << residual << '\n';
      std::cout << std::setprecision(15) << "energy:" << energy << '\n';
      if(residual < thresh) {
        //nodezero_print();
        break;
      }
      jacobi(*d_r1s[off], d_t1, -1.0 * zshiftl, false, p_evl_sorted.data());
      jacobi(*d_r2s[off], d_t2, -2.0 * zshiftl, false, p_evl_sorted.data());
    }
    if(residual < thresh) {
      //nodezero_print();
      break;
    }
    Scheduler sch = ec.scheduler();//{pg, distribution, mgr, irrep, spin_restricted};
    std::vector<std::vector<Tensor<T>*>*> rs{&d_r1s, &d_r2s};
    std::vector<Tensor<T>*> ts{&d_t1, &d_t2};
    // @fixme why not use brace-initiralizer instead of
    // intermediates? possibly use variadic templates?
    diis<T>(sch, rs, ts);
  }

  std::cout << "debug ccsd 2\n";
  for(int i=0; i<ndiis; i++) {
    Tensor<T>::deallocate(*d_r1s[i], *d_r2s[i]);
  }
  d_r1s.clear();
  d_r2s.clear();
  ec.deallocate(d_e, d_r1_residual, d_r2_residual);
  return corr;
}

Irrep operator "" _ir(unsigned long long int val) {
  return Irrep{strongnum_cast<Irrep::value_type>(val)};
}

Spin operator "" _sp(unsigned long long int val) {
  return Spin{strongnum_cast<Spin::value_type>(val)};
}

BlockDim operator "" _bd(unsigned long long int val) {
  return BlockDim{strongnum_cast<BlockDim::value_type>(val)};
}

//std::vector<Spin> spins = {1_sp, 2_sp, 1_sp, 2_sp};
//std::vector<Irrep> spatials = {0_ir, 0_ir, 0_ir, 0_ir};
//std::vector<size_t> sizes = {5,5,2,2};
//BlockDim noa {1};
//BlockDim noab {2};
//BlockDim nva {1};
//BlockDim nvab {2};
//bool spin_restricted = false;
//Irrep irrep_f {0};
//Irrep irrep_v {0};
//Irrep irrep_t {0};
//Irrep irrep_x {0};
//Irrep irrep_y {0};


//std::vector<Spin> spins = {1_sp, 1_sp, 1_sp,
//                           2_sp, 2_sp, 2_sp,
//                           1_sp, 1_sp,
//                           2_sp, 2_sp};
std::vector<Spin> spins = {1_sp, 2_sp,
                           1_sp, 2_sp};
//std::vector<Irrep> spatials = {0_ir, 0_ir, 0_ir,
//                               0_ir, 0_ir, 0_ir,
//                               0_ir, 0_ir,
//                               0_ir, 0_ir};
std::vector<Irrep> spatials = {0_ir, 0_ir,
                               0_ir, 0_ir};
//std::vector<size_t> sizes = {3,1,1, 3,1,1, 1,1, 1,1};
std::vector<size_t> sizes = {5,5, 2,2};
BlockDim noa {1};
BlockDim noab {2};
BlockDim nva {1};
BlockDim nvab {2};
bool spin_restricted = false;
Irrep irrep_f {0};
Irrep irrep_v {0};
Irrep irrep_t {0};
Irrep irrep_x {0};
Irrep irrep_y {0};


// void fortran_init(int noa, int nob, int nva, int nvb, bool intorb, bool restricted,
//                   const std::vector<int> &spins,
//                   const std::vector<int> &syms,
//                   const std::vector<int> &ranges);
// void fortran_finalize();

extern "C" {
void init_fortran_vars_(Integer *noa1, Integer *nob1, Integer *nva1,
                        Integer *nvb1, logical *intorb1, logical *restricted1,
                        Integer *spins, Integer *syms, Integer *ranges);
void finalize_fortran_vars_();
}


void fortran_init(int noa, int nob, int nva, int nvb, bool intorb, bool restricted,
                  const std::vector<int> &spins,
                  const std::vector<int> &syms,
                  const std::vector<int> &ranges) {
  Integer inoa = noa;
  Integer inob = nob;
  Integer inva = nva;
  Integer invb = nvb;

  logical lintorb = intorb ? 1 : 0;
  logical lrestricted = restricted ? 1 : 0;

  assert(spins.size() == noa + nob + nva + nvb);
  assert(syms.size() == noa + nob + nva + nvb);
  assert(ranges.size() == noa + nob + nva + nvb);

  Integer ispins[noa + nob + nvb + nvb];
  Integer isyms[noa + nob + nvb + nvb];
  Integer iranges[noa + nob + nvb + nvb];

  std::copy_n(&spins[0], noa + nob + nva + nvb, &ispins[0]);
  std::copy_n(&syms[0], noa + nob + nva + nvb, &isyms[0]);
  std::copy_n(&ranges[0], noa + nob + nva + nvb, &iranges[0]);

  init_fortran_vars_(&inoa, &inob, &inva, &invb, &lintorb, &lrestricted,
                     &ispins[0], &isyms[0], &iranges[0]);
}

void fortran_finalize() {
  finalize_fortran_vars_();
}


extern std::tuple<Tensor4D> two_four_index_transform(const int ndocc, const Matrix &C, Matrix &F, libint2::BasisSet &shells);

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(MT_DBL, 8000000, 20000000);

  TCE::init(spins, spatials, sizes,
            noa,
            noab,
            nva,
            nvab,
            spin_restricted,
            irrep_f,
            irrep_v,
            irrep_t,
            irrep_x,
            irrep_y);
  {
    bool intorb = false;
    std::vector<int> ispins, isyms, iranges;
    for(auto x: spins) {
      ispins.push_back(x.value());
    }
    for(auto x: spatials) {
      isyms.push_back(x.value());
    }
    for(auto x: sizes) {
      iranges.push_back(x);
    }
    fortran_init(noa.value(), noab.value()-noa.value(),
                 nva.value(), nvab.value()-nva.value(),
                 intorb, spin_restricted, ispins, isyms, iranges);
  }

  using T = double;
  Irrep irrep{0};
  bool spin_restricted = false;
  Tensor<T> d_t1{V|O, irrep, spin_restricted};
  Tensor<T> d_t2{VV|OO, irrep, spin_restricted};
  Tensor<T> d_f1{N|N, irrep, spin_restricted};
  Tensor<T> d_v2{NN|NN, irrep, spin_restricted};
  int maxiter = 100;
  double thresh = 1.0e-10;
  double zshiftl = 0.0;
  int ndiis = 1005;


  auto distribution = Distribution_NW();
  auto pg = ProcGroup{MPI_COMM_WORLD};
  auto mgr = new MemoryManagerGA{pg};

  Tensor<T>::allocate(pg, &distribution, mgr, d_t1, d_t2, d_f1, d_v2);

  const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";

  Matrix C;
  Matrix F;
  Tensor4D V;
  int ndocc{0};
  double hf_energy{0.0};
  libint2::BasisSet shells;

  std::tie(ndocc, hf_energy, shells) = hartree_fock(filename,C,F);
  std::tie(V) = two_four_index_transform(ndocc, C, F, shells);
  std::cerr << "debug2" << '\n';

  //Tensor Map
  tensor_map(d_f1(), [&](auto& block) {
    auto buf = block.buf();
    const auto& block_offset = block.block_offset();
    const auto& block_dims = block.block_dims();
    // std::cout << "block offset:" << block_offset << '\n';
    // std::cout << "block dims:" << block_dims << '\n';
    // std::cout << "block size:" << block.size() << '\n';
    EXPECTS(block.tensor().rank() == 2);
    int c = 0;
    for (auto i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
      for (auto j = block_offset[1]; j < block_offset[1] + block_dims[1];
           j++, c++) {
        buf[c] = F(i.value(), j.value());
      }
    }
  });

  // tensor_print(d_f1);
  // std::cerr << "debug1" << '\n';

  tensor_map(d_v2(), [&](auto& block) {
    auto buf = block.buf();
    const auto& block_offset = block.block_offset();
    const auto& block_dims = block.block_dims();
    EXPECTS(block.tensor().rank() == 4);
    int c = 0;
    for (auto i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
      for (auto j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
        for (auto k = block_offset[2]; k < block_offset[2] + block_dims[2];
             k++) {
          for (auto l = block_offset[3]; l < block_offset[3] + block_dims[3];
               l++, c++) {
            buf[c] = V(i.value(), j.value(), k.value(), l.value());
          }
        }
      }
    }
  });

  std::cerr << "debug3" << '\n';
  ExecutionContext ec {pg, &distribution, mgr, Irrep{0}, false};

  // Scheduler(pg, &distribution, &mgr, Irrep{0}, false)
  ec.scheduler()
    .output(d_t1, d_t2)
      (d_t1() = 0)
      (d_t2() = 0)
    .execute();

  //end tensor map

  ccsd_driver(ec, d_t1, d_t2, d_f1, d_v2,
              maxiter, thresh, zshiftl,
              ndiis);

  std::cerr << "debug4" << '\n';
  // pg,
  //             &distribution, &mgr);
  Tensor<T>::deallocate(d_t1, d_t2, d_f1, d_v2);
  fortran_finalize();
  TCE::finalize();

  GA_Terminate();
  MPI_Finalize();
  return 0;
}
