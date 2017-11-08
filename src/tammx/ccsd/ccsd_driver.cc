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
#include <chrono>

#include "mpi.h"
#include "macdecls.h"
#include "ga.h"
#include "ga-mpi.h"

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
  auto mgr = MemoryManagerLocal(pg);
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
      .io(f1,v2,t1,t2)
      .output(de)
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
    .io(t1,t2,f1,v2)
    .output(i0)
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
      .io(t1,t2,f1,v2)
      .output(i0)
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

void iteration_print(const ProcGroup& pg, int iter, double residual, double energy) {
  if(pg.rank() == 0) {
    std::cout.width(6); std::cout << std::right << iter+1 << "  ";
    std::cout << std::setprecision(13) << residual << "  ";
    std::cout << std::fixed << std::setprecision(13) << energy << " ";
    std::cout << std::string(4, ' ') << "0.0";
    std::cout << std::string(5, ' ') << "0.0";
    std::cout << std::string(5, ' ') << "0.0" << std::endl;
  }
}

double iteration_summarize(const ProcGroup& pg,
                           int iter,
                           Tensor<double>& d_r1_residual,
                           Tensor<double>& d_r2_residual,
                           Tensor<double>& d_e) {
  auto get_scalar = [] (Tensor<double>& tensor) -> double {
    EXPECTS(tensor.rank() == 0);
    Block<double> resblock = tensor.get({});
    return *resblock.buf();
  };

  double r1 = 0.5*std::sqrt(get_scalar(d_r1_residual));
  double r2 = 0.5*std::sqrt(get_scalar(d_r2_residual));
  double residual = std::max(r1, r2);
  double energy = get_scalar(d_e);
  iteration_print(pg, iter, residual, energy);
  return residual;
}


template<typename T>
void
ccsd_iteration_scheduler(Scheduler& sch,
                         Tensor<T>& d_t1,
                         Tensor<T>& d_t2,
                         Tensor<T>& d_f1,
                         Tensor<T>& d_v2,
                         Tensor<T>& d_r1,
                         Tensor<T>& d_r2,
                         std::vector<T>& p_evl_sorted,
                         double zshiftl,
                         double& residual,
                         double& energy
                         ) {
  auto get_scalar = [] (Tensor<double>& tensor) -> double {
    EXPECTS(tensor.rank() == 0);
    Block<double> resblock = tensor.get({});
    return *resblock.buf();
  };

  Tensor<T>& d_t1_local = *sch.tensor<T>(V|O);
  auto& d_r1_residual = *sch.tensor<T>(E|E);
  auto& d_r2_residual = *sch.tensor<T>(E|E);
  auto& d_e = *sch.tensor<T>(E|E);

  sch
      .io(d_t1, d_t2, d_f1, d_v2, d_r1, d_r2)
      .alloc(d_t1_local,d_r1_residual, d_r2_residual, d_e)
      (d_t1_local() = d_t1())
      ;

  ccsd_e(sch, d_f1, d_e, d_t1_local, d_t2, d_v2);
  ccsd_t1(sch, d_f1, d_r1, d_t1_local, d_t2, d_v2);
  ccsd_t2(sch, d_f1, d_r2, d_t1_local, d_t2, d_v2);

  sch
      (d_r1_residual() = d_r1()  * d_r1())
      (d_r2_residual() = d_r2()  * d_r2())
      ( [&](const ProcGroup& ec_pg) {
        double r1 = 0.5*std::sqrt(get_scalar(d_r1_residual));
        double r2 = 0.5*std::sqrt(get_scalar(d_r2_residual));
        residual = std::max(r1, r2);
        energy = get_scalar(d_e);
      })
      ( [&](const ProcGroup& pg) {
        jacobi(pg, d_r1, d_t1, -1.0 * zshiftl, false, p_evl_sorted.data());
      })
      ( [&](const ProcGroup& pg) {
        jacobi(pg, d_r2, d_t2, -2.0 * zshiftl, false, p_evl_sorted.data());
      })
      .dealloc(d_t1_local, d_r1_residual, d_r2_residual, d_e)
      ;
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
                   int ndiis, double hf_energy) {
  std::cout.precision(15);
  // ,
  //   ProcGroup pg,
  //   Distribution* distribution,
  //   MemoryManager* mgr) {
  Irrep irrep{0};
  bool spin_restricted = false;

 long lo_offset, hi_offset;
 long int total_orbitals = 0;
 const auto &flindices = d_f1.flindices();

  BlockIndex blo, bhi;
  std::tie(blo, bhi) = tensor_index_range(flindices[0]);
  lo_offset = TCE::offset(blo);
  hi_offset = TCE::offset(bhi);
  total_orbitals += hi_offset - lo_offset;

  std::cout << "Total orbitals = " << total_orbitals << std::endl;
  std::vector<double> p_evl_sorted(total_orbitals);

  {
    auto lambda = [&] (const auto& blockid) {
      if(blockid[0] == blockid[1]) {
        auto block = d_f1.get(blockid);
        auto dim = d_f1.block_dims(blockid)[0].value();
        auto offset = d_f1.block_offset(blockid)[0].value();
        TAMMX_SIZE i=0;
        for(auto p = offset; p < offset + dim; p++,i++) {
          p_evl_sorted[p] = block.buf()[i*dim + i];
        }
      }
    };
    block_for(ec.pg(), d_f1(), lambda);
  }
  ec.pg().barrier();

  if(ec.pg().rank() == 0) {
    std::cout << "p_evl_sorted:" << '\n';
    for(auto p = 0; p < p_evl_sorted.size(); p++)
      std::cout << p_evl_sorted[p] << '\n';
  }

  if(ec.pg().rank() == 0) {
    std::cout << "\n\n";
    std::cout << " CCSD iterations" << std::endl;
    std::cout << std::string(66, '-') << std::endl;
    std::cout <<
        " Iter          Residuum       Correlation     Cpu    Wall    V2*C2"
              << std::endl;
    std::cout << std::string(66, '-') << std::endl;
  }

  std::vector<Tensor<T>*> d_r1s, d_r2s, d_t1s, d_t2s;

  for(int i=0; i<ndiis; i++) {
    d_r1s.push_back(new Tensor<T>{{V|O}, irrep, spin_restricted});
    d_r2s.push_back(new Tensor<T>{{VV|OO}, irrep, spin_restricted});
    d_t1s.push_back(new Tensor<T>{{V|O}, irrep, spin_restricted});
    d_t2s.push_back(new Tensor<T>{{VV|OO}, irrep, spin_restricted});
    ec.allocate(*d_r1s[i], *d_r2s[i], *d_t1s[i], *d_t2s[i]);
  }

  Tensor<T> d_r1{{V|O}, irrep, spin_restricted};
  Tensor<T> d_r2{{VV|OO}, irrep, spin_restricted};
  ec.allocate(d_r1, d_r2);

  auto get_scalar = [] (Tensor<T>& tensor) -> T {
    EXPECTS(tensor.rank() == 0);
    Block<T> resblock = tensor.get({});
    return *resblock.buf();
  };

  double corr = 0;
  double residual = 0.0;
  double energy = 0.0;

  for(int titer=0; titer<maxiter; titer+=ndiis) {
    for(int iter = titer; iter < std::min(titer+ndiis,maxiter); iter++) {
      int off = iter - titer;

      Scheduler sch = ec.scheduler();
      ccsd_iteration_scheduler(sch,
                               d_t1,
                               d_t2,
                               d_f1,
                               d_v2,
                               d_r1,
                               d_r2,
                               p_evl_sorted,
                               zshiftl,
                               residual,
                               energy);

      ec.scheduler()
          .io(d_t1, d_t2)
          .output(*d_t1s[off], *d_t2s[off])
          ((*d_t1s[off])() = d_t1())
          ((*d_t2s[off])() = d_t2())
          .execute();
      sch.execute();
      ec.scheduler()
          .io(d_r1, d_r2)
          .output(*d_r1s[off], *d_r2s[off])
          ((*d_r1s[off])() = d_r1())
          ((*d_r2s[off])() = d_r2())
          .execute();
      iteration_print(ec.pg(), iter, residual, energy);
      if(residual < thresh) {
        break;
      }
    }
    if(residual < thresh || titer+ndiis >= maxiter) {
      break;
    }
    if(ec.pg().rank() == 0) {
      std::cout << " MICROCYCLE DIIS UPDATE:";
      std::cout.width(21); std::cout << std::right << std::min(titer+ndiis,maxiter)+1;
      std::cout.width(21); std::cout << std::right << "5" << std::endl;
    }
    //Scheduler sch = ec.scheduler();//{pg, distribution, mgr, irrep, spin_restricted};
    std::vector<std::vector<Tensor<T>*>*> rs{&d_r1s, &d_r2s};
    std::vector<std::vector<Tensor<T>*>*> ts{&d_t1s, &d_t2s};
    std::vector<Tensor<T>*> next_t{&d_t1, &d_t2};
    // @fixme why not use brace-initiralizer instead of
    // intermediates? possibly use variadic templates?
    diis<T>(ec, rs, ts, next_t);
  }
  if(ec.pg().rank() == 0) {
    std::cout << std::string(66, '-') << std::endl;
    if(residual < thresh) {
      std::cout << " Iterations converged" << std::endl;
      std::cout.precision(15);
      std::cout << " CCSD correlation energy / hartree =" << std::setw(26) << std::right << energy << std::endl;
      std::cout << " CCSD total energy / hartree       =" << std::setw(26) <<  std::right << energy + hf_energy << std::endl;
    }
  }

  for(int i=0; i<ndiis; i++) {
    Tensor<T>::deallocate(*d_r1s[i], *d_r2s[i], *d_t1s[i], *d_t2s[i]);
  }
  d_r1s.clear();
  d_r2s.clear();
  ec.deallocate(d_r1, d_r2);
  //ec.deallocate(d_e, d_r1_residual, d_r2_residual);
  return corr;
}

Irrep operator "" _ir(unsigned long long int val) {
  return Irrep{strongnum_cast<Irrep::value_type>(val)};
}

Spin operator "" _sp(unsigned long long int val) {
  return Spin{strongnum_cast<Spin::value_type>(val)};
}

BlockIndex operator "" _bd(unsigned long long int val) {
  return BlockIndex{strongnum_cast<BlockIndex::value_type>(val)};
}


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

BlockIndex noa {1};
BlockIndex noab {2};
BlockIndex nva {1};
BlockIndex nvab {2};
bool spin_restricted = false;
Irrep irrep_f {0};
Irrep irrep_v {0};
Irrep irrep_t {0};
Irrep irrep_x {0};
Irrep irrep_y {0};


extern "C" {
void init_fortran_vars_(Integer *noa1, Integer *nob1, Integer *nva1,
                        Integer *nvb1, logical *intorb1, logical *restricted1,
                        Integer *spins, Integer *syms, Integer *ranges);
void finalize_fortran_vars_();
}


void fortran_init(TAMMX_INT32 noa, TAMMX_INT32 nob, TAMMX_INT32 nva, TAMMX_INT32 nvb, bool intorb, bool restricted,
                  const std::vector<TAMMX_INT32> &spins,
                  const std::vector<TAMMX_INT32> &syms,
                  const std::vector<TAMMX_INT32> &ranges) {
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


extern std::tuple<Tensor4D> two_four_index_transform(const TAMMX_SIZE ndocc, const TAMMX_SIZE nao,
                                                     const TAMMX_SIZE freeze_core, const TAMMX_SIZE,
                                                     const Matrix &C, Matrix &F,
                                                     libint2::BasisSet &shells);

int main(int argc, char *argv[]) {


  const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";

  Matrix C;
  Matrix F;
  Tensor4D V2;
  TAMMX_SIZE ov_alpha{0};
  TAMMX_SIZE freeze_core = 0;
  TAMMX_SIZE freeze_virtual = 0;

  double hf_energy{0.0};
  libint2::BasisSet shells;
  TAMMX_SIZE nao{0};

  //omp_set_num_threads(1);
  omp_set_num_threads(omp_get_max_threads());

  auto hf_t1 = std::chrono::high_resolution_clock::now();
  std::tie(ov_alpha, nao, hf_energy, shells) = hartree_fock(filename,C,F);
  auto hf_t2 = std::chrono::high_resolution_clock::now();

  double hf_time = std::chrono::duration_cast<std::chrono::seconds>((hf_t2 - hf_t1)).count();
  std::cout << "Time taken for Hartree-Fock: " << hf_time << " secs\n";

  hf_t1 = std::chrono::high_resolution_clock::now();
  std::tie(V2) = two_four_index_transform(ov_alpha, nao, freeze_core, freeze_virtual, C, F, shells);
  hf_t2 = std::chrono::high_resolution_clock::now();
  double two_4index_time = std::chrono::duration_cast<std::chrono::seconds>((hf_t2 - hf_t1)).count();
  std::cout << "Time taken for 2&4-index transforms: " << two_4index_time << " secs\n";


  TAMMX_SIZE ov_beta{nao-ov_alpha};

  std::cout << "ov_alpha,nao === " << ov_alpha << ":" << nao << std::endl;
  std::vector<TAMMX_SIZE> sizes = {ov_alpha-freeze_core, ov_alpha-freeze_core, ov_beta-freeze_virtual, ov_beta-freeze_virtual};

  std::cout << "sizes vector -- \n";
  for(auto x: sizes) std::cout << x << ", ";
  std::cout << "\n";

#include "tammx/mpi_checks.h"

int mpi_rank, provided;

#define USE_MPI_INIT_THREAD_MULTIPLE

#if defined(USE_MPI_INIT)

MPI_Init( &argc, &argv );
//MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );

#else

int requested = -1;

#  if defined(USE_MPI_INIT_THREAD_MULTIPLE)
requested = MPI_THREAD_MULTIPLE;
#  elif defined(USE_MPI_INIT_THREAD_SERIALIZED)
requested = MPI_THREAD_SERIALIZED;
#  elif defined(USE_MPI_INIT_THREAD_FUNNELED)
requested = MPI_THREAD_FUNNELED;
#  else
requested = MPI_THREAD_SINGLE;
#  endif

MPI_Init_thread( &argc, &argv, requested, &provided );
//MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );

if (provided>requested)
{
if (mpi_rank==0) printf("MPI_Init_thread returned %s instead of %s, but this is okay. \n",
                          MPI_THREAD_STRING(provided), MPI_THREAD_STRING(requested) );
}
if (provided<requested)
{
if (mpi_rank==0) printf("MPI_Init_thread returned %s instead of %s so the test will exit. \n",
                          MPI_THREAD_STRING(provided), MPI_THREAD_STRING(requested) );
MPI_Abort(MPI_COMM_WORLD, 1);
}

#endif

  //MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(MT_DBL, 8000000, 20000000);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  TCE::init(spins, spatials,sizes,
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
    std::vector<TAMMX_INT32> ispins, isyms, iranges;
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
  int maxiter = 50;
  double thresh = 1.0e-10;
  double zshiftl = 0.0;
  int ndiis = 5;

  ProcGroup pg{GA_MPI_Comm()};
  Distribution_NW distribution;
  auto mgr = MemoryManagerGA::create_coll(ProcGroup{GA_MPI_Comm()});

  Tensor<T>::allocate(&distribution, mgr, d_t1, d_t2, d_f1, d_v2);

  {
  ExecutionContext ec {pg, &distribution, mgr, Irrep{0}, false};

  ec.scheduler()
      .output(d_t1, d_t2)
      (d_t1() = 0)
      (d_t2() = 0)
    .execute();

  //Tensor Map
  block_parfor(ec.pg(), d_f1(), [&](auto& blockid) {
      auto block = d_f1.alloc(blockid);
    auto buf = block.buf();
    const auto& block_offset = block.block_offset();
    const auto& block_dims = block.block_dims();
    EXPECTS(block.tensor().rank() == 2);
    TAMMX_INT32 c = 0;
    for (auto i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
      for (auto j = block_offset[1]; j < block_offset[1] + block_dims[1];
           j++, c++) {
        buf[c] = F(i.value(), j.value());
      }
    }
    d_f1.put(blockid, block);
    });

  block_parfor(ec.pg(), d_v2(), [&](auto& blockid) {
      auto block = d_v2.alloc(blockid);
    auto buf = block.buf();
    const auto& block_offset = block.block_offset();
    const auto& block_dims = block.block_dims();
    EXPECTS(block.tensor().rank() == 4);
    TAMMX_INT32 c = 0;
    for (auto i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
      for (auto j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
        for (auto k = block_offset[2]; k < block_offset[2] + block_dims[2];
             k++) {
          for (auto l = block_offset[3]; l < block_offset[3] + block_dims[3];
               l++, c++) {
            buf[c] = V2(i.value(), j.value(), k.value(), l.value());
          }
        }
      }
    }
    d_v2.put(blockid, block);
    });

#if 1
  ccsd_driver(ec, d_t1, d_t2, d_f1, d_v2,
              maxiter, thresh, zshiftl,
              ndiis,hf_energy);
#endif

  Tensor<T>::deallocate(d_t1, d_t2, d_f1, d_v2);

  MemoryManagerGA::destroy_coll(mgr);
  }

  fortran_finalize();
  TCE::finalize();

  GA_Terminate();
  MPI_Finalize();
  return 0;
}
