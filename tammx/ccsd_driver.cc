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

#include "mpi.h"

#include "tammx/tammx.h"
#include "tammx/work.h"
#include "tammx/diis.h"

using namespace std;
using namespace tammx;
using namespace tammx::tensor_dims;
using namespace tammx::tensor_labels;

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

template<typename T>
void compute_residual(Scheduler &sch, Tensor<T>& tensor, Tensor<T>& scalar) {
  sch(scalar() = tensor() * tensor());
}


template<typename T>
void ccsd_e(Scheduler &sch, Tensor<T>& f1, Tensor<T>& de,
            Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& v2) {
  auto &i1 = sch.tensor<T>(O|V);

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

template<typename T>
void ccsd_t1(Scheduler& sch, Tensor<T>& f1, Tensor<T>& i0,
             Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& v2) {
  auto &t1_2_1 = sch.tensor<T>(O|O);
  auto &t1_2_2_1 = sch.tensor<T>(O|V);
  auto &t1_3_1 = sch.tensor<T>(V|V);
  auto &t1_5_1 = sch.tensor<T>(O|V);
  auto &t1_6_1 = sch.tensor<T>(OO|OV);

  sch.alloc(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1)
    .io(t1,t2,f1,v2)
    .output(i0)
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
            .io(t1,t2,f1,v2)
            .output(i0)
      (i0(p3,p4,h1,h2)            =        v2(p3,p4,h1,h2))
      (t2_2_1(h10,p3,h1,h2)       =        v2(h10,p3,h1,h2))
      (t2_2_2_1(h10,h11,h1,h2)    = -1  *  v2(h10,h11,h1,h2))
      (t2_2_2_2_1(h10,h11,h1,p5)  =         v2(h10,h11,h1,p5))
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
      (t2_6_1(h9,h11,h1,h2)       = -1   * v2(h9,h11,h1,h2))
      (t2_6_2_1(h9,h11,h1,p8)     =        v2(h9,h11,h1,p8))
      (t2_6_2_1(h9,h11,h1,p8)    += 0.5  * t1(p6,h1) * v2(h9,h11,p6,p8))
      (t2_6_1(h9,h11,h1,h2)      +=        t1(p8,h1) * t2_6_2_1(h9,h11,h2,p8))
      (t2_6_1(h9,h11,h1,h2)      += -0.5 * t2(p5,p6,h1,h2) * v2(h9,h11,p5,p6))
      (i0(p3,p4,h1,h2)           += -0.5 * t2(p3,p4,h9,h11) * t2_6_1(h9,h11,h1,h2))
      (t2_7_1(h6,p3,h1,p5)        =        v2(h6,p3,h1,p5))
      (t2_7_1(h6,p3,h1,p5)       += -1   * t1(p7,h1) * v2(h6,p3,p5,p7))
      (t2_7_1(h6,p3,h1,p5)       += -0.5 * t2(p3,p7,h1,h8) * v2(h6,h8,p5,p7))
      (i0(p3,p4,h1,h2)           += -1   * t2(p3,p5,h1,h6) * t2_7_1(h6,p4,h2,p5))
      (vt1t1_1(h5,p3,h1,h2)       = 0)
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
double ccsd_driver(ExecutionContext& ec,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_f1, Tensor<T>& d_v2,
                   int maxiter, double thresh,
                   double zshiftl,
                   int ndiis) {
  // ,
  //   ProcGroup pg,
  //   Distribution* distribution,
  //   MemoryManager* mgr) {
  Irrep irrep{0};
  bool spin_restricted = false;
  std::vector<double> p_evl_sorted;

  p_evl_sorted.reserve((TCE::noab() + TCE::nvab()).value());
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
    p_evl_sorted.resize((TCE::noab() + TCE::nvab()).value());
    auto lambda = [&] (const auto& blockid) {
      if(blockid[0] == blockid[1]) {
        auto block = d_f1.get(blockid);
        auto dim = d_f1.block_dims(blockid)[0].value();
        auto offset = d_f1.block_offset(blockid)[0].value();
        for(auto p = offset; p < offset + dim; p++) {
          p_evl_sorted[p] = block.buf()[p*dim + p];
        }
      }
    };
    block_for(d_f1(), lambda);
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
    Expects(tensor.rank() == 0);
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
      Scheduler sch = ec.scheduler();//{pg, distribution, mgr, irrep, spin_restricted};
      int off = iter - titer;
      sch.io(d_t1, d_t2, d_f1, d_v2)
        .output(d_e, *d_r1s[off], *d_r2s[off], d_r1_residual, d_r2_residual);
      ccsd_e(sch, d_f1, d_e, d_t1, d_t2, d_v2);
      ccsd_t1(sch, d_f1, *d_r1s[off], d_t1, d_t2, d_v2);
      ccsd_t2(sch, d_f1, *d_r2s[off], d_t1, d_t2, d_v2);
      sch(d_r1_residual() = 0)
        (d_r1_residual() += (*d_r1s[off])()  * (*d_r1s[off])())
        (d_r2_residual() = 0)
        (d_r2_residual() += (*d_r2s[off])()  * (*d_r2s[off])())
        ;
      sch.execute();
      std::cerr<<"----------------------------------------------"<<std::endl;


      double r1 = 0.5*std::sqrt(get_scalar(d_r1_residual));
      double r2 = 0.5*std::sqrt(get_scalar(d_r2_residual));
      residual = std::max(r1, r2);
      energy = get_scalar(d_e);
      std::cout << "iteration:" << iter << '\n';
      std::cout << "r1=" << r1 <<" r2="<<r2 << '\n';
      tensor_print(*d_r1s[off]);
      tensor_print(d_r1_residual);
      std::cout << "residual:" << residual << '\n';
      std:std::cout << "energy:" << energy << '\n';
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

std::vector<Spin> spins = {1_sp, 2_sp, 1_sp, 2_sp};
std::vector<Irrep> spatials = {0_ir, 0_ir, 0_ir, 0_ir};
std::vector<size_t> sizes = {5,5,2,2};
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

// Eigen matrix algebra library
#include <Eigen/Dense>
// #include <Eigen/Eigenvalues>
#include <unsupported/Eigen/CXX11/Tensor>

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Tensor4D = Eigen::Tensor<double, 4, Eigen::RowMajor>;
extern std::tuple<Matrix, Tensor4D, double> hartree_fock(const string filename);


int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);


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
  using T = double;
  Irrep irrep{0};
  bool spin_restricted = false;
  Tensor<T> d_t1{V|O, irrep, spin_restricted};
  Tensor<T> d_t2{VV|OO, irrep, spin_restricted};
  Tensor<T> d_f1{N|N, irrep, spin_restricted};
  Tensor<T> d_v2{NN|NN, irrep, spin_restricted};
  int maxiter = 10;
  double thresh = 1.0e-6;
  double zshiftl = 1.0;
  int ndiis = 6;


  auto distribution = Distribution_NW();
  auto pg = ProcGroup{MPI_COMM_WORLD};
  auto mgr = MemoryManagerSequential(pg);


#if 1
  Tensor<T>::allocate(pg, &distribution, &mgr, d_t1, d_t2, d_f1, d_v2);

  const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";

  Matrix F;
  Tensor4D V;
  double hf_energy{0.0};

  std::tie(F, V, hf_energy) = hartree_fock(filename);
  std::cerr << "debug2" << '\n';

  //Tensor Map
  tensor_map(d_f1(), [&](auto& block) {
    auto buf = block.buf();
    const auto& block_offset = block.block_offset();
    const auto& block_dims = block.block_dims();
    // std::cout << "block offset:" << block_offset << '\n';
    // std::cout << "block dims:" << block_dims << '\n';
    // std::cout << "block size:" << block.size() << '\n';
    Expects(block.tensor().rank() == 2);
    int c = 0;
    for (auto i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
      for (auto j = block_offset[1]; j < block_offset[1] + block_dims[1];
           j++, c++) {
        buf[c] = F(i.value(), j.value());
      }
    }
  });

  tensor_print(d_f1);
  std::cerr << "debug1" << '\n';

  tensor_map(d_v2(), [&](auto& block) {
    auto buf = block.buf();
    const auto& block_offset = block.block_offset();
    const auto& block_dims = block.block_dims();
    Expects(block.tensor().rank() == 4);
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
  ExecutionContext ec {pg, &distribution, &mgr, Irrep{0}, false};

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
#endif
  TCE::finalize();

  MPI_Finalize();
  return 0;
}
