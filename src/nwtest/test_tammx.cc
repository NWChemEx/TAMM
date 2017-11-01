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
#include <iostream>
#include "gtest/gtest.h"

#include "macdecls.h"

#include <mpi.h>
#include <ga.h>
#include <macdecls.h>
#include "nwtest/test_tammx.h"



void tammx_init(TAMMX_INT32 noa, TAMMX_INT32 nob, TAMMX_INT32 nva, TAMMX_INT32 nvb, bool intorb, bool restricted,
                const std::vector<TAMMX_INT32> &ispins,
                const std::vector<TAMMX_INT32> &isyms,
                const std::vector<TAMMX_SIZE> &isizes) {
  using Irrep = tammx::Irrep;
  using Spin = tammx::Spin;
  using BlockDim = tammx::BlockIndex;

  Irrep irrep_f{0}, irrep_v{0}, irrep_t{0}, irrep_x{0}, irrep_y{0};

  std::vector <Spin> spins;
  std::vector <Irrep> irreps;
  std::vector <TAMMX_SIZE> sizes;

  for (auto s : ispins) {
    spins.push_back(Spin{s});
  }
  for (auto r : isyms) {
    irreps.push_back(Irrep{r});
  }
  for (auto s : isizes) {
    sizes.push_back(TAMMX_SIZE{s});
  }

  tammx::TCE::init(spins, irreps, sizes,
                   BlockDim{noa},
                   BlockDim{noa + nob},
                   BlockDim{nva},
                   BlockDim{nva + nvb},
                   restricted,
                   irrep_f,
                   irrep_v,
                   irrep_t,
                   irrep_x,
                   irrep_y);
}

void tammx_finalize() {
  tammx::TCE::finalize();
}

#define INITVAL_TEST_0D 1
#define INITVAL_TEST_1D 1
#define INITVAL_TEST_2D 1
#define INITVAL_TEST_3D 1
#define INITVAL_TEST_4D 1

#define MULT_TEST_0D_0D 1
#define MULT_TEST_0D_1D 1

#define SYMM_ASSIGN_TEST_3D 1
#define SYMM_ASSIGN_TEST_4D 1

////////////////////////////////////////////////

TensorVec <tammx::TensorSymmGroup>
tammx_label_to_indices(const tammx::IndexLabelVec &labels) {
  tammx::TensorVec <tammx::TensorSymmGroup> ret;
  tammx::RangeTypeVec tdims;

  for (const auto &l : labels) {
    tdims.push_back(l.rt());
  }
  TAMMX_SIZE n = labels.size();
  //tammx::SymmGroup sg;
  size_t grp_size = 0;
  RangeType last_dt;
  for (const auto &dt: tdims) {
    if (grp_size == 0) {
      grp_size = 1;
      last_dt = dt;
    } else if (last_dt == dt) {
      grp_size += 1;
    } else {
      ret.push_back({last_dt, grp_size});
      grp_size = 1;
      last_dt = dt;
    }
  }
  if (grp_size > 0) {
    ret.push_back({last_dt, grp_size});
  }
  return ret;
}

TensorVec <tammx::TensorSymmGroup>
tammx_label_to_indices(const tammx::IndexLabelVec &upper_labels,
                       const tammx::IndexLabelVec &lower_labels,
                       bool all_n) {
  TensorVec<tammx::TensorSymmGroup> ret;
  if(!all_n) {
    ret = tammx_label_to_indices(upper_labels);
    const auto &lower =  tammx_label_to_indices(lower_labels);
    ret.insert_back(lower.begin(), lower.end());    
  } else {
    if(upper_labels.size() > 0) {
      const auto& upper = TensorVec<tammx::TensorSymmGroup>{
        tammx::TensorSymmGroup{tammx::DimType::n,
                               upper_labels.size()}};
      ret.insert_back(upper.begin(), upper.end());
    }
    if(lower_labels.size() > 0) {   
      const auto& lower = TensorVec<tammx::TensorSymmGroup>{
        tammx::TensorSymmGroup{tammx::DimType::n,
                               lower_labels.size()}};
      ret.insert_back(lower.begin(), lower.end());
    }
  }
  return ret;
}

template<typename T>
void
tammx_tensor_dump(const tammx::Tensor <T> &tensor, std::ostream &os) {
  const auto &buf = static_cast<const T *>(tensor.memory_region().access(tammx::Offset{0}));
  TAMMX_SIZE sz = tensor.memory_region().local_nelements().value();
  os << "tensor size=" << sz << std::endl;
  os << "tensor size (from distribution)=" << tensor.distribution()->buf_size(Proc{0}) << std::endl;
  for (TAMMX_SIZE i = 0; i < sz; i++) {
    os << buf[i] << " ";
  }
  os << std::endl;
}


bool
test_initval_no_n(tammx::ExecutionContext &ec,
                  const tammx::IndexLabelVec &upper_labels,
                  const tammx::IndexLabelVec &lower_labels) {
  const auto &upper_indices = tammx_label_to_indices(upper_labels);
  const auto &lower_indices = tammx_label_to_indices(lower_labels);

  tammx::TensorRank nupper{upper_labels.size()};
  tammx::TensorVec <tammx::TensorSymmGroup> indices{upper_indices};
  indices.insert_back(lower_indices.begin(), lower_indices.end());
  tammx::Tensor<double> xta{indices, nupper, tammx::Irrep{0}, false};
  tammx::Tensor<double> xtc{indices, nupper, tammx::Irrep{0}, false};

  double init_val = 9.1;

  g_ec->allocate(xta, xtc);
  g_ec->scheduler()
    .io(xta, xtc)
      (xta() = init_val)
      (xtc() = xta())
    .execute();

  tammx::BlockDimVec id{indices.size(), tammx::BlockIndex{0}};
  auto sz = xta.memory_region().local_nelements().value();

  bool ret = true;
  const double threshold = 1e-14;
  const auto abuf = reinterpret_cast<const double*>(xta.memory_region().access(tammx::Offset{0}));
  const auto cbuf = reinterpret_cast<const double*>(xtc.memory_region().access(tammx::Offset{0}));
  for (TAMMX_INT32 i = 0; i < sz; i++) {
    if (std::abs(abuf[i] - init_val) > threshold) {
      ret = false;
      break;
    }
  }
  if (ret == true) {
    for (TAMMX_INT32 i = 0; i < sz; i++) {
      if (std::abs(cbuf[i] - init_val) > threshold) {
        return false;
      }
    }
  }
  g_ec->deallocate(xta, xtc);
  return ret;
}

bool
test_symm_assign(tammx::ExecutionContext &ec,
                 const tammx::TensorVec <tammx::TensorSymmGroup> &cindices,
                 const tammx::TensorVec <tammx::TensorSymmGroup> &aindices,
                 TAMMX_INT32 nupper_indices,
                 const tammx::IndexLabelVec &clabels,
                 double alpha,
                 const std::vector<double> &factors,
                 const std::vector<tammx::IndexLabelVec> &alabels) {
  assert(factors.size() > 0);
  assert(factors.size() == alabels.size());
  bool restricted = ec.is_spin_restricted();
  tammx::TensorRank nup{static_cast<TensorRank>(nupper_indices)};
  tammx::Tensor<double> tc{cindices, nup, tammx::Irrep{0}, restricted};
  tammx::Tensor<double> ta{aindices, nup, tammx::Irrep{0}, restricted};
  tammx::Tensor<double> tc2{aindices, nup, tammx::Irrep{0}, restricted};

  bool status = true;

  ec.allocate(tc, tc2, ta);

  ec.scheduler()
    .io(ta, tc, tc2)
      (ta() = 0)
      (tc() = 0)
      (tc2() = 0)
    .execute();

  // auto init_lambda = [](tammx::Block<double> &block) {
  //   double n = std::rand() % 100;
  //   auto dbuf = block.buf();
  //   for (TAMMX_SIZE i = 0; i < block.size(); i++) {
  //     dbuf[i] = n + i;
  //     // std::cout<<"init_lambda. dbuf["<<i<<"]="<<dbuf[i]<<std::endl;
  //   }
  //   //std::generate_n(reinterpret_cast<double *>(block.buf()), block.size(), [&]() { return n++; });
  // };

  auto& tensor = ta;
  auto init_lambda = [&](auto& blockid) {
    double n = std::rand() % 5;
    auto block = tensor.alloc(blockid);
    auto dbuf = block.buf();      
    for (size_t i = 0; i < block.size(); i++) {
      dbuf[i] = n + i;
      //std::cout << "init_lambda. dbuf[" << i << "]=" << dbuf[i] << std::endl;
    }
    tensor.put(blockid, block);
  };

  block_parfor(ec.pg(), ta(), init_lambda);
  tammx_symmetrize(ec, ta());
  
  // std::cout<<"TA=\n";
  // tammx_tensor_dump(ta, std::cout);

  //std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
  ec.scheduler()
    .io(tc, ta)
      (tc(clabels) += alpha * ta(alabels[0]))
    .execute();
  //std::cout<<">>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<std::endl;

  for (TAMMX_SIZE i = 0; i < factors.size(); i++) {
    //std::cout<<"++++++++++++++++++++++++++"<<std::endl;
    ec.scheduler()
      .io(tc2, ta)
        (tc2(clabels) += alpha * factors[i] * ta(alabels[i]))
      .execute();
    //std::cout<<"---------------------------"<<std::endl;
  }
  // std::cout<<"TA=\n";
  // tammx_tensor_dump(ta, std::cout);
  // std::cout<<"TC=\n";
  // tammx_tensor_dump(tc, std::cout);
  // std::cout<<"TC2=\n";
  // tammx_tensor_dump(tc2, std::cout);
  // std::cout<<"\n";
  ec.scheduler()
    .io(tc, tc2)
      (tc2(clabels) += -1.0 * tc(clabels))
    .execute();
  // std::cout<<"TC - TC2=\n";
  // tammx_tensor_dump(tc2, std::cout);
  // std::cout<<"\n";

  double threshold = 1e-12;
  auto lambda = [&](auto &val) {
    if (std::abs(val) > threshold) {
      //std::cout<<"----ERROR----\n";
    }
    status &= (std::abs(val) < threshold);
  };
  ec.scheduler()
    .io(tc2)
    .sop(tc2(), lambda)
    .execute();
  ec.deallocate(tc, tc2, ta);
  return status;
}


tammx::TensorVec <tammx::TensorSymmGroup>
tammx_tensor_dim_to_symm_groups(tammx::DimTypeVec dims, TAMMX_INT32 nup) {
  tammx::TensorVec <tammx::TensorSymmGroup> ret;

  TAMMX_INT32 nlo = dims.size() - nup;
  if (nup == 0) {
    //no-op
  } else if (nup == 1) {
    //tammx::SymmGroup sg{dims[0]};
    ret.push_back({dims[0]});
  } else if (nup == 2) {
    if (dims[0] == dims[1]) {
      //tammx::SymmGroup sg{dims[0], dims[1]};
      ret.push_back({dims[0], 2});
    } else {
      //tammx::SymmGroup sg1{dims[0]}, sg2{dims[1]};
      ret.push_back({dims[0]});
      ret.push_back({dims[1]});
    }
  } else {
    assert(0);
  }

  if (nlo == 0) {
    //no-op
  } else if (nlo == 1) {
    // tammx::SymmGroup sg{dims[nup]};
    // ret.push_back(sg);
    ret.push_back({dims[nup]});
  } else if (nlo == 2) {
    if (dims[nup + 0] == dims[nup + 1]) {
      // tammx::SymmGroup sg{dims[nup + 0], dims[nup + 1]};
      // ret.push_back(sg);
      ret.push_back({dims[nup], 2});
    } else {
      // tammx::SymmGroup sg1{dims[nup + 0]}, sg2{dims[nup + 1]};
      // ret.push_back(sg1);
      // ret.push_back(sg2);
      ret.push_back({dims[nup+0]});
      ret.push_back({dims[nup+1]});
    }
  } else {
    assert(0);
  }
  return ret;
}
