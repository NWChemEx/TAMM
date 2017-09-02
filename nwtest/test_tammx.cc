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

//namespace {
//    tammx::ExecutionContext* g_ec;
//}
//
//class TestEnvironment : public testing::Environment {
//public:
//    explicit TestEnvironment(tammx::ExecutionContext* ec) {
//        g_ec = ec;
//    }
//};


//void
//assert_result(bool pass_or_fail, const std::string& msg) {
//  if (!pass_or_fail) {
//    std::cout << "C & F Tensors differ in Test " << msg << std::endl;
//  } else {
//    std::cout << "Congratulations! Test " << msg << " PASSED" << std::endl;
//  }
//}
//
//






void tammx_init(int noa, int nob, int nva, int nvb, bool intorb, bool restricted,
                const std::vector<int> &ispins,
                const std::vector<int> &isyms,
                const std::vector<int> &isizes) {
  using Irrep = tammx::Irrep;
  using Spin = tammx::Spin;
  using BlockDim = tammx::BlockDim;

  Irrep irrep_f{0}, irrep_v{0}, irrep_t{0}, irrep_x{0}, irrep_y{0};

  std::vector <Spin> spins;
  std::vector <Irrep> irreps;
  std::vector <size_t> sizes;

  for (auto s : ispins) {
    spins.push_back(Spin{s});
  }
  for (auto r : isyms) {
    irreps.push_back(Irrep{r});
  }
  for (auto s : isizes) {
    sizes.push_back(size_t{s});
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

//using namespace tammx;

//namespace new_impl {
TensorVec <tammx::TensorSymmGroup>
tammx_label_to_indices(const tammx::TensorLabel &labels) {
  tammx::TensorVec <tammx::TensorSymmGroup> ret;
  tammx::TensorRange tdims;

  for (const auto &l : labels) {
    tdims.push_back(l.rt());
  }
  size_t n = labels.size();
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
tammx_label_to_indices(const tammx::TensorLabel &upper_labels,
                       const tammx::TensorLabel &lower_labels,
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


// tammx::TensorVec<tammx::SymmGroup>
// tammx_tensor_dim_to_symm_groups(tammx::TensorDim dims, int nup) {
//   tammx::TensorVec<tammx::SymmGroup> ret;

//   int nlo = dims.size() - nup;
//   if(nup==0) {
//     //no-op
//   } else if(nup == 1) {
//     tammx::SymmGroup sg{dims[0]};
//     ret.push_back(sg);
//   } else if (nup == 2) {
//     if(dims[0] == dims[1]) {
//       tammx::SymmGroup sg{dims[0], dims[1]};
//       ret.push_back(sg);
//     }
//     else {
//       tammx::SymmGroup sg1{dims[0]}, sg2{dims[1]};
//       ret.push_back(sg1);
//       ret.push_back(sg2);
//     }
//   } else {
//     assert(0);
//   }

//   if(nlo==0) {
//     //no-op
//   } else if(nlo == 1) {
//     tammx::SymmGroup sg{dims[nup]};
//     ret.push_back(sg);
//   } else if (nlo == 2) {
//     if(dims[nup + 0] == dims[nup + 1]) {
//       tammx::SymmGroup sg{dims[nup + 0], dims[nup + 1]};
//       ret.push_back(sg);
//     }
//     else {
//       tammx::SymmGroup sg1{dims[nup + 0]}, sg2{dims[nup + 1]};
//       ret.push_back(sg1);
//       ret.push_back(sg2);
//     }
//   } else {
//     assert(0);
//   }
//   return ret;
// }


template<typename T>
void
tammx_tensor_dump(const tammx::Tensor <T> &tensor, std::ostream &os) {
  const auto &buf = static_cast<const T *>(tensor.memory_manager()->access(tammx::Offset{0}));
  size_t sz = tensor.memory_manager()->local_size_in_elements().value();
  os << "tensor size=" << sz << std::endl;
  os << "tensor size (from distribution)=" << tensor.distribution()->buf_size(Proc{0}) << std::endl;
  for (size_t i = 0; i < sz; i++) {
    os << buf[i] << " ";
  }
  os << std::endl;
}


bool
test_initval_no_n(tammx::ExecutionContext &ec,
                  const tammx::TensorLabel &upper_labels,
                  const tammx::TensorLabel &lower_labels) {
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

  tammx::TensorIndex id{indices.size(), tammx::BlockDim{0}};
  auto sz = xta.memory_manager()->local_size_in_elements().value();

  bool ret = true;
  const double threshold = 1e-14;
  const auto abuf = reinterpret_cast<double *>(xta.memory_manager()->access(tammx::Offset{0}));
  const auto cbuf = reinterpret_cast<double *>(xtc.memory_manager()->access(tammx::Offset{0}));
  for (int i = 0; i < sz; i++) {
    if (std::abs(abuf[i] - init_val) > threshold) {
      ret = false;
      break;
    }
  }
  if (ret == true) {
    for (int i = 0; i < sz; i++) {
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
                 int nupper_indices,
                 const tammx::TensorLabel &clabels,
                 double alpha,
                 const std::vector<double> &factors,
                 const std::vector<tammx::TensorLabel> &alabels) {
  assert(factors.size() > 0);
  assert(factors.size() == alabels.size());
  bool restricted = ec.is_spin_restricted();
  //auto restricted = tamm::Variables::restricted();
  //auto clabels = tamm_label_to_tammx_label(tclabels);
  // std::vector<tammx::TensorLabel> alabels;
  // for(const auto& tal: talabels) {
  //   alabels.push_back(tamm_label_to_tammx_label(tal));
  // }
  tammx::TensorRank nup{nupper_indices};
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

  auto init_lambda = [](tammx::Block<double> &block) {
    double n = std::rand() % 100;
    auto dbuf = block.buf();
    for (size_t i = 0; i < block.size(); i++) {
      dbuf[i] = n + i;
      // std::cout<<"init_lambda. dbuf["<<i<<"]="<<dbuf[i]<<std::endl;
    }
    //std::generate_n(reinterpret_cast<double *>(block.buf()), block.size(), [&]() { return n++; });
  };


  tensor_map(ta(), init_lambda);
  tammx_symmetrize(ec, ta());
  // std::cout<<"TA=\n";
  // tammx_tensor_dump(ta, std::cout);

  //std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
  ec.scheduler()
    .io(tc, ta)
      (tc(clabels) += alpha * ta(alabels[0]))
    .execute();
  //std::cout<<">>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<std::endl;

  for (size_t i = 0; i < factors.size(); i++) {
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
tammx_tensor_dim_to_symm_groups(tammx::TensorDim dims, int nup) {
  tammx::TensorVec <tammx::TensorSymmGroup> ret;

  int nlo = dims.size() - nup;
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
