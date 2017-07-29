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

#include "tensor/corf.h"
#include "tensor/equations.h"
#include "tensor/input.h"
#include "tensor/schedulers.h"
#include "tensor/t_assign.h"
#include "tensor/t_mult.h"
#include "tensor/tensor.h"
#include "tensor/tensors_and_ops.h"
#include "tensor/variables.h"
#include "macdecls.h"

#include "tammx/tammx.h"

#include <mpi.h>
#include <ga.h>
#include <macdecls.h>

namespace {
tammx::ExecutionContext* g_ec;
}

class TestEnvironment : public testing::Environment {
 public:
  explicit TestEnvironment(tammx::ExecutionContext* ec) {
    g_ec = ec;
  }
};

extern "C" {
  void init_fortran_vars_(Integer *noa1, Integer *nob1, Integer *nva1,
                          Integer *nvb1, logical *intorb1, logical *restricted1,
                          Integer *spins, Integer *syms, Integer *ranges);
  void finalize_fortran_vars_();
  void f_calls_setvars_cxx_();
  
  void offset_ccsd_t1_2_1_(Integer *l_t1_2_1_offset, Integer *k_t1_2_1_offset,
                           Integer *size_t1_2_1);

  typedef void add_fn(Integer *ta, Integer *offseta, Integer *irrepa,
                      Integer *tc, Integer *offsetc, Integer *irrepc);
                       
  typedef void mult_fn(Integer *ta, Integer *offseta, Integer *irrepa,
                       Integer *tb, Integer *offsetb, Integer *irrepb,
                       Integer *tc, Integer *offsetc, Integer *irrepc);

  typedef void mult_fn_2(Integer *ta, Integer *offseta,
                       Integer *tb, Integer *offsetb,
                       Integer *tc, Integer *offsetc);

  add_fn ccsd_t1_1_;
  mult_fn ccsd_t1_2_;
  mult_fn_2 cc2_t1_5_;
}

void
assert_result(bool pass_or_fail, const std::string& msg) {
  if (!pass_or_fail) {
    std::cout << "C & F Tensors differ in Test " << msg << std::endl;
  } else {
    std::cout << "Congratulations! Test " << msg << " PASSED" << std::endl;
  }
}


tamm::Tensor
tamm_tensor(const std::vector<tamm::RangeType>& upper_ranges,
            const std::vector<tamm::RangeType>& lower_ranges,
            int irrep = 0,
            tamm::DistType dist_type = tamm::dist_nw) {
  int ndim = upper_ranges.size() + lower_ranges.size();
  int nupper = upper_ranges.size();
  std::vector<tamm::RangeType> rt {upper_ranges};
  std::copy(lower_ranges.begin(), lower_ranges.end(), std::back_inserter(rt));
  return tamm::Tensor(ndim, nupper, irrep, &rt[0], dist_type);
}

// tammx::Tensor<double>*
// tammx_tensor(const std::vector<tamm::RangeType>& upper_ranges,
//              const std::vector<tamm::RangeType>& lower_ranges,
//              int irrep = 0,
//              tamm::DistType dist_type = tamm::dist_nw) {
//   int ndim = upper_ranges.size() + lower_ranges.size();
//   int nupper = upper_ranges.size();
//   std::vector<tamm::RangeType> rt {upper_ranges};
//   std::copy(lower_ranges.begin(), lower_ranges.end(), std::back_inserter(rt));
//   return tammxdc::Tensor(ndim, nupper, irrep, &rt[0], dist_type);
// }


void
tamm_assign(tamm::Tensor* tc,
            const std::vector<tamm::IndexName>& clabel,
            double alpha,
            tamm::Tensor* ta,
            const std::vector<tamm::IndexName>& alabel) {
  tamm::Assignment as(tc, ta, alpha, clabel, alabel);
  as.execute();
}

tammx::TensorLabel
tamm_label_to_tammx_label(const std::vector<tamm::IndexName>& label) {
  tammx::TensorLabel ret;
  for(auto l : label) {
    if(l >= tamm::P1B && l<= tamm::P12B) {
      ret.push_back(tammx::IndexLabel{l - tamm::P1B, tammx::DimType::v});
    }
    else if(l >= tamm::H1B && l<= tamm::H12B) {
      ret.push_back(tammx::IndexLabel{l - tamm::H1B, tammx::DimType::o});
    }
  }
  return ret;
}

tamm::RangeType
tamm_idname_to_tamm_range(const tamm::IndexName& idname) {
  return (idname >= tamm::H1B && idname <= tamm::H12B)
      ? tamm::RangeType::TO : tamm::RangeType::TV;
}

tamm::RangeType
tamm_id_to_tamm_range(const tamm::Index& id) {
  return tamm_idname_to_tamm_range(id.name());
}

tammx::DimType
tamm_range_to_tammx_dim(tamm::RangeType rt) {
  tammx::DimType ret;
  switch(rt) {
    case tamm::RangeType::TO:
      ret = tammx::DimType::o;
      break;
    case tamm::RangeType::TV:
      ret = tammx::DimType::v;
      break;
    default:
      assert(0);
  }
  return ret;
}

tammx::DimType
tamm_id_to_tammx_dim(const tamm::Index& id) {
  return tamm_range_to_tammx_dim(tamm_id_to_tamm_range(id));
}

tammx::TensorVec<tammx::SymmGroup>
tammx_tensor_dim_to_symm_groups(tammx::TensorDim dims, int nup) {
  tammx::TensorVec<tammx::SymmGroup> ret;

  int nlo = dims.size() - nup;
  if(nup == 1) {
    tammx::SymmGroup sg{dims[0]};
    ret.push_back(sg);
  } else if (nup == 2) {
    if(dims[0] == dims[1]) {
      tammx::SymmGroup sg{dims[0], dims[1]};
      ret.push_back(sg);      
    }
    else {
      tammx::SymmGroup sg1{dims[0]}, sg2{dims[1]};
      ret.push_back(sg1);
      ret.push_back(sg2);
    }
  }

  if(nlo == 1) {
    tammx::SymmGroup sg{dims[nup]};
    ret.push_back(sg);
  } else if (nlo == 2) {
    if(dims[nup + 0] == dims[nup + 1]) {
      tammx::SymmGroup sg{dims[nup + 0], dims[nup + 1]};
      ret.push_back(sg);      
    }
    else {
      tammx::SymmGroup sg1{dims[nup + 0]}, sg2{dims[nup + 1]};
      ret.push_back(sg1);
      ret.push_back(sg2);
    }
  }
  return ret;
}

tammx::TensorVec<tammx::SymmGroup>
tamm_tensor_to_tammx_symm_groups(const tamm::Tensor* tensor) {
  const std::vector<tamm::Index>& ids = tensor->ids();
  int nup = tensor->nupper();
  int nlo = ids.size() - nup;

  if (tensor->dim_type() == tamm::DimType::dim_n) {
    using tammx::SymmGroup;
    SymmGroup sgu, sgl;
    for(int i=0; i<nup; i++) {
      sgu.push_back(tammx::DimType::n);
    }
    for(int i=0; i<nlo; i++) {
      sgl.push_back(tammx::DimType::n);
    }
    tammx::TensorVec<SymmGroup> ret;
    if(sgu.size() > 0) {
      ret.push_back(sgu);
    }
    if(sgl.size() > 0) {
      ret.push_back(sgl);
    }
    return ret;
  }

  assert(ids.size() <=4); //@todo @fixme assume for now
  assert(nup <= 2); //@todo @fixme assume for now
  assert(nlo <= 2);  //@todo @fixme assume for now
  
  tammx::TensorDim dims;
  for(const auto& id: ids) {
    dims.push_back(tamm_id_to_tammx_dim(id));
  }

  return tammx_tensor_dim_to_symm_groups(dims, nup);
}


tammx::Tensor<double>*
tamm_tensor_to_tammx_tensor(tammx::ProcGroup pg, tamm::Tensor* ttensor) {
  using tammx::Irrep;
  using tammx::TensorVec;
  using tammx::SymmGroup;

  auto irrep = Irrep{ttensor->irrep()};
  auto nup = ttensor->nupper();
  
  auto restricted = tamm::Variables::restricted();
  const TensorVec<SymmGroup>& indices = tamm_tensor_to_tammx_symm_groups(ttensor);

  auto xtensor = new tammx::Tensor<double>{indices, nup, irrep, restricted};
  auto mgr = std::make_shared<tammx::MemoryManagerGA>(pg, ttensor->ga().ga());
  auto distribution = tammx::Distribution_NW();
  xtensor->attach(&distribution, mgr);
  return xtensor;
}

void
tammx_assign(tammx::ExecutionContext& ec,
             tamm::Tensor* ttc,
             const std::vector<tamm::IndexName>& clabel,
             double alpha,
             tamm::Tensor* tta,
            const std::vector<tamm::IndexName>& alabel) {
  tammx::Tensor<double> *ta = tamm_tensor_to_tammx_tensor(ec.pg(), tta);
  tammx::Tensor<double> *tc = tamm_tensor_to_tammx_tensor(ec.pg(), ttc);
  
  auto al = tamm_label_to_tammx_label(alabel);
  auto cl = tamm_label_to_tammx_label(clabel);

  std::cout<<"----AL="<<al<<std::endl;
  std::cout<<"----CL="<<cl<<std::endl;
  ec.scheduler()
      .io((*tc), (*ta))
      ((*tc)(cl) += alpha * (*ta)(al))
      .execute();

  delete ta;
  delete tc;
}

void
tamm_mult(tamm::Tensor* tc,
          const std::vector<tamm::IndexName>& clabel,
          double alpha,
          tamm::Tensor* ta,
          const std::vector<tamm::IndexName>& alabel,
          tamm::Tensor* tb,
          const std::vector<tamm::IndexName>& blabel) {
  tamm::Multiplication mult(tc, clabel, ta, alabel, tb, blabel, alpha);
  mult.execute();
}

 void
 tammx_mult(tammx::ExecutionContext& ec,
            tamm::Tensor* tc,
            const std::vector<tamm::IndexName>& clabel,
            double alpha,
            tamm::Tensor* ta,
            const std::vector<tamm::IndexName>& alabel,
            tamm::Tensor* tb,
            const std::vector<tamm::IndexName>& blabel) {
   tamm::Multiplication mult(tc, clabel, ta, alabel, tb, blabel, alpha);
   mult.execute();
 }


void
fortran_assign(tamm::Tensor* tc,
               tamm::Tensor* ta,
               add_fn fn) {
  Integer da = static_cast<Integer>(ta->ga().ga()),
      offseta = ta->offset_index(),
      irrepa = ta->irrep();
  Integer dc = static_cast<Integer>(tc->ga().ga()),
      offsetc = tc->offset_index(),
      irrepc = tc->irrep();
  fn(&da, &offseta, &irrepa, &dc, &offsetc, &irrepc);
}

void
fortran_mult(tamm::Tensor* tc,
             tamm::Tensor* ta,
             tamm::Tensor* tb,
             mult_fn fn) {
  Integer da = static_cast<Integer>(ta->ga().ga()),
      offseta = ta->offset_index(),
      irrepa = ta->irrep();
  Integer db = static_cast<Integer>(tb->ga().ga()),
      offsetb = tb->offset_index(),
      irrepb = tb->irrep();
  Integer dc = static_cast<Integer>(tc->ga().ga()),
      offsetc = tc->offset_index(),
      irrepc = tc->irrep();
  fn(&da, &offseta, &irrepa, &db, &offsetb, &irrepb, &dc, &offsetc, &irrepc);
}

void
fortran_mult_vvoo_vo(tamm::Tensor* tc,
             tamm::Tensor* ta,
             tamm::Tensor* tb,
             mult_fn_2 fn) {
  Integer da = static_cast<Integer>(ta->ga().ga()),
      offseta = ta->offset_index(),
      irrepa = ta->irrep();
  Integer db = static_cast<Integer>(tb->ga().ga()),
      offsetb = tb->offset_index(),
      irrepb = tb->irrep();
  Integer dc = static_cast<Integer>(tc->ga().ga()),
      offsetc = tc->offset_index(),
      irrepc = tc->irrep();
  fn(&da, &offseta, &db, &offsetb, &dc, &offsetc);
}

void
tamm_create() {}

template<typename ...Args>
void
tamm_create(tamm::Tensor* tensor, Args ... args) {
  tensor->create();
  tamm_create(args...);
}

void
tamm_destroy() {}

template<typename ...Args>
void
tamm_destroy(tamm::Tensor* tensor, Args ... args) {
  tensor->destroy();
  tamm_destroy(args...);
}

const auto P1B = tamm::P1B;
const auto P2B = tamm::P2B;
const auto P3B = tamm::P3B;
const auto P4B = tamm::P4B;
const auto P5B = tamm::P5B;
const auto P6B = tamm::P6B;
const auto P7B = tamm::P7B;
const auto P8B = tamm::P8B;
const auto P10B = tamm::P10B;
const auto P11B = tamm::P11B;
const auto P12B = tamm::P12B;
const auto P9B = tamm::P9B;
const auto H1B = tamm::H1B;
const auto H2B = tamm::H2B;
const auto H3B = tamm::H3B;
const auto H4B = tamm::H4B;
const auto H5B = tamm::H5B;
const auto H6B = tamm::H6B;
const auto H7B = tamm::H7B;
const auto H8B = tamm::H8B;
const auto H9B = tamm::H9B;
const auto H10B = tamm::H10B;
const auto H11B = tamm::H11B;
const auto H12B = tamm::H12B;

const auto p1 = tamm::P1B;
const auto p2 = tamm::P2B;
const auto p3 = tamm::P3B;
const auto p4 = tamm::P4B;
const auto h1 = tamm::H1B;
const auto h2 = tamm::H2B;
const auto h3 = tamm::H3B;
const auto h4 = tamm::H4B;
const auto TO = tamm::TO;
const auto TV = tamm::TV;

std::vector<tamm::RangeType>
tamm_labels_to_ranges(const std::vector<tamm::IndexName>& labels) {
  std::vector<tamm::RangeType> ret;
  for(auto l : labels) {
    ret.push_back(tamm_idname_to_tamm_range(l));
  }
  return ret;
}

bool test_assign(tammx::ExecutionContext& ec,
                      double alpha,
                      const std::vector<tamm::IndexName>& cupper_labels,
                      const std::vector<tamm::IndexName>& clower_labels,
                      const std::vector<tamm::IndexName>& aupper_labels,
                      const std::vector<tamm::IndexName>& alower_labels) {
  const auto& cupper_ranges = tamm_labels_to_ranges(cupper_labels);
  const auto& clower_ranges = tamm_labels_to_ranges(clower_labels);
  const auto& aupper_ranges = tamm_labels_to_ranges(aupper_labels);
  const auto& alower_ranges = tamm_labels_to_ranges(alower_labels);
  auto tc1 = tamm_tensor(cupper_ranges, clower_ranges);
  auto tc2 = tamm_tensor(cupper_ranges, clower_ranges);
  auto ta = tamm_tensor(aupper_ranges, alower_ranges);

  tamm_create(&tc1, &tc2, &ta);
  ta.fill_random();

  auto clabels = cupper_labels;
  std::copy(clower_labels.begin(), clower_labels.end(), std::back_inserter(clabels));
  auto alabels = aupper_labels;
  std::copy(alower_labels.begin(), alower_labels.end(), std::back_inserter(alabels));

  tamm_assign(&tc1, clabels, alpha, &ta, alabels);
  tammx_assign(ec, &tc2, clabels, alpha, &ta, alabels);
  //fortran_assign(&tc_f, &ta, ccsd_t1_1_);

  //assert_result(tc1.check_correctness(&tc2), __func__);
  bool status = tc1.check_correctness(&tc2);

  tamm_destroy(&tc1, &tc2, &ta);
  return status;
}

bool test_assign_no_n(tammx::ExecutionContext& ec,
                      double alpha,
                      const std::vector<tamm::IndexName>& cupper_labels,
                      const std::vector<tamm::IndexName>& clower_labels,
                      const std::vector<tamm::IndexName>& aupper_labels,
                      const std::vector<tamm::IndexName>& alower_labels) {
  const auto& cupper_ranges = tamm_labels_to_ranges(cupper_labels);
  const auto& clower_ranges = tamm_labels_to_ranges(clower_labels);
  const auto& aupper_ranges = tamm_labels_to_ranges(aupper_labels);
  const auto& alower_ranges = tamm_labels_to_ranges(alower_labels);
  auto tc1 = tamm_tensor(cupper_ranges, clower_ranges);
  auto tc2 = tamm_tensor(cupper_ranges, clower_ranges);
  auto ta = tamm_tensor(aupper_ranges, alower_ranges);

  tamm_create(&tc1, &tc2, &ta);  
  ta.fill_random();

  auto clabels = cupper_labels;
  std::copy(clower_labels.begin(), clower_labels.end(), std::back_inserter(clabels));
  auto alabels = aupper_labels;
  std::copy(alower_labels.begin(), alower_labels.end(), std::back_inserter(alabels));
  
  tamm_assign(&tc1, clabels, alpha, &ta, alabels);
  tammx_assign(ec, &tc2, clabels, alpha, &ta, alabels);
  //fortran_assign(&tc_f, &ta, ccsd_t1_1_);

  //assert_result(tc1.check_correctness(&tc2), __func__);
  bool status = tc1.check_correctness(&tc2);

  tamm_destroy(&tc1, &tc2, &ta);
  return status;
}

bool test_mult_no_n(tammx::ExecutionContext& ec,
                      double alpha,
                      const std::vector<tamm::IndexName>& cupper_labels,
                      const std::vector<tamm::IndexName>& clower_labels,
                      const std::vector<tamm::IndexName>& aupper_labels,
                      const std::vector<tamm::IndexName>& alower_labels,
					  const std::vector<tamm::IndexName>& bupper_labels,
					  const std::vector<tamm::IndexName>& blower_labels) {
  const auto& cupper_ranges = tamm_labels_to_ranges(cupper_labels);
  const auto& clower_ranges = tamm_labels_to_ranges(clower_labels);
  const auto& aupper_ranges = tamm_labels_to_ranges(aupper_labels);
  const auto& alower_ranges = tamm_labels_to_ranges(alower_labels);
  const auto& bupper_ranges = tamm_labels_to_ranges(bupper_labels);
  const auto& blower_ranges = tamm_labels_to_ranges(blower_labels);
  auto tc1 = tamm_tensor(cupper_ranges, clower_ranges);
  auto tc2 = tamm_tensor(cupper_ranges, clower_ranges);
  auto ta = tamm_tensor(aupper_ranges, alower_ranges);
  auto tb = tamm_tensor(aupper_ranges, alower_ranges);

  tamm_create(&tc1, &tc2, &ta, &tb);
  ta.fill_given(2.0);
  tb.fill_random();

  auto clabels = cupper_labels;
  std::copy(clower_labels.begin(), clower_labels.end(), std::back_inserter(clabels));
  auto alabels = aupper_labels;
  std::copy(alower_labels.begin(), alower_labels.end(), std::back_inserter(alabels));
  auto blabels = bupper_labels;
  std::copy(blower_labels.begin(), blower_labels.end(), std::back_inserter(blabels));

  tamm_mult(&tc1, clabels, alpha, &ta, alabels, &tb, blabels);
  tammx_mult(ec, &tc2, clabels, alpha, &ta, alabels, &tb, blabels);

  //assert_result(tc1.check_correctness(&tc2), __func__);
  bool status = tc1.check_correctness(&tc2);

  tamm_destroy(&tc1, &tc2, &ta, &tb);
  return status;
}

#define ASSIGN_TEST_0D 1
#define ASSIGN_TEST_1D 1
#define ASSIGN_TEST_2D 1
#define ASSIGN_TEST_3D 1
#define ASSIGN_TEST_4D 1

#define INITVAL_TEST_0D 1
#define INITVAL_TEST_1D 1
#define INITVAL_TEST_2D 1
#define INITVAL_TEST_3D 1
#define INITVAL_TEST_4D 1

tammx::TensorVec<tammx::SymmGroup>
tamm_labels_to_tammx_indices(const std::vector<tamm::IndexName>& labels) {
  tammx::TensorDim tammx_dims;
  for(const auto l : labels) {
    tammx_dims.push_back(tamm_range_to_tammx_dim(tamm_idname_to_tamm_range(l)));
  }
  return tammx_tensor_dim_to_symm_groups(tammx_dims, tammx_dims.size());
}

//-----------------------------------------------------------------------
//
//                            Initval 0-d
//
//-----------------------------------------------------------------------

bool test_initval_no_n(tammx::ExecutionContext& ec,
                       const std::vector<tamm::IndexName>& upper_labels,
                       const std::vector<tamm::IndexName>& lower_labels) {
  const auto& upper_indices = tamm_labels_to_tammx_indices(upper_labels);
  const auto& lower_indices = tamm_labels_to_tammx_indices(lower_labels);

  tammx::TensorRank nupper {upper_labels.size()};
  tammx::TensorVec<tammx::SymmGroup> indices {upper_indices};
  indices.insert_back(lower_indices.begin(), lower_indices.end());
  tammx::Tensor<double> xta {indices, nupper, tammx::Irrep{0}, false};
  tammx::Tensor<double> xtc {indices, nupper, tammx::Irrep{0}, false};

  double init_val = 9.1;
  
  g_ec->allocate(xta, xtc);
  g_ec->scheduler()
      .io(xta, xtc)
      (xta() = init_val)
      (xtc() = xta())
      .execute();

  tammx::TensorIndex id {indices.size(), tammx::BlockDim{0}};
  auto sz = xta.memory_manager()->local_size_in_elements().value();

  bool ret = true;
  const double threshold = 1e-14;
  const auto abuf = reinterpret_cast<double*>(xta.memory_manager()->access(tammx::Offset{0}));
  const auto cbuf = reinterpret_cast<double*>(xtc.memory_manager()->access(tammx::Offset{0}));
  for(int i=0; i<sz; i++) {
    if(std::abs(abuf[i] - init_val) > threshold) {
      ret = false;
      break;
    }
  }
  if(ret == true) {
    for(int i=0; i<sz; i++) {
      if(std::abs(cbuf[i] - init_val) > threshold) {
        return false;
      }
    }
  }
  g_ec->deallocate(xta, xtc);
  return ret;
}

#if INITVAL_TEST_0D

TEST (InitvalTest, ZeroDim) {
  ASSERT_TRUE(test_initval_no_n(*g_ec, {}, {}));
}
#endif

#if INITVAL_TEST_1D

TEST (InitvalTest, OneDim) {
  ASSERT_TRUE(test_initval_no_n(*g_ec, {}, {h1}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {}, {p1}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1}, {}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1}, {}));
}

#endif

#if INITVAL_TEST_2D

TEST (InitvalTest, TwoDim) {
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1}, {h2}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1}, {p2}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1}, {h2}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1}, {p2}));
}

#endif

#if INITVAL_TEST_3D

TEST (InitvalTest, ThreeDim) {
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1}, {h2, h3}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1}, {h2, p3}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1}, {p2, h3}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1}, {p2, p3}));

  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1}, {h2, h3}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1}, {h2, p3}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1}, {p2, h3}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1}, {p2, p3}));

  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, h2}, {h3}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, h2}, {p3}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, p2}, {h3}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, p2}, {p3}));

  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, h2}, {h3}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, h2}, {p3}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, p2}, {h3}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, p2}, {p3}));
}

#endif

#if INITVAL_TEST_4D

TEST (InitvalTest, FourDim) {
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, h2}, {h3, h4}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, h2}, {h3, p4}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, h2}, {p3, h4}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, h2}, {p3, p4}));

  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, p2}, {h3, h4}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, p2}, {h3, p4}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, p2}, {p3, h4}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {h1, p2}, {p3, p4}));

  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, h2}, {h3, h4}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, h2}, {h3, p4}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, h2}, {p3, h4}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, h2}, {p3, p4}));

  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, p2}, {h3, h4}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, p2}, {h3, p4}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, p2}, {p3, h4}));
  ASSERT_TRUE(test_initval_no_n(*g_ec, {p1, p2}, {p3, p4}));
}

#endif


//-----------------------------------------------------------------------
//
//                            Add 0-d
//
//-----------------------------------------------------------------------

#if ASSIGN_TEST_0D

//@todo tamm might not work with zero dimensions. So directly testing tammx.
TEST (AssignTest, ZeroDim) {
  auto ta = tamm_tensor({}, {});
  auto tc1 = tamm_tensor({}, {});
  auto tc2 = tamm_tensor({}, {});

  tamm_create(&ta, &tc1, &tc2);
  ta.fill_given(0.91);
  //tamm_assign(&tc1, {}, 1.0, &ta, {});
  tammx_assign(*g_ec, &tc2, {}, 1.0, &ta, {});
  bool status = tc2.check_correctness(&ta);
  tamm_destroy(&ta, &tc1, &tc2);
  ASSERT_TRUE(status);

  tammx::Tensor<double> xta {{}, 0, tammx::Irrep{0}, false};
  tammx::Tensor<double> xtc {{}, 0, tammx::Irrep{0}, false};

  double init_val = 9.1;
  
  g_ec->allocate(xta, xtc);
  g_ec->scheduler()
      .io(xta, xtc)
      (xta() = init_val)
      (xtc() = xta())
      .execute();

  auto ablock = xta.get({});
  ASSERT_TRUE(*xta.get({}).buf() == init_val);
  ASSERT_TRUE(*xtc.get({}).buf() == init_val);
  g_ec->deallocate(xta, xtc);
}
#endif

//-----------------------------------------------------------------------
//
//                            Add 1-d
//
//-----------------------------------------------------------------------

#if ASSIGN_TEST_1D

TEST (AssignTest, OneDim_o1e_o1e) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {}, {h1}, {}));
}

TEST (AssignTest, OneDim_eo1_eo1) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {}, {h1}, {}, {h1}));
}

TEST (AssignTest, OneDim_v1e_v1e) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {}, {p1}, {}));
}

TEST (AssignTest, OneDim_ev1_ev1) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {}, {p1}, {}, {p1}));
}

#endif

//-----------------------------------------------------------------------
//
//                            Add 2-d
//
//-----------------------------------------------------------------------


#if ASSIGN_TEST_2D

TEST (AssignTest, TwoDim_O1O2_O1O2) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h4}, {h1}, {h4}, {h1}));
}

TEST (AssignTest, TwoDim_O1O2_O2O1) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 1.23, {h4}, {h1}, {h1}, {h4}));
}

TEST (AssignTest, TwoDim_OV_OV) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h4}, {p1}, {h4}, {p1}));
}

TEST (AssignTest, TwoDim_OV_VO) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 1.23, {h4}, {p1}, {p1}, {h4}));
}

TEST (AssignTest, TwoDim_VO_VO) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {h1}, {p1}, {h1}));
}

TEST (AssignTest, TwoDim_VO_OV) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 1.23, {p1}, {h1}, {h1}, {p1}));
}

TEST (AssignTest, TwoDim_V1V2_V1V2) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p4}, {p1}, {p4}, {p1}));
}

TEST (AssignTest, TwoDim_V1V2_V2V1) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 1.23, {p4}, {p1}, {p1}, {p4}));
}

#endif

//TEST (MultTest, FourDim_TwoDim_V1V2O1O2_O2V2) {
//  ASSERT_TRUE(test_mult_no_n(*g_ec, 1.0, {P1B}, {H1B},
//		  {P1B, P2B}, {H1B, H4B}, {H4B}, {P2B}));
//}

// void test_assign_2d(tammx::ExecutionContext& ec) {
//   test_assign_no_n(ec, 0.24, {H4B}, {H1B}, {H4B}, {H1B});
//   test_assign_no_n(ec, 1.23, {H4B}, {H1B}, {H1B}, {H4B});

//   test_assign_no_n(ec, 0.24, {H4B}, {P1B}, {H4B}, {P1B});
//   test_assign_no_n(ec, 1.23, {H4B}, {P1B}, {P1B}, {H4B});

//   test_assign_no_n(ec, 0.24, {P1B}, {H1B}, {P1B}, {H1B});
//   test_assign_no_n(ec, 1.23, {P1B}, {H1B}, {H1B}, {P1B});

//   test_assign_no_n(ec, 0.24, {P4B}, {P1B}, {P4B}, {P1B});
//   test_assign_no_n(ec, 1.23, {P4B}, {P1B}, {P1B}, {P4B});
// }



//-----------------------------------------------------------------------
//
//                            Add 3-d
//
//-----------------------------------------------------------------------

#if ASSIGN_TEST_3D

TEST (AssignTest, ThreeDim_o1_o2o3__o1_o2o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {h2, h3}, {h1}, {h2, h3}));
}

TEST (AssignTest, ThreeDim_o1_o2o3__o1_o3o2) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {h2, h3}, {h1}, {h3, h2}));
}

TEST (AssignTest, ThreeDim_o1_o2v3__o1_o2v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {h2, p3}, {h1}, {h2, p3}));
}

TEST (AssignTest, ThreeDim_o1_o2v3__o1_v3o2) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {h2, p3}, {h1}, {p3, h2}));
}

TEST (AssignTest, ThreeDim_o1_v2o3__o1_v2o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {p2, h3}, {h1}, {p2, h3}));
}

TEST (AssignTest, ThreeDim_o1_v2o3__o1_o3v2) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {p2, h3}, {h1}, {h3, p2}));
}

TEST (AssignTest, ThreeDim_o1_v2v3__o1_v2v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {p2, p3}, {h1}, {p2, p3}));
}

TEST (AssignTest, ThreeDim_o1_v2v3__o1_v3v2) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {p2, p3}, {h1}, {p3, p2}));
}

///////////

TEST (AssignTest, ThreeDim_v1_o2o3__v1_o2o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {h2, h3}, {p1}, {h2, h3}));
}

TEST (AssignTest, ThreeDim_v1_o2o3__v1_o3o2) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {h2, h3}, {p1}, {h3, h2}));
}

TEST (AssignTest, ThreeDim_v1_o2v3__v1_o2v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {h2, p3}, {p1}, {h2, p3}));
}

TEST (AssignTest, ThreeDim_v1_o2v3__v1_v3o2) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {h2, p3}, {p1}, {p3, h2}));
}

TEST (AssignTest, ThreeDim_v1_v2o3__v1_v2o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {p2, h3}, {p1}, {p2, h3}));
}

TEST (AssignTest, ThreeDim_v1_v2o3__v1_o3v2) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {p2, h3}, {p1}, {h3, p2}));
}

TEST (AssignTest, ThreeDim_v1_v2v3__v1_v2v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {p2, p3}, {p1}, {p2, p3}));
}

TEST (AssignTest, ThreeDim_v1_v2v3__v1_v3v2) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {p2, p3}, {p1}, {p3, p2}));
}

//////////////////

TEST (AssignTest, ThreeDim_o1o2_o3__o1o2_o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3}, {h1, h2}, {h3}));
}

TEST (AssignTest, ThreeDim_o1o2_o3__o2o1_o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3}, {h2, h1}, {h3}));
}

TEST (AssignTest, ThreeDim_o1o2_v3__o1o2_v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3}, {h1, h2}, {p3}));
}

TEST (AssignTest, ThreeDim_o1o2_v3__o2o1_v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3}, {h2, h1}, {p3}));
}

/////////

TEST (AssignTest, ThreeDim_o1v2_o3__o1v2_o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3}, {h1, p2}, {h3}));
}

TEST (AssignTest, ThreeDim_o1v2_o3__v2o1_o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3}, {p2, h1}, {h3}));
}

TEST (AssignTest, ThreeDim_o1v2_v3__o1v2_v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3}, {h1, p2}, {p3}));
}

TEST (AssignTest, ThreeDim_o1v2_v3__v2o1_v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3}, {p2, h1}, {p3}));
}

//////////////////

TEST (AssignTest, ThreeDim_v1o2_o3__v1o2_o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3}, {p1, h2}, {h3}));
}

TEST (AssignTest, ThreeDim_v1o2_o3__o2v1_o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3}, {h2, p1}, {h3}));
}

TEST (AssignTest, ThreeDim_v1o2_v3__v1o2_v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3}, {p1, h2}, {p3}));
}

TEST (AssignTest, ThreeDim_v1o2_v3__o2v1_v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3}, {h2, p1}, {p3}));
}

/////////

TEST (AssignTest, ThreeDim_v1v2_o3__v1v2_o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3}, {p1, p2}, {h3}));
}

TEST (AssignTest, ThreeDim_v1v2_o3__v2v1_o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3}, {p2, p1}, {h3}));
}

TEST (AssignTest, ThreeDim_v1v2_v3__v1v2_v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3}, {p1, p2}, {p3}));
}

TEST (AssignTest, ThreeDim_v1v2_v3__v2v1_v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3}, {p2, p1}, {p3}));
}

//////////

#endif

//-----------------------------------------------------------------------
//
//                            Add 4-d
//
//-----------------------------------------------------------------------

#if ASSIGN_TEST_4D

TEST (AssignTest, FourDim_o1o2o3o4_o1o2o3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, h4}, {h1, h2}, {h3, h4}));
}

TEST (AssignTest, FourDim_o1o2o3o4_o1o2o4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, h4}, {h1, h2}, {h4, h3}));
}

TEST (AssignTest, FourDim_o1o2o3o4_o2o1o3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, h4}, {h2, h1}, {h3, h4}));
}

TEST (AssignTest, FourDim_o1o2o3o4_o2o1o4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, h1}, {h3, h4}, {h2, h1}, {h4, h3}));
}

///////

TEST (AssignTest, FourDim_o1o2o3v4_o1o2o3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, h1}, {h3, p4}, {h1, h2}, {h3, p4}));
}

TEST (AssignTest, FourDim_o1o2o3v4_o1o2v4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, p4}, {h1, h2}, {p4, h3}));
}

TEST (AssignTest, FourDim_o1o2o3v4_o2o1o3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, p4}, {h2, h1}, {h3, p4}));
}

TEST (AssignTest, FourDim_o1o2o3v4_o2o1v4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, p4}, {h2, h1}, {p4, h3}));
}

////////

TEST (AssignTest, FourDim_o1o2v3o4_o1o2v3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, h1}, {p3, h4}, {h1, h2}, {p3, h4}));
}

TEST (AssignTest, FourDim_o1o2v3o4_o1o2o4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, h4}, {h1, h2}, {h4, p3}));
}

TEST (AssignTest, FourDim_o1o2v3o4_o2o1v3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, h4}, {h2, h1}, {p3, h4}));
}

TEST (AssignTest, FourDim_o1o2v3o4_o2o1o4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, h4}, {h2, h1}, {h4, p3}));
}


////////

TEST (AssignTest, FourDim_o1o2v3v4_o1o2v3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, h1}, {p3, p4}, {h1, h2}, {p3, p4}));
}

TEST (AssignTest, FourDim_o1o2v3v4_o1o2v4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, p4}, {h1, h2}, {p4, p3}));
}

TEST (AssignTest, FourDim_o1o2v3v4_o2o1v3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, p4}, {h2, h1}, {p3, p4}));
}

TEST (AssignTest, FourDim_o1o2v3v4_o2o1v4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, p4}, {h2, h1}, {p4, p3}));
}

///////////////////////

TEST (AssignTest, FourDim_o1v2o3o4_o1v2o3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, h4}, {h1, p2}, {h3, h4}));
}

TEST (AssignTest, FourDim_o1v2o3o4_o1v2o4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, h4}, {h1, p2}, {h4, h3}));
}

TEST (AssignTest, FourDim_o1v2o3o4_v2o1o3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, h4}, {p2, h1}, {h3, h4}));
}

TEST (AssignTest, FourDim_o1v2o3o4_v2o1o4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, h1}, {h3, h4}, {p2, h1}, {h4, h3}));
}

///////

TEST (AssignTest, FourDim_o1v2o3v4_o1v2o3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, h1}, {h3, p4}, {h1, p2}, {h3, p4}));
}

TEST (AssignTest, FourDim_o1v2o3v4_o1v2v4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, p4}, {h1, p2}, {p4, h3}));
}

TEST (AssignTest, FourDim_o1v2o3v4_v2o1o3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, p4}, {p2, h1}, {h3, p4}));
}

TEST (AssignTest, FourDim_o1v2o3v4_v2o1v4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, p4}, {p2, h1}, {p4, h3}));
}

////////

TEST (AssignTest, FourDim_o1v2v3o4_o1v2v3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, h1}, {p3, h4}, {h1, p2}, {p3, h4}));
}

TEST (AssignTest, FourDim_o1v2v3o4_o1v2o4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, h4}, {h1, p2}, {h4, p3}));
}

TEST (AssignTest, FourDim_o1v2v3o4_v2o1v3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, h4}, {p2, h1}, {p3, h4}));
}

TEST (AssignTest, FourDim_o1v2v3o4_v2o1o4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, h4}, {p2, h1}, {h4, p3}));
}


////////

TEST (AssignTest, FourDim_o1v2v3v4_o1v2v3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, h1}, {p3, p4}, {h1, p2}, {p3, p4}));
}

TEST (AssignTest, FourDim_o1v2v3v4_o1v2v4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, p4}, {h1, p2}, {p4, p3}));
}

TEST (AssignTest, FourDim_o1v2v3v4_v2o1v3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, p4}, {p2, h1}, {p3, p4}));
}

TEST (AssignTest, FourDim_o1v2v3v4_v2o1v4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, p4}, {p2, h1}, {p4, p3}));
}

//////////////////////////////////////

TEST (AssignTest, FourDim_v1o2o3o4_v1o2o3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, h4}, {p1, h2}, {h3, h4}));
}

TEST (AssignTest, FourDim_v1o2o3o4_v1o2o4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, h4}, {p1, h2}, {h4, h3}));
}

TEST (AssignTest, FourDim_v1o2o3o4_o2v1o3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, h4}, {h2, p1}, {h3, h4}));
}

TEST (AssignTest, FourDim_v1o2o3o4_o2v1o4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, p1}, {h3, h4}, {h2, p1}, {h4, h3}));
}

///////

TEST (AssignTest, FourDim_v1o2o3v4_v1o2o3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, p1}, {h3, p4}, {p1, h2}, {h3, p4}));
}

TEST (AssignTest, FourDim_v1o2o3v4_v1o2v4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, p4}, {p1, h2}, {p4, h3}));
}

TEST (AssignTest, FourDim_v1o2o3v4_o2v1o3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, p4}, {h2, p1}, {h3, p4}));
}

TEST (AssignTest, FourDim_v1o2o3v4_o2v1v4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, p4}, {h2, p1}, {p4, h3}));
}

////////

TEST (AssignTest, FourDim_v1o2v3o4_v1o2v3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, p1}, {p3, h4}, {p1, h2}, {p3, h4}));
}

TEST (AssignTest, FourDim_v1o2v3o4_v1o2o4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, h4}, {p1, h2}, {h4, p3}));
}

TEST (AssignTest, FourDim_v1o2v3o4_o2v1v3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, h4}, {h2, p1}, {p3, h4}));
}

TEST (AssignTest, FourDim_v1o2v3o4_o2v1o4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, h4}, {h2, p1}, {h4, p3}));
}


////////

TEST (AssignTest, FourDim_v1o2v3v4_v1o2v3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, p1}, {p3, p4}, {p1, h2}, {p3, p4}));
}

TEST (AssignTest, FourDim_v1o2v3v4_v1o2v4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, p4}, {p1, h2}, {p4, p3}));
}

TEST (AssignTest, FourDim_v1o2v3v4_o2v1v3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, p4}, {h2, p1}, {p3, p4}));
}

TEST (AssignTest, FourDim_v1o2v3v4_o2v1v4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, p4}, {h2, p1}, {p4, p3}));
}

//////////////////////////////////////

TEST (AssignTest, FourDim_v1v2o3o4_v1v2o3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, h4}, {p1, p2}, {h3, h4}));
}

TEST (AssignTest, FourDim_v1v2o3o4_v1v2o4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, h4}, {p1, p2}, {h4, h3}));
}

TEST (AssignTest, FourDim_v1v2o3o4_v2v1o3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, h4}, {p2, p1}, {h3, h4}));
}

TEST (AssignTest, FourDim_v1v2o3o4_v2v1o4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, p1}, {h3, h4}, {p2, p1}, {h4, h3}));
}

///////

TEST (AssignTest, FourDim_v1v2o3v4_v1v2o3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, p1}, {h3, p4}, {p1, p2}, {h3, p4}));
}

TEST (AssignTest, FourDim_v1v2o3v4_v1v2v4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, p4}, {p1, p2}, {p4, h3}));
}

TEST (AssignTest, FourDim_v1v2o3v4_v2v1o3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, p4}, {p2, p1}, {h3, p4}));
}

TEST (AssignTest, FourDim_v1v2o3v4_v2v1v4o3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, p4}, {p2, p1}, {p4, h3}));
}

////////

TEST (AssignTest, FourDim_v1v2v3o4_v1v2v3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, p1}, {p3, h4}, {p1, p2}, {p3, h4}));
}

TEST (AssignTest, FourDim_v1v2v3o4_v1v2o4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, h4}, {p1, p2}, {h4, p3}));
}

TEST (AssignTest, FourDim_v1v2v3o4_v2v1v3o4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, h4}, {p2, p1}, {p3, h4}));
}

TEST (AssignTest, FourDim_v1v2v3o4_v2v1o4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, h4}, {p2, p1}, {h4, p3}));
}


////////

TEST (AssignTest, FourDim_v1v2v3v4_v1v2v3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, p1}, {p3, p4}, {p1, p2}, {p3, p4}));
}

TEST (AssignTest, FourDim_v1v2v3v4_v1v2v4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, p4}, {p1, p2}, {p4, p3}));
}

TEST (AssignTest, FourDim_v1v2v3v4_v2v1v3v4) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, p4}, {p2, p1}, {p3, p4}));
}

TEST (AssignTest, FourDim_v1v2v3v4_v2v1v4v3) {
  ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, p4}, {p2, p1}, {p4, p3}));
}

#endif

void test_assign_ccsd_e(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {H6B}, {P5B},
                       {H6B}, {P5B});            // ccsd_e_1
}

void test_assign_ccsd_t1(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {P2B}, {H1B},
                       {P2B}, {H1B});            // ccsd_t1_1
  test_assign(ec, 1.0, {H7B}, {H1B},
                       {H7B}, {H1B});            // ccsd_t1_1_2_1
  test_assign(ec, 1.0, {H7B}, {P3B},
                       {H7B}, {P3B});            // ccsd_t1_2_2_1
  test_assign(ec, 1.0, {P2B}, {P3B},
                       {P2B}, {P3B});            // ccsd_t1_3_1
  test_assign(ec, 1.0, {H8B}, {P7B},
                       {H8B}, {P7B});            // ccsd_t1_5_1
  test_assign(ec, 1.0, {H4B, H5B}, {H1B, P3B},
                       {H4B, H5B}, {H1B, P3B});  // ccsd_t1_6_1
}

void test_assign_ccsd_t2(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {P3B, P4B}, {H1B, H2B},
                       {P3B, P4B}, {H1B, H2B});        // ccsd_t2_1
  test_assign(ec, 1.0, {H10B, P3B}, {H1B, H2B},
                       {H10B, P3B}, {H1B, H2B});       // ccsd_t2_2_1
  test_assign(ec, -1.0, {H10B, H11B}, {H1B, H2B},
                        {H10B, H11B}, {H1B, H2B});     // ccsd_t2_2_2_1
  test_assign(ec, 1.0, {H10B, H11B}, {H1B, P5B},
                       {H10B, H11B}, {H1B, P5B});      // ccsd_t2_2_2_2_1
  test_assign(ec, 1.0, {H10B}, {P5B}, {H10B}, {P5B});  // ccsd_t2_2_4_1
  test_assign(ec, 1.0, {H7B, H10B}, {H1B, P9B},
                       {H7B, H10B}, {H1B, P9B});       // ccsd_t2_2_5_1
  test_assign(ec, 1.0, {H9B}, {H1B}, {H9B}, {H1B});    // ccsd_t2_4_1
  test_assign(ec, 1.0, {H9B}, {P8B}, {H9B}, {P8B});    // ccsd_t2_4_2_1
  test_assign(ec, 1.0, {P3B}, {P5B}, {P3B}, {P5B});    // ccsd_t2_5_1
  test_assign(ec, -1.0, {H9B, H11B}, {H1B, H2B},
                        {H9B, H11B}, {H1B, H2B});      // ccsd_t2_6_1
  test_assign(ec, 1.0, {H9B, H11B}, {H1B, P8B},
                       {H9B, H11B}, {H1B, P8B});       // ccsd_t2_6_2_1
  test_assign(ec, 1.0, {H6B, P3B}, {H1B, P5B},
                       {H6B, P3B}, {H1B, P5B});      // ccsd_t2_7_1
}

void test_assign_cc2_t1(tammx::ExecutionContext& ec) {  // copy of ccsd_t1
  test_assign(ec, 1.0, {P2B}, {H1B},
                       {P2B}, {H1B});            // ccsd_t1_1
  test_assign(ec, 1.0, {H7B}, {H1B},
                       {H7B}, {H1B});            // ccsd_t1_1_2_1
  test_assign(ec, 1.0, {H7B}, {P3B},
                       {H7B}, {P3B});            // ccsd_t1_2_2_1
  test_assign(ec, 1.0, {P2B}, {P3B},
                       {P2B}, {P3B});            // ccsd_t1_3_1
  test_assign(ec, 1.0, {H8B}, {P7B},
                       {H8B}, {P7B});            // ccsd_t1_5_1
  test_assign(ec, 1.0, {H4B, H5B}, {H1B, P3B},
                       {H4B, H5B}, {H1B, P3B});  // ccsd_t1_6_1
}

void test_assign_cc2_t2(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {P3B, P4B}, {H1B, H2B},
                       {P3B, P4B}, {H1B, H2B});       // ccsd_t2_1
  test_assign(ec, 1.0, {H10B, P3B}, {H1B, H2B},
                       {H10B, P3B}, {H1B, H2B});      // ccsd_t2_2_1
  test_assign(ec, 1.0, {H8B, H10B}, {H1B, P5B},
                       {H8B, H10B}, {H1B, P5B});      // ccsd_t2_2_2_1
  test_assign(ec, 1.0, {H8B, H10B}, {H1B, P5B},
                       {H8B, H10B}, {H1B, P5B});      // ccsd_t2_2_2_2_1
  test_assign(ec, 1.0, {H10B, P3B}, {H1B, P5B},
                       {H10B, P3B}, {H1B, P5B});      // ccsd_t2_2_2_3
  test_assign(ec, 1.0, {P3B, P4B}, {H1B, P5B},
                       {P3B, P4B}, {H1B, P5B});       // ccsd_t2_3_1
}

void test_assign_cisd_c1(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {P2B}, {H1B},
                       {P2B}, {H1B});                 // cisd_c1_1
}

void test_assign_cisd_c2(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {P3B, P4B}, {H1B, H2B},
                       {P3B, P4B}, {H1B, H2B});       // cisd_c2_1
}

void test_assign_ccsd_lambda1(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {H2B}, {P1B},
                       {H2B}, {P1B});                 // lambda1_1
  test_assign(ec, 1.0, {H2B}, {H7B},
                       {H2B}, {H7B});                 // lambda1_2_1
  test_assign(ec, 1.0, {H2B}, {P3B},
                       {H2B}, {P3B});                 // lambda1_2_2_1
  test_assign(ec, 1.0, {P7B}, {P1B},
                       {P7B}, {P1B});                 // lambda1_3_1
  test_assign(ec, 1.0, {P9B}, {H11B},
                       {P9B}, {H11B});                // lambda1_5_1
  test_assign(ec, 1.0, {H10B}, {H11B},
                       {H10B}, {H11B});               // lambda1_5_2_1
  test_assign(ec, 1.0, {H10B}, {P3B},
                       {H10B}, {P3B});                // lambda1_5_2_2_1
  test_assign(ec, 1.0, {P9B}, {P7B},
                       {P9B}, {P7B});                 // lambda1_5_3_1
  test_assign(ec, 1.0, {H5B}, {P4B},
                       {H5B}, {P4B});                 // lambda1_5_5_1
  test_assign(ec, 1.0, {H5B, H6B}, {H11B, P4B},
                       {H5B, H6B}, {H11B, P4B});      // lambda1_5_6_1
  test_assign(ec, 1.0, {H2B, P9B}, {H11B, H12B},
                       {H2B, P9B}, {H11B, H12B});     // lambda1_6_1
  test_assign(ec, 1.0, {H2B, H7B}, {H11B, H12B},
                       {H2B, H7B}, {H11B, H12B});     // lambda1_6_2_1
  test_assign(ec, 1.0, {H2B, H7B}, {H12B, P3B},
                       {H2B, H7B}, {H12B, P3B});     // lambda1_6_2_2_1
  test_assign(ec, 1.0, {H2B, P9B}, {H12B, P3B},
                       {H2B, P9B}, {H12B, P3B});      // lambda1_6_3_1
  test_assign(ec, 1.0, {H2B}, {P5B},
                       {H2B}, {P5B});                 // lambda1_6_4_1
  test_assign(ec, 1.0, {H2B, H6B}, {H12B, P4B},
                       {H2B, H6B}, {H12B, P4B});      // lambda1_6_5_1
  test_assign(ec, -1.0, {P5B, P8B}, {H7B, P1B},
                       {P5B, P8B}, {H7B, P1B});       // lambda1_7_1
  test_assign(ec, 1.0, {P9B}, {H10B},
                       {P9B}, {H10B});                // lambda1_8_1
}

void test_assign_ccsd_lambda2(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {H3B, H4B}, {P1B, P2B},
                       {H3B, H4B}, {P1B, P2B});      // lambda2_2_1
  test_assign(ec, 1.0, {H3B}, {P1B},
                       {H3B}, {P1B});                // lambda2_2_1
  test_assign(ec, 1.0, {H3B, H4B}, {H7B, P1B},
                       {H3B, H4B}, {H7B, P1B});      // lambda2_3_1
  test_assign(ec, 1.0, {H3B}, {H9B},
                       {H3B}, {H9B});                // lambda2_5_1
  test_assign(ec, 1.0, {H3B}, {P5B},
                       {H3B}, {P5B});                // lambda2_5_2_1
  test_assign(ec, 1.0, {P10B}, {P1B},
                       {P10B}, {P1B});               // lambda2_6_1
  test_assign(ec, 1.0, {H3B, H4B}, {H9B, H10B},
                       {H3B, H4B}, {H9B, H10B});     // lambda2_7_1
  test_assign(ec, 1.0, {H3B, H4B}, {H10B, P5B},
                       {H3B, H4B}, {H10B, P5B});     // lambda2_7_2_1
  test_assign(ec, 1.0, {H3B, P7B}, {H9B, P1B},
                       {H3B, P7B}, {H9B, P1B});      // lambda2_8_1
}

void test_assign_eaccsd_x1(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {P2B}, {P6B},
                       {P2B}, {P6B});               // eaccsd_x1_1_1
  test_assign(ec, 1.0, {P6B}, {P7B},
                       {P6B}, {P7B});               // eaccsd_x1_2_1
  test_assign(ec, 1.0, {H3B}, {P7B},
                       {H3B}, {P7B});               // eaccsd_x1_4_1_1
}

void test_assign_eaccsd_x2(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {H8B}, {H1B},
                       {H8B}, {H1B});               // eaccsd_x2_2_1
  test_assign(ec, 1.0, {H8B}, {P9B},
                       {H8B}, {P9B});               // eaccsd_x2_2_2_1
  test_assign(ec, 1.0, {P3B}, {P8B},
                       {P3B}, {P8B});               // eaccsd_x2_3_1
  test_assign(ec, 1.0, {H7B, P3B}, {H1B, P8B},
                       {H7B, P3B}, {H1B, P8B});     // eaccsd_x2_4_1
  test_assign(ec, 1.0, {H9B}, {P5B},
                       {H9B}, {P5B});               // eaccsd_x2_6_2_1
  test_assign(ec, 1.0, {H8B, H9B}, {H1B, P10B},
                       {H8B, H9B}, {H1B, P10B});    // eaccsd_x2_6_3_1
  test_assign(ec, 1.0, {H5B}, {P9B},
                       {H5B}, {P9B});               // eaccsd_x2_8_1_1
}

void test_assign_icsd_t1(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {P2B}, {H1B},
                       {P2B}, {H1B});               // icsd_t1_1
  test_assign(ec, 1.0, {H7B}, {H1B},
                       {H7B}, {H1B});               // icsd_t1_2_1
  test_assign(ec, 1.0, {H7B}, {P3B},
                       {H7B}, {P3B});               // icsd_t1_2_2_1
  test_assign(ec, 1.0, {P2B}, {P3B},
                       {P2B}, {P3B});               // icsd_t1_3_1
  test_assign(ec, 1.0, {H8B}, {P7B},
                       {H8B}, {P7B});               // icsd_t1_5_1
  test_assign(ec, 1.0, {H4B, H5B}, {H1B, P3B},
                       {H4B, H5B}, {H1B, P3B});     // icsd_t1_6_1
}

void test_assign_icsd_t2(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {P3B, P4B}, {H1B, H2B},
                       {P3B, P4B}, {H1B, H2B});     // icsd_t2_1
  test_assign(ec, 1.0, {H10B, P3B}, {H1B, H2B},
                       {H10B, P3B}, {H1B, H2B});    // icsd_t2_2_1
  test_assign(ec, -1.0, {H10B, H11B}, {H1B, H2B},
                       {H10B, H11B}, {H1B, H2B});   // icsd_t2_2_2_1
  test_assign(ec, 1.0, {H10B, H11B}, {H1B, P5B},
                       {H10B, H11B}, {H1B, P5B});   // icsd_t2_2_2_2_1
  test_assign(ec, 1.0, {H10B}, {P5B},
                       {H10B}, {P5B});              // icsd_t2_2_4_1
  test_assign(ec, 1.0, {H7B, H10B}, {H1B, P9B},
                       {H7B, H10B}, {H1B, P9B});    // icsd_t2_2_5_1
  test_assign(ec, 1.0, {H9B}, {H1B},
                       {H9B}, {H1B});               // icsd_t2_4_1
  test_assign(ec, 1.0, {H9B}, {P8B},
                       {H9B}, {P8B});               // icsd_t2_4_2_1
  test_assign(ec, 1.0, {P3B}, {P5B},
                       {P3B}, {P5B});               // icsd_t2_5_1
  test_assign(ec, -1.0, {H9B, H11B}, {H1B, H2B},
                       {H9B, H11B}, {H1B, H2B});     // icsd_t2_6_1
  test_assign(ec, 1.0, {H9B, H11B}, {H1B, P8B},
                       {H9B, H11B}, {H1B, P8B});     // icsd_t2_6_2_1
  test_assign(ec, 1.0, {H6B, P3B}, {H1B, P5B},
                       {H6B, P3B}, {H1B, P5B});     // icsd_t2_7_1
}

void test_assign_ipccsd_x1(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {H6B}, {H1B},
                       {H6B}, {H1B});               // ipccsd_x1_1_1
  test_assign(ec, 1.0, {H6B}, {P7B},
                       {H6B}, {P7B});               // ipccsd_x1_1_2_1
  test_assign(ec, 1.0, {H6B}, {P7B},
                       {H6B}, {P7B});               // ipccsd_x1_2_1
  test_assign(ec, 1.0, {H6B, H8B}, {H1B, P7B},
                       {H6B, H8B}, {H1B, P7B});     // ipccsd_x1_3_1
}

void test_assign_ipccsd_x2(tammx::ExecutionContext& ec) {
  test_assign(ec, 1.0, {H9B, P3B}, {H1B, H2B},
                       {H9B, P3B}, {H1B, H2B});     // ipccsd_x2_1_1
  test_assign(ec, 1.0, {H9B, P3B}, {H1B, P5B},
                       {H9B, P3B}, {H1B, P5B});     // ipccsd_x2_1_2_1
  test_assign(ec, 1.0, {H9B}, {P8B},
                       {H9B}, {P8B});               // ipccsd_x1_3_1
  test_assign(ec, 1.0, {H6B, H9B}, {H1B, P5B},
                       {H6B, H9B}, {H1B, P5B});     // ipccsd_x2_1_4_1
  test_assign(ec, 1.0, {H8B}, {H1B},
                       {H8B}, {H1B});               // ipccsd_x2_2_1
  test_assign(ec, 1.0, {H8B}, {P9B},
                       {H8B}, {P9B});               // ipccsd_x2_2_2_1
  test_assign(ec, 1.0, {P3B}, {P8B},
                       {P3B}, {P8B});               // ipccsd_x2_3_1
  test_assign(ec, 1.0, {H9B, H10B}, {H1B, H2B},
                       {H9B, H10B}, {H1B, H2B});     // ipccsd_x2_4_1
  test_assign(ec, 1.0, {H9B, H10B}, {H1B, P5B},
                       {H9B, H10B}, {H1B, P5B});     // ipccsd_x2_4_2_1
  test_assign(ec, 1.0, {H7B, P3B}, {H1B, P8B},
                       {H7B, P3B}, {H1B, P8B});     // ipccsd_x2_5_1
  test_assign(ec, 1.0, {H8B, H10B}, {H1B, P5B},
                       {H8B, H10B}, {H1B, P5B});     // ipccsd_x2_6_1_2_1
  test_assign(ec, 1.0, {H10B}, {P5B},
                       {H10B}, {P5B});               // ipccsd_x2_6_2_1
  test_assign(ec, 1.0, {H8B, H10B}, {H1B, P9B},
                       {H8B, H10B}, {H1B, P9B});     // ipccsd_x2_6_3_1
}

void test_assign_4d(tammx::ExecutionContext& ec) {
  //test_assign_no_n(ec, 0.24, {H1B, H2B}, {H3B, H4B}, {H1B, H2B}, {H3B, H4B});
  test_assign_no_n(ec, 0.24, {H1B, H2B}, {H3B, H4B}, {H1B, H2B}, {H4B, H3B});
  // test_assign_no_n(ec, 1.23, {H4B}, {H1B}, {H1B}, {H4B});

  // test_assign_no_n(ec, 0.24, {H4B}, {P1B}, {H4B}, {P1B});
  // test_assign_no_n(ec, 1.23, {H4B}, {P1B}, {P1B}, {H4B});

  // test_assign_no_n(ec, 0.24, {P1B}, {H1B}, {P1B}, {H1B});
  // test_assign_no_n(ec, 1.23, {P1B}, {H1B}, {H1B}, {P1B});

  // test_assign_no_n(ec, 0.24, {P4B}, {P1B}, {P4B}, {P1B});
  // test_assign_no_n(ec, 1.23, {P4B}, {P1B}, {P1B}, {P4B});
}

void test_mult_vo_oo(tammx::ExecutionContext& ec) {
  auto tc_c = tamm_tensor({TV}, {TO});
  auto tc_f = tamm_tensor({TV}, {TO});
  auto ta = tamm_tensor({TV}, {TO}, 0, tamm::dist_nwma);
  auto tb = tamm_tensor({TO}, {TO});

  tamm_create(&ta, &tb, &tc_c, &tc_f);
  tb.fill_random();
  ta.fill_given(2.0);

  tamm_mult(&tc_c, {P1B, H1B}, -1.0, &ta, {P1B, H4B}, &tb, {H4B, H1B});
  //fortran_mult(&tc_f, &ta, &tb, ccsd_t1_2_);

  assert_result(tc_c.check_correctness(&tc_f), __func__);

  tamm_destroy(&ta, &tb, &tc_c, &tc_f);
}

void test_mult_vvoo_ov(tammx::ExecutionContext& ec) {
  auto tc_c = tamm_tensor({TV}, {TO});
  auto tc_f = tamm_tensor({TV}, {TO});
  auto ta = tamm_tensor({TV,TV}, {TO,TO}, 0, tamm::dist_nw);
  auto tb = tamm_tensor({TO}, {TV});

  tamm_create(&ta, &tb, &tc_c, &tc_f);
  ta.fill_random();
  tb.fill_given(2.0);
  tamm_mult(&tc_c, {P1B, H1B}, 1.0, &ta, {P1B, P2B, H1B, H4B},
		  &tb, {H4B, P2B});
  fortran_mult_vvoo_vo(&tc_f, &ta, &tb, cc2_t1_5_);

  assert_result(tc_c.check_correctness(&tc_f), __func__);

  tamm_destroy(&ta, &tb, &tc_c, &tc_f);
}

void fortran_init(int noa, int nob, int nva, int nvb, bool intorb, bool restricted,
                  const std::vector<int>& spins,
                  const std::vector<int>& syms,
                  const std::vector<int>& ranges) {
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


/*
 * @note should be called after fortran_init
 */
void tamm_init(...) {
  f_calls_setvars_cxx_();  
}

void tamm_finalize() {
  //no-op
}

void tammx_init(int noa, int nob, int nva, int nvb, bool intorb, bool restricted,
                const std::vector<int>& ispins,
                const std::vector<int>& isyms,
                const std::vector<int>& isizes) {
  using Irrep = tammx::Irrep;
  using Spin = tammx::Spin;
  using BlockDim = tammx::BlockDim;
  
  Irrep irrep_f{0}, irrep_v{0}, irrep_t{0}, irrep_x{0}, irrep_y{0};

  std::vector<Spin> spins;
  std::vector<Irrep> irreps;
  std::vector<size_t> sizes;

  for(auto s : ispins) {
    spins.push_back(Spin{s});
  }
  for(auto r : isyms) {
    irreps.push_back(Irrep{r});
  }
  for(auto s : isizes) {
    sizes.push_back(size_t{s});
  }
  
  tammx::TCE::init(spins, irreps, sizes,
                   BlockDim{noa},
                   BlockDim{noa+nob},
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


int main(int argc, char *argv[]) {
  // int noa = 1;
  // int nob = 1;
  // int nva = 1;
  // int nvb = 1;
  int noa = 2;
  int nob = 2;
  int nva = 2;
  int nvb = 2;

  bool intorb = false;
  bool restricted = false;

  // std::vector<int> spins = {1, 2, 1, 2};
  // std::vector<int> syms = {0, 0, 0, 0};
  // std::vector<int> ranges = {4, 4, 4, 4};
  std::vector<int> spins = {1, 1, 2, 2, 1, 1, 2, 2};
  std::vector<int> syms = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> ranges = {4, 4, 4, 4, 4, 4, 4, 4};

  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(MT_DBL, 1000000, 8000000);

  fortran_init(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);    
  tamm_init(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);    
  tammx_init(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);    

  tammx::ProcGroup pg {tammx::ProcGroup{MPI_COMM_WORLD}.clone()};
  auto default_distribution = tammx::Distribution_NW();
  tammx::MemoryManagerGA default_memory_manager{pg};
  auto default_irrep = tammx::Irrep{0};
  auto default_spin_restricted = false;

  ::testing::InitGoogleTest(&argc, argv);

  int ret = 0;
  {  
    tammx::ExecutionContext ec {pg, &default_distribution, &default_memory_manager,
          default_irrep, default_spin_restricted};

    testing::AddGlobalTestEnvironment(new TestEnvironment(&ec));
    ret = RUN_ALL_TESTS();
    // test_assign_2d(ec);
    // test_assign_4d(ec);
    // test_assign(ec);
    // test_mult_vo_oo(ec);
    // test_mult_vvoo_ov(ec);
    // CCSD methods
    test_assign_ccsd_e(ec);
    test_assign_ccsd_t1(ec);
    test_assign_ccsd_t2(ec);
    test_assign_cc2_t1(ec);
    test_assign_cc2_t2(ec);

    test_assign_cisd_c1(ec);
    test_assign_cisd_c2(ec);
    test_assign_ccsd_lambda1(ec);
    test_assign_ccsd_lambda2(ec);
    test_assign_eaccsd_x1(ec);
    test_assign_eaccsd_x2(ec);

    test_assign_icsd_t1(ec);
    test_assign_icsd_t2(ec);

    test_assign_ipccsd_x1(ec);
    test_assign_ipccsd_x2(ec);
  }
  pg.destroy();
  tammx_finalize();
  tamm_finalize();
  fortran_finalize();
  
  GA_Terminate();
  MPI_Finalize();
  return ret;
}
