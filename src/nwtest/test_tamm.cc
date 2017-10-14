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
#include "nwtest/test_tamm.h"
#include "nwtest/test_tammx.h"
#include "nwtest/test_fortran.h"
#include "nwtest/test_tamm_tce.h"


tammx::ExecutionContext* g_ec;


//class TestEnvironment : public testing::Environment {
//public:
//    explicit TestEnvironment(tammx::ExecutionContext* ec) {
//        g_ec = ec;
//    }
//};

extern "C" {
void init_fortran_vars_(Integer *noa1, Integer *nob1, Integer *nva1,
                        Integer *nvb1, logical *intorb1, logical *restricted1,
                        Integer *spins, Integer *syms, Integer *ranges);
void finalize_fortran_vars_();
void f_calls_setvars_cxx_();
}

tamm::RangeType
tamm_idname_to_tamm_range(const tamm::IndexName &idname) {
  return (idname >= tamm::H1B && idname <= tamm::H12B)
         ? tamm::RangeType::TO : tamm::RangeType::TV;
}

void
fortran_mult_vvoo_vo(tamm::Tensor *tc,
                     tamm::Tensor *ta,
                     tamm::Tensor *tb,
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

  //void
//fortran_mult_vvoo_vo(tamm::Tensor* tc,
//             tamm::Tensor* ta,
//             tamm::Tensor* tb,
//             mult_fn_2 fn) {
//  Integer da = static_cast<Integer>(ta->ga().ga()),
//      offseta = ta->offset_index(),
//      irrepa = ta->irrep();
//  Integer db = static_cast<Integer>(tb->ga().ga()),
//      offsetb = tb->offset_index(),
//      irrepb = tb->irrep();
//  Integer dc = static_cast<Integer>(tc->ga().ga()),
//      offsetc = tc->offset_index(),
//      irrepc = tc->irrep();
//  fn(&da, &offseta, &db, &offsetb, &dc, &offsetc);
//}

#define ASSIGN_TEST_0D 1
#define ASSIGN_TEST_1D 1
#define ASSIGN_TEST_2D 1
#define ASSIGN_TEST_3D 1
#define ASSIGN_TEST_4D 1


#define MULT_TEST_1D_1D 1
#define MULT_TEST_1D_1D_2D 1
#define MULT_TEST_1D_2D_1D 1
#define MULT_TEST_1D_2D_3D 1
#define MULT_TEST_1D_3D_2D 1
#define MULT_TEST_1D_3D_4D 1
#define MULT_TEST_1D_4D_3D 1
#define MULT_TEST_2D_1D_1D 1
#define MULT_TEST_2D_2D_2D 1
#define MULT_TEST_2D_2D_4D 1
#define MULT_TEST_2D_4D_2D 1
#define MULT_TEST_3D_1D_2D 1
#define MULT_TEST_4D_2D_2D 1



using namespace tammx;


//
////-----------------------------------------------------------------------
////
////                            Symmetrization add
////
////-----------------------------------------------------------------------

  // tamm does not support this functionality

////-----------------------------------------------------------------------
////
////                            Mult 0-d
////
////-----------------------------------------------------------------------

  // tamm does not support this functionality


/*
 * @note should be called after fortran_init
 */
void tamm_init() {
  f_calls_setvars_cxx_();
}

void tamm_finalize() {
  //no-op
}


//////////////////////////////////////////////////////
//
//           tamm stuff
//
/////////////////////////////////////////////////////

void
tamm_assign(tamm::Tensor *tc,
            const std::vector<tamm::IndexName> &clabel,
            double alpha,
            tamm::Tensor *ta,
            const std::vector<tamm::IndexName> &alabel) {
  tamm::Assignment as(tc, ta, alpha, clabel, alabel);
  as.execute();
}

void
tamm_mult(tamm::Tensor *tc,
          const std::vector<tamm::IndexName> &clabel,
          double alpha,
          tamm::Tensor *ta,
          const std::vector<tamm::IndexName> &alabel,
          tamm::Tensor *tb,
          const std::vector<tamm::IndexName> &blabel) {
  tamm::Multiplication mult(tc, clabel, ta, alabel, tb, blabel, alpha);
  mult.execute();
}

/////////////////////////////////////////////////////////
//
//             tamm vx tammx
//
//
////////////////////////////////////////////////////////


tamm::IndexName
tammx_label_to_tamm_label(const tammx::IndexLabel &label) {
  tamm::IndexName ret;

  if (label.rt().dt() == tammx::DimType::o) {
    ret = static_cast<tamm::IndexName>(tamm::H1B + label.label);
  } else if (label.rt().dt() == tammx::DimType::v) {
    ret = static_cast<tamm::IndexName>(tamm::P1B + label.label);
  } else {
    assert(0); //@note unsupported
  }
  return ret;
}

std::vector<tamm::IndexName>
tammx_label_to_tamm_label(const tammx::IndexLabelVec &label) {
  std::vector<tamm::IndexName> ret;
  for (auto l: label) {
    ret.push_back(tammx_label_to_tamm_label(l));
  }
  return ret;
}

std::vector<tamm::RangeType>
tamm_labels_to_ranges(const std::vector<tamm::IndexName> &labels) {
  std::vector<tamm::RangeType> ret;
  for (auto l : labels) {
    ret.push_back(tamm_idname_to_tamm_range(l));
  }
  return ret;
}

tamm::RangeType
tammx_range_to_tamm_rangetype(tammx::RangeType rt) {
  tamm::RangeType ret;
  switch (rt.dt()) {
    case DimType::o:
      return tamm::TO;
      break;
    case DimType::v:
      return tamm::TV;
      break;
    case DimType::n:
      return tamm::TN;
      break;
    default:
      assert(0);
  }
  return ret;
}

std::pair<tamm::Tensor *, Integer *>
tammx_tensor_to_tamm_tensor(tammx::Tensor<double> &ttensor) {
  TAMMX_INT32 ndim = ttensor.rank();
  TAMMX_INT32 nupper = ttensor.nupper_indices();
  TAMMX_INT32 irrep = ttensor.irrep().value();
  tamm::DistType dist_type = tamm::dist_nw;

  std::vector<tamm::RangeType> rt;
  for (auto id: ttensor.flindices()) {
    rt.push_back(tammx_range_to_tamm_rangetype(id));
  }

  auto ptensor = new tamm::Tensor{static_cast<int>(ndim), static_cast<int>(nupper), static_cast<int>(irrep), &rt[0], dist_type};
  auto dst_nw = static_cast<const tammx::Distribution_NW *>(ttensor.distribution());
  auto map = dst_nw->hash();
  auto length = 2 * map[0] + 1;
  Integer *offset_map = new Integer[length];
  for (TAMMX_SIZE i = 0; i < length; i++) {
    offset_map[i] = map[i];
  }

  // std::cout << "tensor tammx -----------\n";
  // for (TAMMX_SIZE i = 0; i < length; i++) {
  //   std::cout << map[i] << ",";
  // }
  // std::cout << std::endl;

  auto mgr_ga = static_cast<tammx::MemoryManagerGA *>(ttensor.memory_manager());

  auto fma_offset_index = offset_map - tamm::Variables::int_mb();
  auto fma_offset_handle = -1; //@todo @bug FIX THIS
  auto array_handle = mgr_ga->ga();
  ptensor->attach(fma_offset_index, fma_offset_handle, array_handle);
  return {ptensor, offset_map};
}

void
tamm_assign(tammx::Tensor<double> &ttc,
            const tammx::IndexLabelVec &tclabel,
            double alpha,
            tammx::Tensor<double> &tta,
            const tammx::IndexLabelVec &talabel) {
  tamm::Tensor *ta, *tc;
  Integer *amap, *cmap;
  std::tie(ta, amap) = tammx_tensor_to_tamm_tensor(tta);
  std::tie(tc, cmap) = tammx_tensor_to_tamm_tensor(ttc);
  const auto &clabel = tammx_label_to_tamm_label(tclabel);
  const auto &alabel = tammx_label_to_tamm_label(talabel);
  tamm_assign(tc, clabel, alpha, ta, alabel);
  delete ta;
  delete tc;
  delete[] amap;
  delete[] cmap;
}

void
tamm_mult(tammx::Tensor<double> &ttc,
          const tammx::IndexLabelVec &tclabel,
          double alpha,
          tammx::Tensor<double> &tta,
          const tammx::IndexLabelVec &talabel,
          tammx::Tensor<double> &ttb,
          const tammx::IndexLabelVec &tblabel) {
  tamm::Tensor *ta, *tb, *tc;
  Integer *amap, *bmap, *cmap;
  std::tie(ta, amap) = tammx_tensor_to_tamm_tensor(tta);
  std::tie(tb, bmap) = tammx_tensor_to_tamm_tensor(ttb);
  std::tie(tc, cmap) = tammx_tensor_to_tamm_tensor(ttc);
  const auto &clabel = tammx_label_to_tamm_label(tclabel);
  const auto &alabel = tammx_label_to_tamm_label(talabel);
  const auto &blabel = tammx_label_to_tamm_label(tblabel);
  tamm_mult(tc, clabel, alpha, ta, alabel, tb, blabel);
  delete ta;
  delete tb;
  delete tc;
  delete[] amap;
  delete[] bmap;
  delete[] cmap;
}


bool
test_assign(tammx::ExecutionContext &ec,
            double alpha,
            const tammx::IndexLabelVec &cupper_labels,
            const tammx::IndexLabelVec &clower_labels,
            const tammx::IndexLabelVec &aupper_labels,
            const tammx::IndexLabelVec &alower_labels,
            AllocationType at) {
  const auto& cindices = tammx_label_to_indices(cupper_labels, clower_labels, is_lhs_n(at));
  const auto& aindices = tammx_label_to_indices(aupper_labels, alower_labels, is_rhs1_n(at));
  auto irrep = ec.irrep();
  auto restricted = ec.is_spin_restricted();
  auto cnup = cupper_labels.size();
  auto anup = aupper_labels.size();

  tammx::Tensor<double> tc1{cindices, cnup, irrep, restricted};
  tammx::Tensor<double> tc2{cindices, cnup, irrep, restricted};
  tammx::Tensor<double> ta{aindices, anup, irrep, restricted};

  ec.allocate(ta, tc1, tc2);

  ec.scheduler()
    .io(ta, tc1, tc2)
      (ta() = 0)
      (tc1() = 0)
      (tc2() = 0)
    .execute();


  tammx_tensor_fill(ec, ta());

  auto clabels = cupper_labels;
  clabels.insert_back(clower_labels.begin(), clower_labels.end());
  auto alabels = aupper_labels;
  alabels.insert_back(alower_labels.begin(), alower_labels.end());

  tamm_assign(tc1, clabels, alpha, ta, alabels);
  tammx_assign(ec, tc2, clabels, alpha, ta, alabels);

  bool status = tammx_tensors_are_equal(ec, tc1, tc2);

  ec.deallocate(tc1, tc2, ta);
  return status;
}

bool
test_assign(tammx::ExecutionContext &ec,
            const tammx::IndexLabelVec &cupper_labels,
            const tammx::IndexLabelVec &clower_labels,
            double alpha,
            const tammx::IndexLabelVec &aupper_labels,
            const tammx::IndexLabelVec &alower_labels,
            AllocationType at) {
  return test_assign(ec, alpha,
                     cupper_labels, clower_labels,
                     aupper_labels, alower_labels,
                     at);
}


bool
test_mult_no_n(tammx::ExecutionContext &ec,
               double alpha,
               const tammx::IndexLabelVec &cupper_labels,
               const tammx::IndexLabelVec &clower_labels,
               const tammx::IndexLabelVec &aupper_labels,
               const tammx::IndexLabelVec &alower_labels,
               const tammx::IndexLabelVec &bupper_labels,
               const tammx::IndexLabelVec &blower_labels) {
  const auto &cupper_indices = tammx_label_to_indices(cupper_labels);
  const auto &clower_indices = tammx_label_to_indices(clower_labels);
  const auto &aupper_indices = tammx_label_to_indices(aupper_labels);
  const auto &alower_indices = tammx_label_to_indices(alower_labels);
  const auto &bupper_indices = tammx_label_to_indices(bupper_labels);
  const auto &blower_indices = tammx_label_to_indices(blower_labels);

  auto cindices = cupper_indices;
  cindices.insert_back(clower_indices.begin(), clower_indices.end());
  auto aindices = aupper_indices;
  aindices.insert_back(alower_indices.begin(), alower_indices.end());
  auto bindices = bupper_indices;
  bindices.insert_back(blower_indices.begin(), blower_indices.end());
  auto irrep = ec.irrep();
  auto restricted = ec.is_spin_restricted();
  auto cnup = cupper_labels.size();
  auto anup = aupper_labels.size();
  auto bnup = bupper_labels.size();

  tammx::Tensor<double> tc1{cindices, cnup, irrep, restricted};
  tammx::Tensor<double> tc2{cindices, cnup, irrep, restricted};
  tammx::Tensor<double> ta{aindices, anup, irrep, restricted};
  tammx::Tensor<double> tb{bindices, bnup, irrep, restricted};

  ec.allocate(ta, tb, tc1, tc2);

  ec.scheduler()
    .io(ta, tb, tc1, tc2)
      (ta() = 0)
      (tb() = 0)
      (tc1() = 0)
      (tc2() = 0)
    .execute();


  tammx_tensor_fill(ec, ta());
  tammx_tensor_fill(ec, tb());

  auto clabels = cupper_labels;
  clabels.insert_back(clower_labels.begin(), clower_labels.end());
  auto alabels = aupper_labels;
  alabels.insert_back(alower_labels.begin(), alower_labels.end());
  auto blabels = bupper_labels;
  blabels.insert_back(blower_labels.begin(), blower_labels.end());

  tamm_mult(tc1, clabels, alpha, ta, alabels, tb, blabels);
  tammx_mult(ec, tc2, clabels, alpha, ta, alabels, tb, blabels);

  bool status = tammx_tensors_are_equal(ec, tc1, tc2);

  ec.deallocate(tc1, tc2, ta, tb);
  return status;
}


bool
test_mult_no_n(tammx::ExecutionContext &ec,
               const tammx::IndexLabelVec &cupper_labels,
               const tammx::IndexLabelVec &clower_labels,
               double alpha,
               const tammx::IndexLabelVec &aupper_labels,
               const tammx::IndexLabelVec &alower_labels,
               const tammx::IndexLabelVec &bupper_labels,
               const tammx::IndexLabelVec &blower_labels) {
  return test_mult_no_n(ec, alpha, cupper_labels, clower_labels,
                        aupper_labels, alower_labels, bupper_labels, blower_labels);
}

tamm::RangeType
tamm_id_to_tamm_range(const tamm::Index &id) {
  return tamm_idname_to_tamm_range(id.name());
}

tammx::DimType
tamm_range_to_tammx_dim(tamm::RangeType rt) {
  tammx::DimType ret;
  switch (rt) {
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
tamm_id_to_tammx_dim(const tamm::Index &id) {
  return tamm_range_to_tammx_dim(tamm_id_to_tamm_range(id));
}


tammx::TensorVec <tammx::TensorSymmGroup>
tamm_labels_to_tammx_indices(const std::vector<tamm::IndexName> &labels) {
  tammx::DimTypeVec tammx_dims;
  for (const auto l : labels) {
    tammx_dims.push_back(tamm_range_to_tammx_dim(tamm_idname_to_tamm_range(l)));
  }
  return tammx_tensor_dim_to_symm_groups(tammx_dims, tammx_dims.size());
}


//-----------------------------------------------------------------------
//
//                            Add 1-d
//
//-----------------------------------------------------------------------

#if ASSIGN_TEST_1D

TEST (AssignTest, OneDim_o1e_o1e
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {}, {h1}, {}));
}

TEST (AssignTest, OneDim_eo1_eo1
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {}, {h1}, {}, {h1}));
}

TEST (AssignTest, OneDim_v1e_v1e
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {}, {p1}, {}));
}

TEST (AssignTest, OneDim_ev1_ev1
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {}, {p1}, {}, {p1}));
}

TEST (AssignTest, OneDim_o1e_o1e_right_n
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {}, {h1}, {}, AllocationType::rhs1_n));
}

TEST (AssignTest, OneDim_eo1_eo1_right_n
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {}, {h1}, {}, {h1}, AllocationType::rhs1_n));
}

TEST (AssignTest, OneDim_v1e_v1e_right_n
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {}, {p1}, {}, AllocationType::rhs1_n));
}

TEST (AssignTest, OneDim_ev1_ev1_right_n
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {}, {p1}, {}, {p1}, AllocationType::rhs1_n));
}

#endif


//-----------------------------------------------------------------------
//
//                            Add 2-d
//
//-----------------------------------------------------------------------


#if ASSIGN_TEST_2D

TEST (AssignTest, TwoDim_O1O2_O1O2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h4}, {h1}, {h4}, {h1}));
}

TEST (AssignTest, TwoDim_O1O2_O2O1
) {
ASSERT_TRUE(test_assign(*g_ec, 1.23, {h4}, {h1}, {h1}, {h4}));
}

TEST (AssignTest, TwoDim_OV_OV
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h4}, {p1}, {h4}, {p1}));
}

TEST (AssignTest, TwoDim_OV_VO
) {
ASSERT_TRUE(test_assign(*g_ec, 1.23, {h4}, {p1}, {p1}, {h4}));
}

TEST (AssignTest, TwoDim_VO_VO
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {h1}, {p1}, {h1}));
}

TEST (AssignTest, TwoDim_VO_OV
) {
ASSERT_TRUE(test_assign(*g_ec, 1.23, {p1}, {h1}, {h1}, {p1}));
}

TEST (AssignTest, TwoDim_V1V2_V1V2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p4}, {p1}, {p4}, {p1}));
}

TEST (AssignTest, TwoDim_V1V2_V2V1
) {
ASSERT_TRUE(test_assign(*g_ec, 1.23, {p4}, {p1}, {p1}, {p4}));
}

//

TEST (AssignTest, TwoDim_O1O2_O1O2_right_n
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h4}, {h1}, {h4}, {h1}, AllocationType::rhs1_n));
}

TEST (AssignTest, TwoDim_O1O2_O2O1_right_n
) {
ASSERT_TRUE(test_assign(*g_ec, 1.23, {h4}, {h1}, {h1}, {h4}, AllocationType::rhs1_n));
}

TEST (AssignTest, TwoDim_OV_OV_right_n
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h4}, {p1}, {h4}, {p1}, AllocationType::rhs1_n));
}

TEST (AssignTest, TwoDim_OV_VO_right_n
) {
ASSERT_TRUE(test_assign(*g_ec, 1.23, {h4}, {p1}, {p1}, {h4}, AllocationType::rhs1_n));
}

TEST (AssignTest, TwoDim_VO_VO_right_n
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {h1}, {p1}, {h1}, AllocationType::rhs1_n));
}

TEST (AssignTest, TwoDim_VO_OV_right_n
) {
ASSERT_TRUE(test_assign(*g_ec, 1.23, {p1}, {h1}, {h1}, {p1}, AllocationType::rhs1_n));
}

TEST (AssignTest, TwoDim_V1V2_V1V2_right_n
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p4}, {p1}, {p4}, {p1}, AllocationType::rhs1_n));
}

TEST (AssignTest, TwoDim_V1V2_V2V1_right_n
) {
ASSERT_TRUE(test_assign(*g_ec, 1.23, {p4}, {p1}, {p1}, {p4}, AllocationType::rhs1_n));
}


#endif

//////////////////////////////////////

#if MULT_TEST_2D_2D_2D

TEST (MultTest, Dim_oo_oo_oo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {h2},
                           {h1}, {h3},
                           {h3}, {h2}));
}

TEST (MultTest, Dim_oo_ov_vo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {h2},
                           {h1}, {p3},
                           {p3}, {h2}));
}

TEST (MultTest, Dim_ov_oo_ov
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {p2},
                           {h1}, {h3},
                           {h3}, {p2}));
}

TEST (MultTest, Dim_ov_ov_vv
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {p2},
                           {h1}, {p3},
                           {p3}, {p2}));
}

TEST (MultTest, Dim_vo_vo_oo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {h2},
                           {p1}, {h3},
                           {h3}, {h2}));
}

TEST (MultTest, Dim_vo_vv_vo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {h2},
                           {p1}, {p3},
                           {p3}, {h2}));
}

TEST (MultTest, Dim_vv_vo_ov
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {p2},
                           {p1}, {h3},
                           {h3}, {p2}));
}

TEST (MultTest, Dim_vv_vv_vv
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {p2},
                           {p1}, {p3},
                           {p3}, {p2}));
}

#endif

#if MULT_TEST_2D_2D_4D

TEST (MultTest, Dim_oo_oo_oooo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {h4},
                           {h2}, {h3},
                           {h1, h3}, {h2, h4}));
}


TEST (MultTest, Dim_oo_ov_ovoo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {h4},
                           {h2}, {p3},
                           {h1, p3}, {h2, h4}));
}

TEST (MultTest, Dim_oo_vo_oovo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {h4},
                           {p2}, {h3},
                           {h1, h3}, {p2, h4}));
}

TEST (MultTest, Dim_oo_vv_ovvo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {h4},
                           {p2}, {p3},
                           {h1, p3}, {p2, h4}));
}


/////////////////////////

TEST (MultTest, Dim_ov_oo_ooov
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {p4},
                           {h2}, {h3},
                           {h1, h3}, {h2, p4}));
}


TEST (MultTest, Dim_ov_ov_ovov
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {p4},
                           {h2}, {p3},
                           {h1, p3}, {h2, p4}));
}

TEST (MultTest, Dim_ov_vo_oovv
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {p4},
                           {p2}, {h3},
                           {h1, h3}, {p2, p4}));
}

TEST (MultTest, Dim_ov_vv_ovvv
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {p4},
                           {p2}, {p3},
                           {h1, p3}, {p2, p4}));
}

////////////////////////

TEST (MultTest, Dim_vo_oo_vooo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {h4},
                           {h2}, {h3},
                           {p1, h3}, {h2, h4}));
}


TEST (MultTest, Dim_vo_ov_vvoo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {h4},
                           {h2}, {p3},
                           {p1, p3}, {h2, h4}));
}

TEST (MultTest, Dim_vo_vo_vovo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {h4},
                           {p2}, {h3},
                           {p1, h3}, {p2, h4}));
}

TEST (MultTest, Dim_vo_vv_vvvo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {h4},
                           {p2}, {p3},
                           {p1, p3}, {p2, h4}));
}

////////////////////////

TEST (MultTest, Dim_vv_oo_voov
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {p4},
                           {h2}, {h3},
                           {p1, h3}, {h2, p4}));
}


TEST (MultTest, Dim_vv_ov_vvov
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {p4},
                           {h2}, {p3},
                           {p1, p3}, {h2, p4}));
}

TEST (MultTest, Dim_vv_vo_vovv
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {p4},
                           {p2}, {h3},
                           {p1, h3}, {p2, p4}));
}

TEST (MultTest, Dim_vv_vv_vvvv
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {p4},
                           {p2}, {p3},
                           {p1, p3}, {p2, p4}));
}


#endif

#if MULT_TEST_2D_4D_2D
#endif

#if MULT_TEST_3D_1D_2D

TEST (MultTest, Dim_ovo_ovo_oo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, p2}, {h3},
                           {h1, p2}, {h4},
                           {h4}, {h3}));
}
#endif


#if MULT_TEST_4D_2D_2D

TEST (MultTest, Dim__oo_oo__o_o__o_o
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, h2}, {h3, h4},
                           {h1}, {h3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__oo_ov__o_o__o_v
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, h2}, {h3, p4},
                           {h1}, {h3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__ov_oo__o_o__v_o
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, p2}, {h3, h4},
                           {h1}, {h3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__ov_ov__o_o__v_v
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, p2}, {h3, p4},
                           {h1}, {h3},
                           {p2}, {p4}));
}

////////////

TEST (MultTest, Dim__oo_vo__o_v__o_o
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, h2}, {p3, h4},
                           {h1}, {p3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__oo_vv__o_v__o_v
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, h2}, {p3, p4},
                           {h1}, {p3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__ov_vo__o_v__v_o
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, p2}, {p3, h4},
                           {h1}, {p3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__ov_vv__o_v__v_v
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, p2}, {p3, p4},
                           {h1}, {p3},
                           {p2}, {p4}));
}


////////////

TEST (MultTest, Dim__vo_oo__v_o__o_o
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, h2}, {h3, h4},
                           {p1}, {h3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__vo_ov__v_o__o_v
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, h2}, {h3, p4},
                           {p1}, {h3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__vv_oo__v_o__v_o
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, p2}, {h3, h4},
                           {p1}, {h3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__vv_ov__v_o__v_v
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, p2}, {h3, p4},
                           {p1}, {h3},
                           {p2}, {p4}));
}

////////////

TEST (MultTest, Dim__vo_vo__v_v__o_o
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, h2}, {p3, h4},
                           {p1}, {p3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__vo_vv__v_v__o_v
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, h2}, {p3, p4},
                           {p1}, {p3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__vv_vo__v_v__v_o
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, p2}, {p3, h4},
                           {p1}, {p3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__vv_vv__v_v__v_v
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, p2}, {p3, p4},
                           {p1}, {p3},
                           {p2}, {p4}));
}

/////////////////////////


TEST (MultTest, Dim__oo_oo__o_o__o_o_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           1,
                           {h2, h1}, {h3, h4},
                           {h1}, {h3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__oo_ov__o_o__o_v_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h2, h1}, {h3, p4},
                           {h1}, {h3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__ov_oo__o_o__v_o_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p2, h1}, {h3, h4},
                           {h1}, {h3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__ov_ov__o_o__v_v_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           4.36,
                           {p2, h1}, {h3, p4},
                           {h1}, {h3},
                           {p2}, {p4}));
}


////////////

TEST (MultTest, Dim__oo_vo__o_v__o_o_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h2, h1}, {p3, h4},
                           {h1}, {p3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__oo_vv__o_v__o_v_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h2, h1}, {p3, p4},
                           {h1}, {p3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__ov_vo__o_v__v_o_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p2, h1}, {p3, h4},
                           {h1}, {p3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__ov_vv__o_v__v_v_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p2, h1}, {p3, p4},
                           {h1}, {p3},
                           {p2}, {p4}));
}


////////////

TEST (MultTest, Dim__vo_oo__v_o__o_o_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h2, p1}, {h3, h4},
                           {p1}, {h3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__vo_ov__v_o__o_v_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h2, p1}, {h3, p4},
                           {p1}, {h3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__vv_oo__v_o__v_o_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p2, p1}, {h3, h4},
                           {p1}, {h3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__vv_ov__v_o__v_v_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p2, p1}, {h3, p4},
                           {p1}, {h3},
                           {p2}, {p4}));
}

////////////

TEST (MultTest, Dim__vo_vo__v_v__o_o_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h2, p1}, {p3, h4},
                           {p1}, {p3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__vo_vv__v_v__o_v_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h2, p1}, {p3, p4},
                           {p1}, {p3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__vv_vo__v_v__v_o_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p2, p1}, {p3, h4},
                           {p1}, {p3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__vv_vv__v_v__v_v_upflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p2, p1}, {p3, p4},
                           {p1}, {p3},
                           {p2}, {p4}));
}

///////////////////////////////

TEST (MultTest, Dim__oo_oo__o_o__o_o_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, h2}, {h4, h3},
                           {h1}, {h3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__oo_ov__o_o__o_v_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, h2}, {p4, h3},
                           {h1}, {h3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__ov_oo__o_o__v_o_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, p2}, {h4, h3},
                           {h1}, {h3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__ov_ov__o_o__v_v_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, p2}, {p4, h3},
                           {h1}, {h3},
                           {p2}, {p4}));
}

////////////

TEST (MultTest, Dim__oo_vo__o_v__o_o_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, h2}, {h4, p3},
                           {h1}, {p3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__oo_vv__o_v__o_v_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, h2}, {p4, p3},
                           {h1}, {p3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__ov_vo__o_v__v_o_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, p2}, {h4, p3},
                           {h1}, {p3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__ov_vv__o_v__v_v_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1, p2}, {p4, p3},
                           {h1}, {p3},
                           {p2}, {p4}));
}


////////////

TEST (MultTest, Dim__vo_oo__v_o__o_o_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, h2}, {h4, h3},
                           {p1}, {h3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__vo_ov__v_o__o_v_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, h2}, {p4, h3},
                           {p1}, {h3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__vv_oo__v_o__v_o_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, p2}, {h4, h3},
                           {p1}, {h3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__vv_ov__v_o__v_v_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, p2}, {p4, h3},
                           {p1}, {h3},
                           {p2}, {p4}));
}

////////////

TEST (MultTest, Dim__vo_vo__v_v__o_o_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, h2}, {h4, p3},
                           {p1}, {p3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__vo_vv__v_v__o_v_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, h2}, {p4, p3},
                           {p1}, {p3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__vv_vo__v_v__v_o_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, p2}, {h4, p3},
                           {p1}, {p3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__vv_vv__v_v__v_v_downflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1, p2}, {p4, p3},
                           {p1}, {p3},
                           {p2}, {p4}));
}

/////////////////////////

TEST (MultTest, Dim__oo_oo__o_o__o_o_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h2, h1}, {h4, h3},
                           {h1}, {h3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__oo_ov__o_o__o_v_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h2, h1}, {p4, h3},
                           {h1}, {h3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__ov_oo__o_o__v_o_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p2, h1}, {h4, h3},
                           {h1}, {h3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__ov_ov__o_o__v_v_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p2, h1}, {p4, h3},
                           {h1}, {h3},
                           {p2}, {p4}));
}

////////////

TEST (MultTest, Dim__oo_vo__o_v__o_o_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h2, h1}, {h4, p3},
                           {h1}, {p3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__oo_vv__o_v__o_v_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h2, h1}, {p4, p3},
                           {h1}, {p3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__ov_vo__o_v__v_o_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p2, h1}, {h4, p3},
                           {h1}, {p3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__ov_vv__o_v__v_v_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p2, h1}, {p4, p3},
                           {h1}, {p3},
                           {p2}, {p4}));
}


////////////

TEST (MultTest, Dim__vo_oo__v_o__o_o_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h2, p1}, {h4, h3},
                           {p1}, {h3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__vo_ov__v_o__o_v_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h2, p1}, {p4, h3},
                           {p1}, {h3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__vv_oo__v_o__v_o_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p2, p1}, {h4, h3},
                           {p1}, {h3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__vv_ov__v_o__v_v_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p2, p1}, {p4, h3},
                           {p1}, {h3},
                           {p2}, {p4}));
}

////////////

TEST (MultTest, Dim__vo_vo__v_v__o_o_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h2, p1}, {h4, p3},
                           {p1}, {p3},
                           {h2}, {h4}));
}

TEST (MultTest, Dim__vo_vv__v_v__o_v_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h2, p1}, {p4, p3},
                           {p1}, {p3},
                           {h2}, {p4}));
}

TEST (MultTest, Dim__vv_vo__v_v__v_o_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p2, p1}, {h4, p3},
                           {p1}, {p3},
                           {p2}, {h4}));
}

TEST (MultTest, Dim__vv_vv__v_v__v_v_bothflip
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p2, p1}, {p4, p3},
                           {p1}, {p3},
                           {p2}, {p4}));
}


#endif


//-----------------------------------------------------------------------
//
//                            Add 3-d
//
//-----------------------------------------------------------------------

#if ASSIGN_TEST_3D

TEST (AssignTest, ThreeDim_o1_o2o3__o1_o2o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {h2, h3}, {h1}, {h2, h3}));
}

TEST (AssignTest, ThreeDim_o1_o2o3__o1_o3o2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {h2, h3}, {h1}, {h3, h2}));
}

TEST (AssignTest, ThreeDim_o1_o2v3__o1_o2v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {h2, p3}, {h1}, {h2, p3}));
}

TEST (AssignTest, ThreeDim_o1_o2v3__o1_v3o2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {h2, p3}, {h1}, {p3, h2}));
}

TEST (AssignTest, ThreeDim_o1_v2o3__o1_v2o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {p2, h3}, {h1}, {p2, h3}));
}

TEST (AssignTest, ThreeDim_o1_v2o3__o1_o3v2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {p2, h3}, {h1}, {h3, p2}));
}

TEST (AssignTest, ThreeDim_o1_v2v3__o1_v2v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {p2, p3}, {h1}, {p2, p3}));
}

TEST (AssignTest, ThreeDim_o1_v2v3__o1_v3v2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {p2, p3}, {h1}, {p3, p2}));
}

///////////

TEST (AssignTest, ThreeDim_v1_o2o3__v1_o2o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {h2, h3}, {p1}, {h2, h3}));
}

TEST (AssignTest, ThreeDim_v1_o2o3__v1_o3o2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {h2, h3}, {p1}, {h3, h2}));
}

TEST (AssignTest, ThreeDim_v1_o2v3__v1_o2v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {h2, p3}, {p1}, {h2, p3}));
}

TEST (AssignTest, ThreeDim_v1_o2v3__v1_v3o2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {h2, p3}, {p1}, {p3, h2}));
}

TEST (AssignTest, ThreeDim_v1_v2o3__v1_v2o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {p2, h3}, {p1}, {p2, h3}));
}

TEST (AssignTest, ThreeDim_v1_v2o3__v1_o3v2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {p2, h3}, {p1}, {h3, p2}));
}

TEST (AssignTest, ThreeDim_v1_v2v3__v1_v2v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {p2, p3}, {p1}, {p2, p3}));
}

TEST (AssignTest, ThreeDim_v1_v2v3__v1_v3v2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {p2, p3}, {p1}, {p3, p2}));
}

//////////////////

TEST (AssignTest, ThreeDim_o1o2_o3__o1o2_o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3}, {h1, h2}, {h3}));
}

TEST (AssignTest, ThreeDim_o1o2_o3__o2o1_o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3}, {h2, h1}, {h3}));
}

TEST (AssignTest, ThreeDim_o1o2_v3__o1o2_v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3}, {h1, h2}, {p3}));
}

TEST (AssignTest, ThreeDim_o1o2_v3__o2o1_v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3}, {h2, h1}, {p3}));
}

/////////

TEST (AssignTest, ThreeDim_o1v2_o3__o1v2_o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3}, {h1, p2}, {h3}));
}

TEST (AssignTest, ThreeDim_o1v2_o3__v2o1_o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3}, {p2, h1}, {h3}));
}

TEST (AssignTest, ThreeDim_o1v2_v3__o1v2_v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3}, {h1, p2}, {p3}));
}

TEST (AssignTest, ThreeDim_o1v2_v3__v2o1_v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3}, {p2, h1}, {p3}));
}

//////////////////

TEST (AssignTest, ThreeDim_v1o2_o3__v1o2_o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3}, {p1, h2}, {h3}));
}

TEST (AssignTest, ThreeDim_v1o2_o3__o2v1_o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3}, {h2, p1}, {h3}));
}

TEST (AssignTest, ThreeDim_v1o2_v3__v1o2_v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3}, {p1, h2}, {p3}));
}

TEST (AssignTest, ThreeDim_v1o2_v3__o2v1_v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3}, {h2, p1}, {p3}));
}

/////////

TEST (AssignTest, ThreeDim_v1v2_o3__v1v2_o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3}, {p1, p2}, {h3}));
}

TEST (AssignTest, ThreeDim_v1v2_o3__v2v1_o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3}, {p2, p1}, {h3}));
}

TEST (AssignTest, ThreeDim_v1v2_v3__v1v2_v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3}, {p1, p2}, {p3}));
}

TEST (AssignTest, ThreeDim_v1v2_v3__v2v1_v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3}, {p2, p1}, {p3}));
}

//////////-------///////////////

TEST (AssignTest, ThreeDim_right_n_o1_o2o3__o1_o2o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {h2, h3}, {h1}, {h2, h3}, AllocationType::rhs1_n));
}

TEST (AssignTest, ThreeDim_right_n_o1_o2o3__o1_o3o2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {h2, h3}, {h1}, {h3, h2}, AllocationType::rhs1_n));
}

TEST (AssignTest, ThreeDim_right_n_o1_o2v3__o1_o2v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {h2, p3}, {h1}, {h2, p3}, AllocationType::rhs1_n));
}

// TEST (AssignTest, ThreeDim_right_n_o1_o2v3__o1_v3o2
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {h2, p3}, {h1}, {p3, h2}, AllocationType::rhs1_n));
// }

// TEST (AssignTest, ThreeDim_right_n_o1_v2o3__o1_v2o3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {p2, h3}, {h1}, {p2, h3}, AllocationType::rhs1_n));
// }

TEST (AssignTest, ThreeDim_right_n_o1_v2o3__o1_o3v2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {p2, h3}, {h1}, {h3, p2}, AllocationType::rhs1_n));
}

TEST (AssignTest, ThreeDim_right_n_o1_v2v3__o1_v2v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {p2, p3}, {h1}, {p2, p3}, AllocationType::rhs1_n));
}

TEST (AssignTest, ThreeDim_right_n_o1_v2v3__o1_v3v2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1}, {p2, p3}, {h1}, {p3, p2}, AllocationType::rhs1_n));
}

///////////

TEST (AssignTest, ThreeDim_right_n_v1_o2o3__v1_o2o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {h2, h3}, {p1}, {h2, h3}, AllocationType::rhs1_n));
}

TEST (AssignTest, ThreeDim_right_n_v1_o2o3__v1_o3o2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {h2, h3}, {p1}, {h3, h2}, AllocationType::rhs1_n));
}

TEST (AssignTest, ThreeDim_right_n_v1_o2v3__v1_o2v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {h2, p3}, {p1}, {h2, p3}, AllocationType::rhs1_n));
}

// TEST (AssignTest, ThreeDim_right_n_v1_o2v3__v1_v3o2
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {h2, p3}, {p1}, {p3, h2}, AllocationType::rhs1_n));
// }

// TEST (AssignTest, ThreeDim_right_n_v1_v2o3__v1_v2o3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {p2, h3}, {p1}, {p2, h3}, AllocationType::rhs1_n));
// }

TEST (AssignTest, ThreeDim_right_n_v1_v2o3__v1_o3v2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {p2, h3}, {p1}, {h3, p2}, AllocationType::rhs1_n));
}

TEST (AssignTest, ThreeDim_right_n_v1_v2v3__v1_v2v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {p2, p3}, {p1}, {p2, p3}, AllocationType::rhs1_n));
}

TEST (AssignTest, ThreeDim_right_n_v1_v2v3__v1_v3v2
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1}, {p2, p3}, {p1}, {p3, p2}, AllocationType::rhs1_n));
}

//////////////////

TEST (AssignTest, ThreeDim_right_n_o1o2_o3__o1o2_o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3}, {h1, h2}, {h3}, AllocationType::rhs1_n));
}

TEST (AssignTest, ThreeDim_right_n_o1o2_o3__o2o1_o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3}, {h2, h1}, {h3}, AllocationType::rhs1_n));
}

TEST (AssignTest, ThreeDim_right_n_o1o2_v3__o1o2_v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3}, {h1, h2}, {p3}, AllocationType::rhs1_n));
}

TEST (AssignTest, ThreeDim_right_n_o1o2_v3__o2o1_v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3}, {h2, h1}, {p3}, AllocationType::rhs1_n));
}

/////////

TEST (AssignTest, ThreeDim_right_n_o1v2_o3__o1v2_o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3}, {h1, p2}, {h3}, AllocationType::rhs1_n));
}

// TEST (AssignTest, ThreeDim_right_n_o1v2_o3__v2o1_o3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3}, {p2, h1}, {h3}, AllocationType::rhs1_n));
// }

TEST (AssignTest, ThreeDim_right_n_o1v2_v3__o1v2_v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3}, {h1, p2}, {p3}, AllocationType::rhs1_n));
}

// TEST (AssignTest, ThreeDim_right_n_o1v2_v3__v2o1_v3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3}, {p2, h1}, {p3}, AllocationType::rhs1_n));
// }

//////////////////

// TEST (AssignTest, ThreeDim_right_n_v1o2_o3__v1o2_o3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3}, {p1, h2}, {h3}, AllocationType::rhs1_n));
// }

TEST (AssignTest, ThreeDim_right_n_v1o2_o3__o2v1_o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3}, {h2, p1}, {h3}, AllocationType::rhs1_n));
}

// TEST (AssignTest, ThreeDim_right_n_v1o2_v3__v1o2_v3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3}, {p1, h2}, {p3}, AllocationType::rhs1_n));
// }

TEST (AssignTest, ThreeDim_right_n_v1o2_v3__o2v1_v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3}, {h2, p1}, {p3}, AllocationType::rhs1_n));
}

/////////

TEST (AssignTest, ThreeDim_right_n_v1v2_o3__v1v2_o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3}, {p1, p2}, {h3}, AllocationType::rhs1_n));
}

TEST (AssignTest, ThreeDim_right_n_v1v2_o3__v2v1_o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3}, {p2, p1}, {h3}, AllocationType::rhs1_n));
}

TEST (AssignTest, ThreeDim_right_n_v1v2_v3__v1v2_v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3}, {p1, p2}, {p3}, AllocationType::rhs1_n));
}

TEST (AssignTest, ThreeDim_right_n_v1v2_v3__v2v1_v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3}, {p2, p1}, {p3}, AllocationType::rhs1_n));
}

#endif

//-----------------------------------------------------------------------
//
//                            Add 4-d
//
//-----------------------------------------------------------------------

#if ASSIGN_TEST_4D

TEST (AssignTest, FourDim_o1o2o3o4_o1o2o3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3, h4}, {h1, h2}, {h3, h4}));
}

TEST (AssignTest, FourDim_o1o2o3o4_o1o2o4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3, h4}, {h1, h2}, {h4, h3}));
}

TEST (AssignTest, FourDim_o1o2o3o4_o2o1o3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3, h4}, {h2, h1}, {h3, h4}));
}

TEST (AssignTest, FourDim_o1o2o3o4_o2o1o4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, h1}, {h3, h4}, {h2, h1}, {h4, h3}));
}

///////

TEST (AssignTest, FourDim_o1o2o3v4_o1o2o3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, h1}, {h3, p4}, {h1, h2}, {h3, p4}));
}

TEST (AssignTest, FourDim_o1o2o3v4_o1o2v4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3, p4}, {h1, h2}, {p4, h3}));
}

TEST (AssignTest, FourDim_o1o2o3v4_o2o1o3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3, p4}, {h2, h1}, {h3, p4}));
}

TEST (AssignTest, FourDim_o1o2o3v4_o2o1v4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3, p4}, {h2, h1}, {p4, h3}));
}

////////

TEST (AssignTest, FourDim_o1o2v3o4_o1o2v3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, h1}, {p3, h4}, {h1, h2}, {p3, h4}));
}

TEST (AssignTest, FourDim_o1o2v3o4_o1o2o4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3, h4}, {h1, h2}, {h4, p3}));
}

TEST (AssignTest, FourDim_o1o2v3o4_o2o1v3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3, h4}, {h2, h1}, {p3, h4}));
}

TEST (AssignTest, FourDim_o1o2v3o4_o2o1o4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3, h4}, {h2, h1}, {h4, p3}));
}


////////

TEST (AssignTest, FourDim_o1o2v3v4_o1o2v3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, h1}, {p3, p4}, {h1, h2}, {p3, p4}));
}

TEST (AssignTest, FourDim_o1o2v3v4_o1o2v4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3, p4}, {h1, h2}, {p4, p3}));
}

TEST (AssignTest, FourDim_o1o2v3v4_o2o1v3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3, p4}, {h2, h1}, {p3, p4}));
}

TEST (AssignTest, FourDim_o1o2v3v4_o2o1v4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3, p4}, {h2, h1}, {p4, p3}));
}

///////////////////////

TEST (AssignTest, FourDim_o1v2o3o4_o1v2o3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3, h4}, {h1, p2}, {h3, h4}));
}

TEST (AssignTest, FourDim_o1v2o3o4_o1v2o4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3, h4}, {h1, p2}, {h4, h3}));
}

TEST (AssignTest, FourDim_o1v2o3o4_v2o1o3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3, h4}, {p2, h1}, {h3, h4}));
}

TEST (AssignTest, FourDim_o1v2o3o4_v2o1o4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, h1}, {h3, h4}, {p2, h1}, {h4, h3}));
}

///////

TEST (AssignTest, FourDim_o1v2o3v4_o1v2o3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, h1}, {h3, p4}, {h1, p2}, {h3, p4}));
}

TEST (AssignTest, FourDim_o1v2o3v4_o1v2v4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3, p4}, {h1, p2}, {p4, h3}));
}

TEST (AssignTest, FourDim_o1v2o3v4_v2o1o3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3, p4}, {p2, h1}, {h3, p4}));
}

TEST (AssignTest, FourDim_o1v2o3v4_v2o1v4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3, p4}, {p2, h1}, {p4, h3}));
}

////////

TEST (AssignTest, FourDim_o1v2v3o4_o1v2v3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, h1}, {p3, h4}, {h1, p2}, {p3, h4}));
}

TEST (AssignTest, FourDim_o1v2v3o4_o1v2o4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3, h4}, {h1, p2}, {h4, p3}));
}

TEST (AssignTest, FourDim_o1v2v3o4_v2o1v3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3, h4}, {p2, h1}, {p3, h4}));
}

TEST (AssignTest, FourDim_o1v2v3o4_v2o1o4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3, h4}, {p2, h1}, {h4, p3}));
}


////////

TEST (AssignTest, FourDim_o1v2v3v4_o1v2v3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, h1}, {p3, p4}, {h1, p2}, {p3, p4}));
}

TEST (AssignTest, FourDim_o1v2v3v4_o1v2v4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3, p4}, {h1, p2}, {p4, p3}));
}

TEST (AssignTest, FourDim_o1v2v3v4_v2o1v3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3, p4}, {p2, h1}, {p3, p4}));
}

TEST (AssignTest, FourDim_o1v2v3v4_v2o1v4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3, p4}, {p2, h1}, {p4, p3}));
}

//////////////////////////////////////

TEST (AssignTest, FourDim_v1o2o3o4_v1o2o3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3, h4}, {p1, h2}, {h3, h4}));
}

TEST (AssignTest, FourDim_v1o2o3o4_v1o2o4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3, h4}, {p1, h2}, {h4, h3}));
}

TEST (AssignTest, FourDim_v1o2o3o4_o2v1o3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3, h4}, {h2, p1}, {h3, h4}));
}

TEST (AssignTest, FourDim_v1o2o3o4_o2v1o4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, p1}, {h3, h4}, {h2, p1}, {h4, h3}));
}

///////

TEST (AssignTest, FourDim_v1o2o3v4_v1o2o3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, p1}, {h3, p4}, {p1, h2}, {h3, p4}));
}

TEST (AssignTest, FourDim_v1o2o3v4_v1o2v4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3, p4}, {p1, h2}, {p4, h3}));
}

TEST (AssignTest, FourDim_v1o2o3v4_o2v1o3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3, p4}, {h2, p1}, {h3, p4}));
}

TEST (AssignTest, FourDim_v1o2o3v4_o2v1v4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3, p4}, {h2, p1}, {p4, h3}));
}

////////

TEST (AssignTest, FourDim_v1o2v3o4_v1o2v3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, p1}, {p3, h4}, {p1, h2}, {p3, h4}));
}

TEST (AssignTest, FourDim_v1o2v3o4_v1o2o4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3, h4}, {p1, h2}, {h4, p3}));
}

TEST (AssignTest, FourDim_v1o2v3o4_o2v1v3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3, h4}, {h2, p1}, {p3, h4}));
}

TEST (AssignTest, FourDim_v1o2v3o4_o2v1o4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3, h4}, {h2, p1}, {h4, p3}));
}


////////

TEST (AssignTest, FourDim_v1o2v3v4_v1o2v3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, p1}, {p3, p4}, {p1, h2}, {p3, p4}));
}

TEST (AssignTest, FourDim_v1o2v3v4_v1o2v4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3, p4}, {p1, h2}, {p4, p3}));
}

TEST (AssignTest, FourDim_v1o2v3v4_o2v1v3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3, p4}, {h2, p1}, {p3, p4}));
}

TEST (AssignTest, FourDim_v1o2v3v4_o2v1v4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3, p4}, {h2, p1}, {p4, p3}));
}

//////////////////////////////////////

TEST (AssignTest, FourDim_v1v2o3o4_v1v2o3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3, h4}, {p1, p2}, {h3, h4}));
}

TEST (AssignTest, FourDim_v1v2o3o4_v1v2o4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3, h4}, {p1, p2}, {h4, h3}));
}

TEST (AssignTest, FourDim_v1v2o3o4_v2v1o3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3, h4}, {p2, p1}, {h3, h4}));
}

TEST (AssignTest, FourDim_v1v2o3o4_v2v1o4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, p1}, {h3, h4}, {p2, p1}, {h4, h3}));
}

///////

TEST (AssignTest, FourDim_v1v2o3v4_v1v2o3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, p1}, {h3, p4}, {p1, p2}, {h3, p4}));
}

TEST (AssignTest, FourDim_v1v2o3v4_v1v2v4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3, p4}, {p1, p2}, {p4, h3}));
}

TEST (AssignTest, FourDim_v1v2o3v4_v2v1o3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3, p4}, {p2, p1}, {h3, p4}));
}

TEST (AssignTest, FourDim_v1v2o3v4_v2v1v4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3, p4}, {p2, p1}, {p4, h3}));
}

////////

TEST (AssignTest, FourDim_v1v2v3o4_v1v2v3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, p1}, {p3, h4}, {p1, p2}, {p3, h4}));
}

TEST (AssignTest, FourDim_v1v2v3o4_v1v2o4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3, h4}, {p1, p2}, {h4, p3}));
}

TEST (AssignTest, FourDim_v1v2v3o4_v2v1v3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3, h4}, {p2, p1}, {p3, h4}));
}

TEST (AssignTest, FourDim_v1v2v3o4_v2v1o4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3, h4}, {p2, p1}, {h4, p3}));
}


////////

TEST (AssignTest, FourDim_v1v2v3v4_v1v2v3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, p1}, {p3, p4}, {p1, p2}, {p3, p4}));
}

TEST (AssignTest, FourDim_v1v2v3v4_v1v2v4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3, p4}, {p1, p2}, {p4, p3}));
}

TEST (AssignTest, FourDim_v1v2v3v4_v2v1v3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3, p4}, {p2, p1}, {p3, p4}));
}

TEST (AssignTest, FourDim_v1v2v3v4_v2v1v4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3, p4}, {p2, p1}, {p4, p3}));
}

#endif

#if ASSIGN_TEST_4D //right n

TEST (AssignTest, FourDim_right_n_o1o2o3o4_o1o2o3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3, h4}, {h1, h2}, {h3, h4}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_o1o2o3o4_o1o2o4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3, h4}, {h1, h2}, {h4, h3}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_o1o2o3o4_o2o1o3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3, h4}, {h2, h1}, {h3, h4}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_o1o2o3o4_o2o1o4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, h1}, {h3, h4}, {h2, h1}, {h4, h3}, AllocationType::rhs1_n));
}

///////

TEST (AssignTest, FourDim_right_n_o1o2o3v4_o1o2o3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, h1}, {h3, p4}, {h1, h2}, {h3, p4}, AllocationType::rhs1_n));
}

// TEST (AssignTest, FourDim_right_n_o1o2o3v4_o1o2v4o3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3, p4}, {h1, h2}, {p4, h3}, AllocationType::rhs1_n));
// }

TEST (AssignTest, FourDim_right_n_o1o2o3v4_o2o1o3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3, p4}, {h2, h1}, {h3, p4}, AllocationType::rhs1_n));
}

// TEST (AssignTest, FourDim_right_n_o1o2o3v4_o2o1v4o3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {h3, p4}, {h2, h1}, {p4, h3}, AllocationType::rhs1_n));
// }

////////

// TEST (AssignTest, FourDim_right_n_o1o2v3o4_o1o2v3o4
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, h1}, {p3, h4}, {h1, h2}, {p3, h4}, AllocationType::rhs1_n));
// }

TEST (AssignTest, FourDim_right_n_o1o2v3o4_o1o2o4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3, h4}, {h1, h2}, {h4, p3}, AllocationType::rhs1_n));
}

// TEST (AssignTest, FourDim_right_n_o1o2v3o4_o2o1v3o4
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3, h4}, {h2, h1}, {p3, h4}, AllocationType::rhs1_n));
// }

TEST (AssignTest, FourDim_right_n_o1o2v3o4_o2o1o4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3, h4}, {h2, h1}, {h4, p3}, AllocationType::rhs1_n));
}


////////

TEST (AssignTest, FourDim_right_n_o1o2v3v4_o1o2v3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, h1}, {p3, p4}, {h1, h2}, {p3, p4}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_o1o2v3v4_o1o2v4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3, p4}, {h1, h2}, {p4, p3}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_o1o2v3v4_o2o1v3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3, p4}, {h2, h1}, {p3, p4}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_o1o2v3v4_o2o1v4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, h2}, {p3, p4}, {h2, h1}, {p4, p3}, AllocationType::rhs1_n));
}

///////////////////////

TEST (AssignTest, FourDim_right_n_o1v2o3o4_o1v2o3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3, h4}, {h1, p2}, {h3, h4}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_o1v2o3o4_o1v2o4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3, h4}, {h1, p2}, {h4, h3}, AllocationType::rhs1_n));
}

// TEST (AssignTest, FourDim_right_n_o1v2o3o4_v2o1o3o4
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3, h4}, {p2, h1}, {h3, h4}, AllocationType::rhs1_n));
// }

// TEST (AssignTest, FourDim_right_n_o1v2o3o4_v2o1o4o3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, h1}, {h3, h4}, {p2, h1}, {h4, h3}, AllocationType::rhs1_n));
// }

///////

TEST (AssignTest, FourDim_right_n_o1v2o3v4_o1v2o3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, h1}, {h3, p4}, {h1, p2}, {h3, p4}, AllocationType::rhs1_n));
}

// TEST (AssignTest, FourDim_right_n_o1v2o3v4_o1v2v4o3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3, p4}, {h1, p2}, {p4, h3}, AllocationType::rhs1_n));
// }

// TEST (AssignTest, FourDim_right_n_o1v2o3v4_v2o1o3v4
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3, p4}, {p2, h1}, {h3, p4}, AllocationType::rhs1_n));
// }

// TEST (AssignTest, FourDim_right_n_o1v2o3v4_v2o1v4o3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {h3, p4}, {p2, h1}, {p4, h3}, AllocationType::rhs1_n));
// }

////////

// TEST (AssignTest, FourDim_right_n_o1v2v3o4_o1v2v3o4
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, h1}, {p3, h4}, {h1, p2}, {p3, h4}, AllocationType::rhs1_n));
// }

TEST (AssignTest, FourDim_right_n_o1v2v3o4_o1v2o4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3, h4}, {h1, p2}, {h4, p3}, AllocationType::rhs1_n));
}

// TEST (AssignTest, FourDim_right_n_o1v2v3o4_v2o1v3o4
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3, h4}, {p2, h1}, {p3, h4}, AllocationType::rhs1_n));
// }

// TEST (AssignTest, FourDim_right_n_o1v2v3o4_v2o1o4v3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3, h4}, {p2, h1}, {h4, p3}, AllocationType::rhs1_n));
// }


////////

TEST (AssignTest, FourDim_right_n_o1v2v3v4_o1v2v3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, h1}, {p3, p4}, {h1, p2}, {p3, p4}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_o1v2v3v4_o1v2v4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3, p4}, {h1, p2}, {p4, p3}, AllocationType::rhs1_n));
}

// TEST (AssignTest, FourDim_right_n_o1v2v3v4_v2o1v3v4
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3, p4}, {p2, h1}, {p3, p4}, AllocationType::rhs1_n));
// }

// TEST (AssignTest, FourDim_right_n_o1v2v3v4_v2o1v4v3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h1, p2}, {p3, p4}, {p2, h1}, {p4, p3}, AllocationType::rhs1_n));
// }

//////////////////////////////////////

// TEST (AssignTest, FourDim_right_n_v1o2o3o4_v1o2o3o4
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3, h4}, {p1, h2}, {h3, h4}, AllocationType::rhs1_n));
// }

// TEST (AssignTest, FourDim_right_n_v1o2o3o4_v1o2o4o3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3, h4}, {p1, h2}, {h4, h3}, AllocationType::rhs1_n));
// }

TEST (AssignTest, FourDim_right_n_v1o2o3o4_o2v1o3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3, h4}, {h2, p1}, {h3, h4}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_v1o2o3o4_o2v1o4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, p1}, {h3, h4}, {h2, p1}, {h4, h3}, AllocationType::rhs1_n));
}

///////

// TEST (AssignTest, FourDim_right_n_v1o2o3v4_v1o2o3v4
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, p1}, {h3, p4}, {p1, h2}, {h3, p4}, AllocationType::rhs1_n));
// }

// TEST (AssignTest, FourDim_right_n_v1o2o3v4_v1o2v4o3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3, p4}, {p1, h2}, {p4, h3}, AllocationType::rhs1_n));
// }

TEST (AssignTest, FourDim_right_n_v1o2o3v4_o2v1o3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3, p4}, {h2, p1}, {h3, p4}, AllocationType::rhs1_n));
}

// TEST (AssignTest, FourDim_right_n_v1o2o3v4_o2v1v4o3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {h3, p4}, {h2, p1}, {p4, h3}, AllocationType::rhs1_n));
// }

////////

// TEST (AssignTest, FourDim_right_n_v1o2v3o4_v1o2v3o4
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, p1}, {p3, h4}, {p1, h2}, {p3, h4}, AllocationType::rhs1_n));
// }

// TEST (AssignTest, FourDim_right_n_v1o2v3o4_v1o2o4v3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3, h4}, {p1, h2}, {h4, p3}, AllocationType::rhs1_n));
// }

// TEST (AssignTest, FourDim_right_n_v1o2v3o4_o2v1v3o4
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3, h4}, {h2, p1}, {p3, h4}, AllocationType::rhs1_n));
// }

TEST (AssignTest, FourDim_right_n_v1o2v3o4_o2v1o4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3, h4}, {h2, p1}, {h4, p3}, AllocationType::rhs1_n));
}


////////

// TEST (AssignTest, FourDim_right_n_v1o2v3v4_v1o2v3v4
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {h2, p1}, {p3, p4}, {p1, h2}, {p3, p4}, AllocationType::rhs1_n));
// }

// TEST (AssignTest, FourDim_right_n_v1o2v3v4_v1o2v4v3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3, p4}, {p1, h2}, {p4, p3}, AllocationType::rhs1_n));
// }

TEST (AssignTest, FourDim_right_n_v1o2v3v4_o2v1v3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3, p4}, {h2, p1}, {p3, p4}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_v1o2v3v4_o2v1v4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, h2}, {p3, p4}, {h2, p1}, {p4, p3}, AllocationType::rhs1_n));
}

//////////////////////////////////////

TEST (AssignTest, FourDim_right_n_v1v2o3o4_v1v2o3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3, h4}, {p1, p2}, {h3, h4}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_v1v2o3o4_v1v2o4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3, h4}, {p1, p2}, {h4, h3}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_v1v2o3o4_v2v1o3o4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3, h4}, {p2, p1}, {h3, h4}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_v1v2o3o4_v2v1o4o3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, p1}, {h3, h4}, {p2, p1}, {h4, h3}, AllocationType::rhs1_n));
}

///////

TEST (AssignTest, FourDim_right_n_v1v2o3v4_v1v2o3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, p1}, {h3, p4}, {p1, p2}, {h3, p4}, AllocationType::rhs1_n));
}

// TEST (AssignTest, FourDim_right_n_v1v2o3v4_v1v2v4o3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3, p4}, {p1, p2}, {p4, h3}, AllocationType::rhs1_n));
// }

TEST (AssignTest, FourDim_right_n_v1v2o3v4_v2v1o3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3, p4}, {p2, p1}, {h3, p4}, AllocationType::rhs1_n));
}

// TEST (AssignTest, FourDim_right_n_v1v2o3v4_v2v1v4o3
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {h3, p4}, {p2, p1}, {p4, h3}, AllocationType::rhs1_n));
// }

////////

// TEST (AssignTest, FourDim_right_n_v1v2v3o4_v1v2v3o4
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, p1}, {p3, h4}, {p1, p2}, {p3, h4}, AllocationType::rhs1_n));
// }

TEST (AssignTest, FourDim_right_n_v1v2v3o4_v1v2o4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3, h4}, {p1, p2}, {h4, p3}, AllocationType::rhs1_n));
}

// TEST (AssignTest, FourDim_right_n_v1v2v3o4_v2v1v3o4
// ) {
// ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3, h4}, {p2, p1}, {p3, h4}, AllocationType::rhs1_n));
// }

TEST (AssignTest, FourDim_right_n_v1v2v3o4_v2v1o4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3, h4}, {p2, p1}, {h4, p3}, AllocationType::rhs1_n));
}


////////

TEST (AssignTest, FourDim_right_n_v1v2v3v4_v1v2v3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p2, p1}, {p3, p4}, {p1, p2}, {p3, p4}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_v1v2v3v4_v1v2v4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3, p4}, {p1, p2}, {p4, p3}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_v1v2v3v4_v2v1v3v4
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3, p4}, {p2, p1}, {p3, p4}, AllocationType::rhs1_n));
}

TEST (AssignTest, FourDim_right_n_v1v2v3v4_v2v1v4v3
) {
ASSERT_TRUE(test_assign(*g_ec, 0.24, {p1, p2}, {p3, p4}, {p2, p1}, {p4, p3}, AllocationType::rhs1_n));
}

#endif

//////


#if MULT_TEST_1D_1D

TEST (MultTest, Dim_0_o_o_up
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {}, {},
                           {h1}, {},
                           {h1}, {}));
}

// TEST (MultTest, Dim_oo_o_o) {
//   ASSERT_TRUE(test_mult_no_n(*g_ec,
//                              3.98,
//                              {h1}, {h2},
//                              {h1}, {},
//                              {},   {h2}));
// }
//
// TEST (MultTest, Dim_ov_o_v) {
//   ASSERT_TRUE(test_mult_no_n(*g_ec,
//                              3.98,
//                              {h1}, {p2},
//                              {h1}, {},
//                              {},   {p2}));
// }
//
// TEST (MultTest, Dim_vo_v_o) {
//   ASSERT_TRUE(test_mult_no_n(*g_ec,
//                              3.98,
//                              {p1}, {h2},
//                              {p1}, {},
//                              {},   {h2}));
// }
//
// TEST (MultTest, Dim_vv_v_v) {
//   ASSERT_TRUE(test_mult_no_n(*g_ec,
//                              3.98,
//                              {p1}, {p2},
//                              {p1}, {},
//                              {},   {p2}));
// }

#endif

#if MULT_TEST_1D_1D_2D

TEST (MultTest, Dim_o_o_oo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {},
                           {}, {h2},
                           {h1}, {h2}));
}

TEST (MultTest, Dim_o_o_oo_lo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {}, {h2},
                           {h1}, {},
                           {h1}, {h2}));
}

TEST (MultTest, Dim_o_v_ov
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {},
                           {}, {p2},
                           {h1}, {p2}));
}

TEST (MultTest, Dim_o_v_vo_lo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {}, {h2},
                           {p1}, {},
                           {p1}, {h2}));
}

TEST (MultTest, Dim_v_o_vo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {},
                           {}, {h2},
                           {p1}, {h2}));
}

TEST (MultTest, Dim_v_o_ov_lo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {}, {p2},
                           {h1}, {},
                           {h1}, {p2}));
}

TEST (MultTest, Dim_v_v_vv
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {},
                           {}, {p2},
                           {p1}, {p2}));
}

TEST (MultTest, Dim_v_v_vv_lo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {}, {p2},
                           {p1}, {},
                           {p1}, {p2}));
}

#endif

#if MULT_TEST_1D_2D_1D

TEST (MultTest, Dim_o_oo_o
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {},
                           {h1}, {h2},
                           {}, {h2}));
}

TEST (MultTest, Dim_o_oo_o_lo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {}, {h2},
                           {h1}, {h2},
                           {h1}, {}));
}

TEST (MultTest, Dim_o_ov_v
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {},
                           {h1}, {p2},
                           {}, {p2}));
}

TEST (MultTest, Dim_o_vo_v_lo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {}, {h2},
                           {p1}, {h2},
                           {p1}, {}));
}

TEST (MultTest, Dim_v_vo_o
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {},
                           {p1}, {h2},
                           {}, {h2}));
}

TEST (MultTest, Dim_v_ov_o_lo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {}, {p2},
                           {h1}, {p2},
                           {h1}, {}));
}

TEST (MultTest, Dim_v_vv_v
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {},
                           {p1}, {p2},
                           {}, {p2}));
}

TEST (MultTest, Dim_v_vv_v_lo
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {}, {p2},
                           {p1}, {p2},
                           {p1}, {}));
}

#endif

#if MULT_TEST_1D_2D_3D

#endif

#if MULT_TEST_1D_3D_2D

#endif

#if MULT_TEST_1D_3D_4D

#endif

#if MULT_TEST_1D_4D_3D

#endif

#if MULT_TEST_2D_1D_1D

TEST (MultTest, Dim_oo_o_o
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {h2},
                           {h1}, {},
                           {}, {h2}));
}

TEST (MultTest, Dim_ov_o_v
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {h1}, {p2},
                           {h1}, {},
                           {}, {p2}));
}

TEST (MultTest, Dim_vo_v_o
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {h2},
                           {p1}, {},
                           {}, {h2}));
}

TEST (MultTest, Dim_vv_v_v
) {
ASSERT_TRUE(test_mult_no_n(*g_ec,
                           3.98,
                           {p1}, {p2},
                           {p1}, {},
                           {}, {p2}));
}

#endif

//TEST (MultTest, FourDim_TwoDim_V1V2O1O2_O2V2) {
//  ASSERT_TRUE(test_mult_no_n(*g_ec, 1.0, {P1B}, {H1B},
//		  {P1B, P2B}, {H1B, H4B}, {H4B}, {P2B}));
//}

// void test_assign_2d(tammx::ExecutionContext& ec) {
//   test_assign(ec, 0.24, {H4B}, {H1B}, {H4B}, {H1B});
//   test_assign(ec, 1.23, {H4B}, {H1B}, {H1B}, {H4B});

//   test_assign(ec, 0.24, {H4B}, {P1B}, {H4B}, {P1B});
//   test_assign(ec, 1.23, {H4B}, {P1B}, {P1B}, {H4B});

//   test_assign(ec, 0.24, {P1B}, {H1B}, {P1B}, {H1B});
//   test_assign(ec, 1.23, {P1B}, {H1B}, {H1B}, {P1B});

//   test_assign(ec, 0.24, {P4B}, {P1B}, {P4B}, {P1B});
//   test_assign(ec, 1.23, {P4B}, {P1B}, {P1B}, {P4B});
// }




int main(int argc, char *argv[]) {
  bool intorb = false;
  bool restricted = false;

#if 0
  TAMMX_INT32 noa = 1;
  TAMMX_INT32 nob = 1;
  TAMMX_INT32 nva = 1;
  TAMMX_INT32 nvb = 1;
  std::vector<TAMMX_INT32> spins = {1, 2, 1, 2};
  std::vector<TAMMX_INT32> syms = {0, 0, 0, 0};
  std::vector<TAMMX_INT32> ranges = {1, 1, 1, 1};
#else
  TAMMX_INT32 noa = 2;
  TAMMX_INT32 nob = 2;
  TAMMX_INT32 nva = 2;
  TAMMX_INT32 nvb = 2;
  std::vector<TAMMX_INT32> spins = {1, 1, 2, 2, 1, 1, 2, 2};
  std::vector<TAMMX_INT32> syms = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<TAMMX_INT32> ranges = {4, 4, 4, 4, 4, 4, 4, 4};
#endif

  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(MT_DBL, 8000000, 20000000);

  fortran_init(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);
  tamm_init() ;//(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);
  tammx_init(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);

  tammx::ProcGroup pg{tammx::ProcGroup{MPI_COMM_WORLD}.clone()};
  auto default_distribution = tammx::Distribution_NW();
  tammx::MemoryManagerGA default_memory_manager{pg};
  auto default_irrep = tammx::Irrep{0};
  auto default_spin_restricted = false;

  ::testing::InitGoogleTest(&argc, argv);

  tammx::ExecutionContext ec{pg, &default_distribution, &default_memory_manager,
                             default_irrep, default_spin_restricted};

  testing::AddGlobalTestEnvironment(new TestEnvironment(&ec));

  int ret = RUN_ALL_TESTS();

  pg.destroy();
  tammx_finalize();
  tamm_finalize();
  fortran_finalize();

  GA_Terminate();
  MPI_Finalize();
  return ret;
}
