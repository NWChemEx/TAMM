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

//void offset_ccsd_t1_2_1_(Integer *l_t1_2_1_offset, Integer *k_t1_2_1_offset,
//                         Integer *size_t1_2_1);
//
//typedef void add_fn(Integer *ta, Integer *offseta, Integer *irrepa,
//                    Integer *tc, Integer *offsetc, Integer *irrepc);
//
//typedef void mult_fn(Integer *ta, Integer *offseta, Integer *irrepa,
//                     Integer *tb, Integer *offsetb, Integer *irrepb,
//                     Integer *tc, Integer *offsetc, Integer *irrepc);
//
//typedef void mult_fn_2(Integer *ta, Integer *offseta,
//                       Integer *tb, Integer *offsetb,
//                       Integer *tc, Integer *offsetc);
//
//// add_fn ccsd_t1_1_;
//mult_fn ccsd_t1_2_;
//mult_fn_2 cc2_t1_5_;
}

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

//
//// tammx::Tensor<double>*
//// tammx_tensor(const std::vector<tamm::RangeType>& upper_ranges,
////              const std::vector<tamm::RangeType>& lower_ranges,
////              int irrep = 0,
////              tamm::DistType dist_type = tamm::dist_nw) {
////   int ndim = upper_ranges.size() + lower_ranges.size();
////   int nupper = upper_ranges.size();
////   std::vector<tamm::RangeType> rt {upper_ranges};
////   std::copy(lower_ranges.begin(), lower_ranges.end(), std::back_inserter(rt));
////   return tammxdc::Tensor(ndim, nupper, irrep, &rt[0], dist_type);
//// }
//
//
//void
//tamm_assign(tamm::Tensor* tc,
//            const std::vector<tamm::IndexName>& clabel,
//            double alpha,
//            tamm::Tensor* ta,
//            const std::vector<tamm::IndexName>& alabel) {
//  tamm::Assignment as(tc, ta, alpha, clabel, alabel);
//  as.execute();
//}
//
//tammx::TensorLabel
//tamm_label_to_tammx_label(const std::vector<tamm::IndexName>& label) {
//  tammx::TensorLabel ret;
//  for(auto l : label) {
//    if(l >= tamm::P1B && l<= tamm::P12B) {
//      ret.push_back(tammx::IndexLabel{l - tamm::P1B, tammx::DimType::v});
//    }
//    else if(l >= tamm::H1B && l<= tamm::H12B) {
//      ret.push_back(tammx::IndexLabel{l - tamm::H1B, tammx::DimType::o});
//    }
//  }
//  return ret;
//}
//
tamm::RangeType
tamm_idname_to_tamm_range(const tamm::IndexName &idname) {
  return (idname >= tamm::H1B && idname <= tamm::H12B)
         ? tamm::RangeType::TO : tamm::RangeType::TV;
}


//


//

//
//void
//tammx_assign(tammx::ExecutionContext& ec,
//             tamm::Tensor* ttc,
//             const std::vector<tamm::IndexName>& clabel,
//             double alpha,
//             tamm::Tensor* tta,
//            const std::vector<tamm::IndexName>& alabel) {
//  tammx::Tensor<double> *ta = tamm_tensor_to_tammx_tensor(ec.pg(), tta);
//  tammx::Tensor<double> *tc = tamm_tensor_to_tammx_tensor(ec.pg(), ttc);
//
//  auto al = tamm_label_to_tammx_label(alabel);
//  auto cl = tamm_label_to_tammx_label(clabel);
//
//  // std::cout<<"----AL="<<al<<std::endl;
//  // std::cout<<"----CL="<<cl<<std::endl;
//  ec.scheduler()
//      .io((*tc), (*ta))
//      ((*tc)(cl) += alpha * (*ta)(al))
//      .execute();
//
//  delete ta;
//  delete tc;
//}
//
//void
//tamm_mult(tamm::Tensor* tc,
//          const std::vector<tamm::IndexName>& clabel,
//          double alpha,
//          tamm::Tensor* ta,
//          const std::vector<tamm::IndexName>& alabel,
//          tamm::Tensor* tb,
//          const std::vector<tamm::IndexName>& blabel) {
//  tamm::Multiplication mult(tc, clabel, ta, alabel, tb, blabel, alpha);
//  mult.execute();
//}
//
//template<typename T>
//void tammx_tensor_dump(const tammx::Tensor<T>& tensor, std::ostream& os) {
//  const auto& buf = static_cast<const T*>(tensor.memory_manager()->access(tammx::Offset{0}));
//  size_t sz = tensor.memory_manager()->local_size_in_elements().value();
//  for(size_t i=0; i<sz; i++) {
//    os<<buf[i]<<" ";
//  }
//  os<<std::endl;
//}
//
// void
// tammx_mult(tammx::ExecutionContext& ec,
//            tamm::Tensor* ttc,
//            const std::vector<tamm::IndexName>& clabel,
//            double alpha,
//            tamm::Tensor* tta,
//            const std::vector<tamm::IndexName>& alabel,
//            tamm::Tensor* ttb,
//            const std::vector<tamm::IndexName>& blabel) {
//   tammx::Tensor<double> *ta = tamm_tensor_to_tammx_tensor(ec.pg(), tta);
//   tammx::Tensor<double> *tb = tamm_tensor_to_tammx_tensor(ec.pg(), ttb);
//   tammx::Tensor<double> *tc = tamm_tensor_to_tammx_tensor(ec.pg(), ttc);
//
//   auto al = tamm_label_to_tammx_label(alabel);
//   auto bl = tamm_label_to_tammx_label(blabel);
//   auto cl = tamm_label_to_tammx_label(clabel);
//
//   {
//     std::cout<<"tammx_mult. A = ";
//     tammx_tensor_dump(*ta, std::cout);
//     std::cout<<"tammx_mult. B = ";
//     tammx_tensor_dump(*tb, std::cout);
//     std::cout<<"tammx_mult. C = ";
//     tammx_tensor_dump(*tc, std::cout);
//   }
//
//   // std::cout<<"----AL="<<al<<std::endl;
//   // std::cout<<"----BL="<<bl<<std::endl;
//   // std::cout<<"----CL="<<cl<<std::endl;
//   ec.scheduler()
//       .io((*tc), (*ta), (*tb))
//       ((*tc)() = 0)
//       ((*tc)(cl) += alpha * (*ta)(al) * (*tb)(bl))
//       .execute();
//
//   delete ta;
//   delete tb;
//   delete tc;
// }
//
//
//void
//fortran_assign(tamm::Tensor* tc,
//               tamm::Tensor* ta,
//               add_fn fn) {
//  Integer da = static_cast<Integer>(ta->ga().ga()),
//      offseta = ta->offset_index(),
//      irrepa = ta->irrep();
//  Integer dc = static_cast<Integer>(tc->ga().ga()),
//      offsetc = tc->offset_index(),
//      irrepc = tc->irrep();
//  fn(&da, &offseta, &irrepa, &dc, &offsetc, &irrepc);
//}
//
//void
//fortran_mult(tamm::Tensor* tc,
//             tamm::Tensor* ta,
//             tamm::Tensor* tb,
//             mult_fn fn) {
//  Integer da = static_cast<Integer>(ta->ga().ga()),
//      offseta = ta->offset_index(),
//      irrepa = ta->irrep();
//  Integer db = static_cast<Integer>(tb->ga().ga()),
//      offsetb = tb->offset_index(),
//      irrepb = tb->irrep();
//  Integer dc = static_cast<Integer>(tc->ga().ga()),
//      offsetc = tc->offset_index(),
//      irrepc = tc->irrep();
//  fn(&da, &offseta, &irrepa, &db, &offsetb, &irrepb, &dc, &offsetc, &irrepc);
//}
//

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
//
//template<typename LabeledTensorType>
//void tammx_symmetrize(tammx::ExecutionContext& ec, LabeledTensorType ltensor) {
//  auto &tensor = *ltensor.tensor_;
//  auto &label = ltensor.label_;
//  const auto &indices = tensor.indices();
//  Expects(tensor.flindices().size() == label.size());
//
//  auto label_copy = label;
//  size_t off = 0;
//  for(auto& sg: indices) {
//    if(sg.size() > 1) {
//      //@todo handle other cases
//      assert(sg.size() == 2);
//      std::swap(label_copy[off], label_copy[off+1]);
//      ec.scheduler()
//          .io(tensor)
//          (tensor(label) += tensor(label_copy))
//          (tensor(label) += -0.5*tensor(label))
//          .execute();
//      std::swap(label_copy[off], label_copy[off+1]);
//    }
//    off += sg.size();
//  }
//}
//

//
//const auto P1B = tamm::P1B;
//const auto P2B = tamm::P2B;
//const auto P3B = tamm::P3B;
//const auto P4B = tamm::P4B;
//const auto P5B = tamm::P5B;
//const auto P6B = tamm::P6B;
//const auto P7B = tamm::P7B;
//const auto P8B = tamm::P8B;
//const auto P10B = tamm::P10B;
//const auto P11B = tamm::P11B;
//const auto P12B = tamm::P12B;
//const auto P9B = tamm::P9B;
//const auto H1B = tamm::H1B;
//const auto H2B = tamm::H2B;
//const auto H3B = tamm::H3B;
//const auto H4B = tamm::H4B;
//const auto H5B = tamm::H5B;
//const auto H6B = tamm::H6B;
//const auto H7B = tamm::H7B;
//const auto H8B = tamm::H8B;
//const auto H9B = tamm::H9B;
//const auto H10B = tamm::H10B;
//const auto H11B = tamm::H11B;
//const auto H12B = tamm::H12B;
//
//const auto h1 = tamm::H1B;
//const auto h2 = tamm::H2B;
//const auto h3 = tamm::H3B;
//const auto h4 = tamm::H4B;
//const auto h5 = tamm::H5B;
//const auto h6 = tamm::H6B;
//const auto h7 = tamm::H7B;
//const auto h8 = tamm::H8B;
//const auto h9 = tamm::H9B;
//const auto h10 = tamm::H10B;
//const auto h11 = tamm::H11B;
//const auto h12 = tamm::H12B;
//
//const auto p1 = tamm::P1B;
//const auto p2 = tamm::P2B;
//const auto p3 = tamm::P3B;
//const auto p4 = tamm::P4B;
//const auto p5 = tamm::P5B;
//const auto p6 = tamm::P6B;
//const auto p7 = tamm::P7B;
//const auto p8 = tamm::P8B;
//const auto p9 = tamm::P9B;
//const auto p10 = tamm::P10B;
//const auto p11 = tamm::P11B;
//const auto p12 = tamm::P12B;
//
//const auto TO = tamm::TO;
//const auto TV = tamm::TV;
//
//std::vector<tamm::RangeType>
//tamm_labels_to_ranges(const std::vector<tamm::IndexName>& labels) {
//  std::vector<tamm::RangeType> ret;
//  for(auto l : labels) {
//    ret.push_back(tamm_idname_to_tamm_range(l));
//  }
//  return ret;
//}
//
//bool test_assign(tammx::ExecutionContext& ec,
//                      double alpha,
//                      const std::vector<tamm::IndexName>& cupper_labels,
//                      const std::vector<tamm::IndexName>& clower_labels,
//                      const std::vector<tamm::IndexName>& aupper_labels,
//                      const std::vector<tamm::IndexName>& alower_labels) {
//  const auto& cupper_ranges = tamm_labels_to_ranges(cupper_labels);
//  const auto& clower_ranges = tamm_labels_to_ranges(clower_labels);
//  const auto& aupper_ranges = tamm_labels_to_ranges(aupper_labels);
//  const auto& alower_ranges = tamm_labels_to_ranges(alower_labels);
//  auto tc1 = tamm_tensor(cupper_ranges, clower_ranges);
//  auto tc2 = tamm_tensor(cupper_ranges, clower_ranges);
//  auto ta = tamm_tensor(aupper_ranges, alower_ranges);
//
//  tamm_create(&tc1, &tc2, &ta);
//  ta.fill_random();
//
//  tamm_symmetrize(ec, &ta);
//
//  auto clabels = cupper_labels;
//  std::copy(clower_labels.begin(), clower_labels.end(), std::back_inserter(clabels));
//  auto alabels = aupper_labels;
//  std::copy(alower_labels.begin(), alower_labels.end(), std::back_inserter(alabels));
//
//  tamm_assign(&tc1, clabels, alpha, &ta, alabels);
//  tammx_assign(ec, &tc2, clabels, alpha, &ta, alabels);
//  //fortran_assign(&tc_f, &ta, ccsd_t1_1_);
//
//  //assert_result(tc1.check_correctness(&tc2), __func__);
//  bool status = tc1.check_correctness(&tc2);
//
//  tamm_destroy(&tc1, &tc2, &ta);
//  return status;
//}
//
//bool test_assign_no_n(tammx::ExecutionContext& ec,
//                      double alpha,
//                      const std::vector<tamm::IndexName>& cupper_labels,
//                      const std::vector<tamm::IndexName>& clower_labels,
//                      const std::vector<tamm::IndexName>& aupper_labels,
//                      const std::vector<tamm::IndexName>& alower_labels,
//					  add_fn fortran_assign_fn) {
//  const auto& cupper_ranges = tamm_labels_to_ranges(cupper_labels);
//  const auto& clower_ranges = tamm_labels_to_ranges(clower_labels);
//  const auto& aupper_ranges = tamm_labels_to_ranges(aupper_labels);
//  const auto& alower_ranges = tamm_labels_to_ranges(alower_labels);
//  auto tc1 = tamm_tensor(cupper_ranges, clower_ranges);
//  auto tc2 = tamm_tensor(cupper_ranges, clower_ranges);
//  auto tc_f = tamm_tensor(cupper_ranges, clower_ranges);
//  auto ta = tamm_tensor(aupper_ranges, alower_ranges);
//
//  tamm_create(&tc1, &tc2, &tc_f, &ta);
//  ta.fill_random();
//
//  auto clabels = cupper_labels;
//  std::copy(clower_labels.begin(), clower_labels.end(), std::back_inserter(clabels));
//  auto alabels = aupper_labels;
//  std::copy(alower_labels.begin(), alower_labels.end(), std::back_inserter(alabels));
//
//  tamm_assign(&tc1, clabels, alpha, &ta, alabels);
//  tammx_assign(ec, &tc2, clabels, alpha, &ta, alabels);
//  fortran_assign(&tc_f, &ta, fortran_assign_fn);
//
//  // assert_result(tc1.check_correctness(&tc2), __func__);
//  // bool status = tc1.check_correctness(&tc2);
//  assert_result(tc1.check_correctness(&tc_f), __func__);
//  bool status = tc1.check_correctness(&tc_f);
//
//
//  tamm_destroy(&tc1, &tc2, &tc_f, &ta);
//  return status;
//}
//
//bool test_assign_no_n(tammx::ExecutionContext& ec,
//                      double alpha,
//                      const std::vector<tamm::IndexName>& cupper_labels,
//                      const std::vector<tamm::IndexName>& clower_labels,
//                      const std::vector<tamm::IndexName>& aupper_labels,
//                      const std::vector<tamm::IndexName>& alower_labels) {
//  const auto& cupper_ranges = tamm_labels_to_ranges(cupper_labels);
//  const auto& clower_ranges = tamm_labels_to_ranges(clower_labels);
//  const auto& aupper_ranges = tamm_labels_to_ranges(aupper_labels);
//  const auto& alower_ranges = tamm_labels_to_ranges(alower_labels);
//  auto tc1 = tamm_tensor(cupper_ranges, clower_ranges);
//  auto tc2 = tamm_tensor(cupper_ranges, clower_ranges);
//  auto ta = tamm_tensor(aupper_ranges, alower_ranges);
//
//  tamm_create(&tc1, &tc2, &ta);
//  ta.fill_random();
//  tamm_symmetrize(ec, &ta);
//
//  auto clabels = cupper_labels;
//  std::copy(clower_labels.begin(), clower_labels.end(), std::back_inserter(clabels));
//  auto alabels = aupper_labels;
//  std::copy(alower_labels.begin(), alower_labels.end(), std::back_inserter(alabels));
//
//  tamm_assign(&tc1, clabels, alpha, &ta, alabels);
//  tammx_assign(ec, &tc2, clabels, alpha, &ta, alabels);
//  //fortran_assign(&tc_f, &ta, ccsd_t1_1_);
//
//  //assert_result(tc1.check_correctness(&tc2), __func__);
//  bool status = tc1.check_correctness(&tc2);
//
//  tamm_destroy(&tc1, &tc2, &ta);
//  return status;
//}
//
//bool test_assign_no_n(tammx::ExecutionContext& ec,
//                      const std::vector<tamm::IndexName>& cupper_labels,
//                      const std::vector<tamm::IndexName>& clower_labels,
//                      double alpha,
//                      const std::vector<tamm::IndexName>& aupper_labels,
//                      const std::vector<tamm::IndexName>& alower_labels) {
//	return test_assign_no_n(ec, alpha, cupper_labels, clower_labels,
//			aupper_labels,
//			alower_labels);
//}
//
//bool
//test_mult_no_n(tammx::ExecutionContext& ec,
//               double alpha,
//               const std::vector<tamm::IndexName>& cupper_labels,
//               const std::vector<tamm::IndexName>& clower_labels,
//               const std::vector<tamm::IndexName>& aupper_labels,
//               const std::vector<tamm::IndexName>& alower_labels,
//               const std::vector<tamm::IndexName>& bupper_labels,
//               const std::vector<tamm::IndexName>& blower_labels) {
//  const auto& cupper_ranges = tamm_labels_to_ranges(cupper_labels);
//  const auto& clower_ranges = tamm_labels_to_ranges(clower_labels);
//  const auto& aupper_ranges = tamm_labels_to_ranges(aupper_labels);
//  const auto& alower_ranges = tamm_labels_to_ranges(alower_labels);
//  const auto& bupper_ranges = tamm_labels_to_ranges(bupper_labels);
//  const auto& blower_ranges = tamm_labels_to_ranges(blower_labels);
//  auto tc1 = tamm_tensor(cupper_ranges, clower_ranges);
//  auto tc2 = tamm_tensor(cupper_ranges, clower_ranges);
//  auto ta = tamm_tensor(aupper_ranges, alower_ranges);
//  auto tb = tamm_tensor(bupper_ranges, blower_ranges);
//
//  tamm_create(&tc1, &tc2, &ta, &tb);
//  // ta.fill_given(2.0);
//  ta.fill_random();
//  tb.fill_random();
//  tamm_symmetrize(ec, &ta);
//  tamm_symmetrize(ec, &tb);
//
//  auto clabels = cupper_labels;
//  std::copy(clower_labels.begin(), clower_labels.end(), std::back_inserter(clabels));
//  auto alabels = aupper_labels;
//  std::copy(alower_labels.begin(), alower_labels.end(), std::back_inserter(alabels));
//  auto blabels = bupper_labels;
//  std::copy(blower_labels.begin(), blower_labels.end(), std::back_inserter(blabels));
//
//  tamm_mult(&tc1, clabels, alpha, &ta, alabels, &tb, blabels);
//  tammx_mult(ec, &tc2, clabels, alpha, &ta, alabels, &tb, blabels);
//
//  //assert_result(tc1.check_correctness(&tc2), __func__);
//  bool status = tc1.check_correctness(&tc2);
//
//  tamm_destroy(&tc1, &tc2, &ta, &tb);
//  return status;
//}
//
//bool
//test_mult_no_n(tammx::ExecutionContext& ec,
//               const std::vector<tamm::IndexName>& cupper_labels,
//               const std::vector<tamm::IndexName>& clower_labels,
//               double alpha,
//               const std::vector<tamm::IndexName>& aupper_labels,
//               const std::vector<tamm::IndexName>& alower_labels,
//               const std::vector<tamm::IndexName>& bupper_labels,
//               const std::vector<tamm::IndexName>& blower_labels) {
//  return test_mult_no_n(*g_ec, alpha, cupper_labels, clower_labels,
//		  aupper_labels, alower_labels, bupper_labels, blower_labels);
//}
//
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
//
//
//bool
//test_symm_assign(tammx::ExecutionContext& ec,
//                 const tammx::TensorVec<tammx::SymmGroup>& cindices,
//                 const tammx::TensorVec<tammx::SymmGroup>& aindices,
//                 int nupper_indices,
//                 const std::vector<tamm::IndexName>& tclabels,
//                 double alpha,
//                 const std::vector<double>& factors,
//                 const std::vector<std::vector<tamm::IndexName>>& talabels) {
//  assert(factors.size() > 0);
//  assert(factors.size() == talabels.size());
//  auto restricted = tamm::Variables::restricted();
//  auto clabels = tamm_label_to_tammx_label(tclabels);
//  std::vector<tammx::TensorLabel> alabels;
//  for(const auto& tal: talabels) {
//    alabels.push_back(tamm_label_to_tammx_label(tal));
//  }
//  tammx::TensorRank nup{nupper_indices};
//  tammx::Tensor<double> tc{cindices, nup, tammx::Irrep{0}, restricted};
//  tammx::Tensor<double> ta{aindices, nup, tammx::Irrep{0}, restricted};
//  tammx::Tensor<double> tc2{aindices, nup, tammx::Irrep{0}, restricted};
//
//  bool status = true;
//
//  ec.allocate(tc, tc2, ta);
//
//  ec.scheduler()
//      .io(ta, tc, tc2)
//      (ta() = 0)
//      (tc() = 0)
//      (tc2() = 0)
//      .execute();
//
//  auto init_lambda = [](tammx::Block<double> &block) {
//    double n = std::rand()%100;
//    auto dbuf = block.buf();
//    for(size_t i=0; i<block.size(); i++) {
//      dbuf[i] = n+i;
//      // std::cout<<"init_lambda. dbuf["<<i<<"]="<<dbuf[i]<<std::endl;
//    }
//    //std::generate_n(reinterpret_cast<double *>(block.buf()), block.size(), [&]() { return n++; });
//  };
//
//
//  tensor_map(ta(), init_lambda);
//  tammx_symmetrize(ec, ta());
//  // std::cout<<"TA=\n";
//  // tammx_tensor_dump(ta, std::cout);
//
//  //std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<<<"<<std::endl;
//  ec.scheduler()
//      .io(tc, ta)
//      (tc(clabels) += alpha * ta(alabels[0]))
//      .execute();
//  //std::cout<<">>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<std::endl;
//
//  for(size_t i=0; i < factors.size(); i++) {
//    //std::cout<<"++++++++++++++++++++++++++"<<std::endl;
//    ec.scheduler()
//        .io(tc2, ta)
//        (tc2(clabels) += alpha * factors[i] * ta(alabels[i]))
//        .execute();
//    //std::cout<<"---------------------------"<<std::endl;
//  }
//  // std::cout<<"TA=\n";
//  // tammx_tensor_dump(ta, std::cout);
//  // std::cout<<"TC=\n";
//  // tammx_tensor_dump(tc, std::cout);
//  // std::cout<<"TC2=\n";
//  // tammx_tensor_dump(tc2, std::cout);
//  // std::cout<<"\n";
//  ec.scheduler()
//      .io(tc, tc2)
//      (tc2(clabels) += -1.0 * tc(clabels))
//      .execute();
//  // std::cout<<"TC - TC2=\n";
//  // tammx_tensor_dump(tc2, std::cout);
//  // std::cout<<"\n";
//
//  double threshold = 1e-12;
//  auto lambda = [&] (auto &val) {
//    if(std::abs(val) > threshold) {
//      //std::cout<<"----ERROR----\n";
//    }
//    status &= (std::abs(val) < threshold);
//  };
//  ec.scheduler()
//      .io(tc2)
//      .sop(tc2(), lambda)
//      .execute();
//  ec.deallocate(tc, tc2, ta);
//  return status;
//}
//

////-----------------------------------------------------------------------
////
////                            Mult 0-d
////
////-----------------------------------------------------------------------
//
//void
//test_tammx_mult(tammx::ExecutionContext& ec,
//           tamm::Tensor* ttc,
//           const std::vector<tamm::IndexName>& clabel,
//           double alpha,
//           tamm::Tensor* tta,
//           const std::vector<tamm::IndexName>& alabel,
//           tamm::Tensor* ttb,
//           const std::vector<tamm::IndexName>& blabel) {
//  tammx::Tensor<double> *ta = tamm_tensor_to_tammx_tensor(ec.pg(), tta);
//  tammx::Tensor<double> *tb = tamm_tensor_to_tammx_tensor(ec.pg(), ttb);
//  tammx::Tensor<double> *tc = tamm_tensor_to_tammx_tensor(ec.pg(), ttc);
//
//  auto al = tamm_label_to_tammx_label(alabel);
//  auto bl = tamm_label_to_tammx_label(blabel);
//  auto cl = tamm_label_to_tammx_label(clabel);
//
//  // std::cout<<"----AL="<<al<<std::endl;
//  // std::cout<<"----CL="<<cl<<std::endl;
//  ec.scheduler()
//      .io((*tc), (*ta), (*tb))
//      ((*tc)(cl) += alpha * (*ta)(al) * (*tb)(bl))
//      .execute();
//
//  delete ta;
//  delete tb;
//  delete tc;
//}
//
//



//
//
//void test_assign_ccsd_e(tammx::ExecutionContext& ec);
//void test_assign_ccsd_t1(tammx::ExecutionContext& ec);
//void test_assign_ccsd_t2(tammx::ExecutionContext& ec);
//void test_assign_cc2_t1(tammx::ExecutionContext& ec);
//void test_assign_cc2_t2(tammx::ExecutionContext& ec);
//void test_assign_cisd_c1(tammx::ExecutionContext& ec);
//void test_assign_cisd_c2(tammx::ExecutionContext& ec);
//void test_assign_ccsd_lambda1(tammx::ExecutionContext& ec);
//void test_assign_ccsd_lambda2(tammx::ExecutionContext& ec);
//void test_assign_eaccsd_x1(tammx::ExecutionContext& ec);
//void test_assign_eaccsd_x2(tammx::ExecutionContext& ec);
//void test_assign_icsd_t1(tammx::ExecutionContext& ec);
//void test_assign_icsd_t2(tammx::ExecutionContext& ec);
//void test_assign_ipccsd_x1(tammx::ExecutionContext& ec);
//void test_assign_ipccsd_x2(tammx::ExecutionContext& ec);
//
//void test_assign_4d(tammx::ExecutionContext& ec) {
//  //test_assign_no_n(ec, 0.24, {H1B, H2B}, {H3B, H4B}, {H1B, H2B}, {H3B, H4B});
//  test_assign_no_n(ec, 0.24, {H1B, H2B}, {H3B, H4B}, {H1B, H2B}, {H4B, H3B});
//  // test_assign_no_n(ec, 1.23, {H4B}, {H1B}, {H1B}, {H4B});
//
//  // test_assign_no_n(ec, 0.24, {H4B}, {P1B}, {H4B}, {P1B});
//  // test_assign_no_n(ec, 1.23, {H4B}, {P1B}, {P1B}, {H4B});
//
//  // test_assign_no_n(ec, 0.24, {P1B}, {H1B}, {P1B}, {H1B});
//  // test_assign_no_n(ec, 1.23, {P1B}, {H1B}, {H1B}, {P1B});
//
//  // test_assign_no_n(ec, 0.24, {P4B}, {P1B}, {P4B}, {P1B});
//  // test_assign_no_n(ec, 1.23, {P4B}, {P1B}, {P1B}, {P4B});
//}
//
//void test_mult_vo_oo(tammx::ExecutionContext& ec) {
//  auto tc_c = tamm_tensor({TV}, {TO});
//  auto tc_f = tamm_tensor({TV}, {TO});
//  auto ta = tamm_tensor({TV}, {TO}, 0, tamm::dist_nwma);
//  auto tb = tamm_tensor({TO}, {TO});
//
//  tamm_create(&ta, &tb, &tc_c, &tc_f);
//  tb.fill_random();
//  ta.fill_given(2.0);
//
//  tamm_mult(&tc_c, {P1B, H1B}, -1.0, &ta, {P1B, H4B}, &tb, {H4B, H1B});
//  //fortran_mult(&tc_f, &ta, &tb, ccsd_t1_2_);
//
//  assert_result(tc_c.check_correctness(&tc_f), __func__);
//
//  tamm_destroy(&ta, &tb, &tc_c, &tc_f);
//}
//
//void test_mult_vvoo_ov(tammx::ExecutionContext& ec) {
//  auto tc_c = tamm_tensor({TV}, {TO});
//  auto tc_f = tamm_tensor({TV}, {TO});
//  auto ta = tamm_tensor({TV,TV}, {TO,TO}, 0, tamm::dist_nw);
//  auto tb = tamm_tensor({TO}, {TV});
//
//  tamm_create(&ta, &tb, &tc_c, &tc_f);
//  ta.fill_random();
//  tb.fill_given(2.0);
//  tamm_mult(&tc_c, {P1B, H1B}, 1.0, &ta, {P1B, P2B, H1B, H4B},
//		  &tb, {H4B, P2B});
//  fortran_mult_vvoo_vo(&tc_f, &ta, &tb, cc2_t1_5_);
//
//  assert_result(tc_c.check_correctness(&tc_f), __func__);
//
//  tamm_destroy(&ta, &tb, &tc_c, &tc_f);
//}

/*
 * @note should be called after fortran_init
 */
void tamm_init(...) {
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

  if (label.dt == tammx::DimType::o) {
    ret = static_cast<tamm::IndexName>(tamm::H1B + label.label);
  } else if (label.dt == tammx::DimType::v) {
    ret = static_cast<tamm::IndexName>(tamm::P1B + label.label);
  } else {
    assert(0); //@note unsupported
  }
  return ret;
}

std::vector<tamm::IndexName>
tammx_label_to_tamm_label(const tammx::TensorLabel &label) {
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
tammx_dim_to_tamm_rangetype(tammx::DimType dt) {
  tamm::RangeType ret;
  switch (dt) {
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
  int ndim = ttensor.rank();
  int nupper = ttensor.nupper_indices();
  int irrep = ttensor.irrep().value();
  tamm::DistType dist_type = tamm::dist_nw;

  std::vector<tamm::RangeType> rt;
  for (auto id: ttensor.flindices()) {
    rt.push_back(tammx_dim_to_tamm_rangetype(id));
  }

  auto ptensor = new tamm::Tensor{ndim, nupper, irrep, &rt[0], dist_type};
  auto dst_nw = static_cast<const tammx::Distribution_NW *>(ttensor.distribution());
  auto map = dst_nw->hash();
  auto length = 2 * map[0] + 1;
  Integer *offset_map = new Integer[length];
  for (size_t i = 0; i < length; i++) {
    offset_map[i] = map[i];
  }

  std::cout << "tensor tammx -----------\n";
  for (size_t i = 0; i < length; i++) {
    std::cout << map[i] << ",";
  }
  std::cout << std::endl;

  auto mgr_ga = static_cast<tammx::MemoryManagerGA *>(ttensor.memory_manager());

  auto fma_offset_index = offset_map - tamm::Variables::int_mb();
  auto fma_offset_handle = -1; //@todo @bug FIX THIS
  auto array_handle = mgr_ga->ga();
  ptensor->attach(fma_offset_index, fma_offset_handle, array_handle);
  return {ptensor, offset_map};
}

void
tamm_assign(tammx::Tensor<double> &ttc,
            const tammx::TensorLabel &tclabel,
            double alpha,
            tammx::Tensor<double> &tta,
            const tammx::TensorLabel &talabel) {
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
          const tammx::TensorLabel &tclabel,
          double alpha,
          tammx::Tensor<double> &tta,
          const tammx::TensorLabel &talabel,
          tammx::Tensor<double> &ttb,
          const tammx::TensorLabel &tblabel) {
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
test_assign_no_n(tammx::ExecutionContext &ec,
                 double alpha,
                 const tammx::TensorLabel &cupper_labels,
                 const tammx::TensorLabel &clower_labels,
                 const tammx::TensorLabel &aupper_labels,
                 const tammx::TensorLabel &alower_labels) {
  const auto &cupper_indices = tammx_label_to_indices(cupper_labels);
  const auto &clower_indices = tammx_label_to_indices(clower_labels);
  const auto &aupper_indices = tammx_label_to_indices(aupper_labels);
  const auto &alower_indices = tammx_label_to_indices(alower_labels);

  auto cindices = cupper_indices;
  cindices.insert_back(clower_indices.begin(), clower_indices.end());
  auto aindices = aupper_indices;
  aindices.insert_back(alower_indices.begin(), alower_indices.end());
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
test_assign_no_n(tammx::ExecutionContext &ec,
                 const tammx::TensorLabel &cupper_labels,
                 const tammx::TensorLabel &clower_labels,
                 double alpha,
                 const tammx::TensorLabel &aupper_labels,
                 const tammx::TensorLabel &alower_labels) {
  return test_assign_no_n(ec, alpha, cupper_labels, clower_labels,
                          aupper_labels,
                          alower_labels);
}


bool
test_mult_no_n(tammx::ExecutionContext &ec,
               double alpha,
               const tammx::TensorLabel &cupper_labels,
               const tammx::TensorLabel &clower_labels,
               const tammx::TensorLabel &aupper_labels,
               const tammx::TensorLabel &alower_labels,
               const tammx::TensorLabel &bupper_labels,
               const tammx::TensorLabel &blower_labels) {
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
               const tammx::TensorLabel &cupper_labels,
               const tammx::TensorLabel &clower_labels,
               double alpha,
               const tammx::TensorLabel &aupper_labels,
               const tammx::TensorLabel &alower_labels,
               const tammx::TensorLabel &bupper_labels,
               const tammx::TensorLabel &blower_labels) {
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


tammx::TensorVec <tammx::SymmGroup>
tamm_labels_to_tammx_indices(const std::vector<tamm::IndexName> &labels) {
  tammx::TensorDim tammx_dims;
  for (const auto l : labels) {
    tammx_dims.push_back(tamm_range_to_tammx_dim(tamm_idname_to_tamm_range(l)));
  }
  return tammx_tensor_dim_to_symm_groups(tammx_dims, tammx_dims.size());
}


//tammx::TensorVec<tammx::SymmGroup>
//tamm_tensor_to_tammx_symm_groups(const tamm::Tensor* tensor) {
//    const std::vector<tamm::Index>& ids = tensor->ids();
//    int nup = tensor->nupper();
//    int nlo = ids.size() - nup;
//
//    if (tensor->dim_type() == tamm::DimType::dim_n) {
//        using tammx::SymmGroup;
//        SymmGroup sgu, sgl;
//        for(int i=0; i<nup; i++) {
//            sgu.push_back(tammx::DimType::n);
//        }
//        for(int i=0; i<nlo; i++) {
//            sgl.push_back(tammx::DimType::n);
//        }
//        tammx::TensorVec<SymmGroup> ret;
//        if(sgu.size() > 0) {
//            ret.push_back(sgu);
//        }
//        if(sgl.size() > 0) {
//            ret.push_back(sgl);
//        }
//        return ret;
//    }
//
//    assert(ids.size() <=4); //@todo @fixme assume for now
//    assert(nup <= 2); //@todo @fixme assume for now
//    assert(nlo <= 2);  //@todo @fixme assume for now
//
//    tammx::TensorDim dims;
//    for(const auto& id: ids) {
//        dims.push_back(tamm_id_to_tammx_dim(id));
//    }
//
//    return tammx_tensor_dim_to_symm_groups(dims, nup);
//}
//
//
//tammx::Tensor<double>*
//tamm_tensor_to_tammx_tensor(tammx::ProcGroup pg, tamm::Tensor* ttensor) {
//    using tammx::Irrep;
//    using tammx::TensorVec;
//    using tammx::SymmGroup;
//
//    auto irrep = Irrep{ttensor->irrep()};
//    auto nup = ttensor->nupper();
//
//    auto restricted = tamm::Variables::restricted();
//    const TensorVec<SymmGroup>& indices = tamm_tensor_to_tammx_symm_groups(ttensor);
//
//    auto xtensor = new tammx::Tensor<double>{indices, nup, irrep, restricted};
//    auto mgr = std::make_shared<tammx::MemoryManagerGA>(pg, ttensor->ga().ga());
//    auto distribution = tammx::Distribution_NW();
//    xtensor->attach(&distribution, mgr);
//    return xtensor;
//}
//
//
//void
//tamm_symmetrize(tammx::ExecutionContext& ec,
//                tamm::Tensor* tensor) {
//    tammx::Tensor<double> *xta = tamm_tensor_to_tammx_tensor(ec.pg(), tensor);
//    tammx_symmetrize(ec, (*xta)());
//}
//
//void
//tamm_create() {}
//
//template<typename ...Args>
//void
//tamm_create(tamm::Tensor* tensor, Args ... args) {
//    tensor->create();
//    tamm_create(args...);
//}
//
//void
//tamm_destroy() {}
//
//template<typename ...Args>
//void
//tamm_destroy(tamm::Tensor* tensor, Args ... args) {
//    tensor->destroy();
//    tamm_destroy(args...);
//}
//
//tamm::Tensor
//tamm_tensor(const std::vector<tamm::RangeType>& upper_ranges,
//            const std::vector<tamm::RangeType>& lower_ranges,
//            int irrep = 0,
//            tamm::DistType dist_type = tamm::dist_nw) {
//    int ndim = upper_ranges.size() + lower_ranges.size();
//    int nupper = upper_ranges.size();
//    std::vector<tamm::RangeType> rt {upper_ranges};
//    std::copy(lower_ranges.begin(), lower_ranges.end(), std::back_inserter(rt));
//    return tamm::Tensor(ndim, nupper, irrep, &rt[0], dist_type);
//}


//-----------------------------------------------------------------------
//
//                            Add 1-d
//
//-----------------------------------------------------------------------

#if ASSIGN_TEST_1D

TEST (AssignTest, OneDim_o1e_o1e
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {}, {h1}, {}));
}

TEST (AssignTest, OneDim_eo1_eo1
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {}, {h1}, {}, {h1}));
}

TEST (AssignTest, OneDim_v1e_v1e
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {}, {p1}, {}));
}

TEST (AssignTest, OneDim_ev1_ev1
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {}, {p1}, {}, {p1}));
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
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h4}, {h1}, {h4}, {h1}));
}

TEST (AssignTest, TwoDim_O1O2_O2O1
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 1.23, {h4}, {h1}, {h1}, {h4}));
}

TEST (AssignTest, TwoDim_OV_OV
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h4}, {p1}, {h4}, {p1}));
}

TEST (AssignTest, TwoDim_OV_VO
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 1.23, {h4}, {p1}, {p1}, {h4}));
}

TEST (AssignTest, TwoDim_VO_VO
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {h1}, {p1}, {h1}));
}

TEST (AssignTest, TwoDim_VO_OV
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 1.23, {p1}, {h1}, {h1}, {p1}));
}

TEST (AssignTest, TwoDim_V1V2_V1V2
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p4}, {p1}, {p4}, {p1}));
}

TEST (AssignTest, TwoDim_V1V2_V2V1
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 1.23, {p4}, {p1}, {p1}, {p4}));
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
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {h2, h3}, {h1}, {h2, h3}));
}

TEST (AssignTest, ThreeDim_o1_o2o3__o1_o3o2
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {h2, h3}, {h1}, {h3, h2}));
}

TEST (AssignTest, ThreeDim_o1_o2v3__o1_o2v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {h2, p3}, {h1}, {h2, p3}));
}

TEST (AssignTest, ThreeDim_o1_o2v3__o1_v3o2
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {h2, p3}, {h1}, {p3, h2}));
}

TEST (AssignTest, ThreeDim_o1_v2o3__o1_v2o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {p2, h3}, {h1}, {p2, h3}));
}

TEST (AssignTest, ThreeDim_o1_v2o3__o1_o3v2
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {p2, h3}, {h1}, {h3, p2}));
}

TEST (AssignTest, ThreeDim_o1_v2v3__o1_v2v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {p2, p3}, {h1}, {p2, p3}));
}

TEST (AssignTest, ThreeDim_o1_v2v3__o1_v3v2
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1}, {p2, p3}, {h1}, {p3, p2}));
}

///////////

TEST (AssignTest, ThreeDim_v1_o2o3__v1_o2o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {h2, h3}, {p1}, {h2, h3}));
}

TEST (AssignTest, ThreeDim_v1_o2o3__v1_o3o2
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {h2, h3}, {p1}, {h3, h2}));
}

TEST (AssignTest, ThreeDim_v1_o2v3__v1_o2v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {h2, p3}, {p1}, {h2, p3}));
}

TEST (AssignTest, ThreeDim_v1_o2v3__v1_v3o2
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {h2, p3}, {p1}, {p3, h2}));
}

TEST (AssignTest, ThreeDim_v1_v2o3__v1_v2o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {p2, h3}, {p1}, {p2, h3}));
}

TEST (AssignTest, ThreeDim_v1_v2o3__v1_o3v2
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {p2, h3}, {p1}, {h3, p2}));
}

TEST (AssignTest, ThreeDim_v1_v2v3__v1_v2v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {p2, p3}, {p1}, {p2, p3}));
}

TEST (AssignTest, ThreeDim_v1_v2v3__v1_v3v2
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1}, {p2, p3}, {p1}, {p3, p2}));
}

//////////////////

TEST (AssignTest, ThreeDim_o1o2_o3__o1o2_o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3}, {h1, h2}, {h3}));
}

TEST (AssignTest, ThreeDim_o1o2_o3__o2o1_o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3}, {h2, h1}, {h3}));
}

TEST (AssignTest, ThreeDim_o1o2_v3__o1o2_v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3}, {h1, h2}, {p3}));
}

TEST (AssignTest, ThreeDim_o1o2_v3__o2o1_v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3}, {h2, h1}, {p3}));
}

/////////

TEST (AssignTest, ThreeDim_o1v2_o3__o1v2_o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3}, {h1, p2}, {h3}));
}

TEST (AssignTest, ThreeDim_o1v2_o3__v2o1_o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3}, {p2, h1}, {h3}));
}

TEST (AssignTest, ThreeDim_o1v2_v3__o1v2_v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3}, {h1, p2}, {p3}));
}

TEST (AssignTest, ThreeDim_o1v2_v3__v2o1_v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3}, {p2, h1}, {p3}));
}

//////////////////

TEST (AssignTest, ThreeDim_v1o2_o3__v1o2_o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3}, {p1, h2}, {h3}));
}

TEST (AssignTest, ThreeDim_v1o2_o3__o2v1_o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3}, {h2, p1}, {h3}));
}

TEST (AssignTest, ThreeDim_v1o2_v3__v1o2_v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3}, {p1, h2}, {p3}));
}

TEST (AssignTest, ThreeDim_v1o2_v3__o2v1_v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3}, {h2, p1}, {p3}));
}

/////////

TEST (AssignTest, ThreeDim_v1v2_o3__v1v2_o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3}, {p1, p2}, {h3}));
}

TEST (AssignTest, ThreeDim_v1v2_o3__v2v1_o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3}, {p2, p1}, {h3}));
}

TEST (AssignTest, ThreeDim_v1v2_v3__v1v2_v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3}, {p1, p2}, {p3}));
}

TEST (AssignTest, ThreeDim_v1v2_v3__v2v1_v3
) {
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

TEST (AssignTest, FourDim_o1o2o3o4_o1o2o3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, h4}, {h1, h2}, {h3, h4}));
}

TEST (AssignTest, FourDim_o1o2o3o4_o1o2o4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, h4}, {h1, h2}, {h4, h3}));
}

TEST (AssignTest, FourDim_o1o2o3o4_o2o1o3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, h4}, {h2, h1}, {h3, h4}));
}

TEST (AssignTest, FourDim_o1o2o3o4_o2o1o4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, h1}, {h3, h4}, {h2, h1}, {h4, h3}));
}

///////

TEST (AssignTest, FourDim_o1o2o3v4_o1o2o3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, h1}, {h3, p4}, {h1, h2}, {h3, p4}));
}

TEST (AssignTest, FourDim_o1o2o3v4_o1o2v4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, p4}, {h1, h2}, {p4, h3}));
}

TEST (AssignTest, FourDim_o1o2o3v4_o2o1o3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, p4}, {h2, h1}, {h3, p4}));
}

TEST (AssignTest, FourDim_o1o2o3v4_o2o1v4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {h3, p4}, {h2, h1}, {p4, h3}));
}

////////

TEST (AssignTest, FourDim_o1o2v3o4_o1o2v3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, h1}, {p3, h4}, {h1, h2}, {p3, h4}));
}

TEST (AssignTest, FourDim_o1o2v3o4_o1o2o4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, h4}, {h1, h2}, {h4, p3}));
}

TEST (AssignTest, FourDim_o1o2v3o4_o2o1v3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, h4}, {h2, h1}, {p3, h4}));
}

TEST (AssignTest, FourDim_o1o2v3o4_o2o1o4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, h4}, {h2, h1}, {h4, p3}));
}


////////

TEST (AssignTest, FourDim_o1o2v3v4_o1o2v3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, h1}, {p3, p4}, {h1, h2}, {p3, p4}));
}

TEST (AssignTest, FourDim_o1o2v3v4_o1o2v4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, p4}, {h1, h2}, {p4, p3}));
}

TEST (AssignTest, FourDim_o1o2v3v4_o2o1v3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, p4}, {h2, h1}, {p3, p4}));
}

TEST (AssignTest, FourDim_o1o2v3v4_o2o1v4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, h2}, {p3, p4}, {h2, h1}, {p4, p3}));
}

///////////////////////

TEST (AssignTest, FourDim_o1v2o3o4_o1v2o3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, h4}, {h1, p2}, {h3, h4}));
}

TEST (AssignTest, FourDim_o1v2o3o4_o1v2o4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, h4}, {h1, p2}, {h4, h3}));
}

TEST (AssignTest, FourDim_o1v2o3o4_v2o1o3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, h4}, {p2, h1}, {h3, h4}));
}

TEST (AssignTest, FourDim_o1v2o3o4_v2o1o4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, h1}, {h3, h4}, {p2, h1}, {h4, h3}));
}

///////

TEST (AssignTest, FourDim_o1v2o3v4_o1v2o3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, h1}, {h3, p4}, {h1, p2}, {h3, p4}));
}

TEST (AssignTest, FourDim_o1v2o3v4_o1v2v4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, p4}, {h1, p2}, {p4, h3}));
}

TEST (AssignTest, FourDim_o1v2o3v4_v2o1o3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, p4}, {p2, h1}, {h3, p4}));
}

TEST (AssignTest, FourDim_o1v2o3v4_v2o1v4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {h3, p4}, {p2, h1}, {p4, h3}));
}

////////

TEST (AssignTest, FourDim_o1v2v3o4_o1v2v3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, h1}, {p3, h4}, {h1, p2}, {p3, h4}));
}

TEST (AssignTest, FourDim_o1v2v3o4_o1v2o4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, h4}, {h1, p2}, {h4, p3}));
}

TEST (AssignTest, FourDim_o1v2v3o4_v2o1v3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, h4}, {p2, h1}, {p3, h4}));
}

TEST (AssignTest, FourDim_o1v2v3o4_v2o1o4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, h4}, {p2, h1}, {h4, p3}));
}


////////

TEST (AssignTest, FourDim_o1v2v3v4_o1v2v3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, h1}, {p3, p4}, {h1, p2}, {p3, p4}));
}

TEST (AssignTest, FourDim_o1v2v3v4_o1v2v4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, p4}, {h1, p2}, {p4, p3}));
}

TEST (AssignTest, FourDim_o1v2v3v4_v2o1v3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, p4}, {p2, h1}, {p3, p4}));
}

TEST (AssignTest, FourDim_o1v2v3v4_v2o1v4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h1, p2}, {p3, p4}, {p2, h1}, {p4, p3}));
}

//////////////////////////////////////

TEST (AssignTest, FourDim_v1o2o3o4_v1o2o3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, h4}, {p1, h2}, {h3, h4}));
}

TEST (AssignTest, FourDim_v1o2o3o4_v1o2o4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, h4}, {p1, h2}, {h4, h3}));
}

TEST (AssignTest, FourDim_v1o2o3o4_o2v1o3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, h4}, {h2, p1}, {h3, h4}));
}

TEST (AssignTest, FourDim_v1o2o3o4_o2v1o4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, p1}, {h3, h4}, {h2, p1}, {h4, h3}));
}

///////

TEST (AssignTest, FourDim_v1o2o3v4_v1o2o3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, p1}, {h3, p4}, {p1, h2}, {h3, p4}));
}

TEST (AssignTest, FourDim_v1o2o3v4_v1o2v4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, p4}, {p1, h2}, {p4, h3}));
}

TEST (AssignTest, FourDim_v1o2o3v4_o2v1o3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, p4}, {h2, p1}, {h3, p4}));
}

TEST (AssignTest, FourDim_v1o2o3v4_o2v1v4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {h3, p4}, {h2, p1}, {p4, h3}));
}

////////

TEST (AssignTest, FourDim_v1o2v3o4_v1o2v3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, p1}, {p3, h4}, {p1, h2}, {p3, h4}));
}

TEST (AssignTest, FourDim_v1o2v3o4_v1o2o4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, h4}, {p1, h2}, {h4, p3}));
}

TEST (AssignTest, FourDim_v1o2v3o4_o2v1v3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, h4}, {h2, p1}, {p3, h4}));
}

TEST (AssignTest, FourDim_v1o2v3o4_o2v1o4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, h4}, {h2, p1}, {h4, p3}));
}


////////

TEST (AssignTest, FourDim_v1o2v3v4_v1o2v3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {h2, p1}, {p3, p4}, {p1, h2}, {p3, p4}));
}

TEST (AssignTest, FourDim_v1o2v3v4_v1o2v4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, p4}, {p1, h2}, {p4, p3}));
}

TEST (AssignTest, FourDim_v1o2v3v4_o2v1v3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, p4}, {h2, p1}, {p3, p4}));
}

TEST (AssignTest, FourDim_v1o2v3v4_o2v1v4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, h2}, {p3, p4}, {h2, p1}, {p4, p3}));
}

//////////////////////////////////////

TEST (AssignTest, FourDim_v1v2o3o4_v1v2o3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, h4}, {p1, p2}, {h3, h4}));
}

TEST (AssignTest, FourDim_v1v2o3o4_v1v2o4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, h4}, {p1, p2}, {h4, h3}));
}

TEST (AssignTest, FourDim_v1v2o3o4_v2v1o3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, h4}, {p2, p1}, {h3, h4}));
}

TEST (AssignTest, FourDim_v1v2o3o4_v2v1o4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, p1}, {h3, h4}, {p2, p1}, {h4, h3}));
}

///////

TEST (AssignTest, FourDim_v1v2o3v4_v1v2o3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, p1}, {h3, p4}, {p1, p2}, {h3, p4}));
}

TEST (AssignTest, FourDim_v1v2o3v4_v1v2v4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, p4}, {p1, p2}, {p4, h3}));
}

TEST (AssignTest, FourDim_v1v2o3v4_v2v1o3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, p4}, {p2, p1}, {h3, p4}));
}

TEST (AssignTest, FourDim_v1v2o3v4_v2v1v4o3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {h3, p4}, {p2, p1}, {p4, h3}));
}

////////

TEST (AssignTest, FourDim_v1v2v3o4_v1v2v3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, p1}, {p3, h4}, {p1, p2}, {p3, h4}));
}

TEST (AssignTest, FourDim_v1v2v3o4_v1v2o4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, h4}, {p1, p2}, {h4, p3}));
}

TEST (AssignTest, FourDim_v1v2v3o4_v2v1v3o4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, h4}, {p2, p1}, {p3, h4}));
}

TEST (AssignTest, FourDim_v1v2v3o4_v2v1o4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, h4}, {p2, p1}, {h4, p3}));
}


////////

TEST (AssignTest, FourDim_v1v2v3v4_v1v2v3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p2, p1}, {p3, p4}, {p1, p2}, {p3, p4}));
}

TEST (AssignTest, FourDim_v1v2v3v4_v1v2v4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, p4}, {p1, p2}, {p4, p3}));
}

TEST (AssignTest, FourDim_v1v2v3v4_v2v1v3v4
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, p4}, {p2, p1}, {p3, p4}));
}

TEST (AssignTest, FourDim_v1v2v3v4_v2v1v4v3
) {
ASSERT_TRUE(test_assign_no_n(*g_ec, 0.24, {p1, p2}, {p3, p4}, {p2, p1}, {p4, p3}));
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
//   test_assign_no_n(ec, 0.24, {H4B}, {H1B}, {H4B}, {H1B});
//   test_assign_no_n(ec, 1.23, {H4B}, {H1B}, {H1B}, {H4B});

//   test_assign_no_n(ec, 0.24, {H4B}, {P1B}, {H4B}, {P1B});
//   test_assign_no_n(ec, 1.23, {H4B}, {P1B}, {P1B}, {H4B});

//   test_assign_no_n(ec, 0.24, {P1B}, {H1B}, {P1B}, {H1B});
//   test_assign_no_n(ec, 1.23, {P1B}, {H1B}, {H1B}, {P1B});

//   test_assign_no_n(ec, 0.24, {P4B}, {P1B}, {P4B}, {P1B});
//   test_assign_no_n(ec, 1.23, {P4B}, {P1B}, {P1B}, {P4B});
// }




int main(int argc, char *argv[]) {
  bool intorb = false;
  bool restricted = false;

#if 0
  int noa = 1;
  int nob = 1;
  int nva = 1;
  int nvb = 1;
  std::vector<int> spins = {1, 2, 1, 2};
  std::vector<int> syms = {0, 0, 0, 0};
  std::vector<int> ranges = {1, 1, 1, 1};
#else
  int noa = 2;
  int nob = 2;
  int nva = 2;
  int nvb = 2;
  std::vector<int> spins = {1, 1, 2, 2, 1, 1, 2, 2};
  std::vector<int> syms = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> ranges = {4, 4, 4, 4, 4, 4, 4, 4};
#endif

  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(MT_DBL, 8000000, 20000000);

  fortran_init(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);
  tamm_init(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);
  tammx_init(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);

  tammx::ProcGroup pg{tammx::ProcGroup{MPI_COMM_WORLD}.clone()};
  auto default_distribution = tammx::Distribution_NW();
  tammx::MemoryManagerGA default_memory_manager{pg};
  auto default_irrep = tammx::Irrep{0};
  auto default_spin_restricted = false;

  ::testing::InitGoogleTest(&argc, argv);

  int ret = 0;

  tammx::ExecutionContext ec{pg, &default_distribution, &default_memory_manager,
                             default_irrep, default_spin_restricted};

  testing::AddGlobalTestEnvironment(new TestEnvironment(&ec));

  // temporarily commented
  ret = RUN_ALL_TESTS();
  // test_assign_2d(ec);
  // test_assign_4d(ec);
  // test_assign(ec);
  // test_mult_vo_oo(ec);
  // test_mult_vvoo_ov(ec);
#if 0
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
#endif

  pg.destroy();
  tammx_finalize();
  tamm_finalize();
  fortran_finalize();

  GA_Terminate();
  MPI_Finalize();
  return ret;
}
