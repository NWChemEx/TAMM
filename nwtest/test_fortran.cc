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

#include "tammx/tammx.h"

#include <mpi.h>
#include <ga.h>
#include <macdecls.h>
#include "nwtest/test_tammx.h"
#include "nwtest/test_fortran.h"
//#include "nwtest/test_fortran_tce.h"

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


// void
// fortran_assign(tamm::Tensor* tc,
//               tamm::Tensor* ta,
//               add_fn fn) {
//  Integer da = static_cast<Integer>(ta->ga().ga()),
//      offseta = ta->offset_index(),
//      irrepa = ta->irrep();
//  Integer dc = static_cast<Integer>(tc->ga().ga()),
//      offsetc = tc->offset_index(),
//      irrepc = tc->irrep();
//  fn(&da, &offseta, &irrepa, &dc, &offsetc, &irrepc);
// }

// void
// fortran_mult(tamm::Tensor* tc,
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
// }


std::pair<Integer, Integer *>
tammx_tensor_to_fortran_info(tammx::Tensor<double> &ttensor) {
  auto adst_nw = static_cast<const tammx::Distribution_NW *>(ttensor.distribution());
  auto ahash = adst_nw->hash();
  auto length = 2 * ahash[0] + 1;
  Integer *offseta = new Integer[length];
  for (size_t i = 0; i < length; i++) {
    offseta[i] = ahash[i];
  }

  auto amgr_ga = static_cast<tammx::MemoryManagerGA *>(ttensor.memory_manager());
  Integer da = amgr_ga->ga();
  return {da, offseta};
}

void
fortran_assign(tammx::Tensor<double> &xtc,
               tammx::Tensor<double> &xta,
               add_fn fn) {
  Integer da, *offseta_map;
  Integer dc, *offsetc_map;
  std::tie(da, offseta_map) = tammx_tensor_to_fortran_info(xta);
  std::tie(dc, offsetc_map) = tammx_tensor_to_fortran_info(xtc);
  Integer irrepa = xta.irrep().value();
  Integer irrepc = xtc.irrep().value();

  Integer offseta = offseta_map - tammx::int_mb();
  Integer offsetc = offsetc_map - tammx::int_mb();

  fn(&da, &offseta, &irrepa, &dc, &offsetc, &irrepc);
  delete[] offseta_map;
  delete[] offsetc_map;
}

void
fortran_mult(tammx::Tensor<double> &xtc,
             tammx::Tensor<double> &xta,
             tammx::Tensor<double> &xtb,
             mult_fn fn) {
  Integer da, *offseta_map;
  Integer db, *offsetb_map;
  Integer dc, *offsetc_map;
  std::tie(da, offseta_map) = tammx_tensor_to_fortran_info(xta);
  std::tie(db, offsetb_map) = tammx_tensor_to_fortran_info(xtb);
  std::tie(dc, offsetc_map) = tammx_tensor_to_fortran_info(xtc);
  Integer irrepa = xta.irrep().value();
  Integer irrepb = xtb.irrep().value();
  Integer irrepc = xtc.irrep().value();

  Integer offseta = offseta_map - tammx::int_mb();
  Integer offsetb = offsetb_map - tammx::int_mb();
  Integer offsetc = offsetc_map - tammx::int_mb();

  fn(&da, &offseta, &irrepa, &db, &offsetb, &irrepb, &dc, &offsetc, &irrepc);

  delete[] offseta_map;
  delete[] offsetb_map;
  delete[] offsetc_map;
}

// void
// fortran_mult_vvoo_vo(tamm::Tensor *tc,
//                      tamm::Tensor *ta,
//                      tamm::Tensor *tb,
//                      mult_fn_2 fn) {
//   Integer da = static_cast<Integer>(ta->ga().ga()),
//     offseta = ta->offset_index(),
//     irrepa = ta->irrep();
//   Integer db = static_cast<Integer>(tb->ga().ga()),
//     offsetb = tb->offset_index(),
//     irrepb = tb->irrep();
//   Integer dc = static_cast<Integer>(tc->ga().ga()),
//     offsetc = tc->offset_index(),
//     irrepc = tc->irrep();
//   fn(&da, &offseta, &db, &offsetb, &dc, &offsetc);
// }
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


bool
test_assign_no_n(tammx::ExecutionContext &ec,
                 const tammx::TensorLabel &cupper_labels,
                 const tammx::TensorLabel &clower_labels,
                 double alpha,
                 const tammx::TensorLabel &aupper_labels,
                 const tammx::TensorLabel &alower_labels,
                 add_fn fortran_assign_fn) {

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
  tammx::Tensor<double> tcf{cindices, cnup, irrep, restricted};
  tammx::Tensor<double> ta{aindices, anup, irrep, restricted};

  ec.allocate(ta, tc1, tc2, tcf);

  ec.scheduler()
    .io(ta, tc1, tc2, tcf)
      (ta() = 0)
      (tc1() = 0)
      (tc2() = 0)
      (tcf() = 0)
    .execute();

  tammx_tensor_fill(ec, ta());

  auto clabels = cupper_labels;
  clabels.insert_back(clower_labels.begin(), clower_labels.end());
  auto alabels = aupper_labels;
  alabels.insert_back(alower_labels.begin(), alower_labels.end());

  //tamm_assign(tc1, clabels, alpha, ta, alabels);
  tammx_assign(ec, tc2, clabels, alpha, ta, alabels);
  fortran_assign(tcf, ta, fortran_assign_fn);

  bool status = tammx_tensors_are_equal(ec, tc2, tcf);

  ec.deallocate(tc1, tc2, ta, tcf);
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
               const tammx::TensorLabel &blower_labels,
               mult_fn fn) {
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
  tammx::Tensor<double> tcf{cindices, cnup, irrep, restricted};
  tammx::Tensor<double> ta{aindices, anup, irrep, restricted};
  tammx::Tensor<double> tb{bindices, bnup, irrep, restricted};

  ec.allocate(ta, tb, tc1, tc2, tcf);

  ec.scheduler()
    .io(ta, tb, tc1, tc2, tcf)
      (ta() = 0)
      (tb() = 0)
      (tc1() = 0)
      (tc2() = 0)
      (tcf() = 0)
    .execute();

  tammx_tensor_fill(ec, ta());
  tammx_tensor_fill(ec, tb());

  auto clabels = cupper_labels;
  clabels.insert_back(clower_labels.begin(), clower_labels.end());
}

int run_fortran_tests(int argc, char *argv[]) {
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
  //tamm_init(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);
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
  //tamm_finalize();
  fortran_finalize();

  GA_Terminate();
  MPI_Finalize();
  return ret;
}
