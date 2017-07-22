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
  
  add_fn ccsd_t1_1_;
  mult_fn ccsd_t1_2_;
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

void
tamm_assign(tamm::Tensor* tc,
            const std::vector<tamm::IndexName>& clabel,
            double alpha,
            tamm::Tensor* ta,
            const std::vector<tamm::IndexName>& alabel) {
  tamm::Assignment as(tc, ta, alpha, clabel, alabel);
  as.execute();
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
const auto H1B = tamm::H1B;
const auto H4B = tamm::H4B;
const auto TO = tamm::TO;
const auto TV = tamm::TV;

void test_assign_vo() {
  auto tc_c = tamm_tensor({TV}, {TO});
  auto tc_f = tamm_tensor({TV}, {TO});
  auto ta = tamm_tensor({TV}, {TO});

  tamm_create(&tc_c, &tc_f, &ta);  
  ta.fill_random();
  
  tamm_assign(&tc_c, {P1B, H1B}, 1.0, &ta, {P1B, H1B});
  fortran_assign(&tc_f, &ta, ccsd_t1_1_);

  assert_result(tc_c.check_correctness(&tc_f), __func__);

  tamm_destroy(&tc_c, &tc_f, &ta);
}

void test_mult_vo_oo() {
  auto tc_c = tamm_tensor({TV}, {TO});
  auto tc_f = tamm_tensor({TV}, {TO});
  auto ta = tamm_tensor({TV}, {TO});
  auto tb = tamm_tensor({TO}, {TO});

  tamm_create(&ta, &tb, &tc_c, &tc_f);
  ta.fill_random();
  tb.fill_given(2.0);

  tamm_mult(&tc_c, {P1B, H1B}, -1.0, &ta, {P1B, H4B}, &tb, {H4B, H1B});
  fortran_mult(&tc_f, &ta, &tb, ccsd_t1_2_);

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
  int noa = 1;
  int nob = 1;
  int nva = 1;
  int nvb = 1;

  bool intorb = false;
  bool restricted = false;

  std::vector<int> spins = {1, 2, 1, 2};
  std::vector<int> syms = {0, 0, 0, 0};
  std::vector<int> ranges = {4, 4, 4, 4};

  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(MT_DBL, 1000000, 8000000);
  
  fortran_init(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);    
  tamm_init(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);    
  tammx_init(noa, nob, nva, nvb, intorb, restricted, spins, syms, ranges);    
  
  test_assign_vo();
  test_mult_vo_oo();
  
  tammx_finalize();
  tamm_finalize();
  fortran_finalize();
  
  GA_Terminate();
  MPI_Finalize();
  return 0;
}
