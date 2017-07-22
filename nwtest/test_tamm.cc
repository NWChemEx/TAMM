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
void ccsd_t1_equations(tamm::Equations *eqs);
//void tensors_and_ops(tamm::Equations *eqs,
//                     std::map<std::string, tamm::Tensor> *tensors,
//                     std::vector<tamm::Operation> *ops);
void ccsd_t1_1_(F77Integer *d_f, F77Integer *k_f_offset, F77Integer *d_i0,
                F77Integer *k_i0_offset);
void offset_ccsd_t1_2_1_(F77Integer *l_t1_2_1_offset, F77Integer *k_t1_2_1_offset,
                         F77Integer *size_t1_2_1);
void ccsd_t1_2_(F77Integer *d_t_vo, F77Integer *k_t_vo_offset, F77Integer *d_t1_2_1,
                F77Integer *k_t1_2_1_offset, F77Integer *d_i0, F77Integer *k_i0_offset);

void f_calls_setvars_cxx_();
// void init_mpi_ga_();
// void finalize_mpi_ga_();
}

void test_assign_vo() {
  auto P1B = tamm::P1B;
  auto H1B = tamm::H1B;

  tamm::RangeType rt_vo[] = {tamm::TV, tamm::TO};
  tamm::Tensor tc_c(2, 1, 0, rt_vo, tamm::dist_nw);
  tamm::Tensor tc_f(2, 1, 0, rt_vo, tamm::dist_nw);
  tamm::Tensor ta(2, 1, 0, rt_vo, tamm::dist_nw);
  tamm::Assignment as_c (&tc_c, &ta, 1.0, {P1B, H1B}, {P1B, H1B});
  tamm::Assignment as_f (&tc_f, &ta, 1.0, {P1B, H1B}, {P1B, H1B});

  tc_c.create();
  tc_f.create();
  ta.create();

  ta.fill_random();

  CorFortran(0, &as_f, ccsd_t1_1_);
  CorFortran(1, &as_c, ccsd_t1_1_);

  bool pass_or_fail = tc_c.check_correctness(&tc_f);
  if (!pass_or_fail) {
    std::cout << "C & F Tensors differ in Test " << __func__ << std::endl;
  } else {
    std::cout << "Congratulations! Test " << __func__ << " PASSED" << std::endl;
  }

  ta.destroy();
  tc_f.destroy();
  tc_c.destroy();
}

void test_mult_vo_oo() {
  auto P1B = tamm::P1B;
  auto H1B = tamm::H1B;
  auto H4B = tamm::H4B;

  tamm::RangeType rt_vo[] = {tamm::TV, tamm::TO};
  tamm::RangeType rt_oo[] = {tamm::TO, tamm::TO};

  tamm::Tensor tc_c(2, 1, 0, rt_vo, tamm::dist_nw);
  tamm::Tensor tc_f(2, 1, 0, rt_vo, tamm::dist_nw);
  tamm::Tensor ta(2, 1, 0, rt_vo, tamm::dist_nw);
  tamm::Tensor tb(2, 1, 0, rt_oo, tamm::dist_nw);
//  tamm::Assignment as_c(&tc_c, &ta, 1.0, {P1B, H1B}, {P1B, H1B});
//  tamm::Assignment as_f(&tc_f, &ta, 1.0, {P1B, H1B}, {P1B, H1B});

  tamm::Multiplication mult_c(&tc_c, {P1B, H1B}, &ta, {P1B, H4B},
          &tb, {H4B, H1B}, 1.0);
  tamm::Multiplication mult_f(&tc_f, {P1B, H1B}, &ta, {P1B, H4B},
          &tb, {H4B, H1B}, 1.0);

  tc_c.create();
  tc_f.create();
  ta.create();
  tb.create();

  ta.fill_random();
  tb.fill_given(2.0);

  CorFortran(0, &mult_f, ccsd_t1_2_);
  CorFortran(1, &mult_c, ccsd_t1_2_);

  bool pass_or_fail = tc_f.check_correctness(&tc_c);
  if (!pass_or_fail) {
    std::cout << "C & F Tensors differ in Test " << __func__ << std::endl;
  } else {
    std::cout << "Congratulations! Test " << __func__ << " PASSED" << std::endl;
  }

  ta.destroy();
  tb.destroy();
  tc_f.destroy();
  tc_c.destroy();
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
  
  tammx_finalize();
  tamm_finalize();
  fortran_finalize();
  
  GA_Terminate();
  MPI_Finalize();
  return 0;
}
