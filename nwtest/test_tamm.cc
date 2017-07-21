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
    std::cout << "C & F Tensors differ" << std::endl;
  } else {
    std::cout << "Congratulations! Fortran & C++ Implementations Match" << std::endl;
  }
    
  ta.destroy();
  tc_f.destroy();
  tc_c.destroy();  
}

void test_mult_vo() {
  auto P1B = tamm::P1B;
  auto H1B = tamm::H1B;

  tamm::RangeType rt_vo[] = {tamm::TV, tamm::TO};
  tamm::Tensor tc_c(2, 1, 0, rt_vo, tamm::dist_nw);
  tamm::Tensor tc_f(2, 1, 0, rt_vo, tamm::dist_nw);
  tamm::Tensor ta(2, 1, 0, rt_vo, tamm::dist_nw);
  tamm::Tensor tb(2, 1, 0, rt_vo, tamm::dist_nw);
  tamm::Assignment as_c (&tc_c, &ta, 1.0, {P1B, H1B}, {P1B, H1B});
  tamm::Assignment as_f (&tc_f, &ta, 1.0, {P1B, H1B}, {P1B, H1B});

  tc_c.create();
  tc_f.create();
  ta.create();

  ta.fill_random();

  CorFortran(0, &as_f, ccsd_t1_1_);
  CorFortran(1, &as_c, ccsd_t1_1_);

  bool pass_or_fail = tc_f.check_correctness(&tc_c);
  if (!pass_or_fail) {
    std::cout << "C & F Tensors differ" << std::endl;
  } else {
    std::cout << "Congratulations! Fortran & C++ Implementations Match" << std::endl;
  }

  ta.destroy();
  tc_f.destroy();
  tc_c.destroy();
}

int main(int argc, char *argv[]) {

    Integer noa1 = 1;
    Integer nob1 = 1;
    Integer nva1 = 1;
    Integer nvb1 = 1;

    logical intorb1 = 0;
    logical restricted1 = 0;

    Integer spins[noa1 + nob1 + nva1 + nvb1]; // = {1, 2, 1, 2};
    spins[0]=1;spins[1]=2;spins[2]=1;spins[3]= 2; 
    Integer syms[noa1+nob1+nva1+nvb1]; // = {0, 0, 0, 0};
    syms[0]=0;syms[1]=0;syms[2]=0;syms[3]=0;
    Integer ranges[noa1+nob1+nva1+nvb1]; // = {4, 4, 4, 4};
    ranges[0]=4;ranges[1]=4,ranges[2]=4;ranges[3]=4;

    MPI_Init(&argc, &argv);
    GA_Initialize();
    MA_init(MT_DBL, 1000000, 8000000);
    
    init_fortran_vars_(&noa1, &nob1, &nva1, &nvb1, &intorb1, &restricted1,
                       &spins[0], &syms[0], &ranges[0]);
    f_calls_setvars_cxx_();
    test_assign_vo();

    std::cout << "File: " << __FILE__ <<"On Line: " << __LINE__ << std::endl;

    finalize_fortran_vars_();
    GA_Terminate();
    MPI_Finalize();

    std::cout << "File: " << __FILE__ <<"On Line: " << __LINE__ << std::endl;
    return 0;
}
