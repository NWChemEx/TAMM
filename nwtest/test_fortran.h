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

#ifndef TEST_FORTRAN_H
#define TEST_FORTRAN_H

extern "C" {
  
  void offset_ccsd_t1_2_1_(Integer *l_t1_2_1_offset, Integer *k_t1_2_1_offset,
                           Integer *size_t1_2_1);

  typedef void add_fn(Integer *ta, Integer *offseta, 
                      Integer *tc, Integer *offsetc);
                       
  typedef void mult_fn(Integer *ta, Integer *offseta,
                       Integer *tb, Integer *offsetb,
                       Integer *tc, Integer *offsetc);



  add_fn ccsd_t1_1_, ccsd_t1_2_1_, ccsd_t1_2_2_1_, ccsd_t1_3_1_;  // ccsd_t1
  add_fn ccsd_t1_5_1_, ccsd_t1_6_1_;  // ccsd_t1
  add_fn ccsd_t2_1_, ccsd_t2_2_1_, ccsd_t2_2_2_1_;  // ccsd_t2
  add_fn ccsd_t2_2_2_2_1_, ccsd_t2_2_4_1_, ccsd_t2_2_5_1_;  // ccsd_t2
  add_fn ccsd_t2_4_1_, ccsd_t2_4_2_1_, ccsd_t2_5_1_;  // ccsd_t2
  add_fn ccsd_t2_6_1_, ccsd_t2_6_2_1_, ccsd_t2_7_1_;  // ccsd_t2
  add_fn cc2_t1_1_, cc2_t1_2_1_, cc2_t1_2_2_1_, cc2_t1_3_1_;  // cc2_t1
  add_fn cc2_t1_5_1_, cc2_t1_6_1_;  // cc2_t1
  add_fn cc2_t2_1_, cc2_t2_2_1_, cc2_t2_2_2_1_, cc2_t2_2_2_2_1_;  // cc2_t2
  add_fn cc2_t2_2_3_1_, cc2_t2_3_1_;  // cc2_t2

  mult_fn ccsd_t1_2_;
}

void fortran_init(int noa, int nob, int nva, int nvb, bool intorb, bool restricted,
                  const std::vector<int> &spins,
                  const std::vector<int> &syms,
                  const std::vector<int> &ranges);

void fortran_finalize();

//bool test_assign_no_n(tammx::ExecutionContext& ec,
//                      double alpha,
//                      const std::vector<tamm::IndexName>& cupper_labels,
//                      const std::vector<tamm::IndexName>& clower_labels,
//                      const std::vector<tamm::IndexName>& aupper_labels,
//                      const std::vector<tamm::IndexName>& alower_labels,
//					  add_fn fortran_assign_fn);


bool test_assign_no_n(tammx::ExecutionContext& ec,
                 const tammx::IndexLabelVec &cupper_labels,
                 const tammx::IndexLabelVec &clower_labels,
                 double alpha,
                 const tammx::IndexLabelVec &aupper_labels,
                 const tammx::IndexLabelVec &alower_labels,
                 add_fn fortran_assign_fn);

bool test_mult_no_n(tammx::ExecutionContext &ec,
               const tammx::IndexLabelVec &cupper_labels,
               const tammx::IndexLabelVec &clower_labels,
               double alpha,
               const tammx::IndexLabelVec &aupper_labels,
               const tammx::IndexLabelVec &alower_labels,
               const tammx::IndexLabelVec &bupper_labels,
               const tammx::IndexLabelVec &blower_labels,
               mult_fn fn );

#endif  // TEST_FORTRAN_H
