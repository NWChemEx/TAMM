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
tamm_id_to_tamm_range(const tamm::Index& id) {
  return (id.name() >= tamm::H1B && id.name() <= tamm::H12B)
      ? tamm::RangeType::TO : tamm::RangeType::TV;
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
  tammx::TensorVec<tammx::SymmGroup> ret;

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

// void
// tammx_mult(tammx::ExecutionContext& ec,
//            tammx::Tensor* tc,
//            const std::vector<tamm::IndexName>& clabel,
//            double alpha,
//            tamm::Tensor* ta,
//            const std::vector<tamm::IndexName>& alabel,
//            tamm::Tensor* tb,
//            const std::vector<tamm::IndexName>& blabel) {
//   tamm::Multiplication mult(tc, clabel, ta, alabel, tb, blabel, alpha);
//   mult.execute();
// }


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

void test_assign_vo(tammx::ExecutionContext& ec) {
  auto tc_c = tamm_tensor({TV}, {TO});
  auto tc_f = tamm_tensor({TV}, {TO});
  auto ta = tamm_tensor({TV}, {TO});

  tamm_create(&tc_c, &tc_f, &ta);  
  ta.fill_random();
  
  // tamm_assign(&tc_c, {P1B, H1B}, 1.0, &ta, {P1B, H1B});
  tammx_assign(ec, &tc_c, {P1B, H1B}, 1.0, &ta, {P1B, H1B});
  fortran_assign(&tc_f, &ta, ccsd_t1_1_);

  assert_result(tc_c.check_correctness(&tc_f), __func__);

  tamm_destroy(&tc_c, &tc_f, &ta);
}

void test_mult_vo_oo(tammx::ExecutionContext& ec) {
  auto tc_c = tamm_tensor({TV}, {TO});
  auto tc_f = tamm_tensor({TV}, {TO});
  auto ta = tamm_tensor({TV}, {TO}, 0, tamm::dist_nwma);
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

  tammx::ProcGroup pg {tammx::ProcGroup{MPI_COMM_WORLD}.clone()};
  auto default_distribution = tammx::Distribution_NW();
  tammx::MemoryManagerGA default_memory_manager{pg};
  auto default_irrep = tammx::Irrep{0};
  auto default_spin_restricted = false;

  
  {  
    tammx::ExecutionContext ec {pg, &default_distribution, &default_memory_manager,
          default_irrep, default_spin_restricted};
    
    //test_assign_vo(ec);
    test_mult_vo_oo(ec);

  }
  pg.destroy();
  tammx_finalize();
  tamm_finalize();
  fortran_finalize();
  
  GA_Terminate();
  MPI_Finalize();
  return 0;
}
