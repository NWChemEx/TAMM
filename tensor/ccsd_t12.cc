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

namespace tamm {

extern "C" {

void ccsd_t1_cxx_(Integer *d_f1, Integer *d_i0, Integer *d_t_vo,
                  Integer *d_t_vvoo, Integer *d_v2, Integer *k_f1_offset,
                  Integer *k_i0_offset, Integer *k_t_vo_offset,
                  Integer *k_t_vvoo_offset, Integer *k_v2_offset);

void ccsd_t2_cxx_(Integer *d_f1, Integer *d_i0, Integer *d_t_vo,
                  Integer *d_t_vvoo, Integer *d_v2, Integer *k_f1_offset,
                  Integer *k_i0_offset, Integer *k_t_vo_offset,
                  Integer *k_t_vvoo_offset, Integer *k_v2_offset);

void ccsd_e_cxx_(Integer *d_f1, Integer *d_i0, Integer *d_t_vo,
                 Integer *d_t_vvoo, Integer *d_v2, Integer *k_f1_offset,
                 Integer *k_i0_offset, Integer *k_t_vo_offset,
                 Integer *k_t_vvoo_offset, Integer *k_v2_offset);
void ccsd_e_(Integer *d_f1, Integer *d_i0, Integer *d_t_vo, Integer *d_t_vvoo,
             Integer *d_v2, Integer *k_f1_offset, Integer *k_i0_offset,
             Integer *k_t_vo_offset, Integer *k_t_vvoo_offset,
             Integer *k_v2_offset);

void ccsd_et12_cxx_(Integer *d_e, Integer *d_f1, Integer *d_v2, Integer *d_r1,
                    Integer *d_r2, Integer *d_t_vo, Integer *d_t_vvoo,
                    Integer *k_e_offset, Integer *k_f1_offset,
                    Integer *k_v2_offset, Integer *k_r1_offset,
                    Integer *k_r2_offset, Integer *k_t_vo_offset,
                    Integer *k_t_vvoo_offset) {
  // icsd_et12_cxx_(d_e, d_f1, d_v2, d_r1, d_r2, d_t_vo, d_t_vvoo,
  //                k_e_offset, k_f1_offset, k_v2_offset,
  //                k_r1_offset, k_r2_offset, k_t_vo_offset, k_t_vvoo_offset);
  // return;

  Equations e_eqs, t1_eqs, t2_eqs;
  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;

  ccsd_e_equations(&e_eqs);
  ccsd_t1_equations(&t1_eqs);
  ccsd_t2_equations(&t2_eqs);

  std::map<std::string, tamm::Tensor> e_tensors, t1_tensors, t2_tensors;
  std::vector<Operation> e_ops, t1_ops, t2_ops;

  tensors_and_ops(&e_eqs, &e_tensors, &e_ops);
  tensors_and_ops(&t1_eqs, &t1_tensors, &t1_ops);
  tensors_and_ops(&t2_eqs, &t2_tensors, &t2_ops);

  {
    // setup e tensors
    Tensor *i0 = &e_tensors["i0"];
    Tensor *f = &e_tensors["f"];
    Tensor *v = &e_tensors["v"];
    Tensor *t_vo = &e_tensors["t_vo"];
    Tensor *t_vvoo = &e_tensors["t_vvoo"];

    v->set_dist(idist);
    t_vo->set_dist(dist_nwma);
    f->attach(*k_f1_offset, 0, *d_f1);
    i0->attach(*k_e_offset, 0, *d_e);
    t_vo->attach(*k_t_vo_offset, 0, *d_t_vo);
    t_vvoo->attach(*k_t_vvoo_offset, 0, *d_t_vvoo);
    v->attach(*k_v2_offset, 0, *d_v2);
  }

  {
    // setup t1 tensors
    Tensor *i0 = &t1_tensors["i0"];
    Tensor *f = &t1_tensors["f"];
    Tensor *v = &t1_tensors["v"];
    Tensor *t_vo = &t1_tensors["t_vo"];
    Tensor *t_vvoo = &t1_tensors["t_vvoo"];

    v->set_dist(idist);
    t_vo->set_dist(dist_nwma);
    f->attach(*k_f1_offset, 0, *d_f1);
    i0->attach(*k_r1_offset, 0, *d_r1);
    t_vo->attach(*k_t_vo_offset, 0, *d_t_vo);
    t_vvoo->attach(*k_t_vvoo_offset, 0, *d_t_vvoo);
    v->attach(*k_v2_offset, 0, *d_v2);
  }

  {
    // setup t2 tensors
    Tensor *i0 = &t2_tensors["i0"];
    Tensor *f = &t2_tensors["f"];
    Tensor *v = &t2_tensors["v"];
    Tensor *t_vo = &t2_tensors["t_vo"];
    Tensor *t_vvoo = &t2_tensors["t_vvoo"];

    v->set_dist(idist);
    t_vo->set_dist(dist_nwma);
    f->attach(*k_f1_offset, 0, *d_f1);
    i0->attach(*k_r2_offset, 0, *d_r2);
    t_vo->attach(*k_t_vo_offset, 0, *d_t_vo);
    t_vvoo->attach(*k_t_vvoo_offset, 0, *d_t_vvoo);
    v->attach(*k_v2_offset, 0, *d_v2);
  }

#if 0
      vector<vector<Tensor> *> tensors(3);
      vector<vector<Operation> *> ops(3);
      tensors[0] = &e_tensors;
      tensors[1] = &t1_tensors;
      tensors[2] = &t2_tensors;
      ops[0] = &e_ops;
      ops[1] = &t1_ops;
      ops[2] = &t2_ops;
      schedule_levels(tensors, ops);

#else
  ccsd_e_cxx_(d_f1, d_e, d_t_vo, d_t_vvoo, d_v2, k_f1_offset, k_e_offset,
              k_t_vo_offset, k_t_vvoo_offset, k_v2_offset);
  ccsd_t1_cxx_(d_f1, d_r1, d_t_vo, d_t_vvoo, d_v2, k_f1_offset, k_r1_offset,
               k_t_vo_offset, k_t_vvoo_offset, k_v2_offset);
  ccsd_t2_cxx_(d_f1, d_r2, d_t_vo, d_t_vvoo, d_v2, k_f1_offset, k_r2_offset,
               k_t_vo_offset, k_t_vvoo_offset, k_v2_offset);
#endif  // 0 -> Fortran functions

  {
    // un-setup e tensors
    Tensor *i0 = &e_tensors["i0"];
    Tensor *f = &e_tensors["f"];
    Tensor *v = &e_tensors["v"];
    Tensor *t_vo = &e_tensors["t_vo"];
    Tensor *t_vvoo = &e_tensors["t_vvoo"];

    f->detach();
    i0->detach();
    t_vo->detach();
    t_vvoo->detach();
    v->detach();
  }

  {
    // un-setup t1 tensors
    Tensor *i0 = &t1_tensors["i0"];
    Tensor *f = &t1_tensors["f"];
    Tensor *v = &t1_tensors["v"];
    Tensor *t_vo = &t1_tensors["t_vo"];
    Tensor *t_vvoo = &t1_tensors["t_vvoo"];

    f->detach();
    i0->detach();
    // std::cout << "DEBUG __ t11\n";
    t_vo->detach();
    // std::cout << "DEBUG __ t12\n";
    t_vvoo->detach();
    v->detach();
  }

  {
    // un-setup t2 tensors
    Tensor *i0 = &t2_tensors["i0"];
    Tensor *f = &t2_tensors["f"];
    Tensor *v = &t2_tensors["v"];
    Tensor *t_vo = &t2_tensors["t_vo"];
    Tensor *t_vvoo = &t2_tensors["t_vvoo"];

    f->detach();
    i0->detach();
    t_vo->detach();
    t_vvoo->detach();
    v->detach();
  }
}
}  // extern C
};  // namespace tamm
