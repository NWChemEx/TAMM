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
#include "tensor/corf.h"
#include "tensor/fapi.h"

namespace tamm {

void CorFortran(int use_c, Assignment * as, add_fn fn) {
  if (use_c) {
    as->execute();
  } else {
    Fint da = static_cast<int>(as->tA().ga()),
         da_offset = as->tA().offset_index();
    Fint dc = static_cast<int>(as->tC().ga()),
         dc_offset = as->tC().offset_index();
    fn(&da, &da_offset, &dc, &dc_offset);
  }
}

void CorFortran(int use_c, Multiplication * m, mult_fn fn) {
  if (use_c) {
    m->execute();
  } else {
    Fint da = static_cast<int>(m->tA().ga()),
         da_offset = m->tA().offset_index();
    Fint db = static_cast<int>(m->tB().ga()),
         db_offset = m->tB().offset_index();
    Fint dc = static_cast<int>(m->tC().ga()),
         dc_offset = m->tC().offset_index();
    fn(&da, &da_offset, &db, &db_offset, &dc, &dc_offset);
  }
}

void CorFortran(int use_c, Assignment * as, icsd_add_fn fn, Integer ctx, Integer count) {
  if (use_c) {
    as->execute();
  } else {
    Fint da = static_cast<int>(as->tA().ga()),
         da_offset = as->tA().offset_index();
    Fint dc = static_cast<int>(as->tC().ga()),
         dc_offset = as->tC().offset_index();
    fn(&da, &da_offset, &dc, &dc_offset, &ctx, &count);
  }
}

void CorFortran(int use_c, Multiplication * m, icsd_mult_fn fn, Integer ctx, Integer count) {
  if (use_c) {
    m->execute();
  } else {
    Fint da = static_cast<int>(m->tA().ga()),
         da_offset = m->tA().offset_index();
    Fint db = static_cast<int>(m->tB().ga()),
         db_offset = m->tB().offset_index();
    Fint dc = static_cast<int>(m->tC().ga()),
         dc_offset = m->tC().offset_index();
    fn(&da, &da_offset, &db, &db_offset, &dc, &dc_offset,&ctx, &count);
  }
}

void CorFortran(int use_c, Tensor *tensor, offset_fn fn) {
  if (use_c) {
    tensor->create();
  } else {
    Fint k_a, l_a, size, d_a;
    fn(&l_a, &k_a, &size);
    fname_and_create(&d_a, &size);
    tensor->attach(k_a, l_a, d_a);
  }
}

void destroy(Tensor *t) {
  if (t->allocated()) {
    t->destroy();
  } else if (t->attached()) {
    Fint d_a = static_cast<int>(t->ga()), l_a = t->offset_handle();
    fdestroy(&d_a, &l_a);
    t->detach();
  } else {
    assert(0);
  }
}

} /*namespace tamm*/
