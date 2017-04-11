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
#ifndef TAMM_TENSOR_CORF_H_
#define TAMM_TENSOR_CORF_H_

#include "tensor/expression.h"
#include "tensor/variables.h"

namespace tamm {
typedef void (*add_fn)(F77Integer *, F77Integer *, F77Integer *, F77Integer *);
typedef void (*mult_fn)(F77Integer *, F77Integer *, F77Integer *, F77Integer *, F77Integer *,
                        F77Integer *);
typedef void (*offset_fn)(F77Integer *, F77Integer *, F77Integer *);

typedef void (*icsd_add_fn)(F77Integer *, F77Integer *, F77Integer *, F77Integer *,
                            F77Integer *, F77Integer *);
typedef void (*icsd_mult_fn)(F77Integer *, F77Integer *, F77Integer *, F77Integer *,
                             F77Integer *, F77Integer *, F77Integer *, F77Integer *);

void CorFortran(int use_c, Assignment *as, add_fn fn);
void CorFortran(int use_c, Multiplication *m, mult_fn fn);

void CorFortran(int use_c, Assignment *as, icsd_add_fn fn, F77Integer ctx,
                F77Integer count);
void CorFortran(int use_c, Multiplication *m, icsd_mult_fn fn, F77Integer ctx,
                F77Integer count);

void CorFortran(int use_c, Tensor *tensor, offset_fn fn);
void destroy(Tensor *t);

} /* namespace tamm */

#endif  // TAMM_TENSOR_CORF_H_
