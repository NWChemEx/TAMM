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

extern "C" {
void init_fortran_vars_(Integer *noa1, Integer *nob1, Integer *nva1,
                        Integer *nvb1, bool intorb1, bool restricted1,
                        Integer *spins, Integer *syms, Integer *ranges);
}

int main() {
    Integer noa1 = 2;
    Integer nob1 = 2;
    Integer nva1 = 2;
    Integer nvb1 = 2;
    bool intorb1 = false;
    bool restricted1 = false;
    Integer spins[noa1+nob1+nva1+nvb1];
    Integer syms[noa1+nob1+nva1+nvb1];
    Integer ranges[noa1+nob1+nva1+nvb1];

    init_fortran_vars_(&noa1, &nob1, &nva1, &nvb1, intorb1, restricted1,
                       &spins[0], &syms[0], &ranges[0]);

    return 0;
}
