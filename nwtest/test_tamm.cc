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
//#include <mpi.h>

extern "C" {
void init_fortran_vars_(Integer *noa1, Integer *nob1, Integer *nva1,
                        Integer *nvb1, logical *intorb1, logical *restricted1,
                        Integer *spins, Integer *syms, Integer *ranges);
void ccsd_t1_equations(tamm::Equations *eqs);
//void tensors_and_ops(tamm::Equations *eqs,
//                     std::map<std::string, tamm::Tensor> *tensors,
//                     std::vector<tamm::Operation> *ops);
void ccsd_t1_1_(F77Integer *d_f, F77Integer *k_f_offset, F77Integer *d_i0,
                F77Integer *k_i0_offset);
void f_calls_setvars_cxx_();
void init_mpi_ga_();
void finalize_mpi_ga_();
}

int main() {

    Integer noa1 = 1;
    Integer nob1 = 1;
    Integer nva1 = 1;
    Integer nvb1 = 1;

    logical intorb1 = 0;
    logical restricted1 = 0;
//    Integer *intorb;
//    Integer *restricted;
//    *intorb = static_cast<Integer> (intorb1);
//    *restricted = static_cast<Integer> (restricted1);
    Integer spins[noa1 + nob1 + nva1 + nvb1] = {1, 2, 1, 2};
    Integer syms[noa1+nob1+nva1+nvb1] = {0, 0, 0, 0};
    Integer ranges[noa1+nob1+nva1+nvb1] = {4, 4, 4, 4};

    // Initialize MPI and GLOBAL ARRAYS
    init_mpi_ga_();

    init_fortran_vars_(&noa1, &nob1, &nva1, &nvb1, &intorb1, &restricted1,
                       &spins[0], &syms[0], &ranges[0]);

//    f_calls_setvars_cxx_();

    std::cout << "File: " << __FILE__ <<"On Line: " << __LINE__ << std::endl;
    f_calls_setvars_cxx_();
    std::cout << "File: " << __FILE__ <<"On Line: " << __LINE__ << std::endl;


    std::cout << "File: " << __FILE__ <<"On Line: " << __LINE__ << std::endl;

    // tensor operation t1_1:  i0[p2,h1] += 1 * f[p2,h1];

    // define tensors using Tensor::Tensor
    // define tensor with variables (int n, int nupper, int irrep_val,
    //                      RangeType rt[], DistType dist_type)
    // tamm::Tensor tc(2, 1, 0, {1, 0}, {0});
    tamm::RangeType rt[2] = {tamm::TV, tamm::TO};
    tamm::DistType d_nwma = {tamm::dist_nwma};
    tamm::Tensor tc(2, 1, 0, rt, d_nwma);
    tamm::Tensor ta(2, 1, 0, rt, d_nwma);

    std::cout << "File: " << __FILE__ <<"On Line: " << __LINE__ << std::endl;
    // create tensors

    static tamm::Equations eqs;
    tamm::ccsd_t1_equations(&eqs);
    std::cout << "File: " << __FILE__ <<"On Line: " << __LINE__ << std::endl;

    std::map<std::string, tamm::Tensor> tensors;
    std::vector<tamm::Operation> ops;
    std::cout << "File: " << __FILE__ <<"On Line: " << __LINE__ << std::endl;
    tensors_and_ops(&eqs, &tensors, &ops);

/*  ccsd_t1.eq file
 *  index h1,h2,h3,h4,h5,h6,h7,h8 = O;
 *  index p1,p2,p3,p4,p5,p6,p7 = V;
 *
 *  array i0[V][O];
 *  array f[V][O]: irrep_f;
 *  t1_1:       i0[p2,h1] += 1 * f[p2,h1];
 */

    tamm::Tensor *i0 = &tensors["i0"];
    tamm::Tensor *f = &tensors["f"];

    i0->create();
    f->create();


    std::cout << "File: " << __FILE__ <<"On Line: " << __LINE__ << std::endl;
    // setup operation
    tamm::Assignment op_t1_1 = ops[0].add;


    // execute
    op_t1_1.execute();

    // Finalize MPI and GLOBAL ARRAYS
    finalize_mpi_ga_();

    std::cout << "File: " << __FILE__ <<"On Line: " << __LINE__ << std::endl;
    return 0;
}
