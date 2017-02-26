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
#ifndef TAMM_TENSOR_EQUATIONS_H_
#define TAMM_TENSOR_EQUATIONS_H_

#include <map>
#include <string>
#include <vector>
#include "tensor/input.h"
#include "frontend/Parse.h"

namespace tamm {

struct Equations {
  std::vector<RangeEntry> range_entries;
  std::vector<IndexEntry> index_entries;
  // std::vector<TensorEntry> tensor_entries;
  std::map<std::string, tamm::TensorEntry> tensor_entries;
  std::vector<OpEntry> op_entries;
};

void cisd_e_equations(Equations *eqs);
void cisd_c1_equations(Equations *eqs);
void cisd_c2_equations(Equations *eqs);
void ccsd_e_equations(Equations *eqs);
void ccsd_t1_equations(Equations *eqs);
void ccsd_lambda1Mod_equations(Equations *eqs);
void ccsd_lambda1_equations(Equations *eqs);
void ccsd_lambda2_equations(Equations *eqs);
void ccsd_t2_equations(Equations *eqs);
void cc2_t1_equations(Equations *eqs);
void cc2_t2_equations(Equations *eqs);
void icsd_t1_equations(Equations *eqs);
void icsd_t2_equations(Equations *eqs);
void ipccsd_x1_equations(Equations *eqs);
void ipccsd_x2_equations(Equations *eqs);
void eaccsd_x1_equations(Equations *eqs);
void eaccsd_x2_equations(Equations *eqs);
void ccsd_1prdm_hh_equations(Equations *eqs);
void ccsd_1prdm_pp_equations(Equations *eqs);
void ccsd_1prdm_hp_equations(Equations *eqs);
void ccsd_1prdm_ph_equations(Equations *eqs);

void parser_eqs_to_tamm_eqs(const tamm::frontend::Equations &eqs, tamm::Equations *ceqs);

/* void icsd_t1_equations(Equations *eqs); */
/* void icsd_t2_equations(Equations *eqs); */

}; /*namespace tamm*/

#endif  // TAMM_TENSOR_EQUATIONS_H_
