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
#include "tensor/equations.h"
#include <string>
#include "tensor/input.h"

using std::string;

namespace tamm {

void parser_eqs_to_tamm_eqs(::Equations *eqs, tamm::Equations *ceqs);

static void parse_equations(const string &filename, tamm::Equations *ceqs);

void ccsd_e_equations(tamm::Equations *eqs) {
  parse_equations("ccsd_e.eq", eqs);
}

void ccsd_t1_equations(tamm::Equations *eqs) {
  parse_equations("ccsd_t1.eq", eqs);
}

void ccsd_lambda1Mod_equations(tamm::Equations *eqs) {
  parse_equations("ccsd_lambda1Mod.eq", eqs);
}

void ccsd_lambda1_equations(tamm::Equations *eqs) {
  parse_equations("ccsd_lambda1.eq", eqs);
}

void ccsd_lambda2_equations(tamm::Equations *eqs) {
  parse_equations("ccsd_lambda2.eq", eqs);
}

void ccsd_t2_equations(tamm::Equations *eqs) {
  parse_equations("ccsd_t2_hand.eq", eqs);
}

void cisd_e_equations(tamm::Equations *eqs) {
  parse_equations("cisd_e.eq", eqs);
}

void cisd_c1_equations(tamm::Equations *eqs) {
  parse_equations("cisd_c1.eq", eqs);
}

void cisd_c2_equations(tamm::Equations *eqs) {
  parse_equations("cisd_c2.eq", eqs);
}

void cc2_t1_equations(tamm::Equations *eqs) {
  parse_equations("cc2_t1.eq", eqs);
}

void cc2_t2_equations(tamm::Equations *eqs) {
  parse_equations("cc2_t2.eq", eqs);
}

void icsd_t1_equations(tamm::Equations *eqs) {
  parse_equations("icsd_t1.eq", eqs);
}

void icsd_t2_equations(tamm::Equations *eqs) {
  parse_equations("icsd_t2_hand.eq", eqs);
}

void ipccsd_x1_equations(tamm::Equations *eqs) {
  parse_equations("ipccsd_x1.eq", eqs);
}

void ipccsd_x2_equations(tamm::Equations *eqs) {
  parse_equations("ipccsd_x2.eq", eqs);
}

void eaccsd_x1_equations(tamm::Equations *eqs) {
  parse_equations("eaccsd_x1.eq", eqs);
}

void eaccsd_x2_equations(tamm::Equations *eqs) {
  parse_equations("eaccsd_x2.eq", eqs);
}

void ccsd_1prdm_hh_equations(tamm::Equations *eqs) {
  parse_equations("ccsd_1prdm_hh.eq", eqs);
}

void ccsd_1prdm_pp_equations(tamm::Equations *eqs) {
  parse_equations("ccsd_1prdm_pp.eq", eqs);
}

void ccsd_1prdm_hp_equations(tamm::Equations *eqs) {
  parse_equations("ccsd_1prdm_hp.eq", eqs);
}

void ccsd_1prdm_ph_equations(tamm::Equations *eqs) {
  parse_equations("ccsd_1prdm_ph.eq", eqs);
}

static void parse_equations(const string &filename, tamm::Equations *ceqs) {
  ::Equations peqs;
  string full_name = string(TAMM_EQ_PATH) + string("/") + filename;
  tamm_parser(full_name.c_str(), &peqs);
  parser_eqs_to_tamm_eqs(peqs, ceqs);
}

void parser_eqs_to_tamm_eqs(const ::Equations &peqs, tamm::Equations *ceqs) {
  int nre, nie, nte, noe;
  nre = peqs.range_entries.size();
  nie = peqs.index_entries.size();
  nte = peqs.tensor_entries.size();
  noe = peqs.op_entries.size();

  for (int i = 0; i < nre; i++) {
    ::RangeEntry * re = peqs.range_entries.at(i);
    tamm::RangeEntry cre;
    // cre.name = strdup(re->name);
    cre.name = string(re->name);
    ceqs->range_entries.push_back(cre);
  }
  for (int i = 0; i < nie; i++) {
    ::IndexEntry *ie = peqs.index_entries.at(i);
    tamm::IndexEntry cie;
    // cie.name = strdup(ie->name);
    cie.name = string(ie->name);
    cie.range_id = ie->range_id;
    // cout<<"range id="<<cie.range_id<<endl;
    assert(cie.range_id >= 0 && cie.range_id < ceqs->range_entries.size());
    ceqs->index_entries.push_back(cie);
  }

  for (int i = 0; i < nte; i++) {
    ::TensorEntry * te = peqs.tensor_entries.at(i);
    tamm::TensorEntry cte;
    // cte.name = strdup(te->name);
    cte.name = string(te->name);
    cte.ndim = te->ndim;
    cte.nupper = te->nupper;

    for (int j = 0; j < MAX_TENSOR_DIMS; j++) {
      cte.range_ids[j] = te->range_ids[j];
    }
    // ceqs.tensor_entries.push_back(cte);
    // ceqs.tensor_entries[string(te->name)] = cte;
    ceqs->tensor_entries.insert(
        std::map<std::string, tamm::TensorEntry>::value_type(string(te->name),
                                                             cte));
  }

  for (int i = 0; i < noe; i++) {
    ::OpEntry *oe = peqs.op_entries.at(i);
    tamm::OpEntry coe;
    // cout<<"optype == "<<oe->optype<<endl;
    coe.op_id = oe->op_id;
    coe.optype =
        (oe->optype == ::OpTypeAdd) ? tamm::OpTypeAdd : tamm::OpTypeMult;
    //      coe.add = oe->add;
    //      coe.mult = oe->mult;

    int j;
    if (coe.optype == tamm::OpTypeAdd) {
      ::TensorEntry *ta =
          peqs.tensor_entries.at(oe->add->ta);
      ::TensorEntry *tc =
          peqs.tensor_entries.at(oe->add->tc);

      coe.add.ta = string(ta->name);  // oe->add->ta;
      coe.add.tc = string(tc->name);  // oe->add->tc;
      coe.add.alpha = oe->add->alpha;
      for (j = 0; j < MAX_TENSOR_DIMS; j++)
        coe.add.tc_ids[j] = oe->add->tc_ids[j];
      for (j = 0; j < MAX_TENSOR_DIMS; j++)
        coe.add.ta_ids[j] = oe->add->ta_ids[j];
    } else {
      ::TensorEntry *ta =
          peqs.tensor_entries.at(oe->mult->ta);
      ::TensorEntry *tb =
          peqs.tensor_entries.at(oe->mult->tb);
      ::TensorEntry *tc =
          peqs.tensor_entries.at(oe->mult->tc);

      coe.mult.ta = string(ta->name);  // oe->mult->ta;
      coe.mult.tb = string(tb->name);  // oe->mult->tb;
      coe.mult.tc = string(tc->name);  // oe->mult->tc;
      coe.mult.alpha = oe->mult->alpha;
      for (j = 0; j < MAX_TENSOR_DIMS; j++)
        coe.mult.tc_ids[j] = oe->mult->tc_ids[j];
      for (j = 0; j < MAX_TENSOR_DIMS; j++)
        coe.mult.ta_ids[j] = oe->mult->ta_ids[j];
      for (j = 0; j < MAX_TENSOR_DIMS; j++)
        coe.mult.tb_ids[j] = oe->mult->tb_ids[j];
    }

    ceqs->op_entries.push_back(coe);
  }
}
};  // namespace tamm
