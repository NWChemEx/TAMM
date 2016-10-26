#include "equations.h"
#include "input.h"

#include <string>

using namespace std;

namespace tamm {

void parser_eqs_to_tamm_eqs(::Equations *eqs, tamm::Equations &ceqs);

static void parse_equations(const string &filename, tamm::Equations &ceqs);

void ccsd_e_equations(tamm::Equations &eqs) {
  parse_equations("ccsd_e.eq", eqs);
}

void ccsd_t1_equations(tamm::Equations &eqs) {
  parse_equations("ccsd_t1.eq", eqs);
}

void ccsd_lambda1Mod_equations(tamm::Equations &eqs) {
  parse_equations("ccsd_lambda1Mod.eq", eqs);
}

void ccsd_lambda1_equations(tamm::Equations &eqs) {
  parse_equations("ccsd_lambda1.eq", eqs);
}

void ccsd_lambda2_equations(tamm::Equations &eqs) {
  parse_equations("ccsd_lambda2.eq", eqs);
}


void ccsd_t2_equations(tamm::Equations &eqs) {
  parse_equations("ccsd_t2_hand.eq", eqs);
}

void cisd_e_equations(tamm::Equations &eqs) {
  parse_equations("cisd_e.eq", eqs);
}

void cisd_c1_equations(tamm::Equations &eqs) {
  parse_equations("cisd_c1.eq", eqs);
}

void cisd_c2_equations(tamm::Equations &eqs) {
  parse_equations("cisd_c2.eq", eqs);
}

void cc2_t1_equations(tamm::Equations &eqs) {
  parse_equations("cc2_t1.eq", eqs);
}

void cc2_t2_equations(tamm::Equations &eqs) {
  parse_equations("cc2_t2.eq", eqs);
}

void icsd_t1_equations(tamm::Equations &eqs) {
  parse_equations("icsd_t1.eq", eqs);
}

void icsd_t2_equations(tamm::Equations &eqs) {
  parse_equations("icsd_t2_hand.eq", eqs);
}

void ipccsd_x1_equations(tamm::Equations &eqs) {
  parse_equations("ipccsd_x1.eq", eqs);
}

void ipccsd_x2_equations(tamm::Equations &eqs) {
  parse_equations("ipccsd_x2.eq", eqs);
}

void eaccsd_x1_equations(tamm::Equations &eqs) {
  parse_equations("eaccsd_x1.eq", eqs);
}

void eaccsd_x2_equations(tamm::Equations &eqs) {
  parse_equations("eaccsd_x2.eq", eqs);
}

static void parse_equations(const string &filename, tamm::Equations &ceqs) {
  ::Equations peqs;
  string full_name = string(TAMM_EQ_PATH) + string("/") + filename;
  tamm_parser(full_name.c_str(), &peqs);
  parser_eqs_to_tamm_eqs(&peqs, ceqs);
}

void parser_eqs_to_tamm_eqs(::Equations *peqs, tamm::Equations &ceqs) {
  assert(peqs);
  int nre, nie, nte, noe;
  nre = vector_count(&peqs->range_entries);
  nie = vector_count(&peqs->index_entries);
  nte = vector_count(&peqs->tensor_entries);
  noe = vector_count(&peqs->op_entries);

  for (int i = 0; i < nre; i++) {
    ::RangeEntry re = (::RangeEntry)vector_get(&peqs->range_entries, i);
    tamm::RangeEntry cre;
    // cre.name = strdup(re->name);
    cre.name = string(re->name);
    ceqs.range_entries.push_back(cre);
  }
  for (int i = 0; i < nie; i++) {
    ::IndexEntry ie = (::IndexEntry)vector_get(&peqs->index_entries, i);
    tamm::IndexEntry cie;
    // cie.name = strdup(ie->name);
    cie.name = string(ie->name);
    cie.range_id = ie->range_id;
    // cout<<"range id="<<cie.range_id<<endl;
    assert(cie.range_id >= 0 && cie.range_id < ceqs.range_entries.size());
    ceqs.index_entries.push_back(cie);
  }

  for (int i = 0; i < nte; i++) {
    ::TensorEntry te = (::TensorEntry)vector_get(&peqs->tensor_entries, i);
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
    ceqs.tensor_entries.insert(
        std::map<std::string, tamm::TensorEntry>::value_type(string(te->name),
                                                             cte));
  }

  for (int i = 0; i < noe; i++) {
    ::OpEntry oe = (::OpEntry)vector_get(&peqs->op_entries, i);
    tamm::OpEntry coe;
    // cout<<"optype == "<<oe->optype<<endl;
    coe.op_id = oe->op_id;
    coe.optype =
        (oe->optype == ::OpTypeAdd) ? tamm::OpTypeAdd : tamm::OpTypeMult;
    //      coe.add = oe->add;
    //      coe.mult = oe->mult;

    int j;
    if (coe.optype == tamm::OpTypeAdd) {
      ::TensorEntry ta =
          (::TensorEntry)vector_get(&peqs->tensor_entries, oe->add->ta);
      ::TensorEntry tc =
          (::TensorEntry)vector_get(&peqs->tensor_entries, oe->add->tc);

      coe.add.ta = string(ta->name);  // oe->add->ta;
      coe.add.tc = string(tc->name);  // oe->add->tc;
      coe.add.alpha = oe->add->alpha;
      for (j = 0; j < MAX_TENSOR_DIMS; j++)
        coe.add.tc_ids[j] = oe->add->tc_ids[j];
      for (j = 0; j < MAX_TENSOR_DIMS; j++)
        coe.add.ta_ids[j] = oe->add->ta_ids[j];
    } else {
      ::TensorEntry ta =
          (::TensorEntry)vector_get(&peqs->tensor_entries, oe->mult->ta);
      ::TensorEntry tb =
          (::TensorEntry)vector_get(&peqs->tensor_entries, oe->mult->tb);
      ::TensorEntry tc =
          (::TensorEntry)vector_get(&peqs->tensor_entries, oe->mult->tc);

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

    ceqs.op_entries.push_back(coe);
  }
}
};
