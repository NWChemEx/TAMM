#include "equations.h"

namespace ctce {
  
  /*@FIXME: @BUG: memory leak in ::Equations */
  void parser_eqs_to_ctce_eqs(::Equations *peqs, ctce::Equations &ceqs) {
    assert(peqs);
    int nre, nie, nte, noe;
    nre = vector_count(&peqs->range_entries);
    nie = vector_count(&peqs->index_entries);
    nte = vector_count(&peqs->tensor_entries);
    noe = vector_count(&peqs->op_entries);

    for(int i=0; i<nre; i++) {
      ::RangeEntry re = (::RangeEntry)vector_get(&peqs->range_entries, i);
      ctce::RangeEntry cre;
      cre.name = strdup(re->name);
      ceqs.range_entries.push_back(cre);
    }
    for(int i=0; i<nie; i++) {
      ::IndexEntry ie = (::IndexEntry)vector_get(&peqs->index_entries, i);
      ctce::IndexEntry cie;
      cie.name = strdup(ie->name);
      cie.range_id = ie->range_id;
      assert(cie.range_id >=0 && cie.range_id < ceqs.range_entries.size());
      ceqs.index_entries.push_back(cie);
    }

    for(int i=0; i<nte; i++) {
      ::TensorEntry te = (::TensorEntry)vector_get(&peqs->tensor_entries, i);
      ctce::TensorEntry cte;
      cte.name = strdup(te->name);
      cte.ndim = te->ndim;
      cte.nupper = te->nupper;

      for(int j=0; j<MAX_TENSOR_DIMS; j++) {
        cte.range_ids[j] = te->range_ids[j];
      }
      ceqs.tensor_entries.push_back(cte);
    }

    for(int i=0; i<noe; i++) {
      ::OpEntry oe = (::OpEntry)vector_get(&peqs->op_entries, i);
      ctce::OpEntry coe;
      coe.optype = (ctce::OpType)oe->optype;
//      coe.add = oe->add;
//      coe.mult = oe->mult;

        int j;
        if (coe.optype == ctce::OpTypeAdd) {
            coe.add.ta = oe->add->ta;
            coe.add.tc = oe->add->tc;
            coe.add.alpha = oe->add->alpha;
            for (j = 0; j < MAX_TENSOR_DIMS; j++) coe.add.tc_ids[j] = oe->add->tc_ids[j];
            for (j = 0; j < MAX_TENSOR_DIMS; j++) coe.add.ta_ids[j] = oe->add->ta_ids[j];
        }
        else {
            coe.mult.ta = oe->mult->ta;
            coe.mult.tb = oe->mult->tb;
            coe.mult.tc = oe->mult->tc;
            coe.mult.alpha = oe->mult->alpha;
            for (j = 0; j < MAX_TENSOR_DIMS; j++) coe.mult.tc_ids[j] = oe->mult->tc_ids[j];
            for (j = 0; j < MAX_TENSOR_DIMS; j++) coe.mult.ta_ids[j] = oe->mult->ta_ids[j];
            for (j = 0; j < MAX_TENSOR_DIMS; j++) coe.mult.tb_ids[j] = oe->mult->tb_ids[j];
        }

      ceqs.op_entries.push_back(coe);
    }
  }


};

