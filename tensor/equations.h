#ifndef __ctce_equations_h__
#define __ctce_equations_h__

#include "input.h"

extern "C" {
#include "../ctce_parser/ctce_parser.h"
}

namespace ctce {

  struct Equations {
    std::vector<RangeEntry> range_entries;
    std::vector<IndexEntry> index_entries;
    std::vector<TensorEntry> tensor_entries;
    std::vector<OpEntry> op_entries;

  };

  void ccsd_t1_equations(Equations &eqs);
  void ccsd_t2_equations(Equations &eqs);
  void cc2_t1_equations(Equations &eqs);
  void cc2_t2_equations(Equations &eqs);
  void ccsd_e_equations(Equations &eqs);
  void icsd_t1_equations(Equations &eqs);
  void icsd_t2_equations(Equations &eqs);

  void parser_eqs_to_ctce_eqs(::Equations *eqs, ctce::Equations &ceqs);

  /* void icsd_t1_equations(Equations &eqs); */
  /* void icsd_t2_equations(Equations &eqs); */

}; /*ctce*/

#endif /*__ctce_equations_h__*/
