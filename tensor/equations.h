#ifndef __tamm_equations_h__
#define __tamm_equations_h__

#include "input.h"
#include <map>

extern "C" {
#include "../tamm_parser/tamm_parser.h"
}

namespace tamm {

  struct Equations {
    std::vector<RangeEntry> range_entries;
    std::vector<IndexEntry> index_entries;
    //std::vector<TensorEntry> tensor_entries;
    std::map<std::string, tamm::TensorEntry> tensor_entries;
    std::vector<OpEntry> op_entries;

  };

  void cisd_e_equations(Equations &eqs);
  void cisd_c1_equations(Equations &eqs);
  void cisd_c2_equations(Equations &eqs);

  void ccsd_e_equations(Equations &eqs);
  void ccsd_t1_equations(Equations &eqs);
  void ccsd_t2_equations(Equations &eqs);
  void cc2_t1_equations(Equations &eqs);
  void cc2_t2_equations(Equations &eqs);
  void icsd_t1_equations(Equations &eqs);
  void icsd_t2_equations(Equations &eqs);
  void ipccsd_x1_equations(Equations &eqs);
  void ipccsd_x2_equations(Equations &eqs);
  void eaccsd_x1_equations(Equations &eqs);
  void eaccsd_x2_equations(Equations &eqs);

  void parser_eqs_to_tamm_eqs(::Equations *eqs, tamm::Equations &ceqs);

  /* void icsd_t1_equations(Equations &eqs); */
  /* void icsd_t2_equations(Equations &eqs); */

}; /*tamm*/

#endif /*__tamm_equations_h__*/
