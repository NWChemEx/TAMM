#ifndef __ctce_equations_h__
#define __ctce_equations_h__

#include "input.h"

namespace ctce {

  struct Equations {
    std::vector<RangeEntry> range_entries;
    std::vector<IndexEntry> index_entries;
    std::vector<TensorEntry> tensor_entries;
    std::vector<OpEntry> op_entries;

  };

  void tensors_and_ops(Equations &eqs,
                       std::vector<Tensor> &tensors,
                       std::vector<Operation> &ops);

  void ccsd_t1_equations(Equations &eqs);
  void ccsd_t2_equations(Equations &eqs);

}; /*ctce*/

#endif /*__ctce_equations_h__*/
