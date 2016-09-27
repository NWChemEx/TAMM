#ifndef __ctce_tops_h__
#define __ctce_tops_h__

#include <map>

namespace ctce {

  void tensors_and_ops(Equations &eqs,
                       std::map<std::string, ctce::Tensor> &tensors,
                       std::vector<Operation> &ops);

}

#endif /* __ctce_tops_h__ */

