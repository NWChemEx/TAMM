#ifndef __tamm_tops_h__
#define __tamm_tops_h__

#include <map>

namespace tamm {

  void tensors_and_ops(Equations &eqs,
                       std::map<std::string, tamm::Tensor> &tensors,
                       std::vector<Operation> &ops);

}

#endif /* __tamm_tops_h__ */

