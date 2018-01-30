#ifndef TAMMY_LABELED_TENSOR_H_
#define TAMMY_LABELED_TENSOR_H_

#include <memory>

#include "boundvec.h"
#include "errors.h"
#include "types.h"
#include "proc_group.h"
#include "ops.h"

namespace tammy {

template<typename T>
class LabeledTensor {
 public:
  LabeledTensor() = default;
  LabeledTensor(const LabeledTensor&) = default;

  LabeledTensor(const Tensor& tensor,
                const IndexLabelVec ilv)
      : tensor_{tensor},
        ilv_{ilv} {}

  AddOp<int,LabeledTensor<T>> operator += (const LabeledTensor<T>& rhs);

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  AddOp<T1,LabeledTensor<T>> operator += (const std::pair<T, LabeledTensor<T>& rhs);

  AddOp<int,LabeledTensor<T>> operator = (const LabeledTensor<T>& rhs);

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  AddOp<T1,LabeledTensor<T>> operator = (const std::pair<T, LabeledTensor<T>>& rhs);

  MultOp<T,LabeledTensor<T>,LabeledTensor<T>>
  operator += (const std::tuple<LabeledTensor, LabeledTensor<T>>& rhs);

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  MultOp<T1,LabeledTensor<T>,LabeledTensor<T>>
  operator += (const std::tuple<T1, LabeledTensor<T>, LabeledTensor<T>>& rhs);

  MultOp<T,LabeledTensor<T>,LabeledTensor<T>>
  operator = (const std::tuple<LabeledTensor<T>, LabeledTensor<T>>& rhs);

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  MultOp<T1,LabeledTensor<T>,LabeledTensor<T>>
  operator = (const std::tuple<T1, LabeledTensor<T>, LabeledTensor<T>>& rhs);

 protected:
  Tensor tensor_;
  IndexLabelVec ilv_;
};

inline std::pair<T, LabeledTensor>
operator * (T val, const LabeledTensor& rhs) {
  return {val, rhs};
}

inline std::pair<LabeledTensor, LabeledTensor>
operator * (const LabeledTensor& rhs1, const LabeledTensor& rhs2) {
  return {rhs1, rhs2};
}

inline std::tuple<T, LabeledTensor, LabeledTensor>
operator * (std::pair<T, LabeledTensor> rhs1, const LabeledTensor& rhs2) {
  return {std::get<0>(rhs1), std::get<1>(rhs1), rhs2};
}


}  // namespace tammy

#endif  // TAMMY_LABELED_TENSOR_H_
