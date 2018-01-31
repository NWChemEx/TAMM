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

  LabeledTensor(const Tensor<T>& tensor,
                const IndexLabelVec ilv)
      : tensor_{tensor},
        ilv_{ilv} {}

  AddOp<T,LabeledTensor<T>> operator += (const LabeledTensor<T>& rhs);

  AddOp<T,LabeledTensor<T>> operator += (const T& rhs);

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  AddOp<T1,LabeledTensor<T>> operator += (const std::pair<T1, LabeledTensor<T>>& rhs);

  AddOp<T,LabeledTensor<T>> operator = (const LabeledTensor<T>& rhs);

  AddOp<T,LabeledTensor<T>> operator = (const T& rhs);

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  AddOp<T1,LabeledTensor<T>> operator = (const std::pair<T1, LabeledTensor<T>>& rhs);

  MultOp<T,LabeledTensor<T>>
  operator += (const std::tuple<LabeledTensor, LabeledTensor<T>>& rhs);

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  MultOp<T1,LabeledTensor<T>>
  operator += (const std::tuple<T1, LabeledTensor<T>, LabeledTensor<T>>& rhs);

  MultOp<T,LabeledTensor<T>>
  operator = (const std::tuple<LabeledTensor<T>, LabeledTensor<T>>& rhs);

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  MultOp<T1,LabeledTensor<T>>
  operator = (const std::tuple<T1, LabeledTensor<T>, LabeledTensor<T>>& rhs);

 protected:
  Tensor<T> tensor_;
  IndexLabelVec ilv_;
};

template<typename T1, typename T2>
inline std::pair<T1, LabeledTensor<T2>>
operator * (T1 val, const LabeledTensor<T2>& rhs) {
  return {val, rhs};
}

template<typename T>
inline std::pair<LabeledTensor<T>, LabeledTensor<T>>
operator * (const LabeledTensor<T>& rhs1, const LabeledTensor<T>& rhs2) {
  return {rhs1, rhs2};
}

template<typename T1, typename T2>
inline std::tuple<T1, LabeledTensor<T2>, LabeledTensor<T2>>
operator * (std::pair<T1, LabeledTensor<T2>> rhs1, const LabeledTensor<T2>& rhs2) {
  return {std::get<0>(rhs1), std::get<1>(rhs1), rhs2};
}


}  // namespace tammy

#endif  // TAMMY_LABELED_TENSOR_H_
