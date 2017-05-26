#ifndef TAMMX_LABELEDTENSOR_H_
#define TAMMX_LABELEDTENSOR_H_

#include "tammx/op-entry.h"
#include "tammx/types.h"

namespace tammx {

class Tensor;

struct LabeledTensor {
  Tensor* tensor_;
  TensorLabel label_;

  template<typename T>
  SetOpEntry<T> operator = (T value) {
    return {*this, value};
  }
  
  AddOpEntry<int> operator = (LabeledTensor rhs) {
    return {*this, 1, 0, rhs}
  }

  template<typename T>
  AddOpEntry<T> operator = (std::tuple<T, LabeledTensor> rhs) {
    return {*this, std::get<0>(rhs), T(0), std::get<1>(rhs)};
  }

  template<typename T>
  MultOpEntry<T> operator = (std::tuple<T, LabeledTensor, LabeledTensor> rhs) {
    return {*this, std::get<0>(rhs), T(0), std::get<1>(rhs), std::get<2>(rhs)};
  }
  
  MultOpEntry<int> operator = (std::tuple<LabeledTensor, LabeledTensor> rhs) {
    return {*this, 1, 0, std::get<1>(rhs), std::get<2>(rhs)};    
  }

  AddOpEntry<int> operator += (LabeledTensor rhs) {
    return {*this, 1, 1, rhs}
  }

  template<typename T>
  AddOpEntry<T> operator += (std::tuple<T, LabeledTensor> rhs) {
    return {*this, std::get<0>(rhs), T(1), std::get<1>(rhs)};
  }

  template<typename T>
  MultOpEntry<T> operator = (std::tuple<T, LabeledTensor, LabeledTensor> rhs) {
    return {*this, std::get<0>(rhs), T(1), std::get<1>(rhs), std::get<2>(rhs)};
  }
  
  MultOpEntry<int> operator = (std::tuple<LabeledTensor, LabeledTensor> rhs) {
    return {*this, 1, 1, std::get<1>(rhs), std::get<2>(rhs)};    
  }
};  // LabeledTensor

}  // namespace tammx

#endif // TAMMX_LABELEDTENSOR_H_
