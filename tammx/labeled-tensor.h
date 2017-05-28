#ifndef TAMMX_LABELEDTENSOR_H_
#define TAMMX_LABELEDTENSOR_H_

#include <type_traits>

//#include "tammx/op-entry.h"
#include "tammx/types.h"
#include "tammx/tensor.h"

namespace tammx {

class Tensor;

template<typename T>
struct SetOpEntry;

template<typename T>
struct AddOpEntry;

template<typename T>
struct MultOpEntry;

struct LabeledTensor {
  Tensor* tensor_;
  TensorLabel label_;

  /**
   * @todo Implemet setop validate
   */
  template<typename T,
           typename = std::enable_if_t<std::is_arithmetic<T>::value>>
  SetOpEntry<T> operator = (T value);

  AddOpEntry<int> operator = (LabeledTensor rhs);

  template<typename T>
  AddOpEntry<T> operator = (std::tuple<T, LabeledTensor> rhs);

  template<typename T>
  MultOpEntry<T> operator = (std::tuple<T, LabeledTensor, LabeledTensor> rhs);

  MultOpEntry<int> operator = (std::tuple<LabeledTensor, LabeledTensor> rhs);

  AddOpEntry<int> operator += (LabeledTensor rhs);

  template<typename T>
  AddOpEntry<T> operator += (std::tuple<T, LabeledTensor> rhs);

  template<typename T>
  MultOpEntry<T> operator += (std::tuple<T, LabeledTensor, LabeledTensor> rhs);

  MultOpEntry<int> operator += (std::tuple<LabeledTensor, LabeledTensor> rhs);
};  // LabeledTensor

//#include "tammx/op-entry.h"

enum class ResultMode { update, set };

template<typename T>
struct SetOpEntry {
  LabeledTensor lhs;
  T value;
};

template<typename T>
struct AddOpEntry {
  LabeledTensor lhs;
  T alpha;
  LabeledTensor rhs;
  ResultMode mode;
};

template<typename T>
struct MultOpEntry {
  LabeledTensor lhs;
  T alpha;
  LabeledTensor rhs1, rhs2;
  ResultMode mode;
};

template<typename Func, unsigned ndim, unsigned nrhs>
struct MapOpEntry {
  LabeledTensor lhs;
  std::vector<LabeledTensor> rhss;
  Func func;
};

/**
 * @todo Should validation be done in *OpEnty constructors?
 */

template<typename T>
void
addop_validate(const LabeledTensor& ltc,
               const std::tuple<T, LabeledTensor>& rhs);

template<typename T>
void
multop_validate(const LabeledTensor& ltc,
                const std::tuple<T, LabeledTensor, LabeledTensor>& rhs);

/**
 * @todo Implement setop validate
 */
template<typename T,
         typename = std::enable_if_t<std::is_arithmetic<T>::value>>
inline SetOpEntry<T>
LabeledTensor::operator = (T value) {
  //setop_validate(*this, value);
  return {*this, value};
}

inline AddOpEntry<int>
LabeledTensor::operator = (LabeledTensor rhs) {
  addop_validate(*this, std::make_tuple(1, rhs));
  return {*this, 1, rhs, ResultMode::set};
}

template<typename T>
inline AddOpEntry<T>
LabeledTensor::operator = (std::tuple<T, LabeledTensor> rhs) {
  addop_validate(*this, rhs);
  return {*this, std::get<0>(rhs), std::get<1>(rhs), ResultMode::set};
}

template<typename T>
inline MultOpEntry<T>
LabeledTensor::operator = (std::tuple<T, LabeledTensor, LabeledTensor> rhs) {
  multop_validate(*this, rhs);
  return {*this, std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs), ResultMode::set};
}

inline MultOpEntry<int>
LabeledTensor::operator = (std::tuple<LabeledTensor, LabeledTensor> rhs) {
  multop_validate(*this, std::make_tuple(1, std::get<0>(rhs), std::get<1>(rhs)));
  return {*this, 1, std::get<0>(rhs), std::get<1>(rhs), ResultMode::set};
}

inline AddOpEntry<int>
LabeledTensor::operator += (LabeledTensor rhs) {
  addop_validate(*this, std::make_tuple(1, rhs));
  return {*this, 1, rhs, ResultMode::update};
}

template<typename T>
inline AddOpEntry<T>
LabeledTensor::operator += (std::tuple<T, LabeledTensor> rhs) {
  addop_validate(*this, std::make_tuple(std::get<0>(rhs), std::get<1>(rhs)));
  return {*this, std::get<0>(rhs), std::get<1>(rhs), ResultMode::update};
}

template<typename T>
inline MultOpEntry<T>
LabeledTensor::operator += (std::tuple<T, LabeledTensor, LabeledTensor> rhs) {
  multop_validate(*this, rhs);
  return {*this, std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs), ResultMode::update};
}

inline MultOpEntry<int>
LabeledTensor::operator += (std::tuple<LabeledTensor, LabeledTensor> rhs) {
  multop_validate(*this, std::make_tuple(1, std::get<0>(rhs), std::get<1>(rhs)));
  return {*this, 1, std::get<0>(rhs), std::get<1>(rhs), ResultMode::update};
}


//@todo for now assume all indices in a symmetry group are sliced
//the same way
inline void
validate_slicing(const TensorVec<SymmGroup>& indices,
                 const TensorLabel& label) {
  int pos = 0;
  for(auto grp : indices)  {
    if (grp.size() > 0){
      for(int i=0; i<grp.size(); i++) {
        Expects(label[pos+i].dt == label[pos].dt);
      }
    }
    pos += grp.size();
  }
}

template<typename T>
inline void
addop_validate(const LabeledTensor& ltc,
               const std::tuple<T, LabeledTensor>& rhs) {
  auto lta = std::get<1>(rhs);
  Expects(ltc.tensor_ != nullptr);
  Expects(lta.tensor_ != nullptr);
  const Tensor& tc = *ltc.tensor_;
  const Tensor& ta = *lta.tensor_;
  Expects(tc.rank() == ta.rank());

  TensorLabel clabel = ltc.label_;
  TensorLabel alabel = lta.label_;

  validate_slicing(tc.indices(), ltc.label_);
  validate_slicing(ta.indices(), lta.label_);

  Expects(alabel.size() == ta.rank());
  Expects(clabel.size() == tc.rank());

  //all labels are of compatible type
  for(int i=0; i<alabel.size(); i++) {
    Expects(is_dim_subset(ta.flindices()[i], alabel[i].dt));
  }
  for(int i=0; i<clabel.size(); i++) {
    Expects(is_dim_subset(tc.flindices()[i], clabel[i].dt));
  }

  std::sort(alabel.begin(), alabel.end());
  std::sort(clabel.begin(), clabel.end());

  //all labels are unique
  Expects(std::adjacent_find(alabel.begin(), alabel.end()) == alabel.end());
  Expects(std::adjacent_find(clabel.begin(), clabel.end()) == clabel.end());

  //all labels in ta are in tb
  for(auto &al: alabel) {
    Expects(std::find(clabel.begin(), clabel.end(), al) != clabel.end());
  }
}


template<typename T>
inline void
multop_validate(const LabeledTensor& ltc,
                const std::tuple<T, LabeledTensor, LabeledTensor>& rhs) {
  auto &lta = std::get<1>(rhs);
  auto &ltb = std::get<2>(rhs);
  Expects(ltc.tensor_ != nullptr);
  Expects(lta.tensor_ != nullptr);
  Expects(ltb.tensor_ != nullptr);
  const Tensor& tc = *ltc.tensor_;
  const Tensor& ta = *lta.tensor_;
  const Tensor& tb = *ltb.tensor_;

  // Expects(element_type<T> == tc.element_type());
  // Expects(element_type<T> == ta.element_type());
  // Expects(element_type<T> == tb.element_type());

  TensorLabel clabel = ltc.label_;
  TensorLabel alabel = lta.label_;
  TensorLabel blabel = ltb.label_;

  Expects(clabel.size() == tc.rank());
  Expects(alabel.size() == ta.rank());
  Expects(blabel.size() == tb.rank());

  validate_slicing(tc.indices(), ltc.label_);
  validate_slicing(ta.indices(), lta.label_);
  validate_slicing(tb.indices(), ltb.label_);

  //all labels are of compatible type
  for(int i=0; i<alabel.size(); i++) {
    Expects(is_dim_subset(ta.flindices()[i], alabel[i].dt));
  }
  for(int i=0; i<blabel.size(); i++) {
    Expects(is_dim_subset(tb.flindices()[i], blabel[i].dt));
  }
  for(int i=0; i<clabel.size(); i++) {
    Expects(is_dim_subset(tc.flindices()[i], clabel[i].dt));
  }

  std::sort(alabel.begin(), alabel.end());
  std::sort(blabel.begin(), blabel.end());
  std::sort(clabel.begin(), clabel.end());

  //all labels are unique
  Expects(std::adjacent_find(alabel.begin(), alabel.end()) == alabel.end());
  Expects(std::adjacent_find(blabel.begin(), blabel.end()) == blabel.end());
  Expects(std::adjacent_find(clabel.begin(), clabel.end()) == clabel.end());

  TensorLabel slabel;
  std::set_intersection(alabel.begin(), alabel.end(),
                        blabel.begin(), blabel.end(),
                        std::back_inserter(slabel));
  //summation index is not in the output
  for(auto &sl: slabel) {
    Expects(std::find(clabel.begin(), clabel.end(), sl) == clabel.end());
  }
  //every label in A/B is either in slabel or clabel
  for(auto &al : alabel) {
    Expects(std::find(slabel.begin(), slabel.end(), al) != slabel.end()
            || std::find(clabel.begin(), clabel.end(), al) != clabel.end());
  }
  for(auto &bl : blabel) {
    Expects(std::find(slabel.begin(), slabel.end(), bl) != slabel.end()
            || std::find(clabel.begin(), clabel.end(), bl) != clabel.end());
  }
  Expects(clabel.size() == alabel.size() + blabel.size() - 2 * slabel.size());
}



}  // namespace tammx

#endif // TAMMX_LABELEDTENSOR_H_
