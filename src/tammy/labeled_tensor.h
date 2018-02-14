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

  Tensor<T> tensor() const {
    return tensor_;
  }

  IndexLabelVec label() const {
    return ilv_;
  }

  AddOp<T,LabeledTensor<T>> operator += (const LabeledTensor<T>& rhs) {
    addop_validate(*this, std::make_tuple(1, rhs));
    bool is_assign = false;
    return {*this, 1, rhs, is_assign};
  }

  SetOp<T,LabeledTensor<T>> operator += (const T& rhs) {
    bool is_assign = false;
    return {*this, rhs, is_assign};
  }

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  AddOp<T1,LabeledTensor<T>> operator += (const std::tuple<T1, LabeledTensor<T>>& rhs) {
    addop_validate(*this, std::make_tuple(std::get<0>(rhs), std::get<1>(rhs)));
    bool is_assign = false;
    
    return {*this, std::get<0>(rhs), std::get<1>(rhs), is_assign};
  }

  AddOp<T,LabeledTensor<T>> operator = (const LabeledTensor<T>& rhs) {
    addop_validate(*this, std::make_tuple(1, rhs));
    bool is_assign = true;

    return {*this, 1, rhs, is_assign};
  }

  SetOp<T,LabeledTensor<T>> operator = (const T& rhs) {
    bool is_assign = true;
    return {*this, rhs, is_assign};
  }

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  AddOp<T1,LabeledTensor<T>> operator = (const std::tuple<T1, LabeledTensor<T>>& rhs) {
    addop_validate(*this, std::make_tuple(std::get<0>(rhs), std::get<1>(rhs)));
    bool is_assign = true;
    
    return {*this, std::get<0>(rhs), std::get<1>(rhs), is_assign};
  }

  MultOp<T,LabeledTensor<T>>
  operator += (const std::tuple<LabeledTensor, LabeledTensor<T>>& rhs) {
    multop_validate(*this, std::make_tuple(1, std::get<0>(rhs), std::get<1>(rhs)));
    bool is_assign = false;

    return {*this, 1, std::get<0>(rhs), std::get<1>(rhs), is_assign};
  }

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  MultOp<T1,LabeledTensor<T>>
  operator += (const std::tuple<T1, LabeledTensor<T>, LabeledTensor<T>>& rhs) {
    multop_validate(*this, rhs);
    bool is_assign = false;

    return {*this, std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs), is_assign};
  }

  MultOp<T,LabeledTensor<T>>
  operator = (const std::tuple<LabeledTensor<T>, LabeledTensor<T>>& rhs) {
    multop_validate(*this, std::make_tuple(1, std::get<0>(rhs), std::get<1>(rhs)));
    bool is_assign = true;

    return {*this, 1, std::get<0>(rhs), std::get<1>(rhs), is_assign};
  }

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  MultOp<T1,LabeledTensor<T>>
  operator = (const std::tuple<T1, LabeledTensor<T>, LabeledTensor<T>>& rhs) {
    multop_validate(*this, rhs);
    bool is_assign = true;

    return {*this, std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs), is_assign};
  }

 protected:
  Tensor<T> tensor_;
  IndexLabelVec ilv_;
};

template<typename T1, typename T2>
inline std::tuple<T1, LabeledTensor<T2>>
operator * (T1 val, const LabeledTensor<T2>& rhs) {
  return {val, rhs};
}

template<typename T>
inline std::tuple<LabeledTensor<T>, LabeledTensor<T>>
operator * (const LabeledTensor<T>& rhs1, const LabeledTensor<T>& rhs2) {
  return {rhs1, rhs2};
}

template<typename T1, typename T2>
inline std::tuple<T1, LabeledTensor<T2>, LabeledTensor<T2>>
operator * (std::tuple<T1, LabeledTensor<T2>> rhs1, const LabeledTensor<T2>& rhs2) {
  return {std::get<0>(rhs1), std::get<1>(rhs1), rhs2};
}


inline void
validate_slicing(const TensorVec<IndexRange>& index_ranges,
                 const IndexLabelVec& label) {
  for(size_t i=0; i<index_ranges.size(); i++) {
    EXPECTS(index_ranges[i].is_superset_of(label[i].ir()));
  }
}

template<typename LabeledTensorType, typename T>
inline void
addop_validate(const LabeledTensorType& ltc,
               const std::tuple<T, LabeledTensorType>& rhs) {
  auto lta = std::get<1>(rhs);
  // EXPECTS(ltc.tensor() != nullptr);
  // EXPECTS(lta.tensor() != nullptr);
  const auto& tc = ltc.tensor();
  const auto& ta = lta.tensor();

  //tensors should have same rank
  EXPECTS(tc.rank() == ta.rank());

  IndexLabelVec clabel = ltc.label();
  IndexLabelVec alabel = lta.label();

  //index range underlying an index label is the same or a subset of the tensor’s index range along that dimension 
  validate_slicing(tc.dim_ranges(), ltc.label());
  validate_slicing(ta.dim_ranges(), lta.label());

  //length of the index label vector matches the rank (number of indices) in the tensor
  EXPECTS(alabel.size() == ta.rank());
  EXPECTS(clabel.size() == tc.rank());

#if 0
  //all labels are of compatible type
  for(int i=0; i<alabel.size(); i++) {
    EXPECTS(is_range_subset(ta.flindices()[i], alabel[i].rt()));
  }
  for(int i=0; i<clabel.size(); i++) {
    EXPECTS(is_range_subset(tc.flindices()[i], clabel[i].rt()));
  }
#endif
  
  std::sort(alabel.begin(), alabel.end());
  std::sort(clabel.begin(), clabel.end());

  //all labels are unique
  EXPECTS(std::adjacent_find(alabel.begin(), alabel.end()) == alabel.end());
  EXPECTS(std::adjacent_find(clabel.begin(), clabel.end()) == clabel.end());

  //all labels in ta are in tb
  for(auto &al: alabel) {
    EXPECTS(std::find(clabel.begin(), clabel.end(), al) != clabel.end());
  }
}


template<typename LabeledTensorType, typename T>
inline void
multop_validate(const LabeledTensorType& ltc,
                const std::tuple<T, LabeledTensorType, LabeledTensorType>& rhs) {
  auto &lta = std::get<1>(rhs);
  auto &ltb = std::get<2>(rhs);
  // EXPECTS(ltc.tensor_ != nullptr);
  // EXPECTS(lta.tensor_ != nullptr);
  // EXPECTS(ltb.tensor_ != nullptr);
  const auto& tc = ltc.tensor();
  const auto& ta = lta.tensor();
  const auto& tb = ltb.tensor();

  IndexLabelVec clabel = ltc.label();
  IndexLabelVec alabel = lta.label();
  IndexLabelVec blabel = ltb.label();

  //length of the index label vector matches the rank (number of indices) in the tensor
  EXPECTS(clabel.size() == tc.rank());
  EXPECTS(alabel.size() == ta.rank());
  EXPECTS(blabel.size() == tb.rank());

  //index range underlying an index label is the same or a subset of the tensor’s index range along that dimension 
  validate_slicing(tc.dim_ranges(), ltc.label());
  validate_slicing(ta.dim_ranges(), lta.label());
  validate_slicing(tb.dim_ranges(), ltb.label());

#if 0
  //all labels are of compatible type
  for(int i=0; i<alabel.size(); i++) {
    EXPECTS(is_range_subset(ta.flindices()[i], alabel[i].rt()));
  }
  for(int i=0; i<blabel.size(); i++) {
    EXPECTS(is_range_subset(tb.flindices()[i], blabel[i].rt()));
  }
  for(int i=0; i<clabel.size(); i++) {
    EXPECTS(is_range_subset(tc.flindices()[i], clabel[i].rt()));
  }
#endif

  std::sort(alabel.begin(), alabel.end());
  std::sort(blabel.begin(), blabel.end());
  std::sort(clabel.begin(), clabel.end());

  //all labels are unique
  EXPECTS(std::adjacent_find(alabel.begin(), alabel.end()) == alabel.end());
  EXPECTS(std::adjacent_find(blabel.begin(), blabel.end()) == blabel.end());
  EXPECTS(std::adjacent_find(clabel.begin(), clabel.end()) == clabel.end());

  IndexLabelVec rhs_labels;
  std::set_union(alabel.begin(), alabel.end(),
             blabel.begin(), blabel.end(),
             std::back_inserter(rhs_labels));
  
  IndexLabelVec inner_labels;
  std::set_difference(rhs_labels.begin(), rhs_labels.end(),
                  clabel.begin(), clabel.end(),
                  std::back_inserter(inner_labels));

  IndexLabelVec slabel;
  std::set_intersection(alabel.begin(), alabel.end(),
                        blabel.begin(), blabel.end(),
                        std::back_inserter(slabel));

  // Every outer index label (clabel) appears in exactly one RHS tensor
  for(auto &ol: clabel) {
    EXPECTS(std::find(slabel.begin(), slabel.end(), ol) == slabel.end()
            && std::find(rhs_labels.begin(), rhs_labels.end(), ol) != rhs_labels.end());
  }

  // Every inner index label appears exactly once in both RHS tensors 
  for(auto &il: inner_labels) {
    EXPECTS(std::find(slabel.begin(), slabel.end(), il) != slabel.end());
  }
  
  // //summation index is not in the output
  // for(auto &sl: slabel) {
  //   EXPECTS(std::find(clabel.begin(), clabel.end(), sl) == clabel.end());
  // }
  // //every label in A/B is either in slabel or clabel
  // for(auto &al : alabel) {
  //   EXPECTS(std::find(slabel.begin(), slabel.end(), al) != slabel.end()
  //           || std::find(clabel.begin(), clabel.end(), al) != clabel.end());
  // }
  // for(auto &bl : blabel) {
  //   EXPECTS(std::find(slabel.begin(), slabel.end(), bl) != slabel.end()
  //           || std::find(clabel.begin(), clabel.end(), bl) != clabel.end());
  // }

  EXPECTS(clabel.size() == alabel.size() + blabel.size() - 2 * slabel.size());
}

}  // namespace tammy

#endif  // TAMMY_LABELED_TENSOR_H_
