#ifndef TAMMY_LABELED_TENSOR_H_
#define TAMMY_LABELED_TENSOR_H_

#include <memory>

#include "boundvec.h"
#include "errors.h"
#include "types.h"
#include "proc_group.h"
#include "ops.h"

namespace tammy {

class LoopSpec {
 public:
  LoopSpec()
      : has_oll_{false},
        has_ill_{false},
        has_symm_factor_{false} {}

  LoopSpec(const LoopSpec&) = default;
  
  LoopSpec(const OuterLabeledLoop& oll)
      : LoopSpec{} {
    set_oll(oll);
  }

  LoopSpec(const InnerLabeledLoop& ill)
      : LoopSpec{} {
    set_ill(ill);
  }

  LoopSpec(const SymmFactor& sf)
      : LoopSpec{} {
    set_symm_factor(sf);
  }

  LoopSpec& set_oll(const OuterLabeledLoop& oll) {
    oll_ = oll;
    has_oll_ = true;
    return *this;
  }

  LoopSpec& set_ill(const InnerLabeledLoop& ill) {
    ill_ = ill;
    has_ill_ = true;
    return *this;
  }

  LoopSpec& set_symm_factor(const SymmFactor& sf) {
    symm_factor_ = sf;
    has_symm_factor_ = true;
    return *this;
  }

  bool has_oll() const {
    return has_oll_;
  }

  bool has_ill() const {
    return has_ill_;
  }

  bool has_symm_factor() const {
    return has_symm_factor_;
  }

  OuterLabeledLoop oll() const{
    return oll_;
  }

  InnerLabeledLoop ill() const{
    return ill_;
  }

  SymmFactor symm_factor() const{
    return symm_factor_;
  }

 private:
  OuterLabeledLoop oll_;
  InnerLabeledLoop ill_;
  SymmFactor symm_factor_;

  bool has_oll_;
  bool has_ill_;
  bool has_symm_factor_;
};



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

  IndexLabelVec labels() const {
    return ilv_;
  }

  AddOp<T,LabeledTensor<T>> operator += (const std::tuple<LoopSpec,
                                         LabeledTensor<T>>& rhs) {
    construct_addop(std::make_tuple(std::get<0>(rhs), 1, std::get<1>(rhs)), false);
    // addop_validate(*this, std::make_tuple(1, std::get<1>(rhs)));
    // bool is_assign = false;
    // const auto& loop_spec = std::get<0>(rhs);
    // if(loop_spec.has_oll()) {
    //   return {*this, 1, rhs, loop_spec.oll(), is_assign};
    // } else {
    //   return {*this, 1, rhs, loop_nest(), is_assign};
    // }
  }

  AddOp<T,LabeledTensor<T>> operator += (LabeledTensor<T> rhs) {
    return *this += loop_nest() * rhs;
  }

  SetOp<T,LabeledTensor<T>>
  operator += (const T& rhs) {
    // bool is_assign = false;
    // return {*this, rhs, loop_nest(), is_assign};
    return *this += loop_nest() * rhs;
  }

  SetOp<T,LabeledTensor<T>>
  operator += (const std::tuple<LoopSpec, T>& rhs) {
    construct_setop(rhs, false);
  }

  SetOp<T,LabeledTensor<T>>
  operator = (T rhs) {
    return *this = loop_nest() * rhs;
  }

  SetOp<T,LabeledTensor<T>>
  operator = (const std::tuple<LoopSpec, T>& rhs) {
    construct_setop(rhs, true);
  }

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  AddOp<T1,LabeledTensor<T>>
  operator += (const std::tuple<LoopSpec, T1, LabeledTensor<T>>& rhs) {
    // addop_validate(*this, std::make_tuple(std::get<0>(rhs), std::get<1>(rhs)));
    // bool is_assign = false;
    
    // return {*this, std::get<0>(rhs), std::get<1>(rhs), loop_nest(), is_assign};
    //return *this += loop_nest() * std::get<0>(rhs) * std::get<1>(rhs);
    construct_addop(rhs, false);
  }

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  AddOp<T1,LabeledTensor<T>>
  operator = (const std::tuple<LoopSpec, T1, LabeledTensor<T>>& rhs) {
    construct_addop(rhs, true);
  }

  AddOp<T,LabeledTensor<T>>
  operator = (const std::tuple<LoopSpec, LabeledTensor<T>> rhs) {
    return *this = std::get<0>(rhs) * T{1} * std::get<1>(rhs);
  }

  AddOp<T,LabeledTensor<T>>
  operator = (const LabeledTensor<T>& rhs) {
    return *this = loop_nest() * T{1} * rhs;
  }

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  MultOp<T1,LabeledTensor<T>>
  operator += (const std::tuple<LoopSpec, T1, LabeledTensor<T>, LabeledTensor<T>>& rhs) {
    return construct_multop(rhs, false);
  }

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  MultOp<T1,LabeledTensor<T>>
  operator = (const std::tuple<LoopSpec, T1, LabeledTensor<T>, LabeledTensor<T>>& rhs) {
    return construct_multop(rhs, true);
  }

  MultOp<T,LabeledTensor<T>>
  operator += (const std::tuple<LoopSpec, LabeledTensor<T>, LabeledTensor<T>>& rhs) {
    return *this += std::get<0>(rhs) * T{1} * std::get<1>(rhs) * std::get<2>(rhs);
  }

  MultOp<T,LabeledTensor<T>>
  operator = (const std::tuple<LoopSpec, LabeledTensor<T>, LabeledTensor<T>>& rhs) {
    return *this = std::get<0>(rhs) * T{1} * std::get<1>(rhs) * std::get<2>(rhs);
  }


 protected:
  SetOp<T,LabeledTensor<T>>
  construct_setop (const std::tuple<LoopSpec, T>& rhs, bool is_assign) {
    const auto& loop_spec = std::get<0>(rhs);
    if(loop_spec.has_oll()) {
      return {*this, std::get<1>(rhs), loop_spec.oll(), is_assign};
    } else {
      return {*this, std::get<1>(rhs), loop_nest(), is_assign};
    }
  }

  template<typename T1,
          typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  AddOp<T1,LabeledTensor<T>>
  construct_addop (const std::tuple<LoopSpec, T1, LabeledTensor<T>>& rhs, bool is_assign) {
    addop_validate(*this, std::make_tuple(std::get<1>(rhs), std::get<2>(rhs)));
    const auto& loop_spec = std::get<0>(rhs);
    T1 alpha = std::get<1>(rhs);
    auto& rhs_tensor = std::get<2>(rhs);
    if(loop_spec.has_oll()) {
      return {*this, alpha, rhs_tensor, loop_spec.oll(), is_assign};
    } else {
      return {*this, alpha, rhs_tensor, loop_nest(), is_assign};
    }    
  }

    template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  MultOp<T1,LabeledTensor<T>>
  construct_multop (const std::tuple<LoopSpec, T1, LabeledTensor<T>, LabeledTensor<T>>& rhs, bool is_assign) {  
    multop_validate(*this, std::make_tuple(std::get<1>(rhs), std::get<2>(rhs), std::get<3>(rhs)));

    const auto& loop_spec = std::get<0>(rhs);
    OuterLabeledLoop oll;
    InnerLabeledLoop ill;
    SymmFactor sf;
    if(loop_spec.has_oll()) {
      oll = loop_spec.oll();
    } else {
      oll = loop_nest();
    }
    if(loop_spec.has_ill()) {
      ill = loop_spec.ill();
    } else {
      ill = inner_loop_nest(std::get<2>(rhs), std::get<3>(rhs));
    }
    if(loop_spec.has_symm_factor()) {
      sf = loop_spec.symm_factor();
    } else {
      sf = SymmFactor{};
    }
    
    return {*this, std::get<1>(rhs), std::get<2>(rhs), std::get<3>(rhs),
          oll, ill, sf, is_assign};
  }

  Tensor<T> tensor_;
  IndexLabelVec ilv_;

  OuterLabeledLoop loop_nest() const {
    return {labels(), tensor().perm_group().sliced_loop_nest(labels())};
  }  

  template<typename T1>
  static InnerLabeledLoop inner_loop_nest(
      const LabeledTensor<T1>& ltensor1,
      const LabeledTensor<T1>& ltensor2) {
    using Itr = IndexSpace::Iterator;
    IndexLabelVec labels1{ltensor1.labels()};
    IndexLabelVec labels2{ltensor2.labels()};

    std::sort(labels1.begin(), labels1.end());
    std::sort(labels2.begin(), labels2.end());

    IndexLabelVec inner_labels;
    std::set_intersection(labels1.begin(), labels1.end(),
                          labels2.begin(), labels2.end(),
                          std::back_inserter(inner_labels));
    std::vector<Itr> begins, ends;
    for(const auto& il : inner_labels) {
      begins.push_back(il.ir().begin());
      ends.push_back(il.ir().end());
    }
    return InnerLabeledLoop{inner_labels, begins, ends, {}};
  }
};

inline LoopSpec
operator * (LoopSpec ls,
            const InnerLabeledLoop& ill) {
  return ls.set_ill(ill);
}

inline LoopSpec
operator * (LoopSpec ls,
            const SymmFactor& sf) {
  return ls.set_symm_factor(sf);
}

template<typename T>
inline std::tuple<LoopSpec, T>
operator * (LoopSpec ls, T rhs) {
  return {ls, rhs};
}

template<typename... Types, typename T>
inline std::tuple<LoopSpec, Types..., T>
operator * (std::tuple<LoopSpec, Types...> lhs, T rhs) {
  return std::tuple_cat(lhs, std::forward_as_tuple(rhs));
}








// template<typename... Types>
// inline std::tuple<InnerLabeledLoop, std::tuple<Types>>
// operator * (InnerLabeledLoop oll, const std::tuple<Tuples...>& rhs) {
//   return {std::forward_as_tuple(oll), rhs};
// }

// template<typename T1>
// inline std::tuple<OuterLabeledLoop, T1>
// operator * (OuterLabeledLoop oll, const T1& rhs) {
//   return {oll, rhs};
// }

// template<typename T1, typename T2, typename T3>
// inline std::tuple<OuterLabeledLoop, T1, T2, T3>
// operator * (OuterLabeledLoop oll,
//             const std::tuple<T1,T2, T3>& rhs) {
//   return {oll, std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs)};
// }

// template<typename T1, typename T2, typename T3, typename T4>
// inline std::tuple<OuterLabeledLoop, T1, T2, T3, T4>
// operator * (OuterLabeledLoop oll,
//             const std::tuple<T1,T2, T3, T4>& rhs) {
//   return {oll,
//         std::get<0>(rhs),
//         std::get<1>(rhs),
//         std::get<2>(rhs),
//         std::get<3>(rhs)};
// }

// template<typename T1, typename T2>
// inline std::tuple<InnerLabeledLoop, T1, T2>
// operator * (InnerLabeledLoop oll, const std::tuple<T1,T2>& rhs) {
//   return {oll, std::get<0>(rhs), std::get<1>(rhs)};
// }

// template<typename T1>
// inline std::tuple<InnerLabeledLoop, T1>
// operator * (InnerLabeledLoop oll, const T1& rhs) {
//   return {oll, rhs};
// }

// template<typename T1, typename T2, typename T3>
// inline std::tuple<InnerLabeledLoop, T1, T2, T3>
// operator * (InnerLabeledLoop oll,
//             const std::tuple<T1,T2, T3>& rhs) {
//   return {oll, std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs)};
// }

// template<typename T1, typename T2, typename T3, typename T4>
// inline std::tuple<InnerLabeledLoop, T1, T2, T3, T4>
// operator * (InnerLabeledLoop oll,
//             const std::tuple<T1,T2, T3, T4>& rhs) {
//   return {oll,
//         std::get<0>(rhs),
//         std::get<1>(rhs),
//         std::get<2>(rhs),
//         std::get<3>(rhs)};
// }

template<typename T1,
         typename T2,
         typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
inline std::tuple<LoopSpec, T1, LabeledTensor<T2>>
operator * (T1 val, const LabeledTensor<T2>& rhs) {
  return {LoopSpec{}, val, rhs};
}

// template<typename T1, typename T2>
// inline std::tuple<T1, LabeledTensor<T2>>
// operator * (T1 val, const LabeledTensor<T2>& rhs) {
//   return {val, rhs};
// }


template<typename T>
inline std::tuple<LoopSpec, LabeledTensor<T>, LabeledTensor<T>>
operator * (const LabeledTensor<T>& rhs1, const LabeledTensor<T>& rhs2) {
  return {LoopSpec{}, rhs1, rhs2};
}

// template<typename T1, typename T2>
// inline std::tuple<T1, LabeledTensor<T2>, LabeledTensor<T2>>
// operator * (std::tuple<T1, LabeledTensor<T2>> rhs1, const LabeledTensor<T2>& rhs2) {
//   return {std::get<0>(rhs1), std::get<1>(rhs1), rhs2};
// }


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

  IndexLabelVec clabel = ltc.labels();
  IndexLabelVec alabel = lta.labels();

  //index range underlying an index label is the same or a subset of the tensor’s index range along that dimension 
  validate_slicing(tc.dim_ranges(), ltc.labels());
  validate_slicing(ta.dim_ranges(), lta.labels());

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

  IndexLabelVec clabel = ltc.labels();
  IndexLabelVec alabel = lta.labels();
  IndexLabelVec blabel = ltb.labels();

  //length of the index label vector matches the rank (number of indices) in the tensor
  EXPECTS(clabel.size() == tc.rank());
  EXPECTS(alabel.size() == ta.rank());
  EXPECTS(blabel.size() == tb.rank());

  //index range underlying an index label is the same or a subset of the tensor’s index range along that dimension 
  validate_slicing(tc.dim_ranges(), ltc.labels());
  validate_slicing(ta.dim_ranges(), lta.labels());
  validate_slicing(tb.dim_ranges(), ltb.labels());

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
