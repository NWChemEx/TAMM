#ifndef TAMM_LABELED_TENSOR_HPP_
#define TAMM_LABELED_TENSOR_HPP_

#include "tamm/loops.hpp"
#include "tamm/ops.hpp"

namespace tamm {

template<typename T>
class Tensor;

template<typename T>
class LabeledTensor {
    public:
    LabeledTensor()                     = default;
    LabeledTensor(const LabeledTensor&) = default;

    LabeledTensor(const Tensor<T>& tensor, const IndexLabelVec& ilv) :
      tensor_{tensor},
      ilv_{ilv} {}

    Tensor<T> tensor() const { return tensor_; }

    IndexLabelVec labels() const { return ilv_; }

    // // @to-do: implement.
    // AddOp<T, LabeledTensor<T>> operator=(
    //   const std::tuple<LabeledTensor<T>, LabeledTensor<T>>& rhs) {
    //   return {};
    // }

    // AddOp<T, LabeledTensor<T>> operator+=(
    //   const std::tuple<LoopSpec, LabeledTensor<T>>& rhs) {
    //     // construct_addop(std::make_tuple(std::get<0>(rhs), 1, std::get<1>(rhs)),
    //     //                 false);
    //   return {};
    // }

    // AddOp<T, LabeledTensor<T>> operator+=(LabeledTensor<T> rhs) {
    //     // return *this += loop_nest() * rhs;
    //   return {};
    // }

    template<typename... RTypes>
//           ,template<typename,typename> typename OpType>
    //OpType<T,LabeledTensor<T>> operator=(RTypes... rhs){
    std::shared_ptr<AddOp<T, LabeledTensor<T>>> operator=(RTypes... rhs) {
        // return *this += loop_nest() * rhs;
      return {};
    }

    template<typename... RTypes>
    std::shared_ptr<AddOp<T, LabeledTensor<T>>> operator+=(RTypes... rhs) {
        // return *this += loop_nest() * rhs;
      return {};
    }


    // SetOp<T, LabeledTensor<T>> operator=(const std::tuple<LoopSpec, T>& rhs) {
    //   //construct_setop(rhs, true);
    //   return {};
    // }

    // AddOp<T, LabeledTensor<T>> operator=(const LabeledTensor<T>& rhs) {
    //   // return *this = loop_nest() * T{1} * rhs;
    //   return {};
    // }

    // template<typename T1,
    //          typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
    // MultOp<T1, LabeledTensor<T>> operator=(
    //   const std::tuple<LoopSpec, T1, LabeledTensor<T>, LabeledTensor<T>>& rhs) {
    //     // return construct_multop(rhs, true);
    //   return {};
    // }


    protected:
    // SetOp<T, LabeledTensor<T>> construct_setop(
    //   const std::tuple<LoopSpec, T>& rhs, bool is_assign) {
    //     const auto& loop_spec = std::get<0>(rhs);
    //     if(loop_spec.has_oll()) {
    //         return {*this, std::get<1>(rhs), loop_spec.oll(), is_assign};
    //     } else {
    //         return {*this, std::get<1>(rhs), loop_nest(), is_assign};
    //     }
    // }

    // template<typename T1,
    //          typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
    // AddOp<T1, LabeledTensor<T>> construct_addop(
    //   const std::tuple<LoopSpec, T1, LabeledTensor<T>>& rhs, bool is_assign) {
    //     addop_validate(*this,
    //                    std::make_tuple(std::get<1>(rhs), std::get<2>(rhs)));
    //     const auto& loop_spec = std::get<0>(rhs);
    //     T1 alpha              = std::get<1>(rhs);
    //     auto& rhs_tensor      = std::get<2>(rhs);
    //     if(loop_spec.has_oll()) {
    //         return {*this, alpha, rhs_tensor, loop_spec.oll(), is_assign};
    //     } else {
    //         return {*this, alpha, rhs_tensor, loop_nest(), is_assign};
    //     }
    // }

    // template<typename T1,
    //          typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
    // MultOp<T1, LabeledTensor<T>> construct_multop(
    //   const std::tuple<LoopSpec, T1, LabeledTensor<T>, LabeledTensor<T>>& rhs,
    //   bool is_assign) {
    //     multop_validate(*this,
    //                     std::make_tuple(std::get<1>(rhs), std::get<2>(rhs),
    //                                     std::get<3>(rhs)));

    //     const auto& loop_spec = std::get<0>(rhs);
    //     OuterLabeledLoop oll;
    //     InnerLabeledLoop ill;
    //     SymmFactor sf;
    //     if(loop_spec.has_oll()) {
    //         oll = loop_spec.oll();
    //     } else {
    //         oll = loop_nest();
    //     }
    //     if(loop_spec.has_ill()) {
    //         ill = loop_spec.ill();
    //     } else {
    //         ill = inner_loop_nest(std::get<2>(rhs), std::get<3>(rhs));
    //     }
    //     if(loop_spec.has_symm_factor()) {
    //         sf = loop_spec.symm_factor();
    //     } else {
    //         sf = SymmFactor{};
    //     }

    //     return {
    //       *this, std::get<1>(rhs), std::get<2>(rhs), std::get<3>(rhs), oll, ill,
    //       sf,    is_assign};
    // }

    Tensor<T> tensor_;
    IndexLabelVec ilv_;

    // OuterLabeledLoop loop_nest() const {
    //     // return {labels(), tensor().perm_group().unique_loop_nest(labels())};
    //   return {};
    // }

    // template<typename T1>
    // static InnerLabeledLoop inner_loop_nest(const LabeledTensor<T1>& ltensor1,
    //                                         const LabeledTensor<T1>& ltensor2) {
    //     using Itr = IndexSpace::Iterator;
    //     IndexLabelVec labels1{ltensor1.labels()};
    //     IndexLabelVec labels2{ltensor2.labels()};

    //     std::sort(labels1.begin(), labels1.end());
    //     std::sort(labels2.begin(), labels2.end());

    //     IndexLabelVec inner_labels;
    //     std::set_intersection(labels1.begin(), labels1.end(), labels2.begin(),
    //                           labels2.end(), std::back_inserter(inner_labels));
    //     std::vector<Itr> begins, ends;
    //     // for(const auto& il : inner_labels) {
    //     //     begins.push_back(il.ir().begin());
    //     //     ends.push_back(il.ir().end());
    //     // }
    //     return InnerLabeledLoop{inner_labels, begins, ends, {}};
    // }
};


template<typename T1, typename T2>
inline std::tuple<T1,T2> operator*(T1 op1, T2 op2) {
    // return *this += loop_nest() * rhs;
  return {};
}

template<typename... RHSTypes,
        typename T1,
        typename = std::enable_if_t<std::is_arithmetic<T1>::value>
        >
inline std::tuple<T1, RHSTypes...> operator*(T1 alpha, RHSTypes... rhs) {
    // return *this += loop_nest() * rhs;
  return {};
}

// template<typename... RArgs, typename TR>
// inline std::tuple<TR, RArgs...> operator*(TR r1, RArgs... rhs) {
//    return {};
// }

// template<typename... Types, typename T>
// inline std::tuple<LoopSpec, Types..., T> operator*(
//   std::tuple<LoopSpec, Types...> lhs, T rhs) {
//     return std::tuple_cat(lhs, std::forward_as_tuple(rhs));
// }


// template<typename T>
// inline std::tuple<LoopSpec, LabeledTensor<T>, LabeledTensor<T>> operator*(
//   const LabeledTensor<T>& rhs1, const LabeledTensor<T>& rhs2) {
//     return {LoopSpec{}, rhs1, rhs2};
// }

// inline void validate_slicing(const TensorVec<IndexRange>& index_ranges,
//                              const IndexLabelVec& label) {
//     for(size_t i = 0; i < index_ranges.size(); i++) {
//         EXPECTS(index_ranges[i].is_superset_of(label[i].ir()));
//     }
// }

template<typename LabeledTensorType, typename T>
inline void addop_validate(const LabeledTensorType& ltc,
                           const std::tuple<T, LabeledTensorType>& rhs) {
#if 0
    auto lta = std::get<1>(rhs);
    // EXPECTS(ltc.tensor() != nullptr);
    // EXPECTS(lta.tensor() != nullptr);
    const auto& tc = ltc.tensor();
    const auto& ta = lta.tensor();

    // tensors should have same rank
    EXPECTS(tc.rank() == ta.rank());

    IndexLabelVec clabel = ltc.labels();
    IndexLabelVec alabel = lta.labels();

    // index range underlying an index label is the same or a subset of the
    // tensor’s index range along that dimension
    validate_slicing(tc.dim_ranges(), ltc.labels());
    validate_slicing(ta.dim_ranges(), lta.labels());

    // length of the index label vector matches the rank (number of indices) in
    // the tensor
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

    // all labels are unique
    EXPECTS(std::adjacent_find(alabel.begin(), alabel.end()) == alabel.end());
    EXPECTS(std::adjacent_find(clabel.begin(), clabel.end()) == clabel.end());

    // all labels in ta are in tb
    for(auto& al : alabel) {
        EXPECTS(std::find(clabel.begin(), clabel.end(), al) != clabel.end());
    }
#endif
}

template<typename LabeledTensorType, typename T>
inline void multop_validate(
  const LabeledTensorType& ltc,
  const std::tuple<T, LabeledTensorType, LabeledTensorType>& rhs) {
#if 0
    auto& lta = std::get<1>(rhs);
    auto& ltb = std::get<2>(rhs);
    // EXPECTS(ltc.tensor_ != nullptr);
    // EXPECTS(lta.tensor_ != nullptr);
    // EXPECTS(ltb.tensor_ != nullptr);
    const auto& tc = ltc.tensor();
    const auto& ta = lta.tensor();
    const auto& tb = ltb.tensor();

    IndexLabelVec clabel = ltc.labels();
    IndexLabelVec alabel = lta.labels();
    IndexLabelVec blabel = ltb.labels();

    // length of the index label vector matches the rank (number of indices) in
    // the tensor
    EXPECTS(clabel.size() == tc.rank());
    EXPECTS(alabel.size() == ta.rank());
    EXPECTS(blabel.size() == tb.rank());

    // index range underlying an index label is the same or a subset of the
    // tensor’s index range along that dimension
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

    // all labels are unique
    EXPECTS(std::adjacent_find(alabel.begin(), alabel.end()) == alabel.end());
    EXPECTS(std::adjacent_find(blabel.begin(), blabel.end()) == blabel.end());
    EXPECTS(std::adjacent_find(clabel.begin(), clabel.end()) == clabel.end());

    IndexLabelVec rhs_labels;
    std::set_union(alabel.begin(), alabel.end(), blabel.begin(), blabel.end(),
                   std::back_inserter(rhs_labels));

    IndexLabelVec inner_labels;
    std::set_difference(rhs_labels.begin(), rhs_labels.end(), clabel.begin(),
                        clabel.end(), std::back_inserter(inner_labels));

    IndexLabelVec slabel;
    std::set_intersection(alabel.begin(), alabel.end(), blabel.begin(),
                          blabel.end(), std::back_inserter(slabel));

    // Every outer index label (clabel) appears in exactly one RHS tensor
    for(auto& ol : clabel) {
        EXPECTS(std::find(slabel.begin(), slabel.end(), ol) == slabel.end() &&
                std::find(rhs_labels.begin(), rhs_labels.end(), ol) !=
                  rhs_labels.end());
    }

    // Every inner index label appears exactly once in both RHS tensors
    for(auto& il : inner_labels) {
        EXPECTS(std::find(slabel.begin(), slabel.end(), il) != slabel.end());
    }

    // //summation index is not in the output
    // for(auto &sl: slabel) {
    //   EXPECTS(std::find(clabel.begin(), clabel.end(), sl) == clabel.end());
    // }
    // //every label in A/B is either in slabel or clabel
    // for(auto &al : alabel) {
    //   EXPECTS(std::find(slabel.begin(), slabel.end(), al) != slabel.end()
    //           || std::find(clabel.begin(), clabel.end(), al) !=
    //           clabel.end());
    // }
    // for(auto &bl : blabel) {
    //   EXPECTS(std::find(slabel.begin(), slabel.end(), bl) != slabel.end()
    //           || std::find(clabel.begin(), clabel.end(), bl) !=
    //           clabel.end());
    // }

    EXPECTS(clabel.size() == alabel.size() + blabel.size() - 2 * slabel.size());
#endif
}
 
} // tamm
#endif // LABELED_TENSOR_HPP_
