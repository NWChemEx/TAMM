#ifndef TAMM_LABELED_TENSOR_HPP_
#define TAMM_LABELED_TENSOR_HPP_

#include "tamm/loops.hpp"
#include "tamm/ops.hpp"
#include <type_traits>

namespace tamm {

namespace internal {
  template <typename> struct is_tuple: std::false_type {};
  template <typename ...T> struct is_tuple<std::tuple<T...>>: std::true_type {};
  template <typename T> inline constexpr bool is_tuple_v = is_tuple<T>::value;
}

template<typename T>
class Tensor;

template<typename T1, typename T2>
auto operator*(T1&& left, T2&& right){
    
  using internal::is_tuple_v;
  if constexpr(is_tuple_v<T1>)
    return std::tuple_cat(left, std::forward_as_tuple(right));
  else 
    return std::tuple_cat(std::forward_as_tuple(left), std::forward_as_tuple(right));

}

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

    using LTT = LabeledTensor<T>;
    template<typename T1> 
    auto operator=(T1&& rhs){
      using std::get;
      using internal::is_tuple_v;
      using std::tuple_size_v;
      using std::remove_reference;
      using std::is_same_v;
      using std::is_convertible_v;

      //LT = alpha
      if constexpr (is_convertible_v<T1, T>)
        return SetOp<T,LTT>{*this,T(rhs),true};
      
      // LT = LT
      else if constexpr (is_same_v<T1, LTT>)
        return AddOp<T,T1>{*this,T{1.0},rhs,true};
      
      else if constexpr (is_tuple_v<T1>){
        static_assert(!(tuple_size_v<T1> > 3) && !(tuple_size_v<T1> < 2), 
        "Operation can only be of the form c [+]= [alpha] * a [* b]");
        using rhs0_t = typename remove_reference<decltype(get<0>(rhs))>::type;
        using rhs1_t = typename remove_reference<decltype(get<1>(rhs))>::type;

        if constexpr(tuple_size_v<T1> == 2){
          // LT = alpha * LT
          if constexpr(is_convertible_v<rhs0_t, T>
              && is_same_v<rhs1_t, LTT>)
              return AddOp<T,LTT>{*this,get<0>(rhs),get<1>(rhs),true};
            //  LT = LT * LT
          else if constexpr(is_same_v<rhs0_t, LTT>
              && is_same_v<rhs1_t, LTT>)
              return MultOp<T,LTT>{*this,T{1.0},get<0>(rhs),get<1>(rhs),true};
        }
        
         // LT = alpha * LT * LT
        else if constexpr(tuple_size_v<T1> == 3){
          using rhs2_t = typename remove_reference<decltype(get<2>(rhs))>::type;
          static_assert(is_convertible_v<rhs0_t, T>
           && is_same_v<rhs1_t, LTT>
           && is_same_v<rhs2_t, LTT>
           ,"Operation can only be of the form c = alpha * a * b");
          return MultOp<T,LTT>{*this,get<0>(rhs),get<1>(rhs),get<2>(rhs),true};
        }
      } 
    } //operator =

    template<typename T1>
    auto operator+=(T1&& rhs){
      using internal::is_tuple_v;
      using std::get;
      using std::tuple_size_v;
      using std::remove_reference;
      using std::is_same_v;      
      using std::is_convertible_v;
      
      //LT = alpha
      if constexpr (is_convertible_v<T1, T>)
        return SetOp<T,LTT>{*this,rhs,false};
      
      // LT = LT
      else if constexpr (is_same_v<T1, LTT>)
        return AddOp<T,T1>{*this,T{1.0},rhs,false};
      
      else if constexpr (is_tuple_v<T1>){
        static_assert(!(tuple_size_v<T1> > 3) && !(tuple_size_v<T1> < 2), 
        "Operation can only be of the form c [+]= [alpha] * a [* b]");
        using rhs0_t = typename remove_reference<decltype(get<0>(rhs))>::type;
        using rhs1_t = typename remove_reference<decltype(get<1>(rhs))>::type;

        if constexpr(tuple_size_v<T1> == 2){
          // LT = alpha * LT
          if constexpr(is_convertible_v<rhs0_t, T>
              && is_same_v<rhs1_t, LTT>)
              return AddOp<T,LTT>{*this,get<0>(rhs),get<1>(rhs),false};
          //  LT = LT * LT
          else if constexpr(is_same_v<rhs0_t, LTT>
              && is_same_v<rhs1_t, LTT>)
              return MultOp<T,LTT>{*this,T{1.0},get<0>(rhs),get<1>(rhs),false};
        }
        
         // alpha * LT * LT
        else if constexpr(tuple_size_v<T1> == 3){
          using rhs2_t = typename remove_reference<decltype(get<2>(rhs))>::type;
          static_assert(is_convertible_v<rhs0_t, T>
           && is_same_v<rhs1_t, LTT>
           && is_same_v<rhs2_t, LTT>
           , "Operation can only be of the form c += alpha * a * b");
          return MultOp<T,LTT>{*this,get<0>(rhs),get<1>(rhs),get<2>(rhs),false};
        }
      }  
    } //operator +=

    protected:
      Tensor<T> tensor_;
      IndexLabelVec ilv_;

};

//     // @to-do: implement.
//     template<typename T1,
//              typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
//     MultOp<T1, LabeledTensor<T>> operator-=(
//       const std::tuple<LoopSpec, T1, LabeledTensor<T>, LabeledTensor<T>>& rhs);

//     // @to-do: implement.
//     AddOp<T, LabeledTensor<T>> operator=(
//       const std::tuple<LabeledTensor<T>, LabeledTensor<T>>& rhs) {
//       return {};
//     }

//     AddOp<T, LabeledTensor<T>> operator+=(
//       const std::tuple<LoopSpec, LabeledTensor<T>>& rhs) {
//         // construct_addop(std::make_tuple(rhs0_t, 1, rhs1_t),
//         //                 false);
//       return {};
//     }

//     AddOp<T, LabeledTensor<T>> operator+=(LabeledTensor<T> rhs) {
//         // return *this += loop_nest() * rhs;
//       return {};
//     }

//     SetOp<T, LabeledTensor<T>> operator+=(const T& rhs) {
//       //return *this += loop_nest() * rhs;
//       return {};
//     }

//     SetOp<T, LabeledTensor<T>> operator+=(const std::tuple<LoopSpec, T>& rhs) {
//         // construct_setop(rhs, false);
//       return {};
//     }

//     SetOp<T, LabeledTensor<T>> operator=(T rhs) {
//         // return *this = loop_nest() * rhs;
//       return {};
//     }

//     SetOp<T, LabeledTensor<T>> operator=(const std::tuple<LoopSpec, T>& rhs) {
//       //construct_setop(rhs, true);
//       return {};
//     }

//     template<typename T1,
//              typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
//     AddOp<T1, LabeledTensor<T>> operator+=(
//       const std::tuple<LoopSpec, T1, LabeledTensor<T>>& rhs) {
//         // construct_addop(rhs, false);
//       return {};
//     }

//     template<typename T1,
//              typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
//     AddOp<T1, LabeledTensor<T>> operator=(
//       const std::tuple<LoopSpec, T1, LabeledTensor<T>>& rhs) {
//         // construct_addop(rhs, true);
//       return {};
//     }

//     AddOp<T, LabeledTensor<T>> operator=(
//       const std::tuple<LoopSpec, LabeledTensor<T>> rhs) {
//         // return *this = rhs0_t * T{1} * rhs1_t;
//       return {};
//     }

//     AddOp<T, LabeledTensor<T>> operator=(const LabeledTensor<T>& rhs) {
//       // return *this = loop_nest() * T{1} * rhs;
//       return {};
//     }

//     template<typename T1,
//              typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
//     MultOp<T1, LabeledTensor<T>> operator+=(
//       const std::tuple<LoopSpec, T1, LabeledTensor<T>, LabeledTensor<T>>& rhs) {
//         // return construct_multop(rhs, false);
//       return {};
//     }

//     template<typename T1,
//              typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
//     MultOp<T1, LabeledTensor<T>> operator=(
//       const std::tuple<LoopSpec, T1, LabeledTensor<T>, LabeledTensor<T>>& rhs) {
//         // return construct_multop(rhs, true);
//       return {};
//     }

//     MultOp<T, LabeledTensor<T>> operator+=(
//       const std::tuple<LoopSpec, LabeledTensor<T>, LabeledTensor<T>>& rhs) {
//         // return *this +=
//         //        rhs0_t * T{1} * rhs1_t * get<2>(rhs);
//       return {};
//     }

//     MultOp<T, LabeledTensor<T>> operator=(
//       const std::tuple<LoopSpec, LabeledTensor<T>, LabeledTensor<T>>& rhs) {
//         // return *this =
//         //          rhs0_t * T{1} * rhs1_t * get<2>(rhs);
//       return {};
//     }

//     protected:
//     SetOp<T, LabeledTensor<T>> construct_setop(
//       const std::tuple<LoopSpec, T>& rhs, bool is_assign) {
//         const auto& loop_spec = rhs0_t;
//         if(loop_spec.has_oll()) {
//             return {*this, rhs1_t, loop_spec.oll(), is_assign};
//         } else {
//             return {*this, rhs1_t, loop_nest(), is_assign};
//         }
//     }

//     template<typename T1,
//              typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
//     AddOp<T1, LabeledTensor<T>> construct_addop(
//       const std::tuple<LoopSpec, T1, LabeledTensor<T>>& rhs, bool is_assign) {
//         addop_validate(*this,
//                        std::make_tuple(rhs1_t, get<2>(rhs)));
//         const auto& loop_spec = rhs0_t;
//         T1 alpha              = rhs1_t;
//         auto& rhs_tensor      = get<2>(rhs);
//         if(loop_spec.has_oll()) {
//             return {*this, alpha, rhs_tensor, loop_spec.oll(), is_assign};
//         } else {
//             return {*this, alpha, rhs_tensor, loop_nest(), is_assign};
//         }
//     }

//     template<typename T1,
//              typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
//     MultOp<T1, LabeledTensor<T>> construct_multop(
//       const std::tuple<LoopSpec, T1, LabeledTensor<T>, LabeledTensor<T>>& rhs,
//       bool is_assign) {
//         multop_validate(*this,
//                         std::make_tuple(rhs1_t, get<2>(rhs),
//                                         get<3>(rhs)));

//         const auto& loop_spec = rhs0_t;
//         OuterLabeledLoop oll;
//         InnerLabeledLoop ill;
//         SymmFactor sf;
//         if(loop_spec.has_oll()) {
//             oll = loop_spec.oll();
//         } else {
//             oll = loop_nest();
//         }
//         if(loop_spec.has_ill()) {
//             ill = loop_spec.ill();
//         } else {
//             ill = inner_loop_nest(get<2>(rhs), get<3>(rhs));
//         }
//         if(loop_spec.has_symm_factor()) {
//             sf = loop_spec.symm_factor();
//         } else {
//             sf = SymmFactor{};
//         }

//         return {
//           *this, rhs1_t, get<2>(rhs), get<3>(rhs), oll, ill,
//           sf,    is_assign};
//     }

//     Tensor<T> tensor_;
//     IndexLabelVec ilv_;

//     OuterLabeledLoop loop_nest() const {
//         // return {labels(), tensor().perm_group().unique_loop_nest(labels())};
//       return {};
//     }

//     template<typename T1>
//     static InnerLabeledLoop inner_loop_nest(const LabeledTensor<T1>& ltensor1,
//                                             const LabeledTensor<T1>& ltensor2) {
//         using Itr = IndexSpace::Iterator;
//         IndexLabelVec labels1{ltensor1.labels()};
//         IndexLabelVec labels2{ltensor2.labels()};

//         std::sort(labels1.begin(), labels1.end());
//         std::sort(labels2.begin(), labels2.end());

//         IndexLabelVec inner_labels;
//         std::set_intersection(labels1.begin(), labels1.end(), labels2.begin(),
//                               labels2.end(), std::back_inserter(inner_labels));
//         std::vector<Itr> begins, ends;
//         // for(const auto& il : inner_labels) {
//         //     begins.push_back(il.ir().begin());
//         //     ends.push_back(il.ir().end());
//         // }
//         return InnerLabeledLoop{inner_labels, begins, ends, {}};
//     }
// };

// inline LoopSpec operator*(LoopSpec ls, const InnerLabeledLoop& ill) {
//     return ls.set_ill(ill);
// }

// inline LoopSpec operator*(LoopSpec ls, const SymmFactor& sf) {
//     return ls.set_symm_factor(sf);
// }

// template<typename T>
// inline std::tuple<LoopSpec, T> operator*(LoopSpec ls, T rhs) {
//     return {ls, rhs};
// }

// template<typename... Types, typename T>
// inline std::tuple<LoopSpec, Types..., T> operator*(
//   std::tuple<LoopSpec, Types...> lhs, T rhs) {
//     return std::tuple_cat(lhs, std::forward_as_tuple(rhs));
// }

// // @to-do: implement properly
// template<typename T>
// inline std::tuple<LabeledTensor<T>, LabeledTensor<T>> operator-(
//   LabeledTensor<T> lhs, LabeledTensor<T> rhs) {
//     return {lhs, rhs};
// }

// template<typename T1, typename T2,
//          typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
// inline std::tuple<LoopSpec, T1, LabeledTensor<T2>> operator*(
//   T1 val, const LabeledTensor<T2>& rhs) {
//     return {LoopSpec{}, val, rhs};
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
    auto lta = rhs1_t;
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
    auto& lta = rhs1_t;
    auto& ltb = get<2>(rhs);
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
