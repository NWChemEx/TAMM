#pragma once

#include "tamm/op_base.hpp"
#include "tamm/setop.hpp"
#include "tamm/scanop.hpp"
#include "tamm/mapop.hpp"
#include "tamm/addop.hpp"
#include "tamm/multop.hpp"
#include "tamm/allocop.hpp"
#include "tamm/deallocop.hpp"

//#define DO_NB
//#define DO_NB_GET

namespace tamm::internal {
template <typename T>
class LabelMap {
 public:
  LabelMap() = default;
  LabelMap(const LabelMap&) = default;
  LabelMap(LabelMap&&) = default;
  LabelMap& operator=(const LabelMap&) = default;
  LabelMap& operator=(LabelMap&&) = default;
  ~LabelMap() = default;

  LabelMap& update(const IndexLabelVec& labels, const std::vector<T>& vals) {
    EXPECTS(labels.size() == vals.size());
    for (size_t i = 0; i < vals.size(); i++) {
      map_[labels[i]] = vals[i];
    }
    return *this;
  }

  std::vector<T> get(const IndexLabelVec& labels) {
    std::vector<T> ret;
    for (const auto& lbl : labels) {
      auto itr = map_.find(lbl);
      EXPECTS(itr != map_.end());
      ret.push_back(itr->second);
    }
    return ret;
  }

 private:
  std::map<TileLabelElement, T> map_;
};
}


namespace tamm {
template<typename T>

template<typename T1>
constexpr auto LabeledTensor<T>::make_op(T1&& rhs, const bool is_assign,
                                         const int sub_v) {
    using internal::is_complex_v;
    using internal::is_tuple_v;
    using std::get;
    using std::is_convertible_v;
    using std::is_same_v;
    using std::remove_reference;
    using std::tuple_size_v;

    // LT = alpha
    if constexpr(is_convertible_v<T1, T>)
        return SetOp<T, LTT>{*this, static_cast<T>(sub_v) * static_cast<T>(rhs),
                             is_assign};

    // LT = LT
    else if constexpr(is_same_v<T1, LTT>)
        return AddOp<T, LTT, T1>{*this, static_cast<T>(sub_v), rhs, is_assign};
    else if constexpr(is_complex_v<T> &&
                      (is_same_v<T1, LTT_int> || is_same_v<T1, LTT_float> ||
                       is_same_v<T1, LTT_double>))
        return AddOp<int, LTT, T1>{*this, sub_v, rhs, is_assign};
    // real=complex
    else if constexpr(std::is_floating_point<T>::value &&
                      (is_same_v<T1, LTT_cfloat> || is_same_v<T1, LTT_cdouble>))
        return AddOp<int, LTT, T1>{*this, sub_v, rhs, is_assign};

    else if constexpr(is_tuple_v<T1>) {
        static_assert(
          !(tuple_size_v<T1>> 3) && !(tuple_size_v<T1> < 2),
          "Operation can only be of the form c [+-]= [alpha *] a [* b]");
        using rhs0_t = typename remove_reference<decltype(get<0>(rhs))>::type;
        using rhs1_t = typename remove_reference<decltype(get<1>(rhs))>::type;

        if constexpr(tuple_size_v<T1> == 2) {
            // LT = alpha * LT
            if constexpr((is_convertible_v<rhs0_t, T>)&&is_same_v<rhs1_t, LTT>)
                return AddOp<T, LTT, rhs1_t>{
                  *this, static_cast<T>(sub_v) * static_cast<T>(get<0>(rhs)),
                  get<1>(rhs), is_assign};
            else if constexpr(is_convertible_v<rhs0_t, T> && is_complex_v<T> &&
                              (is_same_v<rhs1_t, LTT_int> ||
                               is_same_v<rhs1_t, LTT_float> ||
                               is_same_v<rhs1_t, LTT_double>))
                return AddOp<T, LTT, rhs1_t>{
                  *this, static_cast<T>(sub_v) * static_cast<T>(get<0>(rhs)),
                  get<1>(rhs), is_assign};

            //  LT = LT * LT
            else if constexpr(is_same_v<rhs0_t, LTT> && is_same_v<rhs1_t, LTT>)
                return MultOp<T, LTT, LTT, LTT>{*this, static_cast<T>(sub_v),
                                                get<0>(rhs), get<1>(rhs),
                                                is_assign};
            // LHS is complex, rhs1,rhs2 are either complex/real
            else if constexpr(is_complex_v<T>
                              // &&
                              // (
                              //   (is_same_v<rhs0_t, LTT> &&
                              //     (is_same_v<rhs1_t,LTT_int>
                              //     ||is_same_v<rhs1_t,LTT_float>
                              //     ||is_same_v<rhs1_t,LTT_double>)
                              //   )
                              //   ||
                              //   (is_same_v<rhs1_t, LTT> &&
                              //     (is_same_v<rhs0_t,LTT_int>
                              //     ||is_same_v<rhs0_t,LTT_float>
                              //     ||is_same_v<rhs0_t,LTT_double>)
                              //   )
                              // )
            )
                return MultOp<T, LTT, rhs0_t, rhs1_t>{*this, sub_v, get<0>(rhs),
                                                      get<1>(rhs), is_assign};
            // LHS is real, rhs1,rhs2 are either complex/real
            else if constexpr(!is_complex_v<T> &&
                              ((is_same_v<rhs0_t, LTT> &&
                                (is_same_v<rhs1_t, LTT_cfloat> ||
                                 is_same_v<rhs1_t, LTT_cdouble>)) ||
                               (is_same_v<rhs1_t, LTT> &&
                                (is_same_v<rhs0_t, LTT_cfloat> ||
                                 is_same_v<rhs0_t, LTT_cdouble>))))
                return MultOp<T, LTT, rhs0_t, rhs1_t>{
                  *this, sub_v, get<0>(rhs), get<1>(rhs), is_assign};
        }

        // LT = alpha * LT * LT
        else if constexpr(tuple_size_v<T1> == 3) {
            using rhs2_t =
              typename remove_reference<decltype(get<2>(rhs))>::type;

            if constexpr(is_same_v<rhs1_t, LTT> && is_same_v<rhs2_t, LTT>)
                return MultOp<T, LTT, LTT, LTT>{
                  *this, static_cast<T>(sub_v * get<0>(rhs)), get<1>(rhs),
                  get<2>(rhs), is_assign};
            // LHS is complex, rhs1,rhs2 are either complex/real
            else if constexpr(is_complex_v<T>
                              // &&
                              // (
                              //   (is_same_v<rhs1_t, LTT> &&
                              //     (is_same_v<rhs2_t,LTT_int>
                              //     ||is_same_v<rhs2_t,LTT_float>
                              //     ||is_same_v<rhs2_t,LTT_double>)
                              //   )
                              //   ||
                              //   (is_same_v<rhs2_t, LTT> &&
                              //     (is_same_v<rhs1_t,LTT_int>
                              //     ||is_same_v<rhs1_t,LTT_float>
                              //     ||is_same_v<rhs1_t,LTT_double>)
                              //   )
                              // )
            )
                return MultOp<T, LTT, rhs1_t, rhs2_t>{
                  *this, static_cast<rhs0_t>(sub_v * get<0>(rhs)), get<1>(rhs),
                  get<2>(rhs), is_assign};
            // LHS is real, rhs1,rhs2 are either complex/real
            // alpha has to be real in this case
            else if constexpr(!is_complex_v<T> &&
                              ((is_same_v<rhs1_t, LTT> &&
                                (is_same_v<rhs2_t, LTT_cfloat> ||
                                 is_same_v<rhs2_t, LTT_cdouble>)) ||
                               (is_same_v<rhs2_t, LTT> &&
                                (is_same_v<rhs1_t, LTT_cfloat> ||
                                 is_same_v<rhs1_t, LTT_cdouble>))))
                return MultOp<T, LTT, rhs1_t, rhs2_t>{
                  *this, static_cast<rhs0_t>(sub_v * get<0>(rhs)), get<1>(rhs),
                  get<2>(rhs), is_assign};
            // static_assert(
            //   (is_convertible_v<rhs0_t, T>)&&is_same_v<rhs1_t, LTT> &&
            //     is_same_v<rhs2_t, LTT>,
            //   "Operation can only be of the form c [+-] = [alpha *] a * b");
        }
    }
} // end make_op

}  // namespace tamm

