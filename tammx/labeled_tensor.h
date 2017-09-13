#ifndef TAMMX_LABELEDTENSOR_H_
#define TAMMX_LABELEDTENSOR_H_

#include <type_traits>

#include "tammx/types.h"
#include "tammx/tensor.h"
#include "tammx/fortran.h"

namespace tammx {

template<typename T>
class Tensor;

enum class ResultMode { update, set };

template<typename LabeledTensorType, typename T>
struct SetOpEntry {
  LabeledTensorType lhs;
  T value;
  ResultMode mode;
};

enum class ExecutionMode {
  sch,
  fortran
};

template<typename LabeledTensorType, typename T>
struct AddOpEntry {
  LabeledTensorType lhs;
  T alpha;
  LabeledTensorType rhs;
  ResultMode mode;
  add_fn* fn{nullptr};
  ExecutionMode exec_mode{ExecutionMode::sch};


 /* AddOpEntry()
      : fn{nullptr},
        exec_mode{ExecutionMode::sch} {}

  AddOpEntry(LabeledTensorType tlhs,
             T talpha,
             LabeledTensorType trhs,
             ResultMode tmode,
             add_fn *tfn = nullptr,
             ExecutionMode temode = ExecutionMode::sch)
      : lhs{tlhs},
        alpha{talpha},
        rhs{trhs},
        mode{tmode},
        fn{tfn},
        exec_mode{temode} {}
*/
};

template<typename LabeledTensorType, typename T>
struct MultOpEntry {
  LabeledTensorType lhs;
  T alpha;
  LabeledTensorType rhs1, rhs2;
  ResultMode mode;
  mult_fn* fn{nullptr};
  ExecutionMode exec_mode{ExecutionMode::sch};

 /* MultOpEntry()
      : fn{nullptr},
        exec_mode{ExecutionMode::sch} {}

  MultOpEntry(LabeledTensorType tlhs,
              T talpha,
              LabeledTensorType trhs1,
              LabeledTensorType trhs2,
              ResultMode tmode,
              mult_fn *tfn = nullptr,
              ExecutionMode temode = ExecutionMode::sch)
      : lhs{tlhs},
        alpha{talpha},
        rhs1{trhs1},
        rhs2{trhs2},
        mode{tmode},
        fn{tfn},
        exec_mode{temode} {}

*/
};

template<typename T1,
         typename T2,
         typename = std::enable_if_t<std::is_arithmetic<T1>::value>,
         typename = std::enable_if_t<std::is_arithmetic<T2>::value>>
struct InnerProductOpEntry {
  T1& lhs;
  T2 alpha;
  LabeledTensor<T1> rhs1, rhs2; 
};

template<typename Func, typename LabeledTensorType, unsigned ndim, unsigned nrhs>
struct MapOpEntry {
  LabeledTensorType lhs;
  std::vector<LabeledTensorType> rhss;
  Func func;
};

template<typename TensorType>
struct AllocOpEntry {
  TensorType *tensor;
};

template<typename TensorType>
struct DeallocOpEntry {
  TensorType *tensor;
};

template<typename T>
struct LabeledTensor {
  using element_type = T;
  Tensor<T>* tensor_;
  IndexLabelVec label_;

  Tensor<T>& tensor() {
    return *tensor_;
  }

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  SetOpEntry<LabeledTensor<T>, T1> operator = (T1 value) {
    return {*this, value, ResultMode::set};
  }

  AddOpEntry<LabeledTensor<T>, int> operator = (LabeledTensor<T> rhs) {
    addop_validate(*this, std::make_tuple(1, rhs));
    return {*this, 1, rhs, ResultMode::set};
  }

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  AddOpEntry<LabeledTensor<T>, T1> operator = (std::tuple<T1, LabeledTensor<T>> rhs) {
    addop_validate(*this, rhs);
    return {*this, std::get<0>(rhs), std::get<1>(rhs), ResultMode::set};
  }

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  MultOpEntry<LabeledTensor<T>, T1> operator = (std::tuple<T1, LabeledTensor<T>, LabeledTensor<T>> rhs) {
    multop_validate(*this, rhs);
    return {*this, std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs), ResultMode::set};
  }

  MultOpEntry<LabeledTensor<T>, int> operator = (std::tuple<LabeledTensor<T>, LabeledTensor<T>> rhs) {
    multop_validate(*this, std::make_tuple(1, std::get<0>(rhs), std::get<1>(rhs)));
    return {*this, 1, std::get<0>(rhs), std::get<1>(rhs), ResultMode::set};
  }

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  SetOpEntry<LabeledTensor<T>, T1> operator += (T1 value) {
    //std::cerr<<"Constructing setop. value="<<value<<std::endl;
    return {*this, value, ResultMode::update};
  }

  AddOpEntry<LabeledTensor<T>, int> operator += (LabeledTensor<T> rhs) {
    addop_validate(*this, std::make_tuple(1, rhs));
    return {*this, 1, rhs, ResultMode::update};
  }

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  AddOpEntry<LabeledTensor<T>, T1> operator += (std::tuple<T1, LabeledTensor<T>> rhs) {
    addop_validate(*this, std::make_tuple(std::get<0>(rhs), std::get<1>(rhs)));
    return {*this, std::get<0>(rhs), std::get<1>(rhs), ResultMode::update};
  }

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  MultOpEntry<LabeledTensor<T>, T1> operator += (std::tuple<T1, LabeledTensor<T>, LabeledTensor<T>> rhs) {
    multop_validate(*this, rhs);
    return {*this, std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs), ResultMode::update};
  }


  MultOpEntry<LabeledTensor<T>, int> operator += (std::tuple<LabeledTensor<T>, LabeledTensor<T>> rhs) {
    multop_validate(*this, std::make_tuple(1, std::get<0>(rhs), std::get<1>(rhs)));
    return {*this, 1, std::get<0>(rhs), std::get<1>(rhs), ResultMode::update};
  }
};  // LabeledTensor

template<typename T1,
         typename T2,
         typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
inline std::tuple<T1, LabeledTensor<T2>>
operator * (T1 alpha, LabeledTensor<T2> tensor) {
  // static_assert(std::is_arithmetic<T1>::value,
  //               "Multiplying tensor with a non-arithmetic scalar is invalid.");
  return {alpha, tensor};
}

template<typename T1,
         typename T2,
         typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
inline std::tuple<T1, LabeledTensor<T2>>
operator * (LabeledTensor<T2> tensor, T1 alpha) {
  // static_assert(std::is_arithmetic<T1>::value,
  //               "Multiplying tensor with a non-arithmetic scalar is invalid.");
  return {alpha, tensor};
}

template<typename T1, typename T2>
inline std::tuple<T1, LabeledTensor<T2>, LabeledTensor<T2>>
operator * (const std::tuple<T1, LabeledTensor<T2>>& rhs1, LabeledTensor<T2> rhs2)  {
  return std::tuple_cat(rhs1, std::make_tuple(rhs2));
}

template<typename LabeledTensorType>
inline std::tuple<LabeledTensorType, LabeledTensorType>
operator * (LabeledTensorType rhs1, LabeledTensorType rhs2)  {
  return std::make_tuple(rhs1, rhs2);
}

template<typename T1,
         typename T2>
inline std::tuple<T1, LabeledTensor<T2>, LabeledTensor<T2>>
operator * (T1 alpha, std::tuple<LabeledTensor<T2>, LabeledTensor<T2>> rhs) {
  static_assert(std::is_arithmetic<T1>::value,
                "Multiplying tensor with a non-arithmetic scalar is invalid.");
  return std::tuple_cat(std::make_tuple(alpha), rhs);
}

template<typename T,
         typename T1,
         typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
AddOpEntry<LabeledTensor<T>, T1>
operator |= (add_fn fn, AddOpEntry<LabeledTensor<T>, T1> op) {
  op.fn = fn;
  return op;
}

template<typename T,
         typename T1,
         typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
MultOpEntry<LabeledTensor<T>, T1>
operator |= (mult_fn fn, MultOpEntry<LabeledTensor<T>, T1> op) {
  op.fn = fn;
  return op;
}

template<typename T,
         typename T1,
         typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
AddOpEntry<LabeledTensor<T>, T1>
operator |= (ExecutionMode exec_mode, AddOpEntry<LabeledTensor<T>, T1> op) {
  op.exec_mode = exec_mode;
  return op;
}

template<typename T,
         typename T1,
         typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
MultOpEntry<LabeledTensor<T>, T1>
operator |= (ExecutionMode exec_mode, MultOpEntry<LabeledTensor<T>, T1> op) {
  op.exec_mode = exec_mode;
  return op;
}

// /**
//  * @todo Should validation be done in *OpEnty constructors?
//  */

//@todo for now assume all indices in a symmetry group are sliced
//the same way
// inline void
// validate_slicing(const TensorVec<SymmGroup>& indices,
//                  const IndexLabelVec& label) {
//   int pos = 0;
//   for(auto grp : indices)  {
//     if (grp.size() > 0){
//       for(int i=0; i<grp.size(); i++) {
//         //EXPECTS(label[pos+i].dt == label[pos].dt);
//         EXPECTS(is_dim_subset(grp[i], label[pos+i].dt));
//       }
//     }
//     pos += grp.size();
//   }
// }

inline void
validate_slicing(const TensorVec<TensorSymmGroup>& indices,
                 const IndexLabelVec& label) {
  int pos = 0;
  for(auto &grp : indices)  {
    if (grp.size() > 0){
      for(int i=0; i<grp.size(); i++) {
        //EXPECTS(label[pos+i].dt == label[pos].dt);
        EXPECTS(is_range_subset(grp.rt(), {label[pos+i].rt()}));
      }
    }
    pos += grp.size();
  }
}

template<typename LabeledTensorType, typename T>
inline void
addop_validate(const LabeledTensorType& ltc,
               const std::tuple<T, LabeledTensorType>& rhs) {
  auto lta = std::get<1>(rhs);
  EXPECTS(ltc.tensor_ != nullptr);
  EXPECTS(lta.tensor_ != nullptr);
  const auto& tc = *ltc.tensor_;
  const auto& ta = *lta.tensor_;
  EXPECTS(tc.rank() == ta.rank());

  IndexLabelVec clabel = ltc.label_;
  IndexLabelVec alabel = lta.label_;

  validate_slicing(tc.tindices(), ltc.label_);
  validate_slicing(ta.tindices(), lta.label_);

  EXPECTS(alabel.size() == ta.rank());
  EXPECTS(clabel.size() == tc.rank());

  //all labels are of compatible type
  for(int i=0; i<alabel.size(); i++) {
    EXPECTS(is_range_subset(ta.flindices()[i], alabel[i].rt()));
  }
  for(int i=0; i<clabel.size(); i++) {
    EXPECTS(is_range_subset(tc.flindices()[i], clabel[i].rt()));
  }

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
  EXPECTS(ltc.tensor_ != nullptr);
  EXPECTS(lta.tensor_ != nullptr);
  EXPECTS(ltb.tensor_ != nullptr);
  const auto& tc = *ltc.tensor_;
  const auto& ta = *lta.tensor_;
  const auto& tb = *ltb.tensor_;

  IndexLabelVec clabel = ltc.label_;
  IndexLabelVec alabel = lta.label_;
  IndexLabelVec blabel = ltb.label_;

  EXPECTS(clabel.size() == tc.rank());
  EXPECTS(alabel.size() == ta.rank());
  EXPECTS(blabel.size() == tb.rank());

  validate_slicing(tc.tindices(), ltc.label_);
  validate_slicing(ta.tindices(), lta.label_);
  validate_slicing(tb.tindices(), ltb.label_);

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

  std::sort(alabel.begin(), alabel.end());
  std::sort(blabel.begin(), blabel.end());
  std::sort(clabel.begin(), clabel.end());

  //all labels are unique
  EXPECTS(std::adjacent_find(alabel.begin(), alabel.end()) == alabel.end());
  EXPECTS(std::adjacent_find(blabel.begin(), blabel.end()) == blabel.end());
  EXPECTS(std::adjacent_find(clabel.begin(), clabel.end()) == clabel.end());

  IndexLabelVec slabel;
  std::set_intersection(alabel.begin(), alabel.end(),
                        blabel.begin(), blabel.end(),
                        std::back_inserter(slabel));
  //summation index is not in the output
  for(auto &sl: slabel) {
    EXPECTS(std::find(clabel.begin(), clabel.end(), sl) == clabel.end());
  }
  //every label in A/B is either in slabel or clabel
  for(auto &al : alabel) {
    EXPECTS(std::find(slabel.begin(), slabel.end(), al) != slabel.end()
            || std::find(clabel.begin(), clabel.end(), al) != clabel.end());
  }
  for(auto &bl : blabel) {
    EXPECTS(std::find(slabel.begin(), slabel.end(), bl) != slabel.end()
            || std::find(clabel.begin(), clabel.end(), bl) != clabel.end());
  }
  EXPECTS(clabel.size() == alabel.size() + blabel.size() - 2 * slabel.size());
}


}  // namespace tammx

#endif // TAMMX_LABELEDTENSOR_H_
