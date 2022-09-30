#pragma once

#include "tamm/interfaces.hpp"
#include "tamm/op_attributes.hpp"
#include "tamm/scalar.hpp"
#include "tamm/tensor_variant.hpp"
#include "tamm/types.hpp"
#include <algorithm>
#include <cassert>
#include <complex>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <type_traits>
#include <variant>
#include <vector>

namespace tamm {
namespace new_ops {
//@todo Should we implement swap() and move constructor in all base (and virtual) classes.

//@todo Correct use of &&. Can this reduce the number of constructors?

//@todo and should I always call base class swap() just for
// completeness (in case base class has a state later)

//@todo Choose between clone() and move_clone() without having all
// these constructors. Also selectively choose to move_clone() if only
// some args are rvalues.

//@todo correctly handle invalid case for std::variant visit

//@todo: avoid returning non-const references to member
// variables. expose internal data structures and makes refactoring
// harder.

//@note: the constants are specifically setup to implement lub()

/*
OpExpr :=
  | scalar
  | LabeledTensor
  | OpExpr + OpExpr
  | OpExpr * OpExpr
*/

// helper type for the visitor #4
class Op: public Cloneable<Op>, public virtual Visitable {
public:
  Op() = default;

  Op(const Op& op): coeff_{op.coeff_} {
    for(const auto& kv: op.attributes_) {
      attributes_.insert_or_assign(kv.first, kv.second->clone());
    }
  }

  Op(Op&&) = default;

  virtual ~Op() = default;

  friend void swap(Op& first, Op& second) noexcept {
    using std::swap;
    swap(first.coeff_, second.coeff_);
    swap(first.attributes_, second.attributes_);
  }

  virtual void test_print() const { std::cerr << "Op base\n"; }

  template<typename AttrType>
  void set_attribute(const AttrType& attr) {
    attributes_.insert_or_assign(AttrType::id(), std::move(attr.clone()));
  }

  template<typename AttrType>
  void set_attribute(std::unique_ptr<AttrType>&& attr) {
    attributes_.insert_or_assign(AttrType::id(), std::move(attr));
  }

  template<typename AttrType>
  void remove_attribute() {
    auto it = attributes_.find(AttrType::id());
    assert(it != attributes_.end());
    attributes_.erase(it);
  }

  template<typename AttrType>
  const AttrType& get_attribute() const {
    assert(attributes_.find(AttrType::id()) != attributes_.end());
    return *static_cast<AttrType*>(attributes_.find(AttrType::id())->second.get());
  }

  template<typename AttrType>
  AttrType get_attribute(typename AttrType::Typename default_value) const {
    if(attributes_.find(AttrType::id()) != attributes_.end()) {
      return *static_cast<AttrType*>(attributes_.find(AttrType::id())->second.get());
    }
    else { return AttrType{default_value}; }
  }

  template<typename AttrType>
  const bool has_attribute() const {
    return attributes_.find(AttrType::id()) != attributes_.end();
  }

  void clear_attributes() { attributes_.clear(); }

  Scalar& coeff() { return coeff_; }

  const Scalar& coeff() const { return coeff_; }

  void set_coeff(const Scalar& coeff) { coeff_ = coeff; }

protected:
  Scalar                                      coeff_ = 1.0;
  std::map<int, std::unique_ptr<OpAttribute>> attributes_;
};

//@todo Check these with seperate unit tests
template<typename T>
using Unqualified = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

//@todo Check these with seperate unit tests
template<class U, class T, class = std::enable_if_t<std::is_base_of<U, Unqualified<T>>::value, T>>
using LimitTo = T;

class MultOp: public MakeVisitable<MultOp>, public InheritWithCloneable<MultOp, Op> {
public:
  MultOp(std::unique_ptr<Op>&& lhs, std::unique_ptr<Op>&& rhs):
    lhs_{std::move(lhs)}, rhs_{std::move(rhs)} {
    set_coeff(lhs_->coeff() * rhs_->coeff());
    lhs_->set_coeff(Scalar{1.0});
    rhs_->set_coeff(Scalar{1.0});
  }

  //@todo Check that this works as intended
  template<typename T1, typename T2>
  MultOp(LimitTo<Op, T1>&& lhs, LimitTo<Op, T2>&& rhs):
    lhs_{std::forward<T1>(lhs).clone()}, rhs_{std::forward<T2>(rhs).clone()} {
    set_coeff(lhs_->coeff() * rhs_->coeff());
    lhs_->set_coeff(Scalar{1.0});
    rhs_->set_coeff(Scalar{1.0});
  }

  MultOp(const MultOp& multop, const Scalar& scalar): MultOp{multop} {
    set_coeff(coeff() * scalar);
    lhs_->set_coeff(Scalar{1.0});
    rhs_->set_coeff(Scalar{1.0});
  }

  MultOp() = default;
  MultOp(const MultOp& multop): lhs_{multop.lhs_->clone()}, rhs_{multop.rhs_->clone()} {
    set_coeff(multop.coeff());
    lhs_->set_coeff(Scalar{1.0});
    rhs_->set_coeff(Scalar{1.0});
    for(const auto& kv: multop.attributes_) {
      attributes_.insert_or_assign(kv.first, kv.second->clone());
    }
  }

  MultOp(const std::unique_ptr<Op>& lhs, const std::unique_ptr<Op>& rhs):
    lhs_{lhs->clone()}, rhs_{rhs->clone()} {
    set_coeff(lhs_->coeff() * rhs_->coeff());
    lhs_->set_coeff(Scalar{1.0});
    rhs_->set_coeff(Scalar{1.0});
  }

  Op& lhs() { return *lhs_; }

  Op& rhs() { return *rhs_; }

  const Op& lhs() const { return *lhs_; }

  const Op& rhs() const { return *rhs_; }

  MultOp(MultOp&& other) noexcept: MultOp{} { swap(*this, other); }

  MultOp& operator=(MultOp other) noexcept {
    swap(*this, other);
    return *this;
  }

  friend void swap(MultOp& first, MultOp& second) noexcept {
    using std::swap;
    swap(static_cast<Op&>(first), static_cast<Op&>(second));
    swap(first.lhs_, second.lhs_);
    swap(first.rhs_, second.rhs_);
  }

private:
  std::unique_ptr<Op> lhs_;
  std::unique_ptr<Op> rhs_;
};

class AddOp: public MakeVisitable<AddOp>, public InheritWithCloneable<AddOp, Op> {
public:
  AddOp(std::unique_ptr<Op>&& lhs, std::unique_ptr<Op>&& rhs):
    lhs_{std::move(lhs)}, rhs_{std::move(rhs)} {}

  //@todo Check that this works as intended
  template<typename T1, typename T2>
  AddOp(LimitTo<Op, T1>&& lhs, LimitTo<Op, T2>&& rhs):
    lhs_{std::forward<T1>(lhs).clone()}, rhs_{std::forward<T2>(rhs).clone()} {}

  AddOp() = default;
  AddOp(const AddOp& addop): lhs_{addop.lhs_->clone()}, rhs_{addop.rhs_->clone()} {
    set_coeff(addop.coeff());
    for(const auto& kv: addop.attributes_) {
      attributes_.insert_or_assign(kv.first, kv.second->clone());
    }
  }

  AddOp(const AddOp& addop, const Scalar& scalar): AddOp{addop} {
    lhs_->set_coeff(lhs_->coeff() * scalar);
    rhs_->set_coeff(rhs_->coeff() * scalar);
  }

  AddOp(const std::unique_ptr<Op>& lhs, const std::unique_ptr<Op>& rhs):
    lhs_{lhs->clone()}, rhs_{rhs->clone()} {}

  Op& lhs() { return *lhs_; }

  Op& rhs() { return *rhs_; }

  const Op& lhs() const { return *lhs_; }

  const Op& rhs() const { return *rhs_; }

  AddOp(AddOp&& other) noexcept: AddOp{} { swap(*this, other); }

  AddOp& operator=(AddOp other) noexcept {
    swap(*this, other);
    return *this;
  }

  AddOp& operator+=(const AddOp& other) {
    *this = *this + other;
    return *this;
  }

  AddOp operator+(const AddOp& other) { return AddOp{*this, other}; }

  friend void swap(AddOp& first, AddOp& second) noexcept {
    using std::swap;
    swap(static_cast<Op&>(first), static_cast<Op&>(second));
    swap(first.lhs_, second.lhs_);
    swap(first.rhs_, second.rhs_);
  }

private:
  std::unique_ptr<Op> lhs_;
  std::unique_ptr<Op> rhs_;
};

/// @todo: Change raw pointers to shared_ptr once all Tensor infrastructure move
/// to clonable

class LTOp: public MakeVisitable<LTOp>, public InheritWithCloneable<LTOp, Op> {
public:
  template<typename T>
  LTOp(const LabeledTensor<T>& lt):
    tensor_{lt.tensor()}, tensor_type_{eltype<T>}, labels_{lt.labels()} {}

  template<typename T>
  LTOp(const Tensor<T>& tensor, const IndexLabelVec& labels = {}):
    tensor_{tensor}, tensor_type_{eltype<T>}, labels_{labels} {}

  LTOp(const TensorVariant& tensor, const IndexLabelVec& labels):
    tensor_{tensor}, tensor_type_{tensor.to_eltype()}, labels_{labels} {}

  LTOp(): tensor_type_{ElType::inv} {}

  LTOp(const LTOp& ltop):
    tensor_{ltop.tensor_}, tensor_type_{ltop.tensor_type_}, labels_{ltop.labels_} {
    set_coeff(ltop.coeff());
    for(const auto& kv: ltop.attributes_) {
      attributes_.insert_or_assign(kv.first, kv.second->clone());
    }
  }

  LTOp(const LTOp& ltop, const Scalar& scalar): LTOp{ltop} { set_coeff(coeff() * scalar); }

  ElType tensor_type() const { return tensor_type_; }

  TensorVariant tensor() { return tensor_; }

  const TensorVariant& tensor() const { return tensor_; }

  LTOp(LTOp&& other): LTOp{} { swap(*this, other); }

  LTOp& operator=(LTOp other) {
    swap(*this, other);
    return *this;
  }

  bool is_equal(const LTOp& other) {
    return (tensor_.base_ptr() == other.tensor_.base_ptr()) && (labels_ == other.labels_);
  }

  const IndexLabelVec& labels() const { return labels_; }

  friend void swap(LTOp& first, LTOp& second) noexcept {
    using std::swap;
    swap(static_cast<Op&>(first), static_cast<Op&>(second));
    swap(first.tensor_, second.tensor_);
    swap(first.tensor_type_, second.tensor_type_);
    swap(first.labels_, second.labels_);
  }

private:
  TensorVariant tensor_;
  ElType        tensor_type_;
  IndexLabelVec labels_;
};

class EinSumOp: public MakeVisitable<EinSumOp>, public InheritWithCloneable<EinSumOp, Op> {
public:
  EinSumOp() = default;

  EinSumOp(const EinSumOp& einsumop): op_{einsumop.op_->clone()}, labels_{einsumop.labels_} {}

  EinSumOp(const std::unique_ptr<Op>& op, const std::set<TiledIndexLabel>& labels):
    op_{op->clone()}, labels_{labels} {}

  EinSumOp(const std::unique_ptr<Op>& op, const IndexLabelVec& labels):
    op_{op->clone()}, labels_{std::set<TiledIndexLabel>(labels.begin(), labels.end())} {}

  // EinSumOp(const std::unique_ptr<Op>&& op, const std::set<TiledIndexLabel>&& labels) :
  //   op_{std::move(op)},
  //   labels_{std::move(labels)} {}

  void test_print() const override { std::cerr << "EinSumOp\n"; }

  Op& op() { return *op_; }

  const Op& op() const { return *op_; }

  const std::set<TiledIndexLabel>& labels() { return labels_; }

  EinSumOp(EinSumOp&& other): EinSumOp{} { swap(*this, other); }

  EinSumOp& operator=(EinSumOp other) {
    swap(*this, other);
    return *this;
  }

  friend void swap(EinSumOp& first, EinSumOp& second) noexcept {
    using std::swap;
    swap(static_cast<Op&>(first), static_cast<Op&>(second));
    swap(first.op_, second.op_);
    swap(first.labels_, second.labels_);
  }

private:
  std::unique_ptr<Op>       op_;
  std::set<TiledIndexLabel> labels_;
};

class ReshapeOp: public MakeVisitable<ReshapeOp>, public InheritWithCloneable<ReshapeOp, Op> {
public:
  ReshapeOp() = default;

  ReshapeOp(const ReshapeOp& einsumop): op_{einsumop.op_->clone()}, labels_{einsumop.labels_} {}

  ReshapeOp(const std::unique_ptr<Op>& op, const IndexLabelVec& labels):
    op_{op->clone()}, labels_{labels} {}

  // ReshapeOp(const std::unique_ptr<Op>&& op, const IndexLabelVec&& labels) :
  //   op_{std::move(op)},
  //   labels_{std::move(labels)} {}

  void test_print() const override { std::cerr << "ReshapeOp\n"; }

  Op& op() { return *op_; }

  const Op& op() const { return *op_; }

  const IndexLabelVec& labels() { return labels_; }

  ReshapeOp(ReshapeOp&& other): ReshapeOp{} { swap(*this, other); }

  ReshapeOp& operator=(ReshapeOp other) {
    swap(*this, other);
    return *this;
  }

  friend void swap(ReshapeOp& first, ReshapeOp& second) noexcept {
    using std::swap;
    swap(static_cast<Op&>(first), static_cast<Op&>(second));
    swap(first.op_, second.op_);
    swap(first.labels_, second.labels_);
  }

private:
  std::unique_ptr<Op> op_;
  IndexLabelVec       labels_;
};

class LambdaOp: public MakeVisitable<LambdaOp>, public InheritWithCloneable<LambdaOp, Op> {
  using UniqueOpVec = std::vector<std::unique_ptr<Op>>;
  using Func        = std::function<void()>;

public:
  LambdaOp() = default;

  LambdaOp(const LambdaOp& einsumop) {}

  LambdaOp(const IndexLabelVec& labels, const UniqueOpVec& read, const UniqueOpVec& write,
           Func lambda) {}

  // LambdaOp(const std::unique_ptr<Op>&& op, const IndexLabelVec&& labels) :
  //   op_{std::move(op)},
  //   labels_{std::move(labels)} {}

  void test_print() const override { std::cerr << "LambdaOp\n"; }

  const IndexLabelVec& labels() { return labels_; }

  LambdaOp(LambdaOp&& other): LambdaOp{} { swap(*this, other); }

  LambdaOp& operator=(LambdaOp other) {
    swap(*this, other);
    return *this;
  }

  friend void swap(LambdaOp& first, LambdaOp& second) noexcept {
    using std::swap;
    swap(static_cast<Op&>(first), static_cast<Op&>(second));
    swap(first.labels_, second.labels_);
    swap(first.read_, second.read_);
    swap(first.write_, first.write_);
    swap(first.lambda_, second.lambda_);
  }

private:
  IndexLabelVec labels_;
  UniqueOpVec   read_;
  UniqueOpVec   write_;
  Func          lambda_;
};

class ParForOp: public MakeVisitable<ParForOp>, public InheritWithCloneable<ParForOp, Op> {
public:
  ParForOp() = default;

  ParForOp(const ParForOp& parforop): op_{parforop.op_->clone()}, labels_{parforop.labels_} {}

  ParForOp(const std::unique_ptr<Op>& op, const IndexLabelVec& labels):
    op_{op->clone()}, labels_{labels} {}

  ParForOp(std::unique_ptr<Op>&& op, IndexLabelVec& labels): op_{std::move(op)}, labels_{labels} {}

  void test_print() const override { std::cerr << "ParForOp\n"; }

  Op& op() { return *op_; }

  const Op& op() const { return *op_; }

  const IndexLabelVec& labels() { return labels_; }

  ParForOp(ParForOp&& other): ParForOp{} { swap(*this, other); }

  ParForOp& operator=(ParForOp other) {
    swap(*this, other);
    return *this;
  }

  friend void swap(ParForOp& first, ParForOp& second) noexcept {
    using std::swap;
    swap(static_cast<Op&>(first), static_cast<Op&>(second));
    swap(first.op_, second.op_);
    swap(first.labels_, second.labels_);
  }

private:
  std::unique_ptr<Op> op_;
  IndexLabelVec       labels_;
};

// ///////////////////////////////////////////////////////////////////////////
//
//                Operator overloads to construct op trees
//
///////////////////////////////////////////////////////////////////////////

//@todo Can we unify these overloads for const& vs &&

template<typename Op1, typename Op2>
AddOp operator+(LimitTo<Op, Op1>&& lhs, LimitTo<Op, Op2>&& rhs) {
  return {std::forward<Op1>(lhs), std::forward<Op2>(rhs)};
}

template<typename Op1, typename Op2>
AddOp operator-(LimitTo<Op, Op1>&& lhs, LimitTo<Op, Op2>&& rhs) {
  return {std::forward<Op1>(lhs), std::forward<Op2>(rhs)};
}

template<typename T, typename Op1, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
AddOp operator+(LimitTo<Op, Op1>&& lhs, T value) {
  return {std::forward<Op1>(lhs), Scalar{value}};
}

template<typename T, typename Op1, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
AddOp operator+(T value, LimitTo<Op, Op1>&& rhs) {
  return {Scalar{value}, std::forward<Op1>(rhs)};
}

template<typename T, typename Op1, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
AddOp operator-(LimitTo<Op, Op1>&& lhs, T value) {
  return {std::forward<Op1>(lhs), Scalar{value}};
}

template<typename T, typename Op1, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
AddOp operator-(T value, LimitTo<Op, Op1>&& rhs) {
  return {Scalar{value}, std::forward<Op1>(rhs)};
}

template<typename T, typename Op1, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
Op1 operator*(const LimitTo<Op, Op1>& lhs, T value) {
  return {lhs, Scalar{value}};
}

template<typename T, typename Op1, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
Op1 operator*(T value, const LimitTo<Op, Op1>& rhs) {
  return {rhs, Scalar{value}};
}
template<typename Op1>
Op1 operator*(const LimitTo<Op, Op1>& lhs, Scalar value) {
  return {lhs, Scalar{value}};
}

template<typename Op1>
Op1 operator*(Scalar value, const LimitTo<Op, Op1>& rhs) {
  return {rhs, Scalar{value}};
}

template<typename Op1, typename Op2>
MultOp operator*(LimitTo<Op, Op1>&& lhs, LimitTo<Op, Op2>&& rhs) {
  return {std::forward<Op1>(lhs), std::forward<Op2>(rhs)};
}

template<typename Op1>
ParForOp parfor(LimitTo<Op, Op1>&& op, const IndexLabelVec& labels) {
  return ParForOp(op.clone(), labels);
}

} // namespace new_ops

/// @bug: LabeledTensor class method implementation carried here due to
/// compiling errors we should move it to an appropriate location once we solve
/// all cyclic inclide issues
template<typename T>
void LabeledTensor<T>::set(const std::unique_ptr<new_ops::Op>& op) {
  tensor_.add_update(TensorUpdate(ilv_, op));
}

template<typename T>
void LabeledTensor<T>::set(const new_ops::Op& op) {
  tensor_.add_update(TensorUpdate(ilv_, op.clone()));
}

template<typename T>
void LabeledTensor<T>::update(const std::unique_ptr<new_ops::Op>& op) {
  tensor_.add_update(TensorUpdate(ilv_, op, true));
}

template<typename T>
void LabeledTensor<T>::update(const new_ops::Op& op) {
  tensor_.add_update(TensorUpdate(ilv_, op.clone(), true));
}

template<typename T>
LabeledTensor<T>::operator new_ops::LTOp() const {
  return new_ops::LTOp{*this};
}

///////////////
} // namespace tamm
