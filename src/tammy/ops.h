#ifndef TAMMY_OPS_H_
#define TAMMY_OPS_H_

#include <memory>

#include "boundvec.h"
#include "errors.h"
#include "types.h"
#include "proc_group.h"

namespace tammy {

class Op {
 public:
  virtual Op* clone() = 0;
  virtual void execute();
  virtual ~Op() {}
};

template<typename T, typename LabeledTensorT>
class AddOp :  public Op {
 public:
  AddOp(T alpha, LabeledTensorT rhs, bool is_assign)
      : alpha_{alpha},
        rhs_{rhs},
        is_assign_{is_assign} {}

  AddOp(const AddOp<T,LabeledTensorT>&) = default;

  T alpha() const {
    return alpha;
  }

  LabeledTensorT rhs() const {
    return rhs_;
  }

  bool is_assign() const {
    return is_assign_;
  }

  Op* clone() const {
    return new AddOp<T,LabeledTensorT>{*this};
  }

  void execue() {
  }
  
 protected:
  T alpha_;
  LabeledTensorT rhs_;
  bool is_assign_;
};  // class AddOp


template<typename T, typename LabeledTensorT1, typename LabeledTensorT2>
class MultOp {
 public:
  MultOp(T alpha, LabeledTensorT1 rhs1, LabeledTensorT2 rhs2, bool is_assign)
      : alpha_{alpha},
        rhs1_{rhs1},
        rhs2_{rhs2},
        is_assign_{is_assign} {}
  
  MultOp(const MultOp<T,LabeledTensorT1,LabeledTensorT2>&) = default;
  
  T alpha() const {
    return alpha;
  }

  LabeledTensorT1 rhs1() const {
    return rhs1_;
  }

  LabeledTensorT2 rhs2() const {
    return rhs2_;
  }

  bool is_assign() const {
    return is_assign_;
  }

  Op* clone() const {
    return new MultOp{*this};
  }

  void execue() {
  }
  
 protected:
  T alpha_;
  LabeledTensorT1 rhs1_;
  LabeledTensorT2 rhs2_;
  bool is_assign_;
}; //class MultOp

}   // namespace tammy

#endif  // TAMMY_OPS_H_

