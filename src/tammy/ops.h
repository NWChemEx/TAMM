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
  virtual Op* clone() const = 0;
  virtual void execute() = 0;
  virtual ~Op() {}
};

template<typename T, typename LabeledTensorT>
class SetOp :  public Op {
 public:
  using Itr = IndexSpace::Iterator;
  SetOp(LabeledTensorT lhs,
        T alpha,
        const LBLoopNest<Itr>& loop_nest,
        bool is_assign)
      : lhs_{lhs},
        alpha_{alpha},
        loop_nest_{loop_nest},
        is_assign_{is_assign} {}

  // SetOp(LabeledTensorT lhs,
  //       T alpha,
  //       const LBLoopNest<Itr>& loop_nest,
  //       bool is_assign)
  //     : lhs_{lhs},
  //       alpha_{alpha},
  //       loop_nest_{loop_nest},
  //       is_assign_{is_assign} {}

  SetOp(const SetOp<T,LabeledTensorT>&) = default;

  T alpha() const {
    return alpha;
  }

  LabeledTensorT lhs() const {
    return lhs_;
  }

  bool is_assign() const {
    return is_assign_;
  }

  Op* clone() const override {
    return new SetOp<T,LabeledTensorT>{*this};
  }

  void execute() override {
  }
  
 protected:
  T alpha_;
  LabeledTensorT lhs_;
  LBLoopNest<Itr> loop_nest_;
  bool is_assign_;
};  // class AddOp

template<typename T, typename LabeledTensorT>
class AddOp :  public Op {
 public:
  using Itr = IndexSpace::Iterator;
  AddOp(LabeledTensorT lhs,
        T alpha,
        LabeledTensorT rhs,
        const LBLoopNest<Itr>& loop_nest,
        bool is_assign)
      : lhs_{lhs},
        alpha_{alpha},
        rhs_{rhs},
        loop_nest_{loop_nest},
        is_assign_{is_assign} {}

  AddOp(const AddOp<T,LabeledTensorT>&) = default;

  T alpha() const {
    return alpha;
  }

  LabeledTensorT lhs() const {
    return lhs_;
  }

  LabeledTensorT rhs() const {
    return rhs_;
  }

  bool is_assign() const {
    return is_assign_;
  }

  Op* clone() const override {
    return new AddOp<T,LabeledTensorT>{*this};
  }

  void execute() override {
  }
  
 protected:
  LabeledTensorT lhs_;
  T alpha_;
  LabeledTensorT rhs_;
  LBLoopNest<Itr> loop_nest_;
  bool is_assign_;
};  // class AddOp


template<typename T, typename LabeledTensorT>
class MultOp : public Op {
 public:
  using Itr = IndexSpace::Iterator;
  MultOp(LabeledTensorT lhs,
         T alpha,
         LabeledTensorT rhs1,
         LabeledTensorT rhs2,
         LBLoopNest<Itr> outer_loop_nest,
         LBLoopNest<Itr> inner_loop_nest,
         SymmFactor symm_factor,
         bool is_assign)
      : lhs_{lhs},
        alpha_{alpha},
        rhs1_{rhs1},
        rhs2_{rhs2},
        outer_loop_nest_{outer_loop_nest},
        inner_loop_nest_{inner_loop_nest},
        symm_factor_{symm_factor},
        is_assign_{is_assign} {}
  
  MultOp(const MultOp<T,LabeledTensorT>&) = default;
  
  LabeledTensorT lhs() const {
    return lhs_;
  }

  T alpha() const {
    return alpha;
  }

  LabeledTensorT rhs1() const {
    return rhs1_;
  }

  LabeledTensorT rhs2() const {
    return rhs2_;
  }

  bool is_assign() const {
    return is_assign_;
  }

  Op* clone() const override {
    return new MultOp{*this};
  }

  void execute() override {
  }
  
 protected:
  LabeledTensorT lhs_;
  T alpha_;
  LabeledTensorT rhs1_;
  LabeledTensorT rhs2_;
  LBLoopNest<Itr> outer_loop_nest_;
  LBLoopNest<Itr> inner_loop_nest_;
  SymmFactor symm_factor_;
  bool is_assign_;
}; //class MultOp

template<typename TensorType>
class AllocOp : public Op {
 public:
  AllocOp(TensorType tensor)
      : tensor_{tensor} {}
  
  AllocOp(const AllocOp<TensorType>&) = default;

  TensorType tensor() const {
    return tensor_;
  }

  Op* clone() const override {
    return new AllocOp{*this};
  }

  void execute() override {
    tensor_.allocate();
  }
  
 protected:
  TensorType tensor_;
}; // class AllocOp

template<typename TensorType>
class DeallocOp : public Op {
 public:
  DeallocOp(TensorType tensor)
      : tensor_{tensor} {}
  
  DeallocOp(const DeallocOp<TensorType>&) = default;

  TensorType tensor() const {
    return tensor_;
  }

  Op* clone() const override {
    return new DeallocOp{*this};
  }

  void execute() override {
    tensor_.deallocate();
  }
  
 protected:
  TensorType tensor_;
}; // class AllocOp

}   // namespace tammy

#endif  // TAMMY_OPS_H_

