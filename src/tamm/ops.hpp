#ifndef TAMM_OPS_HPP_
#define TAMM_OPS_HPP_

#include <memory>

#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/types.hpp"
#include "tamm/labeled_tensor.hpp"
#include "tamm/tensor_impl.hpp"

namespace tamm {

namespace internal {

template<typename T>
class LabelMap {
 public:
  LabelMap() = default;
  LabelMap(const LabelMap&) = default;
  LabelMap(LabelMap&&) = default;
  LabelMap& operator = (const LabelMap&) = default;
  LabelMap& operator = (LabelMap&&) = default;
  ~LabelMap() = default;
  
  LabelMap& update(const IndexLabelVec& labels,
                   const std::vector<T>& vals) {
    EXPECTS(labels.size() == vals.size());
    for(size_t i=0; i<vals.size(); i++) {
      map_[labels[i]] = vals[i];
    }
    return *this;
  }

  std::vector<T> get(const IndexLabelVec& labels) {
    std::vector<T> ret;
    for(const auto& lbl : labels) {
      auto itr = map_.find(lbl);
      EXPECTS(itr != map_.end());
      ret.push_back(itr->second);
    }
    return ret;
  }
  
 private:
  std::map<TiledIndexLabel, T> map_;
};

}  //namespace internal

class Op {
    public:
    virtual std::shared_ptr<Op> clone() const = 0;
    virtual void execute() = 0;
    virtual ~Op() {}
};

class OpList : public std::vector<std::shared_ptr<Op>>{
    public:
    // Ctors
    OpList() {}

    template <typename T, typename ...Args>
    OpList(T l_op, Args... args) : OpList(args...) {
        insert(begin(), l_op.clone());
    }
}; // OpList

template<typename T, typename LabeledTensorT>
class SetOp : public Op {
    public:
  SetOp() = default;

#if 0
    SetOp(LabeledTensorT lhs, T alpha, const LabeledLoop& loop_nest,
          bool is_assign) :
      lhs_{lhs},
      alpha_{alpha},
      loop_nest_{loop_nest},
      is_assign_{is_assign} {}
#endif

    SetOp(LabeledTensorT lhs,
          T alpha,
/*          const LabeledLoop& loop_nest,*/
          bool is_assign)
        : lhs_{lhs},
          alpha_{alpha},
          //loop_nest_{loop_nest},
          is_assign_{is_assign} {}

    SetOp(const SetOp<T, LabeledTensorT>&) = default;

    T alpha() const { return alpha_; }

    LabeledTensorT lhs() const { return lhs_; }

    bool is_assign() const { return is_assign_; }

    std::shared_ptr<Op>  clone() const override { return std::shared_ptr<Op>(new SetOp<T, LabeledTensorT>{*this}); }

    void execute() override {
      using TensorElType = typename LabeledTensorT::element_type;
      //the iterator to generate the tasks
      auto loop_nest = lhs_.tensor().loop_nest();
      //function to compute one block
      auto lambda = [&] (const IndexVector blockid) {
        auto tensor = lhs_.tensor();
        size_t size = tensor.block_size(blockid);
        std::vector<TensorElType> buf(size, static_cast<TensorElType>(alpha()));
        if(is_assign_) {
          tensor.put(blockid, span<TensorElType>(&buf[0],size));
        } else {
          tensor.add(blockid, span<TensorElType>(&buf[0], size));
        }
      };
      //ec->...(loop_nest, lambda);
      //@todo use a scheduler
      //@todo make parallel
      for(const auto& blockid: loop_nest) {
        lambda(blockid);
      }
    }

    protected:
    T alpha_;
    LabeledTensorT lhs_;
  //LabeledLoop loop_nest_;
    bool is_assign_;
}; // class SetOp

template<typename T, typename LabeledTensorT>
class AddOp : public Op {
    public:
  AddOp() = default;
    AddOp(LabeledTensorT lhs, T alpha, LabeledTensorT rhs,
          /*const LabeledLoop& loop_nest,*/ bool is_assign) :
      lhs_{lhs},
      alpha_{alpha},
      rhs_{rhs},
      //loop_nest_{loop_nest},
      is_assign_{is_assign} {}

    AddOp(const AddOp<T, LabeledTensorT>&) = default;

    T alpha() const { return alpha_; }

    LabeledTensorT lhs() const { return lhs_; }

    LabeledTensorT rhs() const { return rhs_; }

    bool is_assign() const { return is_assign_; }

    std::shared_ptr<Op> clone() const override { return std::shared_ptr<Op>(new AddOp<T, LabeledTensorT>{*this}); }

    void execute() override {
      using TensorElType = typename LabeledTensorT::element_type;
      //the iterator to generate the tasks
      auto loop_nest = lhs_.tensor().loop_nest();
      //function to compute one block
      auto lambda = [&] (const IndexVector lblockid) {
        auto ltensor = lhs_.tensor();
        auto rtensor = rhs_.tensor();
        size_t size = ltensor.block_size(lblockid);
        IndexVector rblockid =
          internal::LabelMap<Index>()
          .update(lhs_.labels(), lblockid)
          .get(rhs_.labels());
        std::vector<TensorElType> buf(size);
        rtensor.get(rblockid, span<TensorElType>(&buf[0], size));
        //@bug @todo Take labels into account when doing the add
        for(auto& v : buf) {
          v *= static_cast<TensorElType>(alpha());
        }
        if(is_assign_) {
          ltensor.put(lblockid, span<TensorElType>(&buf[0],size));
        } else {
          ltensor.add(lblockid, span<TensorElType>(&buf[0],size));
        }
      };
      //ec->...(loop_nest, lambda);
      //@todo use a scheduler
      //@todo make parallel
      for(const auto& lblockid: loop_nest) {
        lambda(lblockid);
      }
    }

    protected:
    LabeledTensorT lhs_;
    T alpha_;
    LabeledTensorT rhs_;
    //LabeledLoop loop_nest_;
    bool is_assign_;
}; // class AddOp

template<typename T, typename LabeledTensorT>
class MultOp : public Op {
    public:
  MultOp() = default;
    MultOp(LabeledTensorT lhs, T alpha, LabeledTensorT rhs1,
           LabeledTensorT rhs2, //LabeledLoop outer_loop_nest,
           /*LabeledLoop inner_loop_nest, SymmFactor symm_factor,*/
           bool is_assign) :
      lhs_{lhs},
      alpha_{alpha},
      rhs1_{rhs1},
      rhs2_{rhs2},
    //   outer_loop_nest_{outer_loop_nest},
    //   inner_loop_nest_{inner_loop_nest},
    //   symm_factor_{symm_factor},
    is_assign_{is_assign} {}

    MultOp(const MultOp<T, LabeledTensorT>&) = default;

    LabeledTensorT lhs() const { return lhs_; }

    T alpha() const { return alpha_; }

    LabeledTensorT rhs1() const { return rhs1_; }

    LabeledTensorT rhs2() const { return rhs2_; }

    bool is_assign() const { return is_assign_; }

    std::shared_ptr<Op> clone() const override { return std::shared_ptr<Op>(new MultOp{*this}); }

    void execute() override {}

    protected:
    LabeledTensorT lhs_;
    T alpha_;
    LabeledTensorT rhs1_;
    LabeledTensorT rhs2_;
    // LabeledLoop outer_loop_nest_;
    // LabeledLoop inner_loop_nest_;
    // SymmFactor symm_factor_;
    bool is_assign_;
}; // class MultOp

template<typename TensorType>
class AllocOp : public Op {
    public:
    AllocOp(TensorType tensor) : tensor_{tensor} {}

    AllocOp(const AllocOp<TensorType>&) = default;

    TensorType tensor() const { return tensor_; }

    std::shared_ptr<Op> clone() const override { return std::shared_ptr<Op>(new AllocOp{*this}); }

    void execute() override { tensor_.allocate(); }

    protected:
    TensorType tensor_;
}; // class AllocOp

template<typename TensorType>
class DeallocOp : public Op {
    public:
    DeallocOp(TensorType tensor) : tensor_{tensor} {}

    DeallocOp(const DeallocOp<TensorType>&) = default;

    TensorType tensor() const { return tensor_; }

    std::shared_ptr<Op> clone() const override { return std::shared_ptr<Op>(new DeallocOp{*this}); }

    void execute() override { tensor_.deallocate(); }

    protected:
    TensorType tensor_;
}; // class AllocOp

} // namespace tamm

#endif // TAMM_OPS_HPP_
