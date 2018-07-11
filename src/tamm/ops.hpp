#ifndef TAMM_OPS_HPP_
#define TAMM_OPS_HPP_

#include <memory>

#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/types.hpp"

namespace tamm {

class Op {
    public:
    virtual std::shared_ptr<Op> clone() const = 0;
    virtual void execute()    = 0;
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

    T alpha() const { return alpha; }

    LabeledTensorT lhs() const { return lhs_; }

    bool is_assign() const { return is_assign_; }

    std::shared_ptr<Op>  clone() const override { return std::shared_ptr<Op>(new SetOp<T, LabeledTensorT>{*this}); }

    void execute() override {
#if 0
      using TensorElType = LabeledTensorT::element_type;
      //the iterator to generate the tasks
      auto loop_nest = lhs_.tensor()->loop_nest();
      //function to compute one block
      auto lambda = [&] (const IndexVector blockid) {
        auto& tensor = *lhs_.tensor();
        size_t size = tensor.block_size(blockid);
        TensorElType *buf = new TensorElType[size];
        std::fill_n(buf, size, static_cast<T>(alpha));
        tensor.put(blockid, span(buf,size));
        delete [] buf;
      };
#endif      
    }

    protected:
    T alpha_;
    LabeledTensorT lhs_;
  //LabeledLoop loop_nest_;
    bool is_assign_;
}; // class SetOp

namespace detail {

template<typename T>
class LabelMap {
 public:
  LabelMap() = default;
  LabelMap(const LabelMap&) = default;
  LabelMap(LabelMap&&) = default;
  LabelMap& operator = (const LabelMap&) = default;
  LabelMap& operator = (LabelMap&&) = default;
  ~LabelMap() = default;
  
  LabelMap& update(const std::vector<TiledIndexLabel>& labels,
                   const std::vector<T> vals) {
    EXPECTS(labels.size() == vals.size());
    for(size_t i=0; i<vals.size(); i++) {
      map_[labels[i]] = vals[i];
    }
    return *this;
  }

  std::vector<T> get(const std::vector<TiledIndexLabel>& labels) {
    std::vector<T> ret;
    for(const auto& lbl : labels) {
      auto itr = map_.find(lbl);
      EXPECTS(itr != map_.end());
      ret.push_back(*itr);
    }
    return ret;
  }

  
 private:
  std::map<TiledIndexLabel, T> map_;
};

}  //namespace detail

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

    T alpha() const { return alpha; }

    LabeledTensorT lhs() const { return lhs_; }

    LabeledTensorT rhs() const { return rhs_; }

    bool is_assign() const { return is_assign_; }

    std::shared_ptr<Op> clone() const override { return std::shared_ptr<Op>(new AddOp<T, LabeledTensorT>{*this}); }

    void execute() override {
#if 0
      using TensorElType = LabeledTensorT::element_type;
      //the iterator to generate the tasks
      auto loop_nest = lhs_.tensor()->loop_nest();
      //function to compute one block
      auto lambda = [&] (const IndexVector lblockid) {
        auto& ltensor = *lhs_.tensor();
        auto& rtensor = *rhs_.tensor();
        size_t size = tensor.block_size(lblockid);
        IndexVector rblockid =
        LabelMap<Index>()
        .update(lhs_.label(), lblockid);
        .get(rhs_.label());
        std::vector<TensorElType> buf(lsize);
        tensor.get(rblockid, span(&buf[0], size));
        for(auto& v : buf) {
          v *= static_cast<TensorElType>(alpha);
        }
        if(is_assign_) {
          tensor.put(lblockid, span(&buf[0],size));
        } else {
          tensor.add(lblockid, span(&buf[0],size));
        }
      };
#endif
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

    T alpha() const { return alpha; }

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
