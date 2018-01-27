#ifndef TAMMY_OPERATORS_H_
#define TAMMY_OPERATORS_H_

#include "tammy/boundvec.h"
#include "tammy/errors.h"
#include "tammy/types.h"
#include "tammy/index_space.h"
#include "tammy/perm_symmetry.h"
#include "tammy/iterator.h"

#include <memory>
#include <tuple>

namespace tammy {

class SummationOperator {
 public:
  SummationOperator(const IndexLabelVec& index_labels,
                    PermGroup&& perm_group)
      : perm_group_{std::move(perm_group)},
        index_labels_{index_labels} {}

  std::unique_ptr<Generator<BlockIndex>> generator() const {
    TensorVec<BlockIndex> lo, hi;

    for(const auto& il: index_labels_) {
      lo.push_back(il.ir().blo());
      hi.push_back(il.ir().bhi());
    }

    return perm_group_.unique_generator(lo, hi);
  }

  const IndexLabelVec& label() const {
    return index_labels_;
  }

 protected:
  PermGroup perm_group_;
  IndexLabelVec index_labels_;
};  // SummationOperator

#if 0
template<typename T>
using Func1 = void (*)(ExecutionContext& ec, const BlockDimVec& lhs, T alpha, const BlockDimVec& rhs1);

template<typename T>
using Func2 = void (*)(ExecutionContext& ec, const BlockDimVec& lhs, T alpha,  const BlockDimVec& rhs1, const BlockDimVec& rhs2);
#endif

// template<typename T>
// const BlockDimVec bid(const LabelMap<BlockIndex>& lmap, LabeledTensor<T> ltensor) {
//   return lmap.get_blockid(ltensor.label_);
// }

template<typename OpType>
using BlockFunc = void (*)(ExecutionContext& ec,
                           const OpType& op,
                           const LabelMap<BlockIndex>& lmap);

template<typename T>
class GenericOp {
 public:
  GenericOp(BlockFunc<GenericOp<T>> block_func,
            LabeledTensor<T>& lhs,
            T alpha,
            SummationOperator& sum_op,
            const std::tuple<LabeledTensor<T>>& rhs)
      : block_func_{block_func},
        lhs_{lhs},
        alpha_{alpha},
        sum_op_{sum_op},
        rhs_{rhs} {}

  virtual ~GenericOp() {}

  void execute(ExecutionContext& ec) {
    auto lhs_gen = lhs_.tensor().unique_generator();

    parallel_work(ec.pg(), lhs_gen, [&ec,this](const BlockDimVec& lhs_bid) {
        if(!lhs_.tensor().nonzero(lhs_bid))
          return;
        LabelMap<BlockIndex> lmap;
        lmap.update(lhs_.label(), lhs_bid);
        auto sum_gen = sum_op_.generator();
        while(sum_gen->has_more()) {
          lmap.update(sum_op_.label(), sum_gen->get());
          block_func_(ec, *this, lmap);
          sum_gen->next();
        }
      }
      );
  }

 protected:
  BlockFunc<GenericOp<T>> block_func_;
  LabeledTensor<T> lhs_;
  T alpha_;
  SummationOperator sum_op_;
  std::tuple<LabeledTensor<T>> rhs_;
  LabelMap<BlockIndex> lmap_;

  friend BlockFunc<GenericOp<T>>;
};

#if 0
template<typename T1, typename T2...>
using ElementOpFunc = T1 (T2...);

template<typename T>
using BlockOpFunc = void (Block<T> lhs, Block<T> rhs);
#endif

// template<typename T, typename Func>
// class GenMapOp {
// };

// template<typename T>
// class GenMapOp<T, ElementOpFunc<T>> {
//  public:
//   GenMapOp(ElementOpFunc<T> func,
//            LabeledTensor<T>& lhs,
//            LabeledTensor<T>& rhs)
//       : func_{func},
//         lhs_{lhs},
//         rhs_{rhs} {}

//   void execute(ExecutionContext& ec) {
//     auto lhs_gen = lhs_.tensor().unique_nonzero_generator();

//     parallel_work(ec.pg(), lhs_gen, [&ec,this](const BlockDimVec& bid) {
//         auto lhs_block = lhs_.tensor().access_for_write(bid);
//         auto rhs_block = rhs_.tensor().access_for_read(bid);

//         when(lhs_block, rhs_block, [lhs_block,rhs_block]() {
//             for(size_t i=0; i<lhs_block.size(); i++) {
//               lhs_block.buf(i) = func_(rhs_block.buf(i));
//             }
//           });
//       }
//       );
//   }
//  protected:
//   ElementOpFunc<T> func_;
//   LabeledTensor<T> lhs_;
//   LabeledTensor<T> rhs_;
// };

class Codelet {
 public:
  virtual bool is_ready() const = 0;
  virtual void execute() = 0;
  virtual ~Codelet() {}
};  // Codelet

template<typename Func, typename... Cond>
class CodeletImpl: public Codelet {
 public:
  CodeletImpl(const Func& func, Cond... cond)
      : cond_(std::make_tuple<Cond...>(cond...)),
        func_{func} {}

  bool is_ready() const override {
    return check_ready(cond_);
  }

  void execute() override {
    func_();
  }
  
 protected:
  static bool check_ready() {
    return true;
  }

  template<typename Cond1, typename... Cond2>
  static bool check_ready(Cond1 cond1, Cond2... cond2) {
    return cond1.is_ready && check_ready(cond2...);
  }

  std::tuple<Cond...> cond_;
  Func func_;
};  // CodeletImpl

#if 0
template<typename T, Func func>
class MapBlockOp {
 public:
  MapBlockOp(Func func,
             LabeledTensor<T>& lhs,
             LabeledTensor<T>& rhs)
      : func_{func},
        lhs_{lhs},
        rhs_{rhs} {}

  void execute(ExecutionContext& ec) {
    ExecutionScope es{ec};
    auto lambda = [&ec,this](const BlockDimVec& bid) {
      auto lhs_block = lhs_.tensor().access_for_write(bid);
      auto rhs_block = rhs_.tensor().access_for_read(bid);      
      ec.when(lhs_block, rhs_block, [func_,lhs_block,rhs_block]() {
          func_(lhs_block, rhs_block);
        });
    };
    parallel_work(ec.pg(),
                  lhs_.tensor().unique_nonzero_generator(),
                  lambda);
  }
 protected:
  Func func_;
  LabeledTensor<T> lhs_;
  LabeledTensor<T> rhs_;
};

template<typename T, Func func>
class MapElementOp {
 public:
  MapElementOp(Func func,
               LabeledTensor<T>& lhs,
               LabeledTensor<T>& rhs)
      : block_op(BlockFunc(func_),
                 lhs,
                 rhs) {}

  void execute(ExecutionContext& ec) {
    ExecutionScope es{ec};
    block_op.execute(ec);
  }
 protected:
  struct BlockFunc {
    Func func_;
    void operator(Block<T> lhs_block, Block<T> rhs_block) {
      for(size_t i=0; i<lhs_block.size(); i++) {
        lhs_block.buf(i) = func_(rhs_block.buf(i));
      }
    }
  };
  MapBlockOp<T, MapElementOp<T,Func>::BlockFunc> block_op;
};
#endif


#if 0
void operation(ExecutionContext& ec,
               Func block_func,
               LabeledTensor<T>& lhs,
               T alpha,
               SummationOperator& sum_op,
               RHS... rhs) {
  auto work = [] (const BlockDimVec& lhs_bid) {
  };

  auto lhs_itr = lhs.tensor_->unique_generator();


  parallel_work(ec.pg(), lhs_gen, [] (const BlockDimVec& lhs_bid) {
  const auto& lhs_label = lhs.label_;
  const auto& sum_label = sum_op.label();
  LabelMap<BlockIndex> lmap;
  while(lhs_itr->has_more()) {
    lmap.update(lhs_label, lhs_itr->get());
    auto sum_itr = sum_op.generator();

    const auto& lhs_bid = lmap.get_blockid(lhs_label);
    while(sum_itr->has_more()) {
      lmap.update(sum_label, sum_itr->get());

      auto bid = [&] (LabeledTensor<T>& ltensor) {
        return lmap.get_blockid(ltensor.label_);
      };
      block_func(ec, lhs_bid, alpha, bid(rhs)...);
      sum_itr->next();
    }
    lhs_itr->next();
  }
}


void fn2(ExecutionContext& ec,
         LabeledTensor<double>& lhs,
         double alpha,
         SummationOperator& sum_op,
         LabeledTensor<double>& rhs1,
         Func2<double> block_func) {
  operation(ec,  block_func, lhs, alpha, sum_op, rhs1, rhs1);
}
#endif

}  // namespace tammy

#endif // TAMMY_OPERATORS_H_
