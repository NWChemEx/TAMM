#ifndef TAMM_OP_BASE_H_
#define TAMM_OP_BASE_H_

#include <memory>
#include <vector>

#include "tamm/boundvec.hpp"
#include "tamm/execution_context.hpp"
#include "tamm/tensor.hpp"

namespace tamm {
enum class ResultMode { update, set };
enum class OpType { alloc, dealloc, set, add, mult, scan, map };

class OpList;

class Op {
 public:
  virtual TensorBase* writes() const = 0;
  virtual TensorBase* accumulates() const = 0;
  virtual std::vector<TensorBase*> reads() const = 0;
  virtual bool is_memory_barrier() const = 0;
  virtual std::shared_ptr<Op> clone() const = 0;
  virtual void execute(ExecutionContext& ec,
                       ExecutionHW hw = ExecutionHW::CPU) = 0;
  virtual OpList canonicalize() const = 0;
  virtual OpType op_type() const = 0;
  virtual ~Op() {}
  std::string opstr_;
  ExecutionHW exhw_ = ExecutionHW::DEFAULT;
};

class OpList : public std::vector<std::shared_ptr<Op>> {
 public:
  // Ctors
  OpList() {}

  template <typename T, typename... Args>
  OpList(T l_op, Args... args) : OpList(args...) {
    insert(begin(), l_op.clone());
  }
};  // OpList

}  // namespace tamm
#endif  // TAMM_OP_BASE_H_
