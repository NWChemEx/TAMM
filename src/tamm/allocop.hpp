#pragma once

#include <algorithm>
#include <chrono>
#include <memory>

#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/labeled_tensor.hpp"
#include "tamm/runtime_engine.hpp"
#include "tamm/tensor.hpp"
#include "tamm/types.hpp"
#include "tamm/utils.hpp"

namespace tamm {
template<typename TensorType>
class AllocOp: public Op {
public:
  AllocOp(TensorType tensor, ExecutionContext& ec): tensor_{tensor}, ec_{ec} {}

  AllocOp(const AllocOp<TensorType>&) = default;

  TensorType tensor() const { return tensor_; }

  OpList canonicalize() const override { return OpList{(*this)}; }

  OpType op_type() const override { return OpType::alloc; }

  std::shared_ptr<Op> clone() const override { return std::shared_ptr<Op>(new AllocOp{*this}); }

  void execute(ExecutionContext& ec, ExecutionHW hw = ExecutionHW::CPU) override {
    tensor_.allocate(&ec_);
  }

  TensorBase* writes() const override { return tensor_.base_ptr(); }

  TensorBase* accumulates() const override { return nullptr; }

  std::vector<TensorBase*> reads() const override { return {}; }

  bool is_memory_barrier() const override { return false; }

protected:
  TensorType        tensor_;
  ExecutionContext& ec_;

public:
  std::string opstr_;
}; // class AllocOp
} // namespace tamm
