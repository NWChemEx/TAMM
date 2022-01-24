#pragma once

#include "tamm/scalar.hpp"
#include "tamm/types.hpp"

namespace tamm {

template <typename T>
class BlockSpan {
 public:
  enum class BufKind { cpu, invalid };
  // BlockSpan(const Tensor<T>& tensor, const IndexVector& blockid, T* buf)
  //     : buf_{buf} {
  //   EXPECTS(buf != nullptr);
  //   block_dims_ = tensor.block_dims(blockid);
  //   num_elements_ = 1;
  //   for (const auto& bd : block_dims) {
  //     num_elements_ *= bd;
  //   }
  // }

  BlockSpan(T* buf, const std::vector<size_t>& block_dims)
      : buf_kind_{BufKind::cpu}, buf_{buf}, block_dims_{block_dims} {
    EXPECTS(buf != nullptr);
    num_elements_ = 1;
    for (const auto& bd : block_dims) {
      num_elements_ *= bd;
    }
  }

  BlockSpan() : buf_kind_{BufKind::invalid}, buf_{nullptr}, num_elements_{0} {}

  BlockSpan(const BlockSpan<T>&) = default;
  BlockSpan(BlockSpan<T>&&) = default;
  ~BlockSpan() = default;
  BlockSpan<T>& operator=(const BlockSpan<T>&) = default;
  BlockSpan<T>& operator=(BlockSpan<T>&&) = default;

  const std::vector<size_t>& block_dims() const {
    return block_dims_;
  }

  T* buf() { return buf_; }

  const T* buf() const { return buf_; }

  size_t num_elements() const { return num_elements_; }

 private:
  BufKind buf_kind_;
  T* buf_;
  std::vector<size_t> block_dims_;
  size_t num_elements_;
};  // class BlockSpan
}  // namespace tamm
