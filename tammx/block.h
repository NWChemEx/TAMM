#ifndef TAMMX_BLOCK_H_
#define TAMMX_BLOCK_H_

#include "tammx/types.h"
#include "tammx/labeled-block.h"

namespace tammx {

/**
 * @todo Check copy semantics and that the buffer is properly managed.
 */
class Block {
 public:
  Block(Tensor& tensor,
        const TensorIndex& block_id);

  Block(Tensor& tensor,
        const TensorIndex& block_id,
        const TensorIndex& block_dims,
        const TensorPerm& layout,
        Sign sign);

  const TensorIndex& blockid() const {
    return block_id_;
  }

  const TensorIndex& block_dims() const {
    return block_dims_;
  }

  LabeledBlock operator () (const TensorLabel &label) {
    return LabeledBlock{this, label};
  }

  LabeledBlock operator () ();
  
  size_t size() const {
    size_t sz = 1;
    for(auto x : block_dims_) {
      sz *= x.value();
    }
    return sz;
  }

  Sign sign() const {
    return sign_;
  }

  const TensorPerm& layout() const {
    return layout_;
  }

  uint8_t* buf() {
    return buf_.get();
  }

  Tensor& tensor() {
    return tensor_;
  }
  
 private:
  Tensor& tensor_;
  TensorIndex block_id_;
  TensorIndex block_dims_;
  std::unique_ptr<uint8_t []> buf_;
  TensorPerm layout_;
  Sign sign_;
};


}  // namespace tammx

#endif  // TAMMX_BLOCK_H_

