#ifndef TAMMX_BLOCK_H_
#define TAMMX_BLOCK_H_

#include <memory>

#include "tammx/types.h"
#include "tammx/tce.h"
//#include "tammx/labeled-block.h"

namespace tammx {

template<typename T>
class LabeledBlock;

template<typename T>
class Tensor;

/**
 * @todo Check copy semantics and that the buffer is properly managed.
 */
template<typename T>
class Block {
 public:
  Block(Tensor<T>& tensor,
        const TensorIndex& block_id);

  Block(Tensor<T>& tensor,
        const TensorIndex& block_id,
        const TensorIndex& block_dims,
        const TensorPerm& layout,
        Sign sign);

  const TensorIndex& blockid() const {
    return block_id_;
  }

  TensorIndex block_offset() const {
    TensorIndex ret;
    for(auto id: block_id_) {
      ret.push_back(BlockDim{TCE::offset(id)});
    }
  }
  
  const TensorIndex& block_dims() const {
    return block_dims_;
  }

  LabeledBlock<T> operator () (const TensorLabel &label);

  LabeledBlock<T> operator () ();
  
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

  Tensor<T>& tensor() {
    return tensor_;
  }
  
 private:
  Tensor<T>& tensor_;
  TensorIndex block_id_;
  TensorIndex block_dims_;
  std::unique_ptr<uint8_t []> buf_;
  TensorPerm layout_;
  Sign sign_;
};


}  // namespace tammx

#endif  // TAMMX_BLOCK_H_

