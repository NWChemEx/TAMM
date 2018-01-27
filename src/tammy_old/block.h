#ifndef TAMMY_BLOCK_H_
#define TAMMY_BLOCK_H_

#include <memory>

#include "tammy/errors.h"
#include "tammy/types.h"
//#include "tammy/tce.h"

namespace tammy {

template<typename T>
class LabeledBlock;

template<typename T>
class Tensor;

/**
 * @brief A data block in a tensor.
 *
 * Communication on tensors happens in the forms of blocks. Blocks track the underlying tensor, block dimensions, and simplify communication management.
 *
 * @tparam T Type of element in the block
 */
template<typename T>
class Block {
 public:

  Block() = delete;
  Block(const Block<T>&) = delete;
  Block<T>& operator = (const Block<T>&) = delete;

  /**
   * @brief Construct a block to hold data in a specific block in a tensor.
   * @param tensor Tensor from which data is to be stored
   * @param block_id Id of block in the tensor
   */
  Block(Tensor<T>& tensor,
        const BlockDimVec& block_id)
    : tensor_{tensor},
      block_id_{block_id} {
        block_dims_ = tensor.block_dims(block_id);
        layout_.resize(tensor.rank());
        std::iota(layout_.begin(), layout_.end(), 0);
        sign_ = 1;
        EXPECTS(size() > 0);
        buf_ = std::make_unique<T[]> (size());
      }

  /**
   * Move constructor from another block.
   * @param block Block being moved from
   */
  Block(Block<T>&& block)
      : tensor_{block.tensor_},
        block_id_{block.block_id_},
        block_dims_{block.block_dims_},
        layout_{block.layout_},
        sign_{block.sign_},
        buf_{std::move(block.buf_)} { }

  /**
   * @brief Construct a block with a specific layout and sign.
   *
   * Applying the layout transformation and the sign pre-factor on the underlying
   * stored data gives the data actually stored at @p block_id in @p tensor. This
   * is used to avoid construct explicit layout transformation for every communication
   * operation. Operations on blocks need to take the underlying layout and sign into account.
   *
   * @param tensor Tensor's whose block is being costructed
   * @param block_id Id of block in the tensor
   * @param layout Layout (index order) of dimensions in the block
   * @param sign sign pre-factor
   */
  Block(Tensor<T>& tensor,
        const BlockDimVec& block_id,
        const PermVec& layout,
        Sign sign)
      : tensor_{tensor},
        block_id_{block_id},
        layout_{layout},
        sign_{sign} {
          block_dims_ = tensor.block_dims(block_id);
          EXPECTS(tensor.rank() == block_id.size());
          EXPECTS(tensor.rank() == block_dims_.size());
          EXPECTS(tensor.rank() == layout.size());
          EXPECTS(size() > 0);
          buf_ = std::make_unique<T[]> (size());
        }

  /**
   * Block id of this block
   * @return This block's block id
   */
  const BlockDimVec& blockid() const {
    return block_id_;
  }

  /**
   * Offset of the block in the tensor
   * @return Block offset
   */
  BlockDimVec block_offset() const {
    BlockDimVec ret;
    const auto& drs = tensor_.dim_ranges();
    for(size_t i=0; i<tensor_.rank(); i++) {
      ret.push_back(drs[i].offset(block_id_[i]));
    }
    return ret;
  }

  /**
   * Dimensions of this block
   * @return Block dimensions
   */
  const BlockDimVec& block_dims() const {
    return block_dims_;
  }

  /**
   * Construct a labeled block from this block
   * @param label Label to be associated with this block
   * @return Constructed labeled block
   */
  LabeledBlock<T> operator () (const IndexLabelVec& label) {
    return {this, label};
  }
  
  /**
   * Construct a labeled block from this block, with default labels
   * @return Constructed labeled block
   */
  LabeledBlock<T> operator () () {
    IndexLabelVec label;
    for(int i=0; i<block_id_.size(); i++) {
      label.push_back({i, tensor_.flindices()[i]});
    }
    return operator ()(label);
  }
  
  /**
   * Number of elements in this block
   * @return Block size (in number of elements)
   */
  size_t size() const {
    size_t sz = 1;
    for(auto x : block_dims_) {
      sz *= x.value();
    }
    return sz;
  }

  /**
   * Sign prefactor accessor
   * @return Sign prefactor
   */
  Sign sign() const {
    return sign_;
  }

  /**
   * layout accessor
   * @return Sign prefactor
   */
  const PermVec& layout() const {
    return layout_;
  }

  /**
   * Get the buffer storing the block's data
   * @return Pointer to underlying buffer
   */
  T* buf() {
    return buf_.get();
  }

  /**
   * @copydoc Block::buf()
   */
  const T* buf() const {
    return buf_.get();
  }

  /**
   * Access to underlying tensor
   * @return Block's tensor
   */
  Tensor<T>& tensor() {
    return tensor_;
  }
  
 private:
  Tensor<T>& tensor_;
  BlockDimVec block_id_;
  BlockDimVec block_dims_;
  std::unique_ptr<T[]> buf_;
  PermVec layout_;
  Sign sign_;
};


}  // namespace tammy

#endif  // TAMMY_BLOCK_H_

