#include "tammx/block.h"
#include "tammx/labeled-block.h"

namespace tammx {

Block::Block(Tensor &tensor,
             const TensorIndex& block_id,
             const TensorIndex& block_dims,
             const TensorPerm& layout,
             Sign sign)
    : tensor_{tensor},
      block_id_{block_id},
      block_dims_{block_dims},
      layout_{layout},
      sign_{sign} {
        Expects(tensor.rank() == block_id.size());
        Expects(tensor.rank() == block_dims.size());
        Expects(tensor.rank() == layout.size());
        buf_ = std::make_unique<uint8_t []> (size() * tensor.element_size());
      }

Block::Block(Tensor &tensor,
             const TensorIndex& block_id)
    : tensor_{tensor},
      block_id_{block_id} {
        block_dims_ = tensor.block_dims(block_id);
        layout_.resize(tensor.rank());
        std::iota(layout_.begin(), layout_.end(), 0);
        sign_ = 1;
        buf_ = std::make_unique<uint8_t []> (size() * tensor.element_size());
      }

LabeledBlock
Block::operator () (const TensorLabel &label) {
  return LabeledBlock{this, label};
}

LabeledBlock
Block::operator () () {
  TensorLabel label; //(block_id_.size());
  for(int i=0; i<label.size(); i++) {
    label.push_back({i, tensor_.flindices()[i]});
  }
  //std::iota(label.begin(), label.end(), 0);
  return operator ()(label); //LabeledBlock{*this, label};
}

}  // namespace tammx
