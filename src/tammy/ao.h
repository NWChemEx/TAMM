// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMY_AO_H_
#define TAMMY_AO_H_

#include <vector>
#include <numeric>

#include "index_space.h"
#include "types.h"
#include "errors.h"
#include "strong_num_indexed_vector.h"

namespace tammy {

class AO : public IndexSpace {
 public:
  AO() = default; //@todo delete after debugging
  AO(BlockCount no, BlockCount nv, 
     const std::vector<Size>& sizes)
      : no_{no},
        nv_{nv},
        sizes_{sizes} {
          // EXPECTS(noa_ >=0 && nva_ >=0);
          // EXPECTS(nob_ >=0 && nvb_ >= 0);
          BlockCount nov = no_ + nv_;
          EXPECTS(sizes_.size() == nov);          
          offsets_.emplace_back(0);
          std::partial_sum(sizes_.begin(), sizes_.end(), std::back_inserter(offsets_));
          for(BlockIndex i{0}; i<nov; i++) {
            block_indices_.push_back({i});
          }
        }
  Size size(BlockIndex bid) const override {
    return sizes_[bid];
  }
  Offset offset(BlockIndex bid) const override {
    return offsets_[bid];
  }
  int num_indep_indices() const override {
    return 0;
  }

  Iterator begin(RangeValue rv,
                 const TensorVec<IndexSpace::Iterator>& bdv={}) const override {
    EXPECTS(bdv.size() == 0);
    auto ret = block_indices_.begin();
    BlockCount off;
    switch(rv) {
      case range_n:
        off = BlockCount{0};
        break;
      case range_o:
        off = BlockCount{0};
        break;
      case range_v:
        off = no_;
        break;
      case range_e:
        off = no_ + nv_;
        break;
      default:
        UNREACHABLE();
    }
    using size_type = decltype(block_indices_)::size_type;
    return ret + off.template value<size_type>();
  }

  Iterator end(RangeValue rv,
               const TensorVec<IndexSpace::Iterator>& bdv={}) const override {
    EXPECTS(bdv.size() == 0);
    auto ret = block_indices_.begin();
    BlockCount off;
    switch(rv) {
      case range_n:
        off = no_ + nv_;
        break;
      case range_o:
        off = no_;
        break;
      case range_v:
        off = no_ + nv_;
        break;
      case range_e:
        off = no_ + nv_;
        break;
      default:
        UNREACHABLE();
    }
    using size_type = decltype(block_indices_)::size_type;
    return ret + off.template value<size_type>();
  }    

  IndexRange NR() const override {
    return {*this, range_n};
  }
  
  IndexRange ER() const override {
    return {*this, range_e};
  }

  IndexRange OR() const {
    return {*this, range_o};
  }
  
  IndexRange VR() const {
    return {*this, range_v};
  }

  IndexLabel O(Label label) const {
    return {OR(), label};
  }
  
  IndexLabel V(Label label) const {
    return {VR(), label};
  }

  template<typename... LabelArgs>
  auto O(LabelArgs... labels) const {
    return OR().labels(labels...);
  }

  template<typename... LabelArgs>
  auto V(LabelArgs... labels) const {
    return VR().labels(labels...);
  }

 protected:
  BlockCount no_;
  BlockCount nv_;  
  std::vector<BlockIndex> block_indices_;
  StrongNumIndexedVector<Size,BlockIndex> sizes_;
  StrongNumIndexedVector<Offset,BlockIndex> offsets_;

  static const RangeValue range_e = 0x0;
  static const RangeValue range_o = 0x1;
  static const RangeValue range_v = 0x2;
  static const RangeValue range_n = 0x3;
};  // AO


}  // namespace tammy

#endif  // TAMMY_AO_H_
