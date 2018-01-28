// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMY_MSO_H_
#define TAMMY_MSO_H_

#include <vector>
#include <numeric>

#include "index_space.h"
#include "types.h"
#include "errors.h"

namespace tammy {

class MSO : public IndexSpace {
 public:
  MSO() = default; //@todo delete after debugging
  MSO(BlockCount noa, BlockCount nob, BlockCount nva, BlockCount nvb,
     const StrongNumIndexedVector<Irrep,BlockIndex>& spatials,
     const StrongNumIndexedVector<Spin,BlockIndex>& spins,
      const StrongNumIndexedVector<Size,BlockIndex>& sizes)
      : noa_{noa},
        nob_{nob},
        nva_{nva},
        nvb_{nvb},
        spatials_{spatials},
        spins_{spins},
        sizes_(sizes) {
          // EXPECTS(noa_ >=0 && nva_ >=0);
          // EXPECTS(nob_ >=0 && nvb_ >= 0);
          BlockCount nov_ab = noa_ + nob_ + nva_ + nvb_;
          EXPECTS(spatials_.size() == nov_ab);
          EXPECTS(spins_.size() == nov_ab);
          EXPECTS(sizes_.size() == nov_ab);
          
          offsets_.push_back(Offset{0});
          std::partial_sum(sizes_.begin(), sizes_.end(), std::back_inserter(offsets_));
          for(auto s: spins_) {
            EXPECTS(s==Spin{1} || s==Spin{2});
          }
          for(BlockIndex i{0}; i<nov_ab; i++) {
            block_indices_.push_back(i);
          }
        }
           
  bool has_spatial() const override {
    return true;
  }
  Irrep spatial(BlockIndex bid) const override {
    return spatials_[bid];
  }
  bool has_spin() const override {
    return true;
  }
  Spin spin(BlockIndex bid) const override {
    return spins_[bid];
  }
  Size size(BlockIndex bid) const override {
    return sizes_[bid];
  }
  Offset offset(BlockIndex bid) const override {
    return offsets_[bid];
  }

  Iterator begin(RangeValue rv,
                 const BlockDimVec& bdv={}) const override {    
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
      case range_oa:
        off = BlockCount{0};
        break;
      case range_ob:
        off = noa_;
        break;
      case range_v:
        off = noa_ + nob_;
        break;
      case range_va:
        off = noa_ + nob_;
        break;
      case range_vb:
        off = noa_ + nob_ + nva_;
        break;
      case range_e:
        off = noa_ + nob_ + nva_ + nvb_;
        break;
      default:
        assert(0);
    }
    return ret + strongnum_cast<size_t>(off.value());
  }
  
  Iterator end(RangeValue rv,
               const BlockDimVec& bdv={}) const override {
    EXPECTS(bdv.size() == 0);
    auto ret = block_indices_.begin();
    BlockCount off;
    switch(rv) {
      case range_n:
        off = noa_ + nob_ + nva_ + nvb_;
        break;
      case range_o:
        off = noa_ + nob_;
        break;
      case range_oa:
        off = noa_;
        break;
      case range_ob:
        off = noa_ + nob_;
        break;
      case range_v:
        off = noa_ + nob_ + nva_ + nvb_;
        break;
      case range_va:
        off = noa_ + nob_ + nva_;
        break;
      case range_vb:
        off = noa_ + nob_ + nva_ + nvb_;
        break;
      case range_e:
        off = noa_ + nob_ + nva_ + nvb_;
        break;
      default:
        assert(0);
    }
    return ret + strongnum_cast<size_t>(off.value());
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
  IndexRange OaR() const {
    return {*this, range_oa};
  }
  IndexRange VaR() const {
    return {*this, range_va};
  }
  IndexRange ObR() const {
    return {*this, range_ob};
  }
  IndexRange VbR() const {
    return {*this, range_vb};
  }

  IndexLabel O(Label label) const {
    return {OR(),label};
  }
  IndexLabel V(Label label) const {
    return {VR(),label};
  }    
  IndexLabel Oa(Label label) const {
    return {OaR(),label};
  }
  IndexLabel Va(Label label) const {
    return {VaR(),label};
  }
  IndexLabel Ob(Label label) const {
    return {ObR(),label};
  }
  IndexLabel Vb(Label label) const {
    return {VbR(),label};
  }

  template<typename... LabelArgs>
  auto O(Label label, LabelArgs... labels) const {
    return std::make_tuple(O(label), O(labels)...);
  }
  template<typename... LabelArgs>
  auto V(Label label, LabelArgs... labels) const {
    return std::make_tuple(V(label), V(labels)...);
  }    
  template<typename... LabelArgs> auto
  Oa(Label label, LabelArgs... labels) const {
    return std::make_tuple(Oa(label), Oa(labels)...);
  }
  template<typename... LabelArgs> auto
  Va(Label label, LabelArgs... labels) const {
    return std::make_tuple(Va(label), Va(labels)...);
  }
  template<typename... LabelArgs> auto
  Ob(Label label, LabelArgs... labels) const {
    return std::make_tuple(Ob(label), Ob(labels)...);
  }
  template<typename... LabelArgs> auto
  Vb(Label label, LabelArgs... labels) const {
    return std::make_tuple(Vb(label), Vb(labels)...);
  }

 protected:
  BlockCount noa_;
  BlockCount nob_;
  BlockCount nva_;
  BlockCount nvb_;
  
  // std::vector<Irrep> spatials_;
  // std::vector<Spin> spins_;
  // std::vector<Size> sizes_;
  // std::vector<Offset> offsets_;
  StrongNumIndexedVector<Irrep,BlockIndex> spatials_;
  StrongNumIndexedVector<Spin,BlockIndex> spins_;
  StrongNumIndexedVector<Size,BlockIndex> sizes_;
  StrongNumIndexedVector<Offset,BlockIndex> offsets_;
  std::vector<BlockIndex> block_indices_;

  static const RangeValue range_e   = 0x00;
  static const RangeValue range_oa  = 0x01;
  static const RangeValue range_ob  = 0x02;
  static const RangeValue range_o   = 0x03;
  static const RangeValue range_va  = 0x10;
  static const RangeValue range_vb  = 0x20;
  static const RangeValue range_v   = 0x30;
  static const RangeValue range_n   = 0x33;
};  // MSO

}  // namespace tammy

#endif  // TAMMY_MSO_H_
