// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMY_TENSOR_BASE_H_
#define TAMMY_TENSOR_BASE_H_

#include <algorithm>
#include <numeric>

#include "tammy/boundvec.h"
#include "tammy/errors.h"
#include "tammy/types.h"
#include "tammy/index_space.h"
#include "tammy/perm_symmetry.h"

namespace tammy {

//@todo @bug initialize perm_group_
#if 0
#endif

class TensorBase {
 public:
  TensorRank rank_;
  TensorVec<IndexRange> dim_ranges_;
  TensorVec<IndexPosition> ipmask_;
  Irrep irrep_;
  Spin spin_total_;
  bool has_spatial_symmetry_;
  bool has_spin_symmetry_;
  TensorVec<SpinMask> spin_mask_;
  PermGroup perm_group_;
  bool spin_restricted_;

  TensorBase(const TensorVec<IndexRange>& dim_ranges,
             const TensorVec<IndexPosition>& ipmask,
             Irrep irrep = Irrep{0},
             Spin spin_total = Spin{0},
             bool spin_restricted = false)
      : dim_ranges_{dim_ranges},
        ipmask_{ipmask},
        irrep_{irrep},
        spin_total_{spin_total},
        has_spatial_symmetry_{true},
        has_spin_symmetry_{false},
        spin_restricted_{spin_restricted} {
    init_spatial_check();
    init_spin_check();
  }

  TensorBase(const std::tuple<
             TensorVec<IndexRange>,
             TensorVec<IndexRange>>& ranges,
             Irrep irrep = Irrep{0},
             Spin spin_total = Spin{0})
      : TensorBase{TensorBase::compute_index_range(ranges),
        compute_ipmask(ranges),
        irrep,
        spin_total} { }

  TensorBase(const std::tuple<
             TensorVec<IndexRange>,
             TensorVec<IndexRange>,
             TensorVec<IndexRange>>& ranges,
             Irrep irrep = Irrep{0},
             Spin spin_total = Spin{0})
      : TensorBase{compute_index_range(ranges),
        compute_ipmask(ranges),
        irrep,
        spin_total} { }

  virtual ~TensorBase() {}

   /**
   * Get the number of dimensions in this tensor.
   * @return This tensor's rank
   */
  TensorRank rank() const {
    return rank_;
  }

  /**
   * Access this tensor's irrep
   * @return this tensor's irrep
   */
  Irrep irrep() const {
    return irrep_;
  }

  const TensorVec<IndexRange>& dim_ranges() const {
    return dim_ranges_;
  }

  /**
   * Size of a block (in number of elements)
   * @param blockid Id of a block
   * @return The block's size
   *
   * @pre block.size() == rank()
   */
  size_t block_size(const BlockDimVec &blockid) const {
    auto blockdims = block_dims(blockid);
    auto ret = std::accumulate(blockdims.begin(), blockdims.end(),
                               BlockIndex{1}, std::multiplies<BlockIndex>());
    return ret.value();
  }

  /**
   * Dimensions of a block
   * @param blockid Id of a block
   * @return Number of elements in this block along each dimension
   *
   * @pre blockid.size() == rank()
   */
  BlockDimVec block_dims(const BlockDimVec &blockid) const {
    BlockDimVec ret;
    for(size_t i=0; i<rank_; i++) {
      ret.push_back(BlockIndex{dim_ranges_[i].size(blockid[i])});
    }
    return ret;
  }
  
  /**
   * @brief Offset of a block in this tensor along each dimension
   *
   * A tensor stores a subrange of the totral index range.
   * Offset returns the offset of a block with respect to the subrange stored in ths tensor.
   * Offset along an index dimension is specified as the sum of block index sizes of all tiles
   * preceding this block id in this tensor.
   *
   * @param blockid Id of a block
   * @return Number of elements in this block along each dimension
   *
   * @pre blockid.size() == rank()
   */
  BlockDimVec block_offset(const BlockDimVec &blockid) const {
    BlockDimVec ret;
    for(auto b : blockid) {
      //ret.push_back(BlockIndex{TCE::offset(b)});
      assert(0);
    }
    return ret;
  }

  /**
   * Number of blocks stored in this tensor along each dimension
   * @return Number of blocks along each dimension.
   *
   * @post return ret such that ret.size() == rank()
   */
  BlockDimVec num_blocks() const {
    BlockDimVec ret;
    for(auto dr: dim_ranges_) {
      BlockIndex blo = dr.blo();
      BlockIndex bhi = dr.bhi();
      ret.push_back(bhi - blo);
    }
    return ret;
  }

  bool spatial_nonzero(const BlockDimVec& bdv) const {
    if(!has_spatial_symmetry_)
      return true;

    Irrep rhs = Irrep{0};
    for(size_t i=0; i<bdv.size(); i++) {
      if(ipmask_[i] == IndexPosition::upper
         || ipmask_[i] == IndexPosition::lower) {
        rhs ^= dim_ranges_[i].is().spatial(bdv[i]);
      }
    }
    return rhs == irrep_;
  }

  bool spin_nonzero(const BlockDimVec& bdv) const {
    if(!has_spin_symmetry_)
      return true;

    Spin spin_upper=Spin{0}, spin_lower=Spin{0};
    for(size_t i=0; i<bdv.size(); i++) {
      switch(spin_mask_[i]) {
        case SpinMask::upper:
          spin_upper += dim_ranges_[i].is().spin(bdv[i]);
          break;
        case SpinMask::lower:
          spin_lower += dim_ranges_[i].is().spin(bdv[i]);
          break;
        case SpinMask::ignore:
          //no-op
          break;
        default:
          break;
      }
    }
    return spin_upper - spin_lower == spin_total_;
  }
  
  /**
   * Is a given block non-zero based on spin and spatial symmetry.
   * @param blockid Id of a block
   * @return true if it is non-zero based on spin and spatial symnmetry
   */
  bool nonzero(const BlockDimVec& blockid) const {
    return spin_nonzero(blockid) &&
        spatial_nonzero(blockid);
  }

  /**
   * @nrief Is a block unique based on spin symmetry.
   *
   * Spin non-unique blocks exists in the spin restricted case.
   * In particular, 𝛽𝛽|𝛽𝛽 blocks are non-unique in the spin-restricted case.
   * @param blockid
   * @return true if @param blockid is spin-unique.
   */
  bool spin_unique(const BlockDimVec& blockid) const {
    if(spin_restricted_ == false) {
      return true;
    }
    Spin spin {0};
    for(size_t i=0; i<rank(); i++) {
      spin += dim_ranges_[i].is().spin(blockid[i]);
    }
    return spin != 2 * rank();
  }
  
  /**
   * @brief Find the unique block if corresponding to given block id based on spin symmetry.
   *
   * If a block is spin unique, the same block id is returned.
   * If not, the spin-unique block id is returned.
   * @param blockid Id of a block
   * @return Id of block equal to @param blockid but also spin-unique.
   */
  BlockDimVec find_spin_unique_block(const BlockDimVec& blockid) const {
    if(!spin_restricted_) {
      return blockid;
    }
    bool all_beta = true;
    for(size_t i=0; i<rank(); i++) {
      all_beta = all_beta & (dim_ranges_[i].is().spin(blockid[i]) == SpinType::beta);
    }
    if(all_beta) {
      assert(0); //@todo @bug implement
#if 0
#endif
    }
  }

  /**
   * Find the unique block corresponding to a given block based on permuutation symmetry.
   * @param blockid Id of block
   * @return Id of block equivalent to @param blockid in terms of permutation symmetry
   *
   * @pre blockid.size() == rank()
   * @post return ret such that ret.size() == blockid.size()
   */
  BlockDimVec find_unique_block(const BlockDimVec& blockid) const {
    return std::get<1>(perm_group_.find_unique(blockid));
  }

  
  std::unique_ptr<Generator<BlockIndex>>
  unique_generator() const {
    BlockDimVec lo, hi;
    for(const auto& dr: dim_ranges_)  {
      lo.push_back(dr.blo());
      hi.push_back(dr.bhi());
    }
    return perm_group_.unique_generator<BlockIndex>(lo, hi);
  }
  
 private:
  void init_spatial_check() {
    for(size_t i=0; i<dim_ranges_.size(); i++) {
      if(ipmask_[i] == IndexPosition::upper
         || ipmask_[i] == IndexPosition::lower) {
        if(!dim_ranges_[i].is().has_spatial()) {
          has_spatial_symmetry_ = false;
        }
      }
    }
  }

  void init_spin_check() {
    spin_mask_.clear();
    for(const auto ip : ipmask_) {
      switch(ip) {
        case IndexPosition::upper:
          spin_mask_.push_back(SpinMask::upper);
          break;
        case IndexPosition::lower:
          spin_mask_.push_back(SpinMask::lower);
          break;
        case IndexPosition::neither:
          spin_mask_.push_back(SpinMask::ignore);
          break;
        default:
          break;
      }
    }
    auto u = std::find(ipmask_.begin(), ipmask_.end(), IndexPosition::upper);
    auto l = std::find(ipmask_.begin(), ipmask_.end(), IndexPosition::lower);
    while(u != ipmask_.end() && l != ipmask_.end()) {
      auto upos = u - ipmask_.begin();
      auto lpos = l - ipmask_.begin();
      if(!dim_ranges_[upos].is().has_spin() ||
         !dim_ranges_[lpos].is().has_spin()) {
        spin_mask_[upos] = SpinMask::ignore;
        spin_mask_[lpos] = SpinMask::ignore;
      }
      u = std::find(u+1, ipmask_.end(), IndexPosition::upper);
      l = std::find(l+1, ipmask_.end(), IndexPosition::lower);
    }
  }

  static TensorVec<IndexRange>
  compute_index_range(const std::tuple<
                      TensorVec<IndexRange>,
                      TensorVec<IndexRange>,
                      TensorVec<IndexRange>>& ranges) {
    TensorVec<IndexRange> ret;
    std::copy(std::get<0>(ranges).begin(),
              std::get<0>(ranges).end(),
              std::back_inserter(ret));
    std::copy(std::get<1>(ranges).begin(),
              std::get<1>(ranges).end(),
              std::back_inserter(ret));
    std::copy(std::get<2>(ranges).begin(),
              std::get<2>(ranges).end(),
              std::back_inserter(ret));
    return ret;
  }

  static TensorVec<IndexRange>
  compute_index_range(const std::tuple<
                      TensorVec<IndexRange>,
                      TensorVec<IndexRange>>& ranges) {
    return compute_index_range(std::make_tuple(
        TensorVec<IndexRange>{},
        std::get<0>(ranges),
        std::get<1>(ranges)));
  }

  static TensorVec<IndexPosition>
  compute_ipmask(const std::tuple<
                 TensorVec<IndexRange>,
                 TensorVec<IndexRange>,
                 TensorVec<IndexRange>>& ranges) {
    TensorVec<IndexPosition> ret;
    std::fill_n(std::back_inserter(ret),
                std::get<0>(ranges).size(),
                IndexPosition::neither);
    std::fill_n(std::back_inserter(ret),
                std::get<1>(ranges).size(),
                IndexPosition::upper);
    std::fill_n(std::back_inserter(ret),
                std::get<2>(ranges).size(),
                IndexPosition::lower);
    return ret;
  }

  static TensorVec<IndexPosition>
  compute_ipmask(const std::tuple<
                 TensorVec<IndexRange>,
                 TensorVec<IndexRange>>& ranges) {
    return compute_ipmask(std::make_tuple(
        TensorVec<IndexRange>{},
        std::get<0>(ranges),
        std::get<1>(ranges)));
  }
};  // TensorBase


}  // namespace tammy

#endif  // TAMMY_TENSOR_BASE_H_
