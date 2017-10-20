#ifndef TAMMX_TENSOR_BASE_H_
#define TAMMX_TENSOR_BASE_H_

#include "tammx/errors.h"
#include "tammx/types.h"

namespace tammx {

/**
 * @brief Base class for tensors.
 *
 * This class handles the indexing logic for tensors. Memory management is done by subclasses.
 * The class supports MO indices that are permutation symmetric with anti-symmetry.
 *
 * @note In a spin-restricted tensor, a 𝛽𝛽|𝛽𝛽 block is mapped to its corresponding to αα|αα block.
 *
 * @todo For now, we cannot handle tensors in which number of upper
 * and lower indices differ by more than one. This relates to
 * correctly handling spin symmetry.
 * @todo SymmGroup has a different connotation. Change name
 *
 */
class TensorBase {
 public:
  /**
   * Constructor.
   * @param indices Vector of groups of indices. Each group corresponds to a permutation symmetric group.
   * @param nupper_indices Number of upper indices
   * @param irrep Irrep of this tensor, to determine spatial symmetry
   * @param spin_restricted Is this tensor spin restricted (as in NWChem)
   *
   * @pre 0<=i<indices.size(): indices[i].size() > 0
   * @pre rank = sum(0<=i<indices.size(): indices[i].size())
   * @pre rank-1 <= 2*nupper_indices <= rank+1
   */
  TensorBase(const TensorVec<TensorSymmGroup>& indices,
             TensorRank nupper_indices,
             Irrep irrep,
             bool spin_restricted)
      : indices_{indices},
        nupper_indices_{nupper_indices},
        irrep_{irrep},
        spin_restricted_{spin_restricted} {
          for(auto sg : indices) {
            EXPECTS(sg.size()>0);
          }
          rank_ = 0;
          for(auto sg : indices) {
            rank_ += sg.size();
          }
          flindices_ = flatten(indices_);
          TensorRank echeck = rank_ > 2*nupper_indices_ ? (rank_ - 2*nupper_indices_) : (2*nupper_indices_-rank_);
          //EXPECTS(std::abs(rank_ - 2*nupper_indices_) <= 1);
          EXPECTS(echeck <= 1);
        }

  virtual ~TensorBase() {}

  /**
   * Get the number of dimensions in this tensor.
   * @return This tensor's rank
   */
  TensorRank rank() const {
    return rank_;
  }

  /**
   * @brief A flattened list of indices.
   * This is derived from the symmetric list of indices in the contructor by explicitly repeating the indices.
   * @return ret such that ret.size() == rank
   */
  RangeTypeVec flindices() const {
    return flindices_;
  }

  /**
   * Access this tensor's irrep
   * @return this tensor's irrep
   */
  Irrep irrep() const {
    return irrep_;
  }

  /**
   * Is this tensor spin restricted?
   * @return true if this tensor is spin restricted
   */
  bool spin_restricted() const {
    return spin_restricted_;
  }

  /**
   * Indices (with permutation symmetry groups) given to the constructor.
   * @return Tensor's indices with permutation symmetry
   */
  TensorVec<TensorSymmGroup> tindices() const {
    return indices_;
  }

  /**
   * Number of upper indices
   * @return Number of upper indices
   */
  TensorRank nupper_indices() const {
    return nupper_indices_;
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
    auto ret = std::accumulate(blockdims.begin(), blockdims.end(), BlockIndex{1}, std::multiplies<BlockIndex>());
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
    for(auto b : blockid) {
      ret.push_back(BlockIndex{TCE::size(b)});
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
      ret.push_back(BlockIndex{TCE::offset(b)});
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
    for(auto i: flindices_) {
      BlockIndex lo, hi;
      std::tie(lo, hi) = tensor_index_range(i);
      ret.push_back(hi - lo);
    }
    return ret;
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
    for(auto b : blockid) {
      spin += TCE::spin(b);
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
    if(spin_unique(blockid)) {
      return blockid;
    }
    BlockDimVec ret;
    for(auto b : blockid) {
      if(b > TCE::noab() + TCE::nva()) {
        b -= TCE::nva();
      } else if(b > TCE::noa() && b <= TCE::noab()) {
        b -= TCE::noa();
      }
      ret.push_back(b);
    }
    return ret;
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
    BlockDimVec ret {blockid};
    int pos = 0;
    for(auto &igrp: indices_) {
      std::sort(ret.begin()+pos, ret.begin()+pos+igrp.size());
      pos += igrp.size();
    }
    return ret;
  }

  /**
   * @brief Copmute the sign change (due to anti-symmetry) involved in computing this @param blockid from its unique blockid.
   * @param blockid Input blockid
   * @return Sign prefactor due to permutation anti-symmetry
   * @todo Why can't this logic use perm_count_inversions?
   */
  std::pair<PermVec,Sign> compute_sign_from_unique_block(const BlockDimVec& blockid) const {
    EXPECTS(blockid.size() == rank());
    PermVec ret_perm(blockid.size());
    std::iota(ret_perm.begin(), ret_perm.end(), 0);
    int num_inversions=0;
    int pos = 0;
    for(auto &igrp: indices_) {
      EXPECTS(igrp.size() <= 2); // @todo Implement general algorithm
      if(igrp.size() == 2 && blockid[pos+0] > blockid[pos+1]) {
        num_inversions += 1;
        std::swap(ret_perm[pos], ret_perm[pos+1]);
      }
      pos += igrp.size();
    }
    return {ret_perm, (num_inversions%2) ? -1 : 1};
  }

  /**
   * Is the given blockid non-zero due to spin
   * @param blockid Given blockid
   * @return Is it zero due to spin?
   */
  bool spin_nonzero(const BlockDimVec& blockid) const {
    Spin spin_upper {0};
    for(auto itr = std::begin(blockid); itr!= std::begin(blockid) + nupper_indices_; ++itr) {
      spin_upper += TCE::spin(*itr);
    }
    Spin spin_lower {0};
    for(auto itr = std::begin(blockid)+nupper_indices_; itr!= std::end(blockid); ++itr) {
      spin_lower += TCE::spin(*itr);
    }
    return spin_lower - spin_upper == rank_ - 2 * nupper_indices_;
  }

  /**
   * Is the given blockid non-zero due to spatial symmetry
   * @param blockid Given blockid
   * @return Is @param blockid zero due to spatial symmetry?
   */
  bool spatial_nonzero(const BlockDimVec& blockid) const {
    Irrep spatial {0};
    for(auto b : blockid) {
      spatial ^= TCE::spatial(b);
    }
    return spatial == irrep_;
  }

  /**
   * Is the given block zero to spin-restricted spin symmetry
   * @param blockid Given blockid
   * @return true if @param blockid is non-zero due to spin retriction
   * @todo Can this function be deleted?
   * @bug A block is only unique or not (due to spin restriction). It is not zero.
   */
  bool spin_restricted_nonzero(const BlockDimVec& blockid) const {
    TensorRank echeck = rank_ > 2*nupper_indices_ ? (rank_ - 2*nupper_indices_) : (2*nupper_indices_-rank_);
    Spin spin {echeck};
    //Spin spin {std::abs(rank_ - 2 * nupper_indices_)};
    TensorRank rank_even = rank_ + (rank_ % 2);
    for(auto b : blockid) {
      spin += TCE::spin(b);
    }
    return (!spin_restricted_ || (rank_ == 0) || (spin != 2 * rank_even));
  }

 private:

  static RangeTypeVec flatten(const TensorVec<TensorSymmGroup>& indices) {
    RangeTypeVec ret;
    for(const auto& tsg: indices) {
      for(size_t i=0; i<tsg.size(); i++) {
        ret.push_back(tsg.rt());
      }
    }
    return ret;
  }

  // static TensorVec<TensorSymmGroup> to_tsymm_indices(const TensorVec<SymmGroup>& indices) {
  //   TensorVec<TensorSymmGroup> ret;
  //   for(auto sg : indices) {
  //     EXPECTS(sg.size()>0);
  //     auto dim = sg[0];
  //     for(auto d : sg) {
  //       EXPECTS(d == dim);
  //     }
  //     ret.push_back(TensorSymmGroup{RangeType{dim}, sg.size()});
  //   }
  //   return ret;
  // }

  //const TensorVec<SymmGroup> indices_;
  const TensorVec<TensorSymmGroup> indices_;
  TensorRank nupper_indices_;
  Irrep irrep_;
  bool spin_restricted_;
  RangeTypeVec flindices_;
  TensorRank rank_;
};  // class TensorBase

inline bool
operator <= (const TensorBase& lhs, const TensorBase& rhs) {
  return (lhs.tindices() <= rhs.tindices())
      && (lhs.nupper_indices() <= rhs.nupper_indices())
      && (lhs.irrep() < rhs.irrep())
      && (lhs.spin_restricted () < rhs.spin_restricted());
}

inline bool
operator == (const TensorBase& lhs, const TensorBase& rhs) {
  return (lhs.tindices() == rhs.tindices())
      && (lhs.nupper_indices() == rhs.nupper_indices())
      && (lhs.irrep() < rhs.irrep())
      && (lhs.spin_restricted () < rhs.spin_restricted());
}

inline bool
operator != (const TensorBase& lhs, const TensorBase& rhs) {
  return !(lhs == rhs);
}

inline bool
operator < (const TensorBase& lhs, const TensorBase& rhs) {
  return (lhs <= rhs) && (lhs != rhs);
}

// @todo Does not work for arbitrary number of dimensions. 
// inline ProductIterator<TriangleLoop>
// loop_iterator(const TensorVec<SymmGroup>& indices ) {
//   TensorVec<TriangleLoop> tloops, tloops_last;
//   for(auto &sg: indices) {
//     size_t i=1;
//     for(; i<sg.size() && sg[i]==sg[0]; i++) { }    
//     BlockIndex lo, hi;
//     std::tie(lo, hi) = tensor_index_range(sg[0]);
//     tloops.push_back(TriangleLoop{i, lo, hi});
//     tloops_last.push_back(tloops.back().get_end());

//     if(i<sg.size()) {
//       auto ii = i+1;
//       for(; ii<sg.size() && sg[ii]==sg[i]; ii++) { }
//       std::tie(lo, hi) = tensor_index_range(sg[i]);
//       tloops.push_back(TriangleLoop{ii-i, lo, hi});
//       tloops_last.push_back(tloops.back().get_end());
//       EXPECTS(ii == sg.size());
//     }
//   }
//   //FIXME:Handle Scalar
//   if(indices.size()==0){
//     tloops.push_back(TriangleLoop{});
//     tloops_last.push_back(tloops.back().get_end());
//   }
//   return ProductIterator<TriangleLoop>(tloops, tloops_last);
// }

inline ProductIterator<TriangleLoop>
loop_iterator(const TensorVec<TensorSymmGroup>& tindices ) {
  TensorVec<TriangleLoop> tloops, tloops_last;
  for(const auto &sg: tindices) {
    BlockIndex lo, hi;
    std::tie(lo, hi) = tensor_index_range(sg.rt());
    tloops.push_back(TriangleLoop{sg.size(), lo, hi});
    tloops_last.push_back(tloops.back().get_end());
  }
  //FIXME:Handle Scalar
  if(tindices.size()==0){
    tloops.push_back(TriangleLoop{});
    tloops_last.push_back(tloops.back().get_end());
  }
  return ProductIterator<TriangleLoop>(tloops, tloops_last);
}

}  // namespace tammx

#endif  // TAMMX_TENSOR_BASE_H_
