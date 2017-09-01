#ifndef TAMMX_TENSOR_BASE_H_
#define TAMMX_TENSOR_BASE_H_

#include "tammx/types.h"

namespace tammx {

// @todo For now, we cannot handle tensors in which number of upper
// and lower indices differ by more than one. This relates to
// correctly handling spin symmetry.

// @fixme SymmGroup has a different connotation. Change name

class TensorBase {
 public:
  // TensorBase(const TensorVec<SymmGroup> &indices,
  //                 TensorRank nupper_indices,
  //                 Irrep irrep,
  //                 bool spin_restricted)
  //     : TensorBase{to_tsymm_indices(indices),
  //       nupper_indices,
  //       irrep,
  //       spin_restricted} {}

    TensorBase(const TensorVec<TensorSymmGroup>& indices,
                  TensorRank nupper_indices,
                  Irrep irrep,
                  bool spin_restricted)
      : indices_{indices},
        nupper_indices_{nupper_indices},
        irrep_{irrep},
        spin_restricted_{spin_restricted} {
          for(auto sg : indices) {
            Expects(sg.size()>0);
          }
          rank_ = 0;
          for(auto sg : indices) {
            rank_ += sg.size();
          }
          flindices_ = flatten(indices_);
          Expects(std::abs(rank_ - 2*nupper_indices_) <= 1);
        }

  virtual ~TensorBase() {}

  TensorRank rank() const {
    return rank_;
  }

  TensorDim flindices() const {
    return flindices_;
  }

  Irrep irrep() const {
    return irrep_;
  }

  bool spin_restricted() const {
    return spin_restricted_;
  }

  // TensorVec<SymmGroup> indices() const {
  //   TensorVec<SymmGroup> ret;
  //   for(const auto& tid: indices_) {
  //     ret.push_back(SymmGroup(tid.size(), tid.dt()));
  //   }
  //   return ret;
  // }

  TensorVec<TensorSymmGroup> tindices() const {
    return indices_;
  }

  TensorRank nupper_indices() const {
    return nupper_indices_;
  }

  size_t block_size(const TensorIndex &blockid) const {
    auto blockdims = block_dims(blockid);
    auto ret = std::accumulate(blockdims.begin(), blockdims.end(), BlockDim{1}, std::multiplies<BlockDim>());
    return ret.value();
  }

  TensorIndex block_dims(const TensorIndex &blockid) const {
    TensorIndex ret;
    for(auto b : blockid) {
      ret.push_back(BlockDim{TCE::size(b)});
    }
    return ret;
  }

  TensorIndex block_offset(const TensorIndex &blockid) const {
    TensorIndex ret;
    for(auto b : blockid) {
      ret.push_back(BlockDim{TCE::offset(b)});
    }
    return ret;
  }

  TensorIndex num_blocks() const {
    TensorIndex ret;
    for(auto i: flindices_) {
      BlockDim lo, hi;
      std::tie(lo, hi) = tensor_index_range(i);
      ret.push_back(hi - lo);
    }
    return ret;
  }

  bool nonzero(const TensorIndex& blockid) const {
    return spin_nonzero(blockid) &&
        spatial_nonzero(blockid);
  }

  bool spin_unique(const TensorIndex& blockid) const {
    if(spin_restricted_ == false) {
      return true;
    }
    Spin spin {0};
    for(auto b : blockid) {
      spin += TCE::spin(b);
    }
    return spin != 2 * rank();
  }

  TensorIndex find_spin_unique_block(const TensorIndex& blockid) const {
    if(spin_unique(blockid)) {
      return blockid;
    }
    TensorIndex ret;
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
  
  TensorIndex find_unique_block(const TensorIndex& blockid) const {
    TensorIndex ret {blockid};
    int pos = 0;
    for(auto &igrp: indices_) {
      std::sort(ret.begin()+pos, ret.begin()+pos+igrp.size());
      pos += igrp.size();
    }
    return ret;
  }

  /**
   * @todo Why can't this logic use perm_count_inversions?
   */
  std::pair<TensorPerm,Sign> compute_sign_from_unique_block(const TensorIndex& blockid) const {
    Expects(blockid.size() == rank());
    TensorPerm ret_perm(blockid.size());
    std::iota(ret_perm.begin(), ret_perm.end(), 0);
    int num_inversions=0;
    int pos = 0;
    for(auto &igrp: indices_) {
      Expects(igrp.size() <= 2); // @todo Implement general algorithm
      if(igrp.size() == 2 && blockid[pos+0] > blockid[pos+1]) {
        num_inversions += 1;
        std::swap(ret_perm[pos], ret_perm[pos+1]);
      }
      pos += igrp.size();
    }
    return {ret_perm, (num_inversions%2) ? -1 : 1};
  }

  bool spin_nonzero(const TensorIndex& blockid) const {
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

  bool spatial_nonzero(const TensorIndex& blockid) const {
    Irrep spatial {0};
    for(auto b : blockid) {
      spatial ^= TCE::spatial(b);
    }
    return spatial == irrep_;
  }

  // @todo @fixme Can this function be deleted?
  bool spin_restricted_nonzero(const TensorIndex& blockid) const {
    Spin spin {std::abs(rank_ - 2 * nupper_indices_)};
    TensorRank rank_even = rank_ + (rank_ % 2);
    for(auto b : blockid) {
      spin += TCE::spin(b);
    }
    return (!spin_restricted_ || (rank_ == 0) || (spin != 2 * rank_even));
  }

 private:

  static TensorDim flatten(const TensorVec<TensorSymmGroup>& indices) {
    TensorDim ret;
    for(const auto& tsg: indices) {
      for(size_t i=0; i<tsg.size(); i++) {
        ret.push_back(tsg.dt());
      }
    }
    return ret;
  }

  // static TensorVec<TensorSymmGroup> to_tsymm_indices(const TensorVec<SymmGroup>& indices) {
  //   TensorVec<TensorSymmGroup> ret;
  //   for(auto sg : indices) {
  //     Expects(sg.size()>0);
  //     auto dim = sg[0];
  //     for(auto d : sg) {
  //       Expects(d == dim);
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
  TensorDim flindices_;
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
inline ProductIterator<TriangleLoop>
loop_iterator(const TensorVec<SymmGroup>& indices ) {
  TensorVec<TriangleLoop> tloops, tloops_last;
  for(auto &sg: indices) {
    size_t i=1;
    for(; i<sg.size() && sg[i]==sg[0]; i++) { }    
    BlockDim lo, hi;
    std::tie(lo, hi) = tensor_index_range(sg[0]);
    tloops.push_back(TriangleLoop{i, lo, hi});
    tloops_last.push_back(tloops.back().get_end());

    if(i<sg.size()) {
      auto ii = i+1;
      for(; ii<sg.size() && sg[ii]==sg[i]; ii++) { }
      std::tie(lo, hi) = tensor_index_range(sg[i]);
      tloops.push_back(TriangleLoop{ii-i, lo, hi});
      tloops_last.push_back(tloops.back().get_end());
      Expects(ii == sg.size());
    }
  }
  //FIXME:Handle Scalar
  if(indices.size()==0){
    tloops.push_back(TriangleLoop{});
    tloops_last.push_back(tloops.back().get_end());
  }
  return ProductIterator<TriangleLoop>(tloops, tloops_last);
}

inline ProductIterator<TriangleLoop>
loop_iterator(const TensorVec<TensorSymmGroup>& tindices ) {
  TensorVec<TriangleLoop> tloops, tloops_last;
  for(const auto &sg: tindices) {
    BlockDim lo, hi;
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
