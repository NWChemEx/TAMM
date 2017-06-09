#ifndef TAMMX_TENSOR_BASE_H_
#define TAMMX_TENSOR_BASE_H_

namespace tammx {

// @todo For now, we cannot handle tensors in which number of upper
// and lower indices differ by more than one. This relates to
// correctly determining spin symmetry.

class TensorBase {
 public:
  TensorBase(const TensorVec<SymmGroup> &indices,
                  TensorRank nupper_indices,
                  Irrep irrep,
                  bool spin_restricted)
      : indices_{indices},
        nupper_indices_{nupper_indices},
        irrep_{irrep},
        spin_restricted_{spin_restricted} {
          for(auto sg : indices) {
            Expects(sg.size()>0);
            auto dim = sg[0];
            for(auto d : sg) {
              Expects (dim == d);
            }
          }
          rank_ = 0;
          for(auto sg : indices) {
            rank_ += sg.size();
          }
          flindices_ = flatten(indices_);
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

  const TensorVec<SymmGroup>& indices() const {
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
        spatial_nonzero(blockid) &&
        spin_restricted_nonzero(blockid);
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

  bool spin_restricted_nonzero(const TensorIndex& blockid) const {
    Spin spin {std::abs(rank_ - 2 * nupper_indices_)};
    TensorRank rank_even = rank_ + (rank_ % 2);
    for(auto b : blockid) {
      spin += TCE::spin(b);
    }
    return (!spin_restricted_ || (rank_ == 0) || (spin != 2 * rank_even));
  }

 private:
  const TensorVec<SymmGroup> indices_;
  TensorRank nupper_indices_;
  Irrep irrep_;
  bool spin_restricted_;
  TensorDim flindices_;
  TensorRank rank_;
};  // class TensorBase

inline bool
operator <= (const TensorBase& lhs, const TensorBase& rhs) {
  return (lhs.indices() <= rhs.indices())
      && (lhs.nupper_indices() <= rhs.nupper_indices())
      && (lhs.irrep() < rhs.irrep())
      && (lhs.spin_restricted () < rhs.spin_restricted());
}

inline bool
operator == (const TensorBase& lhs, const TensorBase& rhs) {
  return (lhs.indices() == rhs.indices())
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

inline ProductIterator<TriangleLoop>
loop_iterator(const TensorVec<SymmGroup>& indices ) {
  TensorVec<TriangleLoop> tloops, tloops_last;
  for(auto &sg: indices) {
    BlockDim lo, hi;
    std::tie(lo, hi) = tensor_index_range(sg[0]);
    tloops.push_back(TriangleLoop{sg.size(), lo, hi});
    tloops_last.push_back(tloops.back().get_end());
  }
  //FIXME:Handle Scalar
  if(indices.size()==0){
    tloops.push_back(TriangleLoop{});
    tloops_last.push_back(tloops.back().get_end());
  }
  return ProductIterator<TriangleLoop>(tloops, tloops_last);
}

}  // namespace tammx

#endif  // TAMMX_TENSOR_BASE_H_
