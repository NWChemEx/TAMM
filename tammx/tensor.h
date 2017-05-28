#ifndef TAMMX_TENSOR_H_
#define TAMMX_TENSOR_H_

#include "tammx/types.h"
#include "tammx/tce.h"
#include "tammx/triangle_loop.h"
#include "tammx/product_iterator.h"
#include "tammx/util.h"
#include "tammx/block.h"

namespace tammx {

class LabeledTensor;

class Tensor {
 public:
  enum class Distribution { tce_nw, tce_nwma, tce_nwi };
  enum class AllocationPolicy { none, create, attach };

  // @todo For now, we cannot handle tensors in which number of upper
  // and lower indices differ by more than one. This relates to
  // correctly determining spin symmetry.
  Tensor(const TensorVec<SymmGroup> &indices,
         ElementType element_type,
         Distribution distribution,
         TensorRank nupper_indices,
         Irrep irrep,
         bool spin_restricted)
      : indices_{indices},
        element_type_{element_type},
        distribution_{distribution},
        nupper_indices_{nupper_indices},
        irrep_{irrep},
        spin_restricted_{spin_restricted},
        constructed_{false},
        policy_{AllocationPolicy::none} {
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
          //Expects(std::abs(flindices_.size() - nupper_indices_) <= 1);
        }

  TensorRank rank() const {
    return rank_;
  }

  ElementType element_type() const {
    return element_type_;
  }

  TensorDim flindices() const {
    return flindices_;
  }

  size_t element_size() const {
    return tammx::element_size(element_type_);
  }

  Distribution distribution() const {
    return distribution_;
  }

  Irrep irrep() const {
    return irrep_;
  }
  
  bool spin_restricted() const {
    return spin_restricted_;
  }
  
  AllocationPolicy allocation_policy() const {
    return policy_;
  }

  const TensorVec<SymmGroup>& indices() const {
    return indices_;
  }

  TensorRank nupper_indices() const {
    return nupper_indices_;
  }
  
  bool constructed() const {
    return constructed_;
  }

  bool allocated() const {
    return constructed_ && policy_ == AllocationPolicy::create;
  }

  bool attached() const {
    return constructed_ && policy_ == AllocationPolicy::attach;
  }

#if 0
  void attach(tamm::gmem::Handle tce_ga, TCE::Int *tce_hash) {
    Expects (!constructed_);
    Expects (distribution_ == Distribution::tce_nwma
             || distribution_ == Distribution::tce_nwi);
    tce_ga_ = tce_ga;
    tce_hash_ = tce_hash;
    constructed_ = true;
    policy_ = AllocationPolicy::attach;
  }
#endif
  
  void attach(uint8_t* tce_data_buf, TCE::Int* tce_hash) {
    Expects (!constructed_);
    Expects (distribution_ == Distribution::tce_nwma);
    tce_data_buf_ = tce_data_buf;
    tce_hash_ = tce_hash;
    constructed_ = true;
    policy_ = AllocationPolicy::attach;    
  }

  void allocate();
  
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
  
  void destruct() {
    if(!constructed_) {
      return;
    }
    if (policy_ == AllocationPolicy::attach) {
      // no-op
    }
    else if (policy_ == AllocationPolicy::create) {
      if (distribution_ == Distribution::tce_nw || distribution_ == Distribution::tce_nwi) {
#if 0
        tamm::gmem::destroy(tce_ga_);
#else
        assert(0);
#endif
        delete [] tce_hash_;
      }
      else if (distribution_ == Distribution::tce_nwma) {
        delete [] tce_data_buf_;
        delete [] tce_hash_;
      }
    }
    constructed_ = false;
  }

  ~Tensor() {
    Expects(!constructed_);
  }

  template<typename T>
  void init(T value);
  
  bool nonzero(const TensorIndex& blockid) const {
    return spin_nonzero(blockid) &&
        spatial_nonzero(blockid) &&
        spin_restricted_nonzero(blockid);
  }

  Block get(const TensorIndex& blockid) {
    Expects(constructed_);
    Expects(nonzero(blockid));
    auto uniq_blockid = find_unique_block(blockid);
    TensorPerm layout;
    Sign sign;
    std::tie(layout, sign) = compute_sign_from_unique_block(blockid);
    Block block = alloc(uniq_blockid, layout, sign);
    if(distribution_ == Distribution::tce_nwi
       || distribution_ == Distribution::tce_nw
       || distribution_ == Distribution::tce_nwma) {
      auto key = TCE::compute_tce_key(flindices_, uniq_blockid);
      auto size = block.size();

      if (distribution_ == Distribution::tce_nwi) {
        Expects(rank_ == 4);
        //std::vector<size_t> is { &block.blockid()[0], &block.blockid()[rank_]};
        assert(0); //cget_hash_block_i takes offset_index, not hash
        //tamm::cget_hash_block_i(tce_ga_, block.buf(), block.size(), tce_hash_, key, is);
      } else if (distribution_ == Distribution::tce_nwma ||
                 distribution_ == Distribution::tce_nw) {
        auto length = tce_hash_[0];
        auto ptr = std::lower_bound(&tce_hash_[1], &tce_hash_[length + 1], key);
        Expects (!(ptr == &tce_hash_[length + 1] || key < *ptr));
        auto offset = *(ptr + length);
        if (distribution_ == Distribution::tce_nwma) {
          type_dispatch(element_type_, [&] (auto type) {
              using dtype = decltype(type);
              std::copy_n(reinterpret_cast<dtype*>(tce_data_buf_) + offset,
                          size,
                          reinterpret_cast<dtype*>(block.buf()));
            });
          // typed_copy(element_type_, tce_data_buf_ + offset, size, block.buf());
        }
        else {
#if 0
          tamm::gmem::get(tce_ga_, block.buf(), offset, offset + size - 1);
#else
          assert(0);
#endif
        }
      }
    }
    else {
      assert(0); //implement
    }
    return block;
  }
  
  /**
   * @todo For now, no index permutations allowed when writing
   */
  void add(Block& block) const {
    Expects(constructed_ == true);
    for(unsigned i=0; i<block.layout().size(); i++) {
      Expects(block.layout()[i] == i);
    }
    if(distribution_ == Distribution::tce_nw) {
#if 0
      auto key = TCE::compute_tce_key(flindices_, block.blockid());
      auto size = block.size();
      auto length = tce_hash_[0];
      auto ptr = std::lower_bound(&tce_hash_[1], &tce_hash_[length + 1], key);
      Expects (!(ptr == &tce_hash_[length + 1] || key < *ptr));
      auto offset = *(ptr + length);
      tamm::gmem::acc(tce_ga_, block.buf(), offset, offset + size - 1);
#else
      assert(0);
#endif
    } else if(distribution_ == Distribution::tce_nwma) {
#warning "THIS WILL NOT WORK IN PARALLEL RUNS. NWMA ACC IS NOT ATOMIC"
      auto size = block.size();
      auto length = tce_hash_[0];
      auto key = TCE::compute_tce_key(flindices_, block.blockid());
      auto ptr = std::lower_bound(&tce_hash_[1], &tce_hash_[length + 1], key);
      Expects (!(ptr == &tce_hash_[length + 1] || key < *ptr));
      auto offset = *(ptr + length);
      type_dispatch(element_type_, [&] (auto type) {
          using dtype = decltype(type);
          auto* sbuf = reinterpret_cast<dtype*>(block.buf());
          auto* dbuf = reinterpret_cast<dtype*>(tce_data_buf_) + offset;
          for(unsigned i=0; i<size; i++) {
            dbuf[i] += sbuf[i];
          }
        });
    } else {
      assert(0); //implement
    }
  }

    /**
   * @todo For now, no index permutations allowed when writing
   */
  void put(Block& block) const {
    Expects(constructed_ == true);
    for(unsigned i=0; i<block.layout().size(); i++) {
      Expects(block.layout()[i] == i);
    }
    if(distribution_ == Distribution::tce_nw) {
#if 0
      auto key = TCE::compute_tce_key(flindices_, block.blockid());
      auto size = block.size();
      auto length = tce_hash_[0];
      auto ptr = std::lower_bound(&tce_hash_[1], &tce_hash_[length + 1], key);
      Expects (!(ptr == &tce_hash_[length + 1] || key < *ptr));
      auto offset = *(ptr + length);
      tamm::gmem::put(tce_ga_, block.buf(), offset, offset + size - 1);
#else
      assert(0);
#endif
    } else if(distribution_ == Distribution::tce_nwma) {
#warning "THIS WILL NOT WORK IN PARALLEL RUNS. NWMA ACC IS NOT ATOMIC"
      auto size = block.size();
      auto length = tce_hash_[0];
      auto key = TCE::compute_tce_key(flindices_, block.blockid());
      auto ptr = std::lower_bound(&tce_hash_[1], &tce_hash_[length + 1], key);
      Expects (!(ptr == &tce_hash_[length + 1] || key < *ptr));
      auto offset = *(ptr + length);
      type_dispatch(element_type_, [&] (auto type) {
          using dtype = decltype(type);
          auto* sbuf = reinterpret_cast<dtype*>(block.buf());
          auto* dbuf = reinterpret_cast<dtype*>(tce_data_buf_) + offset;
          for(unsigned i=0; i<size; i++) {
            dbuf[i] = sbuf[i];
          }
        });
    } else {
      assert(0); //implement
    }
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

  Block alloc(const TensorIndex& blockid) {
    return Block{*this, blockid};
    // const TensorIndex& blockdims = block_dims(blockid);
    // TensorPerm layout;
    // int sign;
    // std::tie(layout, sign) = find_unique_block(blockid);
    // return Block{*this, blockid, blockdims, layout, sign};
  }

  Block alloc(const TensorIndex& blockid, const TensorPerm& layout, int sign) {
    auto blockdims = block_dims(blockid);
    Expects(layout.size() == rank());
    return Block{*this, blockid, blockdims, layout, sign};
  }
  // ProductIterator<TriangleLoop> iterator() {
  //   TensorVec<TriangleLoop> tloops, tloops_last;
  //   for(auto &sg: indices_) {
  //     BlockDim lo, hi;
  //     std::tie(lo, hi) = tensor_index_range(sg[0]);
  //     tloops.push_back(TriangleLoop{sg.size(), lo, hi});
  //     tloops_last.push_back(tloops.back().get_end());
  //   }
  //   return ProductIterator<TriangleLoop>(tloops, tloops_last);
  // }

  LabeledTensor operator () (const TensorLabel& label);

  // LabeledTensor operator () (const TensorVec<int>& ilabel) {
  //   TensorLabel label;
  //   Expects(ilabel.size() == rank());
  //   for(int i=0; i<ilabel.size(); i++) {
  //     label.push_back(IndexLabel{ilabel[i], flindices_[i]});
  //   }
  //   return operator()(label);
  // }

  LabeledTensor operator () ();
  

 private:

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

  TensorVec<SymmGroup> indices_;
  ElementType element_type_;
  Distribution distribution_;
  TensorRank nupper_indices_;
  Irrep irrep_;
  bool spin_restricted_; //spin restricted
  bool constructed_;
  AllocationPolicy policy_;
  TensorRank rank_;
  TensorDim flindices_;

#if 0
  tamm::gmem::Handle tce_ga_;
#endif
  uint8_t* tce_data_buf_{};
  TCE::Int *tce_hash_{};
};  // class Tensor

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

#endif  // TAMMX_TENSOR_H_

