#ifndef TAMM_TENSOR_TAMMX_H_
#define TAMM_TENSOR_TAMMX_H_

#include <array>
#include <vector>
#include <cassert>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <map>
#include <iostream>
#if 0
#  include "tensor/gmem.h"
#  include "tensor/capi.h"
#endif
#include "tammx/strong_int.h"
#include "tammx/boundvec.h"
#include "tammx/types.h"
#include "tammx/index_sort.h"
#include "tammx/util.h"
#include "tammx/tce.h"
#include "tammx/work.h"
#include "tammx/combination.h"
#include "tammx/product_iterator.h"
#include "tammx/triangle_loop.h"
#include "tammx/copy_symmetrizer.h"

/**
 * @todo Check pass-by-value, reference, or pointer, especially for
 * Block and Tensor
 *
 * @todo Parallelize parallel_work
 *
 * @todo Implement TCE::init() and TCE::finalize()
 *
 * @todo should TCE be a "singleton" or an object? Multiple distinct
 * CC calculations in one run?
 *
 * @todo Make everything process-group aware
 *
 * @todo BoundVec should properly destroy objects
 *
 * @todo A general expression template formulation
 *
 * @todo Move/copy semantics for Tensor and Block
 *
 * @todo Scoped allocation for Tensor & Block
 *
 */

namespace tammx {

inline ProductIterator<TriangleLoop>
loop_iterator(const TensorVec<SymmGroup>& indices ) {
  TensorVec<TriangleLoop> tloops, tloops_last;
  for(auto &sg: indices) {
    BlockDim lo, hi;
    std::tie(lo, hi) = tensor_index_range(sg[0]);
    tloops.push_back(TriangleLoop{sg.size(), lo, hi});
    tloops_last.push_back(tloops.back().get_end());
  }
  //std::cerr<<"loop itr size="<<tloops.size()<<std::endl;
  return ProductIterator<TriangleLoop>(tloops, tloops_last);
}

class Tensor;
class Block;
struct LabeledBlock {
  Block *block_;
  TensorLabel label_;

  void init(double value);
};

struct LabeledTensor  {
  Tensor *tensor_;
  TensorLabel label_;
};

/**
 * @todo Check copy semantics and that the buffer is properly managed.
 */
class Block {
 public:
  Block(Tensor& tensor,
        const TensorIndex& block_id);

  Block(Tensor& tensor,
        const TensorIndex& block_id,
        const TensorIndex& block_dims,
        const TensorPerm& layout,
        Sign sign);

  const TensorIndex& blockid() const {
    return block_id_;
  }

  const TensorIndex& block_dims() const {
    return block_dims_;
  }

  LabeledBlock operator () (const TensorLabel &label) {
    return LabeledBlock{this, label};
  }

  LabeledBlock operator () () {
    TensorLabel label(block_id_.size());
    std::iota(label.begin(), label.end(), 0);
    return this->operator ()(label); //LabeledBlock{*this, label};
  }
  
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

  Tensor& tensor() {
    return tensor_;
  }
  
 private:
  Tensor& tensor_;
  TensorIndex block_id_;
  TensorIndex block_dims_;
  std::unique_ptr<uint8_t []> buf_;
  TensorPerm layout_;
  Sign sign_;
};

class Tensor {
 public:
  enum class Distribution { tce_nw, tce_nwma, tce_nwi };
  enum class AllocationPolicy { none, create, attach };
  enum class Type { integer, single_precision, double_precision};

  Tensor(const TensorVec<SymmGroup> &indices,
         Type element_type,
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
        }

  TensorRank rank() const {
    return rank_;
  }

  Type element_type() const {
    return element_type_;
  }

  TensorDim flindices() const {
    return flindices_;
  }

  static constexpr size_t elsize(Type eltype) {
    size_t ret = 0;
    switch(eltype) {
      case Type::integer:
        ret = sizeof(int);
        break;
      case Type::single_precision:
        ret = sizeof(float);
        break;
      case Type::double_precision:
        ret = sizeof(double);
        break;
    }
    return ret;
  }

  size_t element_size() const {
    return Tensor::elsize(element_type_);
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
  
  void attach(void *tce_data_buf, TCE::Int *tce_hash) {
    Expects (!constructed_);
    Expects (distribution_ == Distribution::tce_nwma);
    tce_data_buf_ = static_cast<double *>(tce_data_buf);
    tce_hash_ = tce_hash;
    constructed_ = true;
    policy_ = AllocationPolicy::attach;    
  }

  void allocate() {
    if (distribution_ == Distribution::tce_nwma || distribution_ == Distribution::tce_nw) {
      ProductIterator<TriangleLoop> pdt =  loop_iterator(indices_);
      auto last = pdt.get_end();
      int length = 0;
      int x=0;
      for(auto itr = pdt; itr != last; ++itr) {
        //std::cout<<x++<<std::endl;
        //std::cout<<"allocate. itr="<<*itr<<std::endl;
        if(nonzero(*itr)) {
          length += 1;
        }
      }
      tce_hash_ = new TCE::Int [2 * length + 1];
      tce_hash_[0] = length;
      //start over
      pdt =  loop_iterator(indices_);
      last = pdt.get_end();
      TCE::Int size = 0;
      int addr = 1;
      for(auto itr = pdt; itr != last; ++itr) {
        auto blockid = *itr;
        if(nonzero(blockid)) {
          //std::cout<<"allocate. set keys. itr="<<*itr<<std::endl;
          tce_hash_[addr] = TCE::compute_tce_key(flindices(), blockid);
          tce_hash_[length + addr] = size;
          size += block_size(blockid);
          addr += 1;
        }
      }
      size = (size == 0) ? 1 : size;
      if (distribution_ == Distribution::tce_nw) {
#if 0
        tce_ga_ = tamm::gmem::create(tamm::gmem::Double, size, std::string{"noname1"});
        tamm::gmem::zero(tce_ga_);
#else
        assert(0);
#endif
      }
      else {
        tce_data_buf_ = new double [size];
        std::fill_n(tce_data_buf_, size, 0);
      }
    }
    else {
      assert(0); // implement
    }
    constructed_ = true;
    policy_ = AllocationPolicy::create;
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
          std::copy_n(static_cast<double*>(tce_data_buf_ + offset),
                      size,
                      reinterpret_cast<double*>(block.buf()));
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
      auto* sbuf = reinterpret_cast<double*>(block.buf());
      auto* dbuf = reinterpret_cast<double*>(tce_data_buf_ + offset);
      for(unsigned i=0; i<size; i++) {
        dbuf[i] += sbuf[i];
      }
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

  LabeledTensor operator () (const TensorLabel& label) {
    Expects(label.size() == rank());
    return LabeledTensor{this, label};
  }

  LabeledTensor operator () () {
    TensorLabel label(rank());
    std::iota(label.begin(), label.end(), 0);
    return this->operator ()(label);
  }
  

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
  Type element_type_;
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
  double* tce_data_buf_{};
  TCE::Int *tce_hash_{};
};  // class Tensor


inline
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

inline
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

inline void
LabeledBlock::init(double value) {
  auto *dbuf = reinterpret_cast<double*>(block_->buf());
  for(unsigned i=0; i<block_->size(); i++) {
    dbuf[i] = value;
  }
}


inline std::tuple<double, const LabeledBlock>
operator * (double alpha, const LabeledBlock& block) {
  return {alpha, block};
}

inline std::tuple<const LabeledBlock, const LabeledBlock>
operator * (const LabeledBlock& rhs1, const LabeledBlock& rhs2)  {
  return std::make_tuple(rhs1, rhs2);
}


inline std::tuple<double, const LabeledBlock, const LabeledBlock>
operator * (const std::tuple<double, LabeledBlock>& rhs1, const LabeledBlock& rhs2)  {
  return std::tuple_cat(rhs1, std::make_tuple(rhs2));
}

inline std::tuple<double, const LabeledBlock, const LabeledBlock>
operator * (double alpha, const std::tuple<LabeledBlock, LabeledBlock>& rhs) {
  return std::tuple_cat(std::make_tuple(alpha), rhs);
}

void
operator += (LabeledBlock block1, std::tuple<double, LabeledBlock> rhs);

void
operator += (LabeledBlock block1, std::tuple<double, LabeledBlock, LabeledBlock> rhs);

inline void
operator += (LabeledBlock block1, LabeledBlock block2) {
  block1 += 1.0 * block2;
}

inline void
operator += (LabeledBlock block1, std::tuple<LabeledBlock, LabeledBlock> rhs) {
  block1 += 1.0 * rhs;
}


inline std::tuple<double, LabeledTensor>
operator * (double alpha, LabeledTensor block) {
  return {alpha, block};
}

inline std::tuple<double, LabeledTensor, LabeledTensor>
operator * (const std::tuple<double, LabeledTensor>& rhs1, LabeledTensor rhs2)  {
  return std::tuple_cat(rhs1, std::make_tuple(rhs2));
}

inline std::tuple<LabeledTensor, LabeledTensor>
operator * (LabeledTensor rhs1, LabeledTensor rhs2)  {
  return std::make_tuple(rhs1, rhs2);
}

inline std::tuple<double, LabeledTensor, LabeledTensor>
operator * (double alpha, std::tuple<LabeledTensor, LabeledTensor> rhs) {
  return std::tuple_cat(std::make_tuple(alpha), rhs);
}

void
operator += (LabeledTensor block1, std::tuple<double, LabeledTensor> rhs);

void
operator += (LabeledTensor block1, std::tuple<double, LabeledTensor, LabeledTensor> rhs);

inline void
operator += (LabeledTensor block1, LabeledTensor block2) {
  block1 += 1.0 * block2;
}

inline void
operator += (LabeledTensor block1, std::tuple<LabeledTensor, LabeledTensor> rhs) {
  block1 += 1.0 * rhs;
}

inline TensorVec<TensorVec<TensorLabel>>
summation_labels(const LabeledTensor& /*ltc*/,
                  const LabeledTensor& lta,
                  const LabeledTensor& ltb) {
  return group_partition(lta.tensor_->indices(), lta.label_,
                         ltb.tensor_->indices(), ltb.label_);

}

inline std::pair<TensorVec<SymmGroup>,TensorLabel>
summation_indices(const LabeledTensor& /*ltc*/,
                  const LabeledTensor& lta,
                  const LabeledTensor& ltb) {
  auto aindices = flatten(lta.tensor_->indices());
  //auto bindices = flatten(ltb.tensor_.indices());
  auto alabels = group_labels(lta.tensor_->indices(), lta.label_);
  auto blabels = group_labels(ltb.tensor_->indices(), ltb.label_);
  TensorVec<SymmGroup> ret_indices;
  TensorLabel sum_labels;
  int apos = 0;
  for (auto &alg : alabels) {
    for (auto &blg : blabels) {
      SymmGroup sg;
      for (auto &a : alg) {
        int apos1 = 0;
        for (auto &b : blg) {
          if (a == b) {
            sg.push_back(aindices[apos + apos1]);
            sum_labels.push_back(a);
          }
        }
        apos1++;
      }
      if (sg.size() > 0) {
        ret_indices.push_back(sg);
      }
    }
    apos += alg.size();
  }
  return {ret_indices, sum_labels};
}

inline TensorVec<TensorVec<TensorLabel>>
nonsymmetrized_external_labels(const LabeledTensor& ltc,
                               const LabeledTensor& lta) {
  auto ca_labels = group_partition(ltc.tensor_->indices(), ltc.label_,
                                   lta.tensor_->indices(), lta.label_);

  TensorVec<TensorVec<TensorLabel>> ret_labels;
  for(unsigned i=0; i<ca_labels.size(); i++)  {
    Expects(ca_labels[i].size() > 0);
    ret_labels.push_back(TensorVec<TensorLabel>());
    ret_labels.back().insert_back(ca_labels[i].begin(), ca_labels[i].end());
  }
  return ret_labels;
}

/**
 * @todo Specify where symmetrization is allowed and what indices in
 * the input tensors can form a symmetry group (or go to distinct
 * groups) in the output tensor.
 */
inline TensorVec<TensorVec<TensorLabel>>
nonsymmetrized_external_labels(const LabeledTensor& ltc,
                               const LabeledTensor& lta,
                               const LabeledTensor& ltb) {
  auto ca_labels = group_partition(ltc.tensor_->indices(), ltc.label_,
                                   lta.tensor_->indices(), lta.label_);
  auto cb_labels = group_partition(ltc.tensor_->indices(), ltc.label_,
                                   ltb.tensor_->indices(), ltb.label_);
  Expects(ca_labels.size() == cb_labels.size());

  TensorVec<TensorVec<TensorLabel>> ret_labels;
  for(unsigned i=0; i<ca_labels.size(); i++)  {
    Expects(ca_labels[i].size() + cb_labels[i].size() > 0);
    ret_labels.push_back(TensorVec<TensorLabel>());
    if(ca_labels[i].size() > 0) {
      ret_labels.back().insert_back(ca_labels[i].begin(), ca_labels[i].end());
    }
    if(cb_labels[i].size() > 0) {
      ret_labels.back().insert_back(cb_labels[i].begin(), cb_labels[i].end());
    }
  }
  return ret_labels;
}

inline TensorVec<TensorVec<TensorLabel>>
symmetrized_external_labels(const LabeledTensor& ltc,
                            const LabeledTensor&  /*lta*/,
                            const LabeledTensor&  /*ltb*/) {
  TensorVec<TensorLabel> ret {ltc.label_};
  return {ret};
}


inline ProductIterator<TriangleLoop>
nonsymmetrized_iterator(const LabeledTensor& ltc,
                        const LabeledTensor& lta,
                        const LabeledTensor& ltb) {
  auto part_labels = nonsymmetrized_external_labels(ltc ,lta, ltb);
  //auto flat_labels = flatten(flatten(part_labels));
  std::map<IndexLabel, DimType> dim_of_label;

  auto cflindices = flatten(ltc.tensor_->indices());
  for(unsigned i=0; i<ltc.label_.size(); i++) {
    dim_of_label[ltc.label_[i]] = cflindices[i];
  }
  auto aflindices = flatten(lta.tensor_->indices());
  for(unsigned i=0; i<lta.label_.size(); i++) {
    dim_of_label[lta.label_[i]] = aflindices[i];
  }
  TensorVec<TriangleLoop> tloops, tloops_last;
  for(auto dim_grps: part_labels) {
    for(auto lbl: dim_grps) {
      if(lbl.size() > 0) {
        BlockDim lo, hi;
        std::tie(lo, hi) = tensor_index_range(dim_of_label[lbl[0]]);
        tloops.push_back(TriangleLoop(lbl.size(), lo, hi));
        tloops_last.push_back(tloops.back().get_end());
      }
    }
  }
  return ProductIterator<TriangleLoop>(tloops, tloops_last);
}


/**
 * @todo abstract the two copy_symmetrizer versions into one function
 * with the logic and two interfaces
 */
inline TensorVec<CopySymmetrizer>
copy_symmetrizer(const LabeledTensor& ltc,
                 const LabeledTensor& lta,
                 const LabeledTensor& ltb,
                 const LabelMap<BlockDim>& lmap) {
  auto part_labels = nonsymmetrized_external_labels(ltc ,lta, ltb);
  TensorVec<CopySymmetrizer> csv;
  for(auto lbls: part_labels) {
    Expects(lbls.size()>0 && lbls.size() <= 2);

    TensorLabel lbl(lbls[0].begin(), lbls[0].end());
    if(lbls.size() == 2 ) {
      lbl.insert_back(lbls[1].begin(), lbls[1].end());
    }

    auto size = lbl.size();
    Expects(size > 0);
    Expects(size <=2); // @todo implement other cases

    auto blockid = lmap.get_blockid(lbl);
    auto uniq_blockid{blockid};
    //find unique block
    std::sort(uniq_blockid.begin(), uniq_blockid.end());
    Expects(size > 0);
    //std::cout<<"CONSTRUCTING COPY SYMMETRIZER FOR LABELS="<<lbl<<std::endl;
    csv.push_back(CopySymmetrizer{size, lbls[0].size(), lbl, blockid, uniq_blockid});
  }
  return csv;
}

inline TensorVec<TensorLabel>
group_labels(const TensorLabel& label,
             const TensorIndex& group_sizes) {
  TensorVec<TensorLabel> ret;
  int pos = 0;
  for(auto grp: group_sizes) {
    ret.push_back(TensorLabel(grp.value()));
    std::copy_n(label.begin() + pos, grp.value(), ret.back().begin());
    pos += grp.value();
  }
  return ret;  
}

inline TensorVec<TensorVec<TensorLabel>>
compute_extra_symmetries(const TensorLabel& lhs_label,
                         const TensorIndex& lhs_group_sizes,
                         const TensorLabel& rhs_label,
                         const TensorIndex& rhs_group_sizes) {
  Expects(lhs_label.size() == lhs_group_sizes.size());
  Expects(rhs_label.size() == rhs_group_sizes.size());
  Expects(lhs_label.size() == rhs_label.size());

  auto lhs_label_groups = group_labels(lhs_label, lhs_group_sizes);
  auto rhs_label_groups = group_labels(rhs_label, rhs_group_sizes);

  TensorVec<TensorVec<TensorLabel>> ret_labels;
  for (auto &glhs : lhs_label_groups) {
    TensorVec<TensorLabel> ret_group;
    for (auto &grhs : rhs_label_groups) {
      auto lbls = intersect(glhs, grhs);
      if (lbls.size() > 0) {
        ret_group.push_back(lbls);
      }
    }
    ret_labels.push_back(ret_group);
  }
  return ret_labels;  
}


inline TensorVec<CopySymmetrizer>
copy_symmetrizer(const LabeledTensor& ltc,
                 const LabeledTensor& lta,
                 const LabelMap<BlockDim>& lmap) {
  auto part_labels = nonsymmetrized_external_labels(ltc ,lta);
  TensorVec<CopySymmetrizer> csv;
  for(auto lbls: part_labels) {
    Expects(lbls.size()>0 && lbls.size() <= 2);

    TensorLabel lbl(lbls[0].begin(), lbls[0].end());
    if(lbls.size() == 2 ) {
      lbl.insert_back(lbls[1].begin(), lbls[1].end());
    }

    auto size = lbl.size();
    Expects(size > 0);
    Expects(size <=2); // @todo implement other cases

    auto blockid = lmap.get_blockid(lbl);
    auto uniq_blockid{blockid};
    //find unique block
    std::sort(uniq_blockid.begin(), uniq_blockid.end());
    Expects(size > 0);
    //std::cout<<"CONSTRUCTING COPY SYMMETRIZER FOR LABELS="<<lbl<<std::endl;
    csv.push_back(CopySymmetrizer{size, lbls[0].size(), lbl, blockid, uniq_blockid});
  }
  return csv;
}



inline ProductIterator<CopySymmetrizer::Iterator>
copy_iterator(const TensorVec<CopySymmetrizer>& sitv) {
  TensorVec<CopySymmetrizer::Iterator> itrs_first, itrs_last;
  for(auto &sit: sitv) {
    // std::cerr<<__FUNCTION__<<" symmetrizer SIZE="<< sit.group_size_<<std::endl;
    itrs_first.push_back(sit.begin());
    itrs_last.push_back(sit.end());
    Expects(itrs_first.back().itr_size() == itrs_last.back().itr_size());
    Expects(itrs_first.back().itr_size() == sit.group_size_);
    // std::cerr<<__FUNCTION__<<" iterator first cs_ ptr="<< itrs_first.back().cs_<<std::endl;    
    // std::cerr<<__FUNCTION__<<" iterator last cs_ ptr="<< itrs_last.back().cs_<<std::endl;    
  }
  return {itrs_first, itrs_last};
}


//working but for unsymmetrization
/**
 * @todo We assume there is no un-symmetrization in the output.
 */
inline void
operator += (LabeledTensor ltc, std::tuple<double, LabeledTensor> rhs) {
  Expects(ltc.tensor_->rank() == std::get<1>(rhs).tensor_->rank());
  double alpha = std::get<0>(rhs);
  std::cerr<<"ALPHA="<<alpha<<std::endl;
  const LabeledTensor& lta = std::get<1>(rhs);
  Tensor& ta = *lta.tensor_;
  Tensor& tc = *ltc.tensor_;
  //check for validity of parameters
  auto aitr = loop_iterator(ta.indices());
  auto lambda = [&] (const TensorIndex& ablockid) {
    size_t dima = ta.block_size(ablockid);
    if(ta.nonzero(ablockid) && dima>0) {
      auto label_map = LabelMap<BlockDim>()
          .update(lta.label_, ablockid);
      // auto cblockid = tc.find_unique_block(label_map.get_blockid(ltc.label_));
      auto cblockid = label_map.get_blockid(ltc.label_);
      auto abp = ta.get(ablockid);
      //auto cbp = tc.alloc(cblockid);
      //cbp(ltc.label_) += alpha * abp(lta.label_);

      auto csbp = tc.alloc(tc.find_unique_block(cblockid));
      csbp().init(0);
      
      std::cerr<<"COPY ITR. ablockid="<<ablockid<<std::endl;
      std::cerr<<"COPY ITR. alabel="<<lta.label_<<std::endl;

      auto copy_symm = copy_symmetrizer(ltc, lta, label_map);      
      auto copy_itr = copy_iterator(copy_symm);
      auto copy_itr_last = copy_itr.get_end();
      auto copy_label = TensorLabel(ltc.label_.size());
      std::iota(copy_label.begin(), copy_label.end(), 0);
      for(auto citr = copy_itr; citr != copy_itr_last; ++citr) {
        auto perm = *citr;
        auto num_inversions = perm_count_inversions(perm);
        Sign sign = (num_inversions%2) ? -1 : 1;
        std::cerr<<"COPY ITR. csbp blockid="<<csbp.blockid()<<std::endl;
        std::cerr<<"COPY ITR. csbp label="<<copy_label<<std::endl;
        std::cerr<<"COPY ITR. cbp blockid="<<cblockid<<std::endl;
        std::cerr<<"COPY ITR. cbp label="<<perm<<std::endl;
        std::cerr<<"COPY ITR. num_inversions="<<num_inversions<<std::endl;
        std::cerr<<"COPY ITR. sign="<<sign<<std::endl;
        //csbp(copy_label) += sign * cbp(perm);
        auto perm_comp = perm_apply(perm, perm_compute(ltc.label_, lta.label_));
        csbp(copy_label) += sign * alpha * abp(perm_comp);
      }
      tc.add(csbp);
    }
  };
  parallel_work(aitr, aitr.get_end(), lambda);
}


inline void
assert_equal(Tensor &tc, double value) {
  auto citr = loop_iterator(tc.indices());
  auto lambda = [&] (const TensorIndex& cblockid) {
    size_t dimc = tc.block_size(cblockid);
    if(tc.nonzero(cblockid) && dimc>0) {
      auto cbp = tc.get(cblockid);
      auto cdbuf = reinterpret_cast<double*>(cbp.buf());
      auto size = cbp.size();
      for(int i=0; i<size; i++) {
        std::cerr<<__FUNCTION__<<": block="<<cblockid<<std::endl;        
        std::cerr<<__FUNCTION__<<": buf["<<i<<"]="<<cdbuf[i]<<std::endl;
        assert(std::abs(cdbuf[i]-value) < 1.0e-6);
      }
    }
  };
  parallel_work(citr, citr.get_end(), lambda);
}

inline void
assert_zero(Tensor& tc) {
  assert_equal(tc, 0);
}


/**
 * Check that all iterators and operators work for rank 0 tensors, and rank 0 symmetry groups.
 */
inline void
operator += (LabeledTensor ltc, std::tuple<double, LabeledTensor, LabeledTensor> rhs) {
  //check for validity of parameters
  double alpha = std::get<0>(rhs);
  LabeledTensor& lta = std::get<1>(rhs);
  LabeledTensor& ltb = std::get<2>(rhs);
  Tensor& ta = *lta.tensor_;
  Tensor& tb = *ltb.tensor_;
  Tensor& tc = *ltc.tensor_;

  TensorLabel sum_labels;
  TensorVec<SymmGroup> sum_indices;
  std::tie(sum_indices, sum_labels) = summation_indices(ltc, lta, ltb);
  auto lambda = [&] (const TensorIndex& cblockid) {
    auto dimc = tc.block_size(cblockid);
    if(!tc.nonzero(cblockid) || dimc == 0) {
      return;
    }
    auto sum_itr_first = loop_iterator(sum_indices);
    auto sum_itr_last = sum_itr_first.get_end();
    auto label_map = LabelMap<BlockDim>().update(ltc.label_, cblockid);
    auto cbp = tc.alloc(cblockid);
    cbp().init(0);
    for(auto sitr = sum_itr_first; sitr!=sum_itr_last; ++sitr) {
      label_map.update(sum_labels, *sitr);
      auto ablockid = label_map.get_blockid(lta.label_);
      auto bblockid = label_map.get_blockid(ltb.label_);

      if(!ta.nonzero(ablockid) || !tb.nonzero(bblockid)) {
        continue;
      }
      auto abp = ta.get(ablockid);
      auto bbp = tb.get(bblockid);

      cbp(ltc.label_) += alpha * abp(lta.label_) * bbp(ltb.label_);
    }
    auto csbp = tc.alloc(tc.find_unique_block(cblockid));
    csbp().init(0);
    auto copy_symm = copy_symmetrizer(ltc, lta, ltb, label_map);      
    auto copy_itr = copy_iterator(copy_symm);
    auto copy_itr_last = copy_itr.get_end();
    auto copy_label = TensorLabel(ltc.label_);
    std::sort(copy_label.begin(), copy_label.end());
    for(auto citr = copy_itr; citr != copy_itr_last; ++citr) {
      auto perm = *citr;
      std::cerr<<"---perm="<<perm<<std::endl;
      auto num_inversions = perm_count_inversions(perm);
      Sign sign = (num_inversions%2) ? -1 : 1;
      csbp(copy_label) += sign * cbp(perm);
    }
    tc.add(csbp);
  };
  auto itr = nonsymmetrized_iterator(ltc, lta, ltb);
  parallel_work(itr, itr.get_end(), lambda);
}


/**
 * performs: cbuf[dims] = scale *abuf[perm(dims)]
 */
inline void
index_permute_acc(uint8_t* dbuf, uint8_t* sbuf, const TensorPerm& perm, const TensorIndex& ddims, double scale) {
  Expects(dbuf!=nullptr && sbuf!=nullptr);
  Expects(perm.size() == ddims.size());

  std::cerr<<__FUNCTION__<<" perm = "<<perm<<std::endl;
  std::cerr<<__FUNCTION__<<" ddims = "<<ddims<<std::endl;
  auto inv_perm = perm_invert(perm);
  auto inv_sizes = perm_apply(ddims, inv_perm);
  std::cerr<<__FUNCTION__<<" inv_perm = "<<inv_perm<<std::endl;
  std::cerr<<__FUNCTION__<<" inv_sizes = "<<inv_sizes<<std::endl;
  TensorVec<size_t> sizes;
  TensorVec<int> iperm;
  for(unsigned i=0; i<ddims.size(); i++) {
    sizes.push_back(inv_sizes[i].value());
    iperm.push_back(inv_perm[i]+1);
  }

  std::cerr<<"sbuf = "<<(void*)sbuf<<std::endl;
  std::cerr<<"dbuf = "<<(void *)dbuf<<std::endl;
  tamm::index_sortacc(reinterpret_cast<double*>(sbuf),
                      reinterpret_cast<double*>(dbuf),
                      sizes.size(), &sizes[0], &iperm[0], scale);
}

inline void
index_permute(uint8_t* dbuf, uint8_t* sbuf, const TensorPerm& perm, const TensorIndex& ddims, double scale) {
  Expects(dbuf!=nullptr && sbuf!=nullptr);
  Expects(perm.size() == ddims.size());

  auto inv_perm = perm_invert(perm);
  auto inv_sizes = perm_apply(ddims, inv_perm);
  TensorVec<size_t> sizes;
  TensorVec<int> iperm;
  for(unsigned i=0; i<ddims.size(); i++) {
    sizes.push_back(inv_sizes[i].value());
    iperm.push_back(inv_perm[i]+1);
  }

  tamm::index_sort(reinterpret_cast<double*>(sbuf),
                   reinterpret_cast<double*>(dbuf),
                   sizes.size(), &sizes[0], &iperm[0], scale);
}

inline void
operator += (LabeledBlock clb, std::tuple<double, LabeledBlock> rhs) {
  const LabeledBlock& alb = std::get<1>(rhs);

  auto &ablock = *alb.block_;
  auto &cblock = *clb.block_;

  auto &clabel = clb.label_;
  auto &alabel = alb.label_;

  auto label_perm = perm_compute(alabel, clabel);
  for(unsigned i=0; i<label_perm.size(); i++) {
    Expects(cblock.block_dims()[i] == ablock.block_dims()[label_perm[i]]);
  }

  auto &alayout = ablock.layout();
  auto &clayout = cblock.layout();

  std::cerr<<__FUNCTION__<<":"<<__LINE__<<": alabel="<<alabel<<std::endl;
  std::cerr<<__FUNCTION__<<":"<<__LINE__<<": alayout="<<alayout<<std::endl;

  Expects(clayout.size() == cblock.tensor().rank());
  Expects(clabel.size() == perm_invert(clayout).size());
  Expects(alabel.size() == perm_invert(alayout).size());
  auto cstore = perm_apply(clabel, perm_invert(clayout));
  auto astore = perm_apply(alabel, perm_invert(alayout));

  auto store_perm = perm_compute(astore, cstore);
  double alpha = std::get<0>(rhs);
  index_permute_acc(cblock.buf(), ablock.buf(), store_perm, cblock.block_dims(), alpha);
}

// C storage order: A[m,k], B[k,n], C[m,n]
inline void
matmul(int m, int n, int k, double *A, int lda, double *B, int ldb, double *C, int ldc, double alpha) {
  Expects(m>0 && n>0 && k>0);
  Expects(A!=nullptr && B!=nullptr && C!=nullptr);

  for(int x=0; x<m; x++) {
    for(int y=0; y<n; y++) {
      double value = 0;
      for(int z=0; z<k; z++) {
        value += A[x*lda + z] * B[z*ldb + y];
      }
      C[x*ldc + y] = alpha * C[x*ldc + y] + value;
    }
  }
}

inline void
operator += (LabeledBlock clb, std::tuple<double, LabeledBlock, LabeledBlock> rhs) {
  const LabeledBlock& alb = std::get<1>(rhs);
  const LabeledBlock& blb = std::get<2>(rhs);

  auto &ablock = *alb.block_;
  auto &bblock = *blb.block_;
  auto &cblock = *clb.block_;

  auto &alabel = alb.label_;
  auto &blabel = blb.label_;
  auto &clabel = clb.label_;

  auto aext_labels = intersect(clabel, alabel);
  auto bext_labels = intersect(clabel, blabel);
  auto sum_labels = intersect(alabel, blabel);

  auto alabel_sort = aext_labels;
  alabel_sort.insert_back(sum_labels.begin(), sum_labels.end());
  auto blabel_sort = sum_labels;
  blabel_sort.insert_back(bext_labels.begin(), bext_labels.end());
  auto clabel_sort = aext_labels;
  clabel_sort.insert_back(bext_labels.begin(), bext_labels.end());
  
  auto ablock_sort = ablock.tensor().alloc(ablock.blockid());
  auto bblock_sort = bblock.tensor().alloc(bblock.blockid());
  auto cblock_sort = cblock.tensor().alloc(cblock.blockid());  

  //TTGT
  ablock_sort(alabel_sort) += ablock(alabel); //T
  bblock_sort(blabel_sort) += bblock(blabel); //T
  // G
  auto alpha = std::get<0>(rhs);
  auto lmap = LabelMap<BlockDim>()
      .update(alabel, ablock.block_dims())
      .update(blabel, bblock.block_dims());
  auto aext_dims = lmap.get_blockid(aext_labels);
  auto bext_dims = lmap.get_blockid(bext_labels);
  auto sum_dims = lmap.get_blockid(sum_labels);
  int m = std::accumulate(aext_dims.begin(), aext_dims.end(), BlockDim{1}, std::multiplies<>()).value();
  int n = std::accumulate(bext_dims.begin(), bext_dims.end(), BlockDim{1}, std::multiplies<>()).value();
  int k = std::accumulate(sum_dims.begin(), sum_dims.end(), BlockDim{1}, std::multiplies<>()).value();
  matmul(m, n, k, reinterpret_cast<double*>(ablock_sort.buf()), k,
         reinterpret_cast<double*>(bblock_sort.buf()), n,
         reinterpret_cast<double*>(cblock_sort.buf()), n, alpha);
  cblock(clabel) += cblock_sort(clabel_sort); //T
}


template<typename Lambda>
inline void
tensor_map (LabeledTensor ltc, Lambda func) {
  Tensor& tc = *ltc.tensor_;
  auto citr = loop_iterator(tc.indices());
  auto lambda = [&] (const TensorIndex& cblockid) {
    size_t dimc = tc.block_size(cblockid);
    if(tc.nonzero(cblockid) && dimc>0) {
      auto cblock = tc.alloc(cblockid);
      func(cblock);
      tc.add(cblock);
    }
  };
  parallel_work(citr, citr.get_end(), lambda);  
}

template<typename Lambda>
inline void
tensor_map (LabeledTensor ltc, LabeledTensor lta, Lambda func) {
  Tensor& tc = *ltc.tensor_;
  Tensor& ta = *lta.tensor_;
  auto citr = loop_iterator(tc.indices());
  auto lambda = [&] (const TensorIndex& cblockid) {
    size_t dimc = tc.block_size(cblockid);
    if(tc.nonzero(cblockid) && dimc>0) {
      auto cblock = tc.alloc(cblockid);
      auto ablock = ta.alloc(cblockid);
      func(cblock, ablock);
      tc.add(cblock);
    }
  };
  parallel_work(citr, citr.get_end(), lambda);
}

inline void
tensor_print(Tensor& tc, std::ostream &os) {
  auto citr_first = loop_iterator(tc.indices());
  auto citr_last = citr_first.get_end();
  for(auto citr = citr_first; citr != citr_last; ++citr) {
    auto cblockid = *citr;
    size_t dimc = tc.block_size(cblockid);
    if(tc.nonzero(cblockid) && dimc>0) {
      auto cblock = tc.get(cblockid);
      os<<"block id = "<<cblockid<<std::endl;
      auto size = cblock.size();
      double *cdbuf = reinterpret_cast<double*>(cblock.buf());
      for(int i=0; i<cblock.size(); i++) {
        os<<cdbuf[i]<<" ";
      }
      os<<std::endl;
    }
  }
}

}  // namespace tammx

#endif  // TAMM_TENSOR_TAMMX_H_

