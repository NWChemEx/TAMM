// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_COPY_SYMMETRIZER_H__
#define TAMMX_COPY_SYMMETRIZER_H__

#include "tammx/expects.h"
#include "tammx/boundvec.h"
#include "tammx/types.h"

namespace tammx {

class CopySymmetrizer {
 public:
  using size_type = TensorIndex::size_type;
      
  CopySymmetrizer()
      : CopySymmetrizer(0, 0, TensorLabel{}, TensorIndex{}, TensorIndex{}) {}
        

  CopySymmetrizer(size_type group_size,
                  size_type part_size,
                  const TensorLabel& label,
                  const TensorIndex& blockid,
                  const TensorIndex& uniq_blockid)
      : group_size_{group_size},
        part_size_{part_size},
        label_{label},
        blockid_{blockid},
        uniq_blockid_{uniq_blockid},
        bag_(group_size) {
          std::iota(bag_.begin(), bag_.end(), 0);
          Expects(label.size() == group_size);
          Expects(blockid.size() == group_size);
          Expects(uniq_blockid.size() == group_size);
          //Expects(group_size > 0);
          //Expects(bag_.size() > 0);
          comb_ = Combination<int>(bag_, part_size);
          //std::cerr<<"Copy symmetrizer constructor. itr size="<<group_size<<std::endl;
        }

  CopySymmetrizer& operator = (const CopySymmetrizer& csm) = default;
  CopySymmetrizer(const CopySymmetrizer& csm) = default;
  // {
  //   group_size_ = csm.group_size_;
  //   part_size_ = csm.part_size_;
  //   blockid_ = csm.blockid_;
  //   return *this;
  // }

  class Iterator {
   public:
    using ItrType = TensorLabel;

    Iterator() : cs_{nullptr} {}
 
    explicit Iterator(const CopySymmetrizer* cs)
        : cs_{const_cast<CopySymmetrizer*>(cs)} {
      if(cs_) {
        itr_ = cs_->comb_.begin();
        end_ = cs_->comb_.end();

        while (itr_ != end_) {
          auto perm = *itr_;
          Expects(perm.size() == cs_->blockid_.size());
          auto perm_blockid = perm_apply(cs_->blockid_, perm);
          auto &uniq_blockid = cs_->uniq_blockid_;
          if (std::equal(uniq_blockid.begin(), uniq_blockid.end(),
                         perm_blockid.begin(), perm_blockid.end())) {
            break;
          }
          ++itr_;
        }
      }
      //std::cerr<<"CopySymmetrizer::Iteratoe constructor. itr_size="<<cs_->group_size_<<std::endl;
    }

    // Iterator& operator = (Iterator& rhs) = default;

    Iterator& operator = (const Iterator& rhs) = default;

    size_t itr_size() const {
      return cs_->group_size_;
    }
    
    TensorLabel operator * () {
      // std::cerr<<"CopyYmmetrizer::Iterator. perm permutation. ="<<*itr_<<std::endl;
      // std::cerr<<"CopyYmmetrizer::Iterator. perm on label. ="<<cs_->label_<<std::endl;      
      return perm_apply(cs_->label_, *itr_);
    }

    Iterator& operator ++ () {
      while (++itr_ != end_) {
        auto perm = *itr_;
        Expects(perm.size() == cs_->blockid_.size());
        auto perm_blockid = perm_apply(cs_->blockid_, perm);
        auto &uniq_blockid = cs_->uniq_blockid_;
        if (std::equal(uniq_blockid.begin(), uniq_blockid.end(),
                       perm_blockid.begin(), perm_blockid.end())) {
          std::cerr<<"COPY. Comb::Iterator. internal perm="<<perm<<std::endl;
          std::cerr<<"COPY. Comb::Iterator. perm blockid="<<perm_blockid<<std::endl;
          std::cerr<<"COPY. Comb::Iterator. unique blockid="<<uniq_blockid<<std::endl;
          break;
        }
      }
      return *this;
    }    
    
   public:
    Combination<int>::Iterator itr_, end_;
    CopySymmetrizer *cs_;

    friend bool operator == (const typename CopySymmetrizer::Iterator& itr1,
                             const typename CopySymmetrizer::Iterator& itr2) {
      return (itr1.cs_  == itr2.cs_)
          &&  itr1.itr_ == itr2.itr_
          &&  itr1.end_ == itr2.end_;
    }
    
    friend bool operator != (const typename CopySymmetrizer::Iterator& itr1,
                             const typename CopySymmetrizer::Iterator& itr2) {
      return !(itr1 == itr2);
    }
    friend class CopySymmetrizer;
  };

  Iterator begin() const {
    return Iterator(this);
  }

  Iterator end() const {
    auto itr = Iterator(this);
    itr.itr_ = comb_.end();
    return itr;
  }

 public:
  size_type group_size_;
  size_type part_size_;
  TensorLabel label_;
  TensorIndex blockid_;
  TensorIndex uniq_blockid_;
  TensorVec<int> bag_;
  Combination<int> comb_;
};



}; // namespace tammx

#endif  // TAMMX_COPY_SYMMETRIZER_H__

