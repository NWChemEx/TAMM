// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_TRIANGLE_LOOP_H__
#define TAMMX_TRIANGLE_LOOP_H__

#include "tammx/expects.h"
#include "tammx/boundvec.h"
#include "tammx/types.h"

namespace tammx {

class TriangleLoop {
 public:
  using Type = BlockDim;
  using ItrType = TensorVec<Type>;
  
  TriangleLoop()
      : nloops_{0} {}

  TriangleLoop(size_t nloops, Type first, Type last)
      : nloops_{nloops},
        first_{first},
        last_{last},
        itr_(nloops, first) {}

  TriangleLoop& operator ++ () {
    int i;
    //std::cout<<"TriangleLoop. itr="<<itr_<<std::endl;
    for(i=itr_.size()-1; i>=0 && ++itr_[i] == last_; i--) {
      //no-op
    }
    if(i>=0) {
      for(unsigned j = i+1; j<itr_.size(); j++) {
        itr_[j] = itr_[j-1];
      }
    }
    return *this;
  }

  TriangleLoop operator ++ (int) {
    auto ret = *this;
    ++ *this;
    return ret;
  }

  const ItrType& operator * () {
    return itr_;
  }

  const ItrType* operator -> () const {
    return &itr_;
  }

  size_t itr_size() const {
    return itr_.size();
  }
  
  TriangleLoop get_end() const {
    TriangleLoop tl {nloops_, first_, last_};
    tl.itr_ = TensorVec<Type>(nloops_, last_);
    return tl;
  }

 private:
  size_t nloops_;
  Type first_{};
  Type last_{};
  TensorVec<Type> itr_;

  friend bool operator == (const TriangleLoop& tl1, const TriangleLoop& tl2);
  friend bool operator != (const TriangleLoop& tl1, const TriangleLoop& tl2);
};

inline bool
operator == (const TriangleLoop& tl1, const TriangleLoop& tl2) {
  return std::equal(tl1.itr_.begin(), tl1.itr_.end(), tl2.itr_.begin());
}

inline bool
operator != (const TriangleLoop& tl1, const TriangleLoop& tl2) {
  return !(tl1 == tl2);
}




}; // namespace tammx

#endif  // TAMMX_TRIANGLE_LOOP_H__

