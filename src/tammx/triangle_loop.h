// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_TRIANGLE_LOOP_H_
#define TAMMX_TRIANGLE_LOOP_H_

#include "tammx/boundvec.h"
#include "tammx/types.h"

namespace tammx {

/**
 * @brief Set of triangular loops.
 *
 * This is used to iterate over symmetric index groups. All loops in the loop nest run over the same range of indices.
 */
class TriangleLoop {
 public:
  using Type = BlockIndex;
  using ItrType = TensorVec<Type>;
  
  TriangleLoop()
      : nloops_{0},
        done_{false} {}

  /**
   * Construct a triangle loop nest
   * @param nloops Number of loops in the loop nest
   * @param first First element in the range for each loop in the loop nest
   * @param last Last element in the range for each loop in the loop nest. @p last is not included in the loop.
   */
  TriangleLoop(size_t nloops, Type first, Type last)
      : nloops_{nloops},
        first_{first},
        last_{last},
        itr_(nloops, first),
        done_{false} {}

  /**
   * Pre-increment the triangular loop
   * @return Return the current loop index (after increment)
   */
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
    } else {
      done_ = true;
    }
    return *this;
  }

  /**
   * Post-increment the triangular loop
   * @return Return the loop index before the increment
   */
  TriangleLoop operator ++ (int) {
    auto ret = *this;
    ++ *this;
    return ret;
  }

  /**
   * Get the vector of values for the loop nest
   * @return Current loop index values
   */
  const ItrType& operator * () {
    return itr_;
  }

  /**
   * Get pointer to vector of values for the loop nest
   * @return Current loop index values
   */
  const ItrType* operator -> () const {
    return &itr_;
  }

  /**
   * Number of loops in the triangle loop
   * @return Loop nest depth
   */
  size_t itr_size() const {
    return itr_.size();
  }

  /**
   * The last element in the triangle loop
   * @return Get the end for this iterator
   */
  TriangleLoop get_end() const {
    TriangleLoop tl {nloops_, first_, last_};
    tl.itr_ = TensorVec<Type>(nloops_, last_);
    tl.done_ = true;
    return tl;
  }

 private:
  size_t nloops_;
  Type first_{};
  Type last_{};
  TensorVec<Type> itr_;
  bool done_;

  friend bool operator == (const TriangleLoop& tl1, const TriangleLoop& tl2);
  friend bool operator != (const TriangleLoop& tl1, const TriangleLoop& tl2);
};

inline bool
operator == (const TriangleLoop& tl1, const TriangleLoop& tl2) {
  return tl1.done_ == tl2.done_
      && tl1.nloops_ == tl2.nloops_
      && std::equal(tl1.itr_.begin(), tl1.itr_.end(), tl2.itr_.begin())
      && tl1.first_ == tl2.first_
      && tl2.last_ == tl2.last_;
}

inline bool
operator != (const TriangleLoop& tl1, const TriangleLoop& tl2) {
  return !(tl1 == tl2);
}

}; // namespace tammx

#endif  // TAMMX_TRIANGLE_LOOP_H_

