// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_PRODUCT_ITERATOR_H__
#define TAMMX_PRODUCT_ITERATOR_H__

#include "tammx/errors.h"
#include "tammx/boundvec.h"
#include "tammx/types.h"

namespace tammx {

/**
 * @brief Cartesian product iterator
 *
 * @todo operator * () needs to be avoid constructing a vector everytime
 *
 * @todo support efficient * and -> operators
 *
 * @todo operator * is return an array
 *
 * @todo support assignment operator on blocks (to initialize a block to 0)
 *
 * @todo check that blockids passed and the order of labels in
 * group_partition are consistenst.
 *
 * @todo the symmetrization might not (will not) work when the
 * symmetric sub-groups are disjoint. Fix this.
 *
 * Eg: C[a,b,c] = A[a,c] x B[b].
 *
 */
template<typename Itr>
class ProductIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  
  ProductIterator(const TensorVec<Itr>& ifirst,
                  const TensorVec<Itr>& ilast)
      : ifirst_{ifirst},
        ilast_{ilast},
        ival_{ifirst} {
          EXPECTS(ifirst.size() == ilast.size());
          // std::cerr<<"Product Iterator constructor. num itrs="<<ifirst.size()<<std::endl;
          // std::cerr<<"Product iterator sizes: ";
          // for(auto it: ifirst) {
          //   std::cerr<<it.itr_size()<<" ";
          // }
          // std::cerr<<std::endl;
          for(int i=0; i<ifirst.size(); i++){
            EXPECTS(ifirst[i] != ilast[i]);
          }
        }

  ~ProductIterator() = default;

  ProductIterator& operator = (const ProductIterator& itr) {
    ifirst_ = itr.ifirst_;
    ilast_ = itr.ilast_;
    ival_ = itr.ival_;
    return *this;
  }

  ProductIterator& operator ++ () {
    int i;
    for(i=ival_.size()-1; i>=0 && ++ival_[i] == ilast_[i]; i--) {
      //no-op
    }
    if(i>=0) {
      for(unsigned j=i+1; j<ival_.size(); j++) {
        ival_[j] = ifirst_[j];
      }
      for(unsigned j=0; j<i; j++) {
        EXPECTS(ival_[j] != ilast_[j]);
      }
    }
    return *this;
  }

  ProductIterator operator ++ (int) {
    auto &ret = *this;
    ++ *this;
    return ret;
  }

  typename Itr::ItrType operator * ()  {
    //std::cerr<<__FUNCTION__<<"*. itr sizes:"<<std::endl;
    for(auto it : ival_){
      //std::cerr<<it.itr_size()<<std::endl;
    }
    TensorVec<typename Itr::ItrType> itrs(ival_.size());
    std::transform(ival_.begin(), ival_.end(), itrs.begin(),
                   [] (Itr& tloop) {
                     return *tloop;
                   });
    return flatten(itrs);
  }

  // const std::vector<Itr>* operator -> () const {
  //   return &ival_;
  // }

  template<typename Itr1>
  friend bool operator==(const ProductIterator<Itr1>& itr1, const ProductIterator<Itr1>& itr2);

  template<typename Itr1>
  friend bool operator!=(const ProductIterator<Itr1>& itr1, const ProductIterator<Itr1>& itr2);

  ProductIterator get_end() const {
    ProductIterator pdt{ifirst_, ilast_};
    pdt.ival_ = ilast_;
    return pdt;
  }

 private:
  TensorVec<Itr> ifirst_, ilast_;
  TensorVec<Itr> ival_;
};

template<typename Itr>
bool
operator == (const ProductIterator<Itr>& itr1,
             const ProductIterator<Itr>& itr2) {
  return std::equal(itr1.ival_.begin(), itr1.ival_.end(), itr2.ival_.begin());
}

template<typename Itr>
bool
operator != (const ProductIterator<Itr>& itr1,
             const ProductIterator<Itr>& itr2) {
  return !(itr1 == itr2);
}


}; // namespace tammx

#endif  // TAMMX_PRODUCT_ITERATOR_H__
