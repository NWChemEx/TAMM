#ifndef TAMM_TENSOR_TAMMX_H_
#define TAMM_TENSOR_TAMMX_H_

#include <array>
#include <vector>
#include <cassert>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <iostream>
#if 0
#  include "tensor/gmem.h"
#  include "tensor/capi.h"
#endif
#include "tammx/StrongInt.h"
#include "tammx/index_sort.h"

using Fint = int64_t;

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
 */

namespace tammx {

// inline void Expects(bool cond) {
//   assert(cond);
// }

#define Expects(cond) assert(cond)

template<typename T, int maxsize>
class BoundVec : public std::array<T, maxsize> {
 public:
  using size_type = typename std::array<T, maxsize>::size_type;
  using value_type = typename std::array<T, maxsize>::value_type;
  using iterator =  typename std::array<T, maxsize>::iterator;
  using const_iterator = typename std::array<T, maxsize>::const_iterator;
  using reverse_iterator = typename std::array<T, maxsize>::reverse_iterator;
  using const_reverse_iterator = typename std::array<T, maxsize>::const_reverse_iterator;
  using reference =  typename std::array<T, maxsize>::reference;
  using const_reference =  typename std::array<T, maxsize>::const_reference;

  using std::array<T, maxsize>::begin;
  using std::array<T, maxsize>::rend;

  BoundVec()
      : size_{0} {}

  explicit BoundVec(size_type count,
           const T& value = T())
      : size_{0} {
    for(size_type i=0; i<count; i++) {
      push_back(value);
    }
  }

  BoundVec(const BoundVec& bv)
      : size_{0} {
    for (auto &value : bv) {
      push_back(value);
    }
  }

  // BoundVec(BoundVec& bv)
  //     : size_{0} {
  //   for (auto &value : bv) {
  //     push_back(value);
  //   }
  // }

  template<typename Itr>
  BoundVec(Itr first, Itr last)
      : size_{0} {
    for(auto itr = first; itr!= last; ++itr) {
      push_back(*itr);
    }
  }

  BoundVec(std::initializer_list<T> init)
      : size_{0} {
    for(auto v: init) {
      push_back(v);
    }
  }

  size_type size() const {
    return size_;
  }

  size_type max_size() const {
    return maxsize;
  }

  bool empty() const {
    return size_ == 0;
  }

  void clear() {
    size_ = 0;
  }

  void push_back( const T& value) {
    this->at(size_++) = value;
  }

  // void push_back(T& value) {
  //   this->at(size_++) = value;
  // }

  void push_back(T&& value ) {
    this->at(size_++) = value;
  }

  void pop_back() {
    size_ -= 1;
  }
  
  template<typename InputIt>
  void insert_back(InputIt first, InputIt last) {
    Expects(size_ + (last - first) <= maxsize);
    for(auto itr = first; itr != last; ++itr) {
      push_back(*itr);
    }
  }

  void insert_back(size_type count, const T& value) {
    Expects(size_ + count <= maxsize);
    for(size_type i=0; i<count; i++) {
      push_back(value);
    }
  }

  // BoundVec<T, maxsize>& operator = (BoundVec<T, maxsize>& bvec) {
  //   size_ = bvec.size_;
  //   std::copy(bvec.begin(), bvec.end(), begin());
  //   return *this;
  // }

  BoundVec<T, maxsize>& operator = (const BoundVec<T, maxsize>& bvec) {
    size_ = bvec.size_;
    std::copy(bvec.begin(), bvec.end(), begin());
    return *this;
  }

  iterator end() {
    return std::array<T, maxsize>::begin() + size_;
  }

  const_iterator end() const {
    return std::array<T, maxsize>::begin() + size_;
  }

  reverse_iterator rbegin() const {
    return std::array<T, maxsize>::begin() + size_;
  }

  reference front() {
    Expects(size_>0);
    return this->at(0);
  }

  const_reference front() const {
    Expects(size_>0);
    return this->at(0);
  }

  reference back() {
    Expects(size_>0);
    return this->at(size_-1);
  }

  const_reference back() const {
    Expects(size_>0);
    return this->at(size_-1);
  }

 private:
  size_type size_;
  friend bool operator == (const BoundVec<T, maxsize>& lhs, const BoundVec<T, maxsize>& rhs) {
    return lhs.size() == rhs.size()
        && std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
  }

  friend bool operator != (const BoundVec<T, maxsize>& lhs, const BoundVec<T, maxsize>& rhs) {
    return !(lhs == rhs);
  }
};

template<typename T, int maxsize>
std::ostream& operator << (std::ostream& os, const BoundVec<T, maxsize>& bvec) {
  os<<"[ ";
  for(auto el: bvec) {
    os<<el<<" ";
  }
  os<<"]";
  return os;
}

using BlockDim = int64_t;
using TensorRank = int;
struct IrrepSpace;
using Irrep = StrongInt<IrrepSpace, int>;
struct SpinSpace;
using Spin = StrongInt<SpinSpace, int>;
using IndexLabel = int;
using Sign = int;

enum class DimType { o, v, n };

inline std::ostream&
operator << (std::ostream& os, DimType dt) {
  switch(dt) {
    case DimType::o:
      os<<"DimType::o";
      break;
    case DimType::v:
      os<<"DimType::v";
      break;
    case DimType::n:
      os<<"DimType::n";
      break;
    default:
      assert(0);
  }
  return os;
}

const TensorRank maxrank{8};

template<typename T>
using TensorVec = BoundVec<T, maxrank>;

using TensorIndex = TensorVec<BlockDim>;
using TensorLabel = TensorVec<IndexLabel>;
using SymmGroup = TensorVec<DimType>;
using TensorDim = TensorVec<DimType>;
using TensorPerm = TensorVec<int>;

template<typename T>
TensorVec<T>
flatten(const TensorVec<TensorVec<T>> vec) {
  TensorVec<T> ret;
  for(auto &v : vec) {
    ret.insert_back(v.begin(), v.end());
  }
  return ret;
}


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
    std::cout<<"TriangleLoop. itr="<<itr_<<std::endl;
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

template<typename T>
struct Combination {
  enum class Case : int { case0=0, case1=1, case2=2 };
  using Int = typename TensorVec<T>::size_type;
  struct StackFrame {
    Case case_value;
    Int i;
    explicit StackFrame(Int ival = 0) {
      case_value = Case::case0;
      i = ival;
    }
    StackFrame(Case csval, Int ival) {
      case_value = csval;
      i = ival;
    }
    void step() {
      case_value  = Case(static_cast<int>(case_value)+1);
    }
    friend bool operator == (const StackFrame& lhs, const StackFrame& rhs) {
      return lhs.case_value == rhs.case_value
          && lhs.i == rhs.i;
    }
    friend bool operator != (const StackFrame& lhs, const StackFrame& rhs) {
      return !(lhs == rhs);
    }    
  };

  typename TensorVec<T>::size_type n_, k_;
  TensorVec<T> bag_;

  Combination()
      : n_{0},
        k_{0} {}

  Combination(const TensorVec<T> &bag, Int k)
      : n_ {bag.size()},
        k_{std::max(k, bag.size() - k)},
        bag_{bag} {
          Expects (n_ > 0);
          Expects (n_ >= k);
          std::sort(bag_.begin(), bag_.end());
        }

  Combination& operator = (const Combination& comb) {
    n_ = comb.n_;
    k_ = comb.k_;
    bag_ = comb.bag_;
    return *this;
  }

  class Iterator {
   public:
    using ItrType = TensorVec<T>;

    Iterator() : comb_{nullptr}  {}

    explicit Iterator(Combination<T>* comb)
        : comb_{comb} {
      for(Int x=0; x<comb_->k_; x++) {
        stack_.push_back({Case::case1, x});
        sub_.push_back(x);
      }
    }

    // Iterator& operator = (Iterator& itr) {
    //   comb_ = itr.comb_;
    //   stack_ = itr.stack_;
    //   sub_ = itr.sub_;
    //   return *this;
    // }

    Iterator& operator = (const Iterator& itr) {
      comb_ = itr.comb_;
      stack_ = itr.stack_;
      sub_ = itr.sub_;
      return *this;
    }

    TensorVec<T> operator *() {
      TensorVec<T> gp1, gp2;
      Expects(sub_.size() == comb_->k_);
      gp2.insert_back(comb_->bag_.begin(), comb_->bag_.begin() + sub_[0]);
      for(Int i=0; i<sub_.size()-1; i++) {
        gp1.push_back(comb_->bag_[sub_[i]]);
        gp2.insert_back(comb_->bag_.begin()+sub_[i]+1,
                   comb_->bag_.begin() + sub_[i+1]);
      }
      gp1.push_back(comb_->bag_[sub_.back()]);
      gp2.insert_back( comb_->bag_.begin()+sub_.back()+1,
                 comb_->bag_.end());
      gp1.insert_back(gp2.begin(), gp2.end());
      return gp1;
    }

    Iterator& operator ++ () {
      do {
        iterate();
      } while(stack_.size()>0 && sub_.size() < comb_->k_);
      if(stack_.size() == 0) {
        assert(sub_.size() == 0);
      }
      return *this;
    }

   private:

    void iterate() {
      Expects(stack_.size() > 0);
      auto case_value = stack_.back().case_value;
      auto i = stack_.back().i;
      Int i1;
      switch(case_value) {
        case Case::case0:
          sub_.push_back(i);
          stack_.back().step();
          if(sub_.size() < comb_->k_ && i+1 < comb_->n_) {
            stack_.push_back(StackFrame{i+1});
          }
          break;
        case Case::case1:
          sub_.pop_back();
          stack_.back().step();
          i1 = comb_->index_of_next_unique_item(i);
          if(i1 < comb_->n_) {
            stack_.push_back(StackFrame{i1});
          }
          break;
        case Case::case2:
          stack_.pop_back();
          break;
      }
    }

   public:
    TensorVec<Int> sub_;
    TensorVec<StackFrame> stack_;
    Combination* comb_;

    friend bool operator == (const typename Combination::Iterator& itr1,
                             const typename Combination::Iterator& itr2) {
      return (itr1.comb_ == itr2.comb_)
          &&  std::equal(itr1.stack_.begin(), itr1.stack_.end(),
                         itr2.stack_.begin(), itr2.stack_.end())
          &&  std::equal(itr1.sub_.begin(), itr1.sub_.end(),
                         itr2.sub_.begin(), itr2.sub_.end());
    }

    friend bool operator != (const typename Combination::Iterator& itr1,
                             const typename Combination::Iterator& itr2) {
      return !(itr1 == itr2);
    }

  };

  Combination<T>::Iterator begin() {
    return Iterator(this);
  }

  Combination<T>::Iterator end() {
    auto itr = Iterator(this);
    itr.stack_.clear();
    itr.sub_.clear();
    return itr;
  }


  Int index_of_next_unique_item(Int i) const {
    unsigned j;
    for(j = i+1; j<n_ && bag_[j] == bag_[i]; j++) {
      // no-op
    }
    return j;
  }

};


/**
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
          Expects(ifirst.size() == ilast.size());
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
    }
    return *this;
  }

  ProductIterator operator ++ (int) {
    auto &ret = *this;
    ++ *this;
    return ret;
  }

  typename Itr::ItrType operator * ()  {
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


class LabelMap {
 public:
  LabelMap& update(const TensorLabel& labels,
                   const TensorIndex& ids) {
    Expects(labels.size() + labels_.size()  <= labels_.max_size());
    labels_.insert_back(labels.begin(), labels.end());
    ids_.insert_back(ids.begin(), ids.end());
    return *this;
  }

  TensorIndex get_blockid(const TensorLabel& labels) const {
    TensorIndex ret(labels.size());
    using size_type = TensorLabel::size_type;
    for(size_type i=0; i<labels.size(); i++) {
      auto itr = std::find(begin(labels_), end(labels_), labels[i]);
      Expects(itr != end(labels_));
      ret[i] = ids_[itr - begin(labels_)];
    }
    return ret;
  }

 private:
  TensorRank rank_{};
  TensorLabel labels_;
  TensorIndex ids_;
};


// class SymmetrizationIterator {
//  public:
//   using iterator_category = std::forward_iterator_tag;
//   using ItrType = TensorLabel;

//   SymmetrizationIterator(const TensorLabel& labels0,
//                          const TensorLabel& labels1,
//                          const TensorLabel& input_label,
//                          const LabelMap& lmap) :
//       input_label_{input_label},
//       lmap_{lmap},
//       perm_(labels0.size() + labels1.size()) {
//         labels01_ = labels0;
//         labels01_.insert_back(labels1.begin(), labels1.end());

//         rank0_ = labels0.size();
//         rank1_ = labels1.size();
//         rank_ = rank0_ + rank1_;

//         //perm_ = std::vector<int>(rank_);
//         std::fill(perm_.begin(), perm_.begin()+rank0_, 0);
//         std::fill(perm_.begin()+rank0_, perm_.end(), 1);
//         update_ival();
//       }

//   SymmetrizationIterator(const SymmetrizationIterator& sit) = default;
//   ~SymmetrizationIterator() = default;
//   SymmetrizationIterator& operator = (const SymmetrizationIterator& itr)  = default;

//   SymmetrizationIterator& operator ++ () {
//     while(std::next_permutation(perm_.begin(), perm_.end())) {
//       update_ival();
//       if(std::is_sorted(ival_.begin(), ival_.end())) {
//         break;
//       }
//     }
//     return *this;
//   }

//   SymmetrizationIterator operator ++ (int) {
//     auto &ret = *this;
//     ++ *this;
//     return ret;
//   }

//   TensorRank size() const {
//     return rank_;
//   }

//   const TensorIndex& operator * () const {
//     return ival_;
//   }

//   SymmetrizationIterator get_end() const {
//     SymmetrizationIterator sit {*this};
//     std::prev_permutation(sit.perm_.begin(), sit.perm_.end());
//     sit.update_ival();
//     return sit;
//   }

//  private:
//   void update_ival() {
//     auto ilabel = apply(labels01_, perm_);
//     ival_ = lmap_.get_blockid(ilabel);
//   }

//   TensorLabel apply(TensorLabel& label,
//                     TensorPerm &perm) const {
//     TensorLabel ret(label.size());
//     int pos[2] = {0, rank0_};
//     for(int i=0; i<label.size(); i++) {
//       ret[i] = label[pos[perm[i]]++];
//     }
//     return ret;
//   }

//   friend bool operator==(const SymmetrizationIterator& itr1, const SymmetrizationIterator& itr2);

//   friend bool operator!=(const SymmetrizationIterator& itr1, const SymmetrizationIterator& itr2);

//   TensorLabel labels01_;
//   const TensorLabel input_label_;
//   TensorPerm perm_;

//   int rank0_, rank1_, rank_;
//   LabelMap lmap_;
//   TensorIndex ival_;
// };

// class SymmetrizationIterator {
//  public:
//   using iterator_category = std::forward_iterator_tag;
//   using ItrType = TensorLabel;

//   SymmetrizationIterator() {
//     // no-op
//   }

//   SymmetrizationIterator(const TensorVec<TensorLabel>& part_labels,
//                          const TensorLabel& input_label,
//                          const LabelMap& lmap) :
//       input_label_{input_label},
//       lmap_{lmap} {
//         // @todo for now only handle symmetrization of two parts
//         Expects(part_labels.size() <= 2);
//         perm_.insert_back(part_labels[0].size(), 1);
//         perm_.insert_back(part_labels[1].size(), 0);
//         for(int i=0; i<part_labels.size(); i++) {
//           flat_labels_.insert_back(part_labels[i].begin(), part_labels[i].end());
//         }
//         //exclusive prefix sum (aka scan) of offs_
//         // offs_.push_back(0);
//         // for(int i=1; i<part_labels.size(); i++) {
//         //   offs_.push_back(offs_.back() + part_labels[i-1].size());
//         // }
//         update_ival();
//       }

//   SymmetrizationIterator& operator = (const SymmetrizationIterator& itr)  {
//     input_label_ = itr.input_label_;
//     lmap_ = itr.lmap_;
//     perm_ = itr.perm_;
//     flat_labels_ = itr.flat_labels_;
//     //offs_ = itr.offs_;
//     update_ival();
//     return *this;
//   }

//   SymmetrizationIterator(const SymmetrizationIterator& sit) = default;
//   ~SymmetrizationIterator() = default;

//   SymmetrizationIterator& operator ++ () {
//     while(std::next_permutation(perm_.begin(), perm_.end())) {
//       update_ival();
//       //  if(std::is_sorted(ival_.begin(), ival_.end())) {
//       //   break;
//       // }
//     }
//     return *this;
//   }

//   SymmetrizationIterator operator ++ (int) {
//     auto &ret = *this;
//     ++ *this;
//     return ret;
//   }

//   TensorRank size() const {
//     return rank_;
//   }

//   const TensorIndex& operator * () const {
//     return ival_;
//   }

//   SymmetrizationIterator get_end() const {
//     SymmetrizationIterator sit {*this};
//     std::prev_permutation(sit.perm_.begin(), sit.perm_.end());
//     sit.update_ival();
//     return sit;
//   }

//  private:
//   void update_ival() {
//     ival_.clear();
//     for(int i=0; i<perm_.size(); i++) {
//       if(perm_[i]) {
//         ival_.push_back(flat_labels_[i]);
//       }
//     }
//     for(int i=0; i<perm_.size(); i++) {
//       if(perm_[0]) {
//         ival_.push_back(flat_labels_[i]);
//       }
//     }
//   }

//   TensorLabel apply(TensorLabel& label,
//                     TensorPerm &perm) const {
//     TensorLabel ret(label.size());
//     auto pos = offs_;
//     for(int i=0; i<label.size(); i++) {
//       ret[i] = label[pos[perm[i]]++];
//     }
//     return ret;
//   }

//   friend bool operator==(const SymmetrizationIterator& itr1, const SymmetrizationIterator& itr2);

//   friend bool operator!=(const SymmetrizationIterator& itr1, const SymmetrizationIterator& itr2);

//   TensorLabel flat_labels_;
//   TensorVec<int> offs_;
//   TensorLabel input_label_;
//   TensorPerm perm_;

//   int rank0_, rank1_, rank_;
//   LabelMap lmap_;
//   TensorIndex ival_;
// };


// bool
// operator == (const SymmetrizationIterator& itr1,
//              const SymmetrizationIterator& itr2) {
//   return std::equal(itr1.perm_.begin(), itr1.perm_.end(), itr2.perm_.begin());
// }

// bool
// operator != (const SymmetrizationIterator& itr1,
//              const SymmetrizationIterator& itr2) {
//   return !(itr1 == itr2);
// }

inline std::pair<BlockDim, BlockDim>
tensor_index_range(DimType dt);

inline ProductIterator<TriangleLoop>
loop_iterator(const TensorVec<SymmGroup>& indices ) {
  TensorVec<TriangleLoop> tloops, tloops_last;
  for(auto &sg: indices) {
    BlockDim lo, hi;
    std::tie(lo, hi) = tensor_index_range(sg[0]);
    tloops.push_back(TriangleLoop{sg.size(), lo, hi});
    tloops_last.push_back(tloops.back().get_end());
  }
  std::cerr<<"loop itr size="<<tloops.size()<<std::endl;
  return ProductIterator<TriangleLoop>(tloops, tloops_last);
}


class TCE {
 public:
  static void init(const std::vector<Spin>& spins,
                   const std::vector<Irrep>& spatials,
                   const std::vector<size_t>& sizes,
                   BlockDim noa,
                   BlockDim noab,
                   BlockDim nva,
                   BlockDim nvab,
                   bool spin_restricted,
                   Irrep irrep_f,
                   Irrep irrep_v,
                   Irrep irrep_t,
                   Irrep irrep_x,
                   Irrep irrep_y) {
    spins_ = spins;
    spatials_ = spatials;
    sizes_ = sizes;
    noa_ = noa;
    noab_ = noab;
    nva = nva_;
    nvab_ = nvab;
    spin_restricted_ = spin_restricted;
    irrep_f_ = irrep_f;
    irrep_v_ = irrep_v;
    irrep_t_ = irrep_t;
    irrep_x_ = irrep_x;
    irrep_y_ = irrep_y;
  }

  static void finalize() {
    // no-op
  }

  static Spin spin(BlockDim block) {
    return spins_[block];
  }

  static Irrep spatial(BlockDim block) {
    return spatials_[block];
  }

  static size_t size(BlockDim block) {
    return sizes_[block];
  }

  static bool restricted() {
    return spin_restricted_;
  }

  static BlockDim noab() {
    return noab_;
  }

  static BlockDim nvab() {
    return nvab_;
  }

  using Int = Fint;

  static Int compute_tce_key(const TensorDim& flindices,
                             const TensorIndex& is) {
    //auto flindices = flatten(indices);
    TensorVec<Int> offsets(flindices.size()), bases(flindices.size());
    std::transform(flindices.begin(), flindices.end(), offsets.begin(),
                   [] (DimType dt) -> Int {
                     if (dt == DimType::o) {
                       return noab();
                     } else if (dt == DimType::v) {
                       return nvab();
                     } else if (dt == DimType::n) {
                       return noab() + nvab();
                     } else {
                       assert(0); //implement
                     }
                   });

    std::transform(flindices.begin(), flindices.end(), bases.begin(),
                   [] (DimType dt) -> Int {
                     if (dt == DimType::o) {
                       return 1;
                     } else if (dt == DimType::v) {
                       return noab() + 1;
                     } else if (dt == DimType::n) {
                       return 1;
                     } else {
                       assert(0); //implement
                     }
                   });

    int rank = flindices.size();
    Int key = 0, offset = 1;
    for(int i=rank-1; i>=0; i--) {
      key += (is[i] - bases[i]) * offset;
      offset *= offsets[i];
    }
    return key;
  }

  // class Tensor {
  //   enum class DistType { nw, nwi, nwma };
  //   enum class AllocationPolicy { construct, attach };

  //   Tensor()
  //       : constructed_{false} {}

  //   Tensor(const std::vector<SymmGroup> &indices)
  //       : indices_{indices},
  //         constructed_{false} {}

  //   void construct() {
  //     ProductIterator<TriangleLoop> pdt =  loop_iterator(indices_);
  //     //ProductIterator<TraingleLoop> pdt =  tensor.iterator();
  //     auto last = pdt.get_end();
  //     int length = count_if(pdt, last, [] (const TensorIndex& id) {
  //         return ta.nonzero(id);
  //       });
  //     offset_map_ = new Fint [2 * length + 1];
  //     offset_map_[0] = length;
  //     //start over
  //     pdt =  loop_iterator(indices_);
  //     last = pdt.get_end();
  //     size_t size = 0, addr = 1;
  //     for(auto itr = pdt; itr != last; ++itr) {
  //       auto blockid = *itr;
  //       if(tc.nonzero(blockid)) {
  //           size_t key = compute_tce_key(blockid);
  //           offset_map_[addr] = key;
  //           offset_map_[length + addr] = size;
  //           size += tc.size(blockid);
  //           addr += 1;
  //         }
  //     }
  //     policy_ = AllocationPolicy::construct;
  //     constructed_ = true;
  //   }

  //   void attach(Fint fma_offset_index, Fint *offset_map, Fint *hash, Fint fma_handle, Fint array_handle) {
  //     ga_ = array_handle;
  //     offset_index_ = fma_offset_index;
  //     offset_handle_ = fma_handle;
  //     offset_map_ = offset_map;
  //     policy_ = AllocationPolicy::attach;
  //     constructed_ = true;
  //   }

  //   void detach() {
  //     Expects(constructed_ && policy_==AllocationPolicy::attach);
  //     constructed_ = false;
  //   }

  //   void destruct() {
  //     Expects(constructed_ && policy_==AllocationPolicy::attach);
  //     tamm::gmem::destroy(ga_);
  //     free(offset_map_);
  //     constructed_ = false;
  //   }

  //   void get(int rank,
  //            const std::array<BlockDim, maxrank> &is, void *buf, size_t size) {
  //     Expects(constructed_);
  //     size_t key = compute_tce_key(indices, rank, is);
  //     if (distribution_ == DistType::tce_nwi) {
  //       assert(Variables::intorb() != 0);
  //       Expects(rank_ == 4);
  //       tamm::cget_hash_block_i(ga(), buf, size, offset_index(), key, is);
  //     } else if (distribution_ == DistType::tce_nwma) {
  //       tamm::cget_hash_block_ma(ga(), buf, size, offset_index(), key);
  //     } else if (distribution_ == DistType::tce_nw) {
  //       tamm::cget_hash_block(ga(), buf, size, offset_index(), key);
  //     } else {
  //       assert(false);
  //     }
  //   }

  //   void add(int rank,
  //            const std::array<BlockDim, maxrank> &is, double *buf, size_t size) {
  //     Expects(constructed_ == true);
  //     size_t key = compute_tce_key(indices, rank, is);
  //     cadd_hash_block(ga(), buf, size, hash, key);
  //   }


  //   ~Tensor() {
  //     Expects(constructed_ == false);
  //   }

  //  private:
  //   std::vector<SymmGroup> indices_;
  //   size_t offset_index_;
  //   Fint *offset_map_;
  //   Fint *hash_;
  //   int offset_handle_;
  //   bool constructed_;
  //   DistType distribution_;
  //   AllocationPolicy policy_;
  //   tamm::gmem::Handle ga;
  // };

 private:
  static std::vector<Spin> spins_;
  static std::vector<Irrep> spatials_;
  static std::vector<size_t> sizes_;
  static bool spin_restricted_;
  static Irrep irrep_f_, irrep_v_, irrep_t_;
  static Irrep irrep_x_, irrep_y_;
  static BlockDim noa_, noab_;
  static BlockDim nva_, nvab_;
};

inline std::pair<BlockDim, BlockDim>
tensor_index_range(DimType dt) {
  switch(dt) {
    case DimType::o:
      return {0, TCE::noab()};
      break;
    case DimType::v:
      return {TCE::noab(), TCE::noab()+TCE::nvab()};
      break;
    case DimType::n:
      return {0, TCE::noab() + TCE::nvab()};
      break;
    default:
      assert(0);
  }
}

class Tensor;
class Block;
struct LabeledBlock {
  Block& block_;
  TensorLabel label_;

  void init(double value);
};

struct LabeledTensor  {
  Tensor &tensor_;
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
    return LabeledBlock{*this, label};
  }

  LabeledBlock operator () () {
    TensorLabel label(block_id_.size());
    std::iota(label.begin(), label.end(), 0);
    return this->operator ()(label); //LabeledBlock{*this, label};
  }

  size_t size() const {
    size_t sz = 1;
    for(auto x : block_dims_) {
      sz *= x;
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

 private:
  Tensor& tensor_;
  TensorIndex block_id_;
  TensorIndex block_dims_;
  std::unique_ptr<uint8_t []> buf_;
  TensorPerm layout_;
  Sign sign_;

  friend void operator += (LabeledBlock& block1, const LabeledBlock& block2);
  friend void operator += (LabeledBlock& block1, const std::pair<LabeledBlock, LabeledBlock>& blocks);
};

inline TensorPerm
perm_compute(const TensorLabel& from, const TensorLabel& to) {
  TensorPerm layout;

  for(auto p : to) {
    auto itr = std::find(from.begin(), from.end(), p);
    Expects(itr != from.end());
    layout.push_back(itr - from.begin());
  }
  return layout;
}

inline int
perm_count_inversions(const TensorPerm& perm) {
  int num_inversions = 0;
  using size_type = TensorPerm::size_type;
  for(size_type i=0; i<perm.size(); i++) {
    auto itr = std::find(perm.begin(), perm.end(), i);
    Expects(itr != perm.end());
    num_inversions += (itr - perm.begin()) - i;
  }
  return num_inversions;
}

template<typename T>
inline TensorVec<T>
perm_apply(const TensorVec<T>& label, const TensorPerm& perm) {
  TensorVec<T> ret;
  Expects(label.size() == perm.size());
  using size_type = TensorPerm::size_type;
  for(size_type i=0; i<label.size(); i++) {
    ret.push_back(label[perm[i]]);
  }
  return ret;
}

inline TensorPerm
perm_compose(const TensorPerm& p1, const TensorPerm& p2) {
  TensorPerm ret(p1.size());
  Expects(p1.size() == p2.size());
  for(unsigned i=0; i<p1.size(); i++) {
    ret[i] = p1[p2[i]];
  }
  return ret;
}

inline TensorPerm
perm_invert(const TensorPerm& perm) {
  TensorPerm ret(perm.size());
  for(unsigned i=0; i<perm.size(); i++) {
    auto itr = std::find(perm.begin(), perm.end(), i);
    Expects(itr != perm.end());
    ret[i] = itr - perm.begin();
  }
  return ret;
}

class Tensor {
 public:
  enum class Distribution { tce_nw, tce_nwma, tce_nwi };
  enum class AllocationPolicy { none, create, attach };
  enum class Type { integer, single_precision, double_precision};

  // class Request {
  //  public:
  //   Block force();
  //  private:
  //   Tensor &tensor_;
  // };

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

  AllocationPolicy allocation_policy() const {
    return policy_;
  }

  const TensorVec<SymmGroup>& indices() const {
    return indices_;
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
        std::cout<<x++<<std::endl;
        std::cout<<"allocate. itr="<<*itr<<std::endl;
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
          std::cout<<"allocate. set keys. itr="<<*itr<<std::endl;
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
    return std::accumulate(blockdims.begin(), blockdims.end(), 1, std::multiplies<int>());
  }

  TensorIndex block_dims(const TensorIndex &blockid) const {
    TensorIndex ret;
    for(auto b : blockid) {
      ret.push_back(TCE::size(b));
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
    Expects(constructed_);
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
    Sign sign = compute_sign_from_unique_block(blockid);
    Block block = alloc(uniq_blockid, layout, sign);
    if(distribution_ == Distribution::tce_nwi
       || distribution_ == Distribution::tce_nw
       || distribution_ == Distribution::tce_nwma) {
      auto key = TCE::compute_tce_key(flindices_, uniq_blockid);
      auto size = block.size();

      if (distribution_ == Distribution::tce_nwi) {
        Expects(rank_ == 4);
        std::vector<size_t> is { &block.blockid()[0], &block.blockid()[rank_]};
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

  Sign compute_sign_from_unique_block(const TensorIndex& blockid) const {
    int num_inversions=0;
    int pos = 0;
    for(auto &igrp: indices_) {
      Expects(igrp.size() <= 2); // @todo Implement general algorithm
      if(igrp.size() == 2 && blockid[pos+0] > blockid[pos+1]) {
        num_inversions += 1;
      }
      pos += igrp.size();
    }
    return (num_inversions%2) ? -1 : 1;
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

  LabeledTensor operator () (const TensorLabel& perm) {
    return LabeledTensor{*this, perm};
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
             int sign)
    : tensor_{tensor},
      block_id_{block_id},
      block_dims_{block_dims},
      layout_{layout},
      sign_{sign} {
        buf_ = std::make_unique<uint8_t []> (size() * tensor.element_size());
      }

inline
Block::Block(Tensor &tensor,
             const TensorIndex& block_id)
    : tensor_{tensor},
      block_id_{block_id} {
        block_dims_ = tensor.block_dims(block_id);
        sign_ = 1;
        buf_ = std::make_unique<uint8_t []> (size() * tensor.element_size());
      }

inline void
LabeledBlock::init(double value) {
  auto *dbuf = reinterpret_cast<double*>(block_.buf());
  for(unsigned i=0; i<block_.size(); i++) {
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
operator += (LabeledBlock block1, const std::tuple<double, const LabeledBlock>& rhs);

void
operator += (LabeledBlock block1, const std::tuple<double, LabeledBlock, LabeledBlock>& rhs);

inline void
operator += (LabeledBlock block1, const LabeledBlock& block2) {
  block1 += 1.0 * block2;
}

inline void
operator += (LabeledBlock block1, const std::tuple<LabeledBlock, LabeledBlock>& rhs) {
  block1 += 1.0 * rhs;
}


inline std::tuple<double, const LabeledTensor>
operator * (double alpha, const LabeledTensor& block) {
  return {alpha, block};
}

inline std::tuple<double, const LabeledTensor, const LabeledTensor>
operator * (const std::tuple<double, LabeledTensor>& rhs1, const LabeledTensor& rhs2)  {
  return std::tuple_cat(rhs1, std::make_tuple(rhs2));
}

inline std::tuple<const LabeledTensor, const LabeledTensor>
operator * (const LabeledTensor& rhs1, const LabeledTensor& rhs2)  {
  return std::make_tuple(rhs1, rhs2);
}

inline std::tuple<double, const LabeledTensor, const LabeledTensor>
operator * (double alpha, const std::tuple<LabeledTensor, LabeledTensor>& rhs) {
  return std::tuple_cat(std::make_tuple(alpha), rhs);
}

void
operator += (LabeledTensor block1, const std::tuple<double, const LabeledTensor>& rhs);

void
operator += (LabeledTensor block1, const std::tuple<double, LabeledTensor, LabeledTensor>& rhs);

inline void
operator += (LabeledTensor block1, const LabeledTensor& block2) {
  block1 += 1.0 * block2;
}

inline void
operator += (LabeledTensor block1, const std::tuple<LabeledTensor, LabeledTensor>& rhs) {
  block1 += 1.0 * rhs;
}


// inline TensorIndex compute_blockid(int rank,
//                                    const TensorIndex& ids,
//                                    const TensorIndex& values,
//                                    const TensorIndex& out_ids) {
//   Expects(rank <= maxrank);
//   TensorIndex ret;

//   for(int i=0; i<rank; i++) {
//     auto oid = out_ids[i];
//     auto itr = std::find(begin(ids), end(ids), oid);
//     Expects(itr != end(ids));
//     ret[i] = values[itr - begin(ids)];
//   }
//   return ret;
// }


// @todo Parallelize
template<typename Itr, typename Fn>
void  parallel_work(Itr first, Itr last, Fn fn) {
  for(; first != last; ++first) {
    fn(*first);
  }
}

inline TensorVec<TensorLabel>
group_labels(const TensorVec<SymmGroup>& groups, const TensorLabel& labels) {
  // std::accumulate(groups.begin(), groups.end(), 0,
  //                 [] (const SymmGroup& sg, int sz) {
  //                   return sg.size() + sz;
  //                 });
  unsigned sz = 0;
  for(auto v : groups) {
    sz += v.size();
  }
  Expects(sz == labels.size());

  int pos = 0;
  TensorVec<TensorLabel> ret;
  for(auto sg : groups) {
    ret.push_back(TensorLabel{labels.begin()+pos, labels.begin()+pos+sg.size()});
    pos += sg.size();
  }
  return ret;
}

// inline std::tuple<TensorVec<TensorDim>, TensorLabel>
// group_partition(const TensorVec<TensorDim>& indices1,
//                 const TensorLabel& label1,
//                 const TensorVec<TensorDim>& indices2,
//                 const TensorLabel& label2) {
//   auto label_groups_1 = group_labels(indices1, label1);
//   auto label_groups_2 = group_labels(indices2, label2);
//   auto flindices1 = flatten(indices1);

//   TensorVec<TensorDim> ret_indices;
//   TensorLabel ret_labels;
//   int pos = 0;
//   for (auto &lg1 : label_groups_1) {
//     for (auto &lg2 : label_groups_2) {
//       SymmGroup sg;
//       for (auto &x : lg1) {
//         int pos1 = 0;
//         for (auto &y : lg2) {
//           if (x == y) {
//             sg.push_back(flindices1[pos + pos1]);
//             ret_labels.push_back(x);
//           }
//         }
//         pos1++;
//       }
//       if (sg.size() > 0) {
//         ret_indices.push_back(sg);
//       }
//     }
//     pos += lg1.size();
//   }
//   return {ret_indices, ret_labels};
// }

template<typename Container>
auto intersect(const Container &ctr1, const Container &ctr2) {
  Container ret;
  for (auto &x : ctr1) {
    for (auto &y : ctr2) {
      if (x == y) {
        ret.push_back(x);
      }
    }
  }
  return ret;
}

inline TensorVec<TensorVec<TensorLabel>>
group_partition(const TensorVec<TensorLabel>& label_groups_1,
                const TensorVec<TensorLabel>& label_groups_2) {
  TensorVec<TensorVec<TensorLabel>> ret_labels;
  for (auto &lg1 : label_groups_1) {
    TensorVec<TensorLabel> ret_group;
    for (auto &lg2 : label_groups_2) {
      auto lbls = intersect(lg1, lg2);
      if (lbls.size() > 0) {
        ret_group.push_back(lbls);
      }
    }
    ret_labels.push_back(ret_group);
  }
  return ret_labels;
}

inline TensorVec<TensorVec<TensorLabel>>
group_partition(const TensorVec<SymmGroup>& indices1,
                const TensorLabel& label1,
                const TensorVec<SymmGroup>& indices2,
                const TensorLabel& label2) {
  auto label_groups_1 = group_labels(indices1, label1);
  auto label_groups_2 = group_labels(indices2, label2);
  return group_partition(label_groups_1, label_groups_2);
}

inline TensorVec<TensorVec<TensorLabel>>
summation_labels(const LabeledTensor& /*ltc*/,
                  const LabeledTensor& lta,
                  const LabeledTensor& ltb) {
  return group_partition(lta.tensor_.indices(), lta.label_,
                         ltb.tensor_.indices(), ltb.label_);

}

inline std::pair<TensorVec<SymmGroup>,TensorLabel>
summation_indices(const LabeledTensor& /*ltc*/,
                  const LabeledTensor& lta,
                  const LabeledTensor& ltb) {
  auto aindices = flatten(lta.tensor_.indices());
  //auto bindices = flatten(ltb.tensor_.indices());
  auto alabels = group_labels(lta.tensor_.indices(), lta.label_);
  auto blabels = group_labels(ltb.tensor_.indices(), ltb.label_);
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

/**
 * @todo Specify where symmetrization is allowed and what indices in
 * the input tensors can form a symmetry group (or go to distinct
 * groups) in the output tensor.
 */
inline TensorVec<TensorVec<TensorLabel>>
nonsymmetrized_external_labels(const LabeledTensor& ltc,
                               const LabeledTensor& lta,
                               const LabeledTensor& ltb) {
  auto ca_labels = group_partition(ltc.tensor_.indices(), ltc.label_,
                                   lta.tensor_.indices(), lta.label_);
  auto cb_labels = group_partition(ltc.tensor_.indices(), ltc.label_,
                                   ltb.tensor_.indices(), ltb.label_);
  Expects(ca_labels.size() == cb_labels.size());
  auto &ret_labels = ca_labels;
  for(unsigned i=0; i<ret_labels.size(); i++) {
    ret_labels[i].insert_back(cb_labels[i].begin(), cb_labels[i].end());
  }
  return ret_labels;

  // auto aindices = flatten(lta.tensor_.indices());
  // auto bindices = flatten(ltb.tensor_.indices());
  // auto cindices = flatten(ltc.tensor_.indices());
  // auto alabels = flatten(group_labels(lta.tensor_.indices(), lta.label_));
  // auto blabels = group_labels(ltb.tensor_.indices(), ltb.label_);
  // auto clabels = group_labels(ltc.tensor_.indices(), ltc.label_);
  // TensorVec<SymmGroup> ret_indices;
  // TensorLabel ext_labels;
  // int pos = 0;
  // for (auto &clg : clabels) {
  //   SymmGroup asg;
  //   for (auto &c : clg) {
  //     for (auto &a : lta.label_) {
  //       if (c == a) {
  //         asg.push_back(cindices[pos++]);
  //         ext_labels.push_back(c);
  //       }
  //     }
  //   }
  //   if (asg.size() > 0) {
  //     ret_indices.push_back(asg);
  //   }

  //   SymmGroup bsg;
  //   pos = 0;
  //   for (auto &c : clg) {
  //     for (auto &b : ltb.label_) {
  //       if (c == b) {
  //         bsg.push_back(cindices[pos++]);
  //         ext_labels.push_back(c);
  //       }
  //     }
  //   }
  //   if (bsg.size() > 0) {
  //     ret_indices.push_back(bsg);
  //   }
  // }
  // return {ret_indices, ext_labels};
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

  auto cflindices = flatten(ltc.tensor_.indices());
  for(unsigned i=0; i<ltc.label_.size(); i++) {
    dim_of_label[ltc.label_[i]] = cflindices[i];
  }
  auto aflindices = flatten(lta.tensor_.indices());
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


// inline ProductIterator<SymmetrizationIterator>
// symmetrization_iterator(const LabeledTensor& ltc,
//                         const LabeledTensor& lta,
//                         const LabeledTensor& ltb,
//                         const LabelMap& lmap) {
//   auto part_labels = nonsymmetrized_external_labels(ltc ,lta, ltb);
//   TensorVec<SymmetrizationIterator> itrs, itrs_last;
//   for(auto lbls: part_labels) {
//     itrs.push_back(SymmetrizationIterator{lbls, ltc.label_, lmap});
//     itrs_last.push_back(itrs.back().get_end());
//   }
//   return ProductIterator<SymmetrizationIterator>(itrs, itrs_last);
// }

class SymmetrizationIterator {
 public:
  SymmetrizationIterator() = default;

  SymmetrizationIterator(const TensorIndex& blockid,
                         int group_size)
      : comb_(blockid, group_size) {}

  SymmetrizationIterator& operator = (const SymmetrizationIterator& sit) = default;

  Combination<BlockDim>::Iterator begin() {
    return comb_.begin();
  }

  Combination<BlockDim>::Iterator end() {
    return comb_.end();
  }

 private:
  Combination<BlockDim> comb_;
};

inline TensorVec<SymmetrizationIterator>
symmetrization_combination(const LabeledTensor& ltc,
                           const LabeledTensor& lta,
                           const LabeledTensor& ltb,
                           const LabelMap& lmap) {
  auto part_labels = nonsymmetrized_external_labels(ltc ,lta, ltb);
  TensorVec<SymmetrizationIterator> sits;
  for(auto lbls: part_labels) {
    Expects(lbls.size()>0 && lbls.size() <= 2);
    TensorLabel lbl(lbls[0].begin(), lbls[0].end());
    int group_size = lbls[0].size();
    if(lbls.size() == 2 ) {
      lbl.insert_back(lbls[1].begin(), lbls[1].end());
    }
    auto blockid = lmap.get_blockid(lbl);
    sits.push_back({blockid, group_size});
  }
  return sits;
}

inline ProductIterator<Combination<BlockDim>::Iterator>
symmetrization_iterator(const TensorVec<SymmetrizationIterator>& sitv) {
  TensorVec<Combination<BlockDim>::Iterator> itrs_first, itrs_last;
  for(auto sit: sitv) {
    itrs_first.push_back(sit.begin());
    itrs_last.push_back(sit.end());
  }
  return {itrs_first, itrs_last};
}

// inline TensorVec<Combination<IndexLabel>>
// symmetrization_copy_combination(const TensorIndex& blockid,
//                                 const LabeledTensor& ltc,
//                                 const LabeledTensor& lta,
//                                 const LabeledTensor& ltb,
//                                 const LabelMap& lmap) {
//   auto part_labels = nonsymmetrized_external_labels(ltc ,lta, ltb);
//   TensorVec<Combination<IndexLabel>> combs;
//   for(auto lbls: part_labels) {
//     Expects(lbls.size()>0 && lbls.size() <= 2);

//     TensorLabel lbl(lbls[0].begin(), lbls[0].end());
//     int group_size = lbls[0].size();
//     if(lbls.size() == 2 ) {
//       lbl.insert_back(lbls[1].begin(), lbls[1].end());
//     }
//     auto blockid = lmap.get_blockid(lbl);

//     Expects(lbl.size() > 0);
//     Expects(lbl.size() <=2); // @todo implement other cases
//     if(lbl.size() == 1) {
//       combs.push_back({lbl, 1});
//     }
//     else if (lbl.size() == 2) {
//       if(lbls[0].size() == 2 || lbls[1].size()==2) {
//         combs.push_back({lbl, 2});
//       }
//       else if(blockid[0] != blockid[1])  {
//         combs.push_back({lbl, 2});
//       }
//       else {
//         combs.push_back({lbl, 1});
//       }
//     }
//   }
//   return combs;
// }

class CopySymmetrizer {
 public:
  using size_type = TensorIndex::size_type;
      
  CopySymmetrizer()
      : CopySymmetrizer(0, 0, TensorIndex{}, TensorIndex{}) {}
        

  CopySymmetrizer(size_type group_size,
                  size_type part_size,
                  const TensorIndex& blockid,
                  const TensorIndex& uniq_blockid)
      : group_size_{group_size},
        part_size_{part_size},
        blockid_{blockid},
        uniq_blockid_{uniq_blockid},
        bag_(group_size) {
          std::iota(bag_.begin(), bag_.end(), 0);
          comb_ = Combination<int>(bag_, part_size);
        }

  CopySymmetrizer& operator = (const CopySymmetrizer& csm) {
    group_size_ = csm.group_size_;
    part_size_ = csm.part_size_;
    blockid_ = csm.blockid_;
    return *this;
  }

  class Iterator {
   public:
    using ItrType = TensorVec<int>;

    Iterator() : cs_{nullptr} {}
 
    explicit Iterator(CopySymmetrizer* cs)
        : cs_{cs} {
      itr_ = cs_->comb_.begin();
      end_ = cs_->comb_.end();
    }

    // Iterator& operator = (Iterator& rhs) = default;

    Iterator& operator = (const Iterator& rhs) = default;

    TensorVec<int> operator * () {
      return *itr_;
    }

    Iterator& operator ++ () {
      do {
        ++itr_;
        auto perm = *itr_;
        auto perm_blockid = perm_apply(cs_->blockid_, perm);
        auto &uniq_blockid = cs_->uniq_blockid_;
        if (std::equal(uniq_blockid.begin(), uniq_blockid.end(),
                       perm_blockid.begin(), perm_blockid.end())) {
          break;
        }
      } while(itr_ != end_);
      return *this;
    }    
    
   private:
    Combination<int>::Iterator itr_, end_;
    CopySymmetrizer *cs_;

    friend bool operator == (const typename CopySymmetrizer::Iterator& itr1,
                             const typename CopySymmetrizer::Iterator& itr2) {
      return (itr1.cs_ == itr2.cs_)
          &&  (itr1.itr_ == itr2.itr_)
          &&  (itr1.end_ == itr2.end_);
    }
    
    friend bool operator != (const typename CopySymmetrizer::Iterator& itr1,
                             const typename CopySymmetrizer::Iterator& itr2) {
      return !(itr1 == itr2);
    }
    friend class CopySymmetrizer;
  };

  Iterator begin() {
    return Iterator(this);
  }

  Iterator end() {
    auto itr = Iterator(this);
    itr.itr_ = comb_.end();
    return itr;
  }

 public:
  size_type group_size_;
  size_type part_size_;
  TensorIndex blockid_;
  TensorIndex uniq_blockid_;
  TensorVec<int> bag_;
  Combination<int> comb_;
};

inline TensorVec<CopySymmetrizer>
copy_symmetrizer(const LabeledTensor& ltc,
                 const LabeledTensor& lta,
                 const LabeledTensor& ltb,
                 const LabelMap& lmap) {
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
    csv.push_back(CopySymmetrizer{size, lbls[0].size(), blockid, uniq_blockid});
  }
  return csv;
}



inline ProductIterator<CopySymmetrizer::Iterator>
copy_iterator(const TensorVec<CopySymmetrizer>& sitv) {
  TensorVec<CopySymmetrizer::Iterator> itrs_first, itrs_last;
  for(auto sit: sitv) {
    itrs_first.push_back(sit.begin());
    itrs_last.push_back(sit.end());
  }
  return {itrs_first, itrs_last};
}


#if 0
inline void
operator += (LabeledTensor ltc, const std::tuple<double, const LabeledTensor>& rhs) {
  double alpha = std::get<0>(rhs);
  const LabeledTensor& lta = std::get<1>(rhs);
  Tensor& ta = lta.tensor_;
  Tensor& tc = ltc.tensor_;
  //check for validity of parameters
  auto citr = loop_iterator(tc.indices());
  auto lambda = [&] (const TensorIndex& cblockid) {
    size_t dimc = tc.block_size(cblockid);
    if(tc.nonzero(cblockid) && dimc>0) {
      auto label_map = LabelMap()
          .update(ltc.label_, cblockid);
      auto ablockid = label_map.get_blockid(lta.label_);
      auto abp = ta.get(ablockid);
      auto cbp = tc.alloc(cblockid);
      cbp(ltc.label_) += alpha * abp(lta.label_);
      tc.add(cbp);
    }
  };
  parallel_work(citr, citr.get_end(), lambda);
}
#endif

/**
 * @todo We assume there is no un-symmetrization in the output.
 */
inline void
operator += (LabeledTensor ltc, const std::tuple<double, const LabeledTensor>& rhs) {
  double alpha = std::get<0>(rhs);
  const LabeledTensor& lta = std::get<1>(rhs);
  Tensor& ta = lta.tensor_;
  Tensor& tc = ltc.tensor_;
  //check for validity of parameters
  auto aitr = loop_iterator(ta.indices());
  auto lambda = [&] (const TensorIndex& ablockid) {
    size_t dima = ta.block_size(ablockid);
    if(ta.nonzero(ablockid) && dima>0) {
      auto label_map = LabelMap()
          .update(lta.label_, ablockid);
      auto cblockid = label_map.get_blockid(ltc.label_);
      auto abp = ta.get(ablockid);
      auto cbp = tc.alloc(cblockid);
      cbp(ltc.label_) += alpha * abp(lta.label_);

      auto csbp = tc.alloc(tc.find_unique_block(cblockid));
      csbp().init(0);
      // @todo make below function also have option to not take ltb
      auto copy_symm = copy_symmetrizer(ltc, lta, ltc, label_map);
      auto copy_itr = copy_iterator(copy_symm);
      auto copy_itr_last = copy_itr.get_end();
      auto copy_label = TensorLabel(ltc.label_.size());
      std::iota(copy_label.begin(), copy_label.end(), 0);
      for(auto citr = copy_itr; citr != copy_itr_last; ++citr) {
        auto perm = *citr;
        auto num_inversions = perm_count_inversions(perm);
        Sign sign = (num_inversions%2) ? -1 : 1;
        csbp(copy_label) += sign * alpha * cbp(perm);
      }
      tc.add(csbp);
    }
  };
  parallel_work(aitr, aitr.get_end(), lambda);
}




/**
 * Check that all iterators and operators work for rank 0 tensors, and rank 0 symmetry groups.
 */

#if 0
inline void operator += (LabeledTensor& ltc, std::tuple<double, LabeledTensor, LabeledTensor> rhs) {
  //check for validity of parameters
  double alpha = std::get<0>(rhs);
  LabeledTensor& lta = std::get<1>(rhs);
  LabeledTensor& ltb = std::get<2>(rhs);
  Tensor& ta = lta.tensor_;
  Tensor& tb = ltb.tensor_;
  Tensor& tc = ltc.tensor_;

  TensorLabel sum_labels;
  TensorVec<SymmGroup> sum_indices;
  std::tie(sum_indices, sum_labels) = summation_indices(ltc, lta, ltb); //label_to_indices(lta, flatten(flatten(sum_labels)));
  //auto nonsymm_external_labels = nonsymmetrized_external_labels(ltc, lta, ltb);
  auto lambda = [&] (const TensorIndex& cblockid) {
    auto sum_itr_first = loop_iterator(sum_indices);
    auto sum_itr_last = sum_itr_first.get_end();
    auto lmap = LabelMap().update(ltc.label_, cblockid);
    auto cbp = tc.alloc(cblockid);
    for(auto sitr = sum_itr_first; sitr!=sum_itr_last; ++sitr) {
      lmap.update(sum_labels, *sitr);
      auto ablockid = lmap.get_blockid(lta.label_);
      auto bblockid = lmap.get_blockid(ltb.label_);
      auto abp = ta.get(ablockid);
      auto bbp = tb.get(bblockid);

      cbp(ltc.label_) += alpha * abp(lta.label_) * bbp(ltb.label_);
    }
    auto cp_combination = symmetrization_copy_combination(cblockid,
                                                          ltc ,lta, ltb, lmap);
    auto cp_itr = copy_iterator(cp_combination);
    auto uniq_cblockid = tc.find_unique_block(cblockid);
    auto csbp = tc.alloc(uniq_cblockid);
    //csbp() = 0;
    for(auto cpitr = cp_itr_first; cp_itr != cp_itr_last; ++cp_itr) {
      cbp(uniq_cblockid) += cbp(ltc.label_);
    }
    tc.add(csbp);
  };
  auto itr = nonsymmetrized_iterator(ltc, lta, ltb);
  parallel_work(itr, itr.get_end(), lambda);
}
#endif

#if 0
inline void operator += (LabeledTensor& ltc, std::tuple<double, LabeledTensor, LabeledTensor> rhs) {
  //check for validity of parameters
  double alpha = std::get<0>(rhs);
  LabeledTensor& lta = std::get<1>(rhs);
  LabeledTensor& ltb = std::get<2>(rhs);
  Tensor& ta = lta.tensor_;
  Tensor& tb = ltb.tensor_;
  Tensor& tc = ltc.tensor_;

  TensorVec<SymmGroup> sum_indices;
  TensorLabel sum_label;
  auto sum_labels = summation_indices(ltc, lta, ltb);
  auto lambda = [&] (const TensorIndex& cblockid) {
    size_t dimc = tc.block_size(cblockid);
    if(tc.nonzero(cblockid) && dimc>0) {
      auto cbp = tc.alloc(cblockid);
      //cbp() = 0;

#if 0
      auto symm_itr_first = symmetization_iterator();
      auto symm_itr_last = symm_itr_first.get_end();
      for(auto symitr = symm_itr_firstl symitr != symm_itr_last; ++symmitr) {
        auto csblockid = ;
        auto cslabel = ;

        auto csbp = tc.alloc(csblockid);
        auto sitr_first = loop_iterator(sum_indices);
        auto sitr_last = sitr_first.get_end();
        for(auto sitr = sitr_first; sitr != sitr_last; sitr++) {
          auto sblockid = *sitr;
          auto label_map = LabelMap()
              .update(cs, csblockid)
              .update(sum_label, sblockid);
          auto ablockid = label_map.get_blockid(lta.label_);
          auto bblockid = label_map.get_blockid(ltb.label_);
          if(!ta.nonzero(ablockid) || !tb.nonzero(bblockid)) {
            continue;
          }
          auto abp = ta.get(ablockid);
          auto bbp = tb.get(bblockid);
          csbp(ext_label) += alpha * abp(lta.label_) * bbp(ltb.label_);
        }

        //handling diagonal blocks
        auto copy_itr_first = copy_iterator();
        auto copy_itr_end = copy_itr_first.get_end();
        for(auto citr = copy_itr_first; citr != copy_itr_last; ++citr) {
          cbp(ltc.label_) += csbp(cslabel);
        }
        tc.add(cbp);
      }
#endif
    }
  };
  auto itr = loop_iterator(tc.indices());
  parallel_work(itr, itr.get_end(), lambda);
}
#endif

/**
 * performs: cbuf[dims] = scale *abuf[perm(dims)]
 */
inline void
index_permute_acc(uint8_t* dbuf, uint8_t* sbuf, const TensorPerm& perm, const TensorIndex& dims, double scale) {
  Expects(dbuf!=nullptr && sbuf!=nullptr);
  Expects(perm.size() == dims.size());

  auto inv_perm = perm_invert(perm);
  TensorVec<size_t> sizes;
  TensorVec<int> iperm;
  for(unsigned i=0; i<dims.size(); i++) {
    sizes.push_back(dims[i]);
    iperm.push_back(inv_perm[i]+1);
  }

  tamm::index_sortacc(reinterpret_cast<double*>(sbuf),
                      reinterpret_cast<double*>(dbuf),
                      sizes.size(), &sizes[0], &perm[0], scale);
}

inline void
index_permute(uint8_t* dbuf, uint8_t* sbuf, const TensorPerm& perm, const TensorIndex& dims, double scale) {
  Expects(dbuf!=nullptr && sbuf!=nullptr);
  Expects(perm.size() == dims.size());

  auto inv_perm = perm_invert(perm);
  TensorVec<size_t> sizes;
  TensorVec<int> iperm;
  for(unsigned i=0; i<dims.size(); i++) {
    sizes.push_back(dims[i]);
    iperm.push_back(inv_perm[i]+1);
  }

  tamm::index_sort(reinterpret_cast<double*>(sbuf),
                   reinterpret_cast<double*>(dbuf),
                   sizes.size(), &sizes[0], &perm[0], scale);
}

inline void
operator += (LabeledBlock clb, const std::tuple<double, const LabeledBlock>& rhs) {
  double alpha = std::get<0>(rhs);
  const LabeledBlock& alb = std::get<1>(rhs);

  auto &ablock = alb.block_;
  auto &cblock = clb.block_;

  auto &clabel = clb.label_;
  auto &alabel = alb.label_;

  auto label_perm = perm_compute(alabel, clabel);
  for(unsigned i=0; i<label_perm.size(); i++) {
    Expects(cblock.block_dims()[i] == ablock.block_dims()[label_perm[i]]);
  }

  auto &alayout = alb.block_.layout();
  auto &clayout = clb.block_.layout();

  auto cstore = perm_apply(clabel, perm_invert(clayout));
  auto astore = perm_apply(alabel, perm_invert(alayout));

  auto store_perm = perm_compute(astore, cstore);
  index_permute_acc(cblock.buf(), ablock.buf(), store_perm, cblock.block_dims(), alpha);
}


template<typename Lambda>
inline void
tensor_map (LabeledTensor& ltc, Lambda func) {
  Tensor& tc = ltc.tensor_;
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
tensor_map (LabeledTensor& ltc, LabeledTensor& lta, Lambda func) {
  Tensor& tc = ltc.tensor_;
  Tensor& ta = lta.tensor_;
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

}  // namespace tammx

#endif  // TAMM_TENSOR_TAMMX_H_

