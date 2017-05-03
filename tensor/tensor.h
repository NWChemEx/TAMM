//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------
#ifndef TAMM_TENSOR_TENSOR_H_
#define TAMM_TENSOR_TENSOR_H_

#include <algorithm>
#include <cassert>
#include <vector>
#include "tensor/index.h"
#include "tensor/variables.h"

namespace tamm {

enum DistType { dist_nwi, dist_nwma, dist_nw };
enum DimType { dim_ov, dim_n };

class Tensor {
 public:
  /**
   * Constructor
   */
  Tensor() {}
  /**
   * Destructor
   */
  ~Tensor() {}

  Tensor(int n, int nupper, int irrep_val, RangeType rt[], DistType dist_type);

  bool attached() const;
  bool allocated() const;
  gmem::Handle ga() const;
  size_t offset_index() const;
  int offset_handle() const;

  /**
   * Get the dimension of this tensor
   * @return dim as a int
   */
  int dim() const;

  int nupper() const;

  /**
   * Get the indices of the tensor
   * @return ids as a vector of Index
   */
  const std::vector<Index> &ids() const;

  /**
   * Get the corresponding irrep value from the global variables
   * @return irrep as integer
   */
  int irrep() const;

  int set_irrep(int irrep);
  int set_dist(DistType dist);

  void get(const std::vector<size_t> &pvalue_r, double *buf, size_t size) const;

  void add(const std::vector<size_t> &pvalue_r, double *buf, size_t size) const;

  bool is_spin_restricted_nonzero(const std::vector<size_t> &ids) const;

  bool is_spin_nonzero(const std::vector<size_t> &ids) const;

  bool is_spatial_nonzero(const std::vector<size_t> &ids) const;

  /**
   * Generate restricted value from value by calling tce_restricted2/4
   */
  void gen_restricted(const std::vector<size_t> &value,
                      std::vector<size_t> *pvalue_r) const;

  void create();
  void attach(Fint fma_offset_index, Fint fma_offset_handle, Fint array_handle);

  void destroy();
  void detach();

 private:
  int dim_;                /*< dimension of this tensor */
  std::vector<Index> ids_; /*< indices of the tensor, actual data */

  bool allocated_; /*true if this tensor were created using create()*/
  bool attached_;
  gmem::Handle ga_;     /*underlying ga if this tensor was created*/
  Fint *offset_map_;    /*offset map used as part of creation*/
  size_t offset_index_; /*index to offset map usable with int_mb */
  int offset_handle_; /*MA handle for the offset map when allocated in fortran*/
  int irrep_;         /*irrep for spatial symmetry*/
  int nupper_;        /* number of upper indices*/

  DistType dist_type_;
  DimType dim_type_;
};  // Tensor

/**
 * Inline implementations
 */

inline bool Tensor::attached() const { return attached_; }

inline bool Tensor::allocated() const { return allocated_; }

inline gmem::Handle Tensor::ga() const { return ga_; }

inline size_t Tensor::offset_index() const { return offset_index_; }

inline int Tensor::offset_handle() const { return offset_handle_; }

inline int Tensor::dim() const { return dim_; }

inline int Tensor::nupper() const { return nupper_; }

inline const std::vector<Index> &Tensor::ids() const { return ids_; }

inline int Tensor::irrep() const { return irrep_; }

inline int Tensor::set_irrep(int irrep) { irrep_ = irrep; }

inline int Tensor::set_dist(DistType dist) { dist_type_ = dist; }

inline bool Tensor::is_spin_restricted_nonzero(
    const std::vector<size_t> &ids) const {
  int lval = std::abs(dim_ - 2 * nupper_);

  assert(ids.size() == dim_);
  int dim_even = dim_ + (dim_ % 2);
  Fint *int_mb = Variables::int_mb();
  size_t k_spin = Variables::k_spin() - 1;
  size_t restricted = Variables::restricted();
  for (int i = 0; i < ids.size(); i++) lval += int_mb[k_spin + ids[i]];
  // assert ((dim_%2==0) || (!restricted) || (lval != 2*dim_even));
  return ((!restricted) || (dim_ == 0) || (lval != 2 * dim_even));
}

inline bool Tensor::is_spin_nonzero(const std::vector<size_t> &ids) const {
  int lval = 0, rval = 0;
  Fint *int_mb = Variables::int_mb();
  Fint k_spin = Variables::k_spin() - 1;
  for (int i = 0; i < nupper_; i++) lval += int_mb[k_spin + ids[i]];
  for (int i = nupper_; i < dim_; i++) rval += int_mb[k_spin + ids[i]];
  return (rval - lval == dim_ - 2 * nupper_);
}

inline bool Tensor::is_spatial_nonzero(const std::vector<size_t> &ids) const {
  Fint lval = 0;
  Fint *int_mb = Variables::int_mb();
  Fint k_sym = Variables::k_sym() - 1;
  for (int i = 0; i < ids.size(); i++) lval ^= int_mb[k_sym + ids[i]];
  return (lval == irrep_);
}

Tensor Tensor0_1(RangeType r1, DistType dt, int irrep);

Tensor Tensor2(RangeType r1, RangeType r2, DistType dt);

Tensor Tensor1_2(RangeType r1, RangeType r2, RangeType r3, DistType dt,
                 int irrep);

Tensor Tensor4(RangeType r1, RangeType r2, RangeType r3, RangeType r4,
               DistType dt);

} /* namespace tamm */

#include <array>
#include <vector>
#include <cassert>
#include <memory>
#include <numeric>
#include <algorithm>
#include <map>
#include "tensor/gmem.h"
#include "tensor/capi.h"

namespace tammx {

inline void Expects(bool cond) {
  assert(cond);
}

template<typename T, int maxsize>
class BoundVec : public std::array<T, maxsize> {
 public:
  using size_type = typename std::array<T, maxsize>::size_type;
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

  BoundVec(size_type count,
           const T& value = T()) {
    for(size_type i=0; i<count; i++) {
      push_back(value);
    }
  }

  BoundVec(const BoundVec& bv) {
    for (auto &value : bv) {
      push_back(value);
    }
  }

  BoundVec(BoundVec& bv) {
    for (auto &value : bv) {
      push_back(value);
    }
  }

  template<typename Itr>
  BoundVec(Itr first, Itr last) {
    for(auto itr = first; itr!= last; ++itr) {
      push_back(*itr);
    }
  }

  BoundVec(std::initializer_list<T> init) {
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

  void push_back(T& value) {
    this->at(size_++) = value;
  }

  void push_back(T&& value ) {
    this->at(size_++) = value;
  }

  template<typename InputIt>
  void insert_back(InputIt first, InputIt last) {
    Expects(size_ + (last - first) <= maxsize);
    for(auto itr = first; itr != last; ++itr) {
      this->at(size_++) = *itr;
    }
  }

  void insert_back(size_type count, const T& value) {
    Expects(size_ + count <= maxsize);
    for(int i=0; i<count; i++) {
      push_back(value);
    }
  }

  BoundVec<T, maxsize>& operator = (BoundVec<T, maxsize>& bvec) {
    size_ = bvec.size_;
    std::copy(bvec.begin(), bvec.end(), begin());
    return *this;
  }

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
};

using BlockDim = int64_t;
using TensorRank = int;
using Irrep = int;
using Spin = int;
using Spatial = int;
using IndexLabel = int;

enum class DimType { o, v, n };

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
    for(i=itr_.size()-1; i>=0 && ++itr_[i] == last_; i--) {
      //no-op
    }
    if(i>=0) {
      for(int j = i+1; j<itr_.size(); j++) {
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

  const ItrType& operator * () const {
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
  Type first_;
  Type last_;
  TensorVec<Type> itr_;

  friend bool operator == (const TriangleLoop& tl1, const TriangleLoop& tl2);
  friend bool operator != (const TriangleLoop& tl1, const TriangleLoop& tl2);
};

bool
operator == (const TriangleLoop& tl1, const TriangleLoop& tl2) {
  return std::equal(tl1.itr_.begin(), tl1.itr_.end(), tl2.itr_.begin());
}

bool
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
    StackFrame(Int ival = 0) {
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

  Combination& operator = (Combination& comb) {
    n_ = comb.n_;
    k_ = comb.k_;
    bag_ = comb.bag_;
    return *this;
  }
  
  class Iterator {
   public:
    using ItrType = TensorVec<T>;

    Iterator() : comb_{nullptr}  {}
    
    Iterator(Combination<T>* comb)
        : comb_{comb} {
      for(Int x=0; x<comb_->k_; x++) {
        stack_.push_back({Case::case1, x});
        sub_.push_back(x);
      }
    }

    Iterator& operator = (Iterator& itr) {
      comb_ = itr.comb_;
      stack_ = itr.stack_;
      sub_ = itr.sub_;
    }

    Iterator& operator = (const Iterator& itr) {
      comb_ = itr.comb_;
      stack_ = itr.stack_;
      sub_ = itr.sub_;
    }

    TensorVec<T> operator *() {
      TensorVec<T> gp1, gp2;
      Expects(sub_.size() == comb_->k_);
      gp2.insert(gp2.end(), comb_->bag_.begin(), comb_->bag_.begin() + sub_[0]);
      for(Int i=0; i<sub_.size()-1; i++) {
        gp1.push_back(comb_->bag_[sub_[i]]);
        gp2.insert(gp2.end(), comb_->bag_.begin()+sub_[i]+1,
                   comb_->bag_.begin() + sub_[i+1]);
      }
      gp1.push_back(comb_->bag_[sub_.back()]);
      gp2.insert(gp2.end(), comb_->bag_.begin()+sub_.back()+1,
                 comb_->bag_.end());
      gp1.insert(gp1.end(), gp2.begin(), gp2.end());
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
            stack_.push_back({i+1});
          }        
          break;
        case Case::case1:
          sub_.pop_back();
          stack_.back().step();
          i1 = comb_->index_of_next_unique_item(i);
          if(i1 < comb_->n_) {
            stack_.push_back({i1});
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

  ~ProductIterator() {}

  ProductIterator& operator = (const ProductIterator& itr) {
    ifirst_ = itr.ifirst_;
    ilast_ = itr.ilast_;
    ival_ = itr.ival_;
    return *this;
  }

  ProductIterator& operator ++ () {
    int i;
    for(int i=ival_.size()-1; i>=0 && ++ival_[i] == ilast_[i]; i--) {
      //no-op
    }
    if(i>=0) {
      for(int j=i+1; j<ival_.size(); j++) {
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

  typename Itr::ItrType operator * () const {
    TensorVec<typename Itr::ItrType> itrs(ival_.size());
    std::transform(ival_.begin(), ival_.end(), itrs.begin(),
                   [] (const Itr& tloop) {
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
    for(int i=0; i<labels.size(); i++) {
      auto itr = std::find(begin(labels_), end(labels_), labels[i]);
      Expects(itr != end(labels_));
      ret[i] = ids_[itr - begin(labels_)];
    }
    return ret;
  }

 private:
  TensorRank rank_;
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

static std::pair<BlockDim, BlockDim>
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
  return ProductIterator<TriangleLoop>(tloops, tloops_last);
}


class TCE {
 public:
  static void init();
  static void finalize();

  static Spin spin(BlockDim block) {
    return spins_[block];
  }

  static Spatial spatial(BlockDim block) {
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
  static std::vector<Spatial> spatials_;
  static std::vector<size_t> sizes_;
  static bool spin_restricted_;
  static Irrep irrep_f_, irrep_v_, irrep_t_;
  static Irrep irrep_x_, irrep_y_;
  static BlockDim noa_, noab_;
  static BlockDim nva_, nvab_;
};

static std::pair<BlockDim, BlockDim>
tensor_index_range(DimType dt) {
  switch(dt) {
    case DimType::o:
      return {1, TCE::noab()+1};
      break;
    case DimType::v:
      return {TCE::noab()+1, TCE::noab()+TCE::nvab()+1};
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
        int sign);

  const TensorIndex& blockid() const {
    return block_id_;
  }

  const TensorIndex& block_dims() const {
    return block_dims_;
  }

  LabeledBlock operator () (const TensorLabel &label) {
    return LabeledBlock{*this, label};
  }

  size_t size() const {
    size_t sz = 1;
    for(auto x : block_dims_) {
      sz *= x;
    }
    return sz;
  }

  int sign() const {
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
  int sign_;

  friend void operator += (LabeledBlock& block1, const LabeledBlock& block2);
  friend void operator += (LabeledBlock& block1, const std::pair<LabeledBlock, LabeledBlock>& blocks);
};

inline std::pair<TensorPer, int>
compute_layout(const TensorIndex& from, const TensorIndex& to) {
  TensorPerm layout;
  int num_inversions;

  for(auto p : to) {
    auto itr = std::find(from.begin(), from.end(), p);
    Expects(itr != from.end());
    layout.push_back(itr - from.begin());
    num_inversions += i;
  }

  return {layout, num_inversions};
}


class Tensor {
 public:
  enum class Distribution { tce_nw, tce_nwma, tce_nwi };
  enum class AllocationPolicy { create, attach };
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
        constructed_{false} {
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

  void attach(tamm::gmem::Handle tce_ga, TCE::Int *tce_hash) {
    Expects (!constructed_);
    Expects (distribution_ == Distribution::tce_nwma
             || distribution_ == Distribution::tce_nwi);
    policy_ = AllocationPolicy::attach;
    tce_ga_ = tce_ga;
    tce_hash_ = tce_hash;
    constructed_ = true;
  }

  void attach(void *tce_data_buf, TCE::Int *tce_hash) {
    Expects (!constructed_);
    Expects (distribution_ == Distribution::tce_nwma);
    policy_ = AllocationPolicy::attach;
    tce_data_buf_ = static_cast<double *>(tce_data_buf);
    tce_hash_ = tce_hash;
    constructed_ = true;
  }

  void allocate() {
    if (distribution_ == Distribution::tce_nwma || distribution_ == Distribution::tce_nw) {
      ProductIterator<TriangleLoop> pdt =  loop_iterator(indices_);
      auto last = pdt.get_end();
      int length = 0;
      for(auto itr = pdt; itr != last; ++itr) {
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
          tce_hash_[addr] = TCE::compute_tce_key(flindices(), blockid);
          tce_hash_[length + addr] = size;
          size += block_size(blockid);
          addr += 1;
        }
      }
      size = (size == 0) ? 1 : size;
      if (distribution_ == Distribution::tce_nw) {
        tce_ga_ = tamm::gmem::create(tamm::gmem::Double, size, std::string{"noname1"});
        tamm::gmem::zero(tce_ga_);
      }
      else {
        tce_data_buf_ = new double [size];
        std::fill_n(tce_data_buf_, size, 0);
      }
    }
    else {
      assert(0); // implement
    }
    policy_ = AllocationPolicy::create;
  }

  size_t block_size(const TensorIndex &blockid) const {
    auto blockdims = block_dims(blockid);
    return std::accumulate(blockdims.begin(), blockdims.end(), 1, std::multiplies<int>());
  }

  TensorIndex block_dims(const TensorIndex &blockid) const {
    int pos = 0;
    TensorIndex ret;
    for(auto b : blockid) {
      ret.push_back(TCE::size(b));
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
        tamm::gmem::destroy(tce_ga_);
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
    int num_inversions;
    assert(0);
    //std::tie(layout, num_inversions) = compute_layout(uniq_blockid, blockid);
    int sign = num_inversions % 2;
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
          tamm::gmem::get(tce_ga_, block.buf(), offset, offset + size - 1);
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
    if(distribution_ == Distribution::tce_nw) {
      auto key = TCE::compute_tce_key(flindices_, block.blockid());
      auto size = block.size();
      auto length = tce_hash_[0];
      auto ptr = std::lower_bound(&tce_hash_[1], &tce_hash_[length + 1], key);
      Expects (!(ptr == &tce_hash_[length + 1] || key < *ptr));
      auto offset = *(ptr + length);
      assert(0); // do the index permutation needed
      tamm::gmem::acc(tce_ga_, block.buf(), offset, offset + size - 1);
    } else {
      assert(0); //implement
    }
  }

  TensorIndex find_unique_block(const TensorIndex& blockid) const {
    TensorIndex ret {blockid};
    int pos = 0;
    for(auto &igrp: indices_) {
      std::sort(ret.begin()+pos, ret.begin()+pos+igrp.size());
    }
    return ret;
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
    Spin spin_upper = 0;
    for(auto itr = std::begin(blockid); itr!= std::begin(blockid) + nupper_indices_; ++itr) {
      spin_upper += TCE::spin(*itr);
    }
    Spin spin_lower = 0;
    for(auto itr = std::begin(blockid)+nupper_indices_; itr!= std::end(blockid); ++itr) {
      spin_lower += TCE::spin(*itr);
    }
    return spin_lower - spin_upper == rank_ - 2 * nupper_indices_;
  }

  bool spatial_nonzero(const TensorIndex& blockid) const {
    Spatial spatial = 0;
    for(auto b : blockid) {
      spatial ^= TCE::spatial(b);
    }
    return spatial == irrep_;
  }

  bool spin_restricted_nonzero(const TensorIndex& blockid) const {
    Spin spin = std::abs(rank_ - 2 * nupper_indices_);
    TensorRank rank_even = rank_ + (rank_ % 2);
    for(auto b : blockid) {
      spin += TCE::spin(b);
    }
    return (!spin_restricted_ || (rank_ == 0) || (spin != 2 * rank_even));
  }

  LabeledTensor operator () (const TensorLabel& perm) {
    return LabeledTensor{*this, perm};
  }

  Type element_type_;
  Distribution distribution_;
  TensorVec<SymmGroup> indices_;
  TensorRank nupper_indices_;
  Irrep irrep_;
  bool spin_restricted_; //spin restricted
  bool constructed_;
  AllocationPolicy policy_;
  TensorRank rank_;
  TensorDim flindices_;

  tamm::gmem::Handle tce_ga_;
  double* tce_data_buf_;
  TCE::Int *tce_hash_;
};  // class Tensor


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

Block::Block(Tensor &tensor,
             const TensorIndex& block_id)
    : tensor_{tensor},
      block_id_{block_id} {
        block_dims_ = tensor.block_dims(block_id);
        sign_ = 1;
        buf_ = std::make_unique<uint8_t []> (size() * tensor.element_size());
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


template<typename Itr, typename Fn>
void  parallel_work(Itr first, Itr last, Fn fn) {
  for(; first != last; ++first) {
    fn(*first);
  }
}

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



inline TensorVec<TensorLabel>
group_labels(const TensorVec<SymmGroup>& groups, const TensorLabel& labels) {
  // std::accumulate(groups.begin(), groups.end(), 0,
  //                 [] (const SymmGroup& sg, int sz) {
  //                   return sg.size() + sz;
  //                 });
  int sz = 0;
  for(auto v : groups) {
    sz += groups.size();
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
  auto bindices = flatten(ltb.tensor_.indices());
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
  for(int i=0; i<ret_labels.size(); i++) {
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
                            const LabeledTensor& lta,
                            const LabeledTensor& ltb) {
  TensorVec<TensorLabel> ret {ltc.label_};
  return {ret};
}


inline ProductIterator<TriangleLoop>
nonsymmetrized_iterator(const LabeledTensor& ltc,
                        const LabeledTensor& lta,
                        const LabeledTensor& ltb) {
  auto part_labels = nonsymmetrized_external_labels(ltc ,lta, ltb);
  auto flat_labels = flatten(flatten(part_labels));
  std::map<IndexLabel, DimType> dim_of_label;

  auto cflindices = flatten(ltc.tensor_.indices());
  for(int i=0; i<ltc.label_.size(); i++) {
    dim_of_label[ltc.label_[i]] = cflindices[i];
  }
  auto aflindices = flatten(lta.tensor_.indices());
  for(int i=0; i<lta.label_.size(); i++) {
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
  SymmetrizationIterator() {}
  
  SymmetrizationIterator(const TensorIndex& blockid,
                         int group_size)
      : comb_(blockid, group_size) {}

  SymmetrizationIterator& operator = (SymmetrizationIterator& sit) {
    comb_ = sit.comb_;
    return *this;
  }
  
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

inline TensorVec<Combination<IndexLabel>>
symmetrization_copy_combination(const TensorIndex& blockid,
                                const LabeledTensor& ltc,
                                const LabeledTensor& lta,
                                const LabeledTensor& ltb,
                                const LabelMap& lmap) {
  auto part_labels = nonsymmetrized_external_labels(ltc ,lta, ltb);
  TensorVec<Combination<IndexLabel>> combs;
  for(auto lbls: part_labels) {
    Expects(lbls.size()>0 && lbls.size() <= 2);

    TensorLabel lbl(lbls[0].begin(), lbls[0].end());
    int group_size = lbls[0].size();
    if(lbls.size() == 2 ) {
      lbl.insert_back(lbls[1].begin(), lbls[1].end());
    }
    auto blockid = lmap.get_blockid(lbl);

    Expects(lbl.size() > 0);
    Expects(lbl.size() <=2); // @todo implement other cases
    if(lbl.size() == 1) {
      combs.push_back({lbl, 1});
    }
    else if (lbl.size() == 2) {
      if(lbls[0].size() == 2 || lbls[1].size()==2) {
        combs.push_back({lbl, 2});
      }
      else if(blockid[0] != blockid[1])  {
        combs.push_back({lbl, 2});
      }
      else {
        combs.push_back({lbl, 1});
      }
    }
  }
  return combs;
}


inline ProductIterator<Combination<IndexLabel>::Iterator>
copy_iterator(const TensorVec<Combination<IndexLabel>>& sitv) {
  TensorVec<Combination<IndexLabel>::Iterator> itrs_first, itrs_last;
  for(auto sit: sitv) {
    itrs_first.push_back(sit.begin());
    itrs_last.push_back(sit.end());
  }
  return {itrs_first, itrs_last};
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

};  // tammx

#endif  // TAMM_TENSOR_TENSOR_H_
