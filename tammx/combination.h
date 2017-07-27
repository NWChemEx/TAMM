// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_COMBINATION_H__
#define TAMMX_COMBINATION_H__

#include "tammx/expects.h"
#include "tammx/boundvec.h"
#include "tammx/types.h"

namespace tammx {

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
          //Expects (n_ > 0);
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

    Iterator() : comb_{nullptr}, done_{false}  {}

    explicit Iterator(Combination<T>* comb)
        : comb_{comb}, done_{false} {
      for(Int x=0; x<comb_->k_; x++) {
        stack_.push_back({Case::case1, x});
        sub_.push_back(x);
      }
      Expects(sub_.size() == comb_->k_);
    }

    // Iterator& operator = (Iterator& itr) {
    //   comb_ = itr.comb_;
    //   stack_ = itr.stack_;
    //   sub_ = itr.sub_;
    //   return *this;
    // }

    Iterator& operator = (const Iterator& itr) {
      done_ = itr.done_;
      comb_ = itr.comb_;
      stack_ = itr.stack_;
      sub_ = itr.sub_;
      return *this;
    }

    size_t itr_size() const {
      return sub_.size();
    }
    
    TensorVec<T> operator *() {
      TensorVec<T> gp1, gp2;
      Expects(!done_);
      Expects(sub_.size() == comb_->k_);
      if(comb_->k_ > 0) {
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
      }
      return gp1;
    }

    Iterator& operator ++ () {
      Expects(!done_);
      do {
        iterate();
      } while(stack_.size()>0 && sub_.size() < comb_->k_);
      if(stack_.size() == 0) {
        assert(sub_.size() == 0);
        done_ = true;
      } else {
        Expects(sub_.size() == comb_->k_);
      }
      return *this;
    }

   private:

    void iterate() {
      if(comb_->k_ == 0) {
        Expects(sub_.size() == 0);
        return;
      }
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
    bool done_;
    
    friend bool operator == (const typename Combination::Iterator& itr1,
                             const typename Combination::Iterator& itr2) {
      return (itr1.done_ == itr2.done_)
          && (itr1.comb_ == itr2.comb_)
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

  Combination<T>::Iterator begin() const {
    return Iterator(const_cast<Combination<T>*>(this));
  }

  Combination<T>::Iterator end() const {
    auto itr = Iterator(const_cast<Combination<T>*>(this));
    itr.stack_.clear();
    itr.sub_.clear();
    itr.done_ = true;
    return itr;
  }


  Int index_of_next_unique_item(Int i) const {
    unsigned j;
    for(j = i+1; j<n_ && bag_[j] == bag_[i]; j++) {
      // no-op
    }
    return j;
  }

};  // class Combination


}; // namespace tammx

#endif  // TAMMX_WORK_H__
