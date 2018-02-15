// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMY_LOOPS_H_
#define TAMMY_LOOPS_H_

#include <vector>
#include <algorithm>
#include <memory>

#include "boundvec.h"
#include "types.h"
#include "errors.h"

namespace tammy {

// template<typename T, typename Itr>
// class Loop {
//  public:
//   //using Iterator = std::vector<T>::const_iterator;
//   virtual ~Loop() {}
//   virtual void reset() = 0;
//   virtual void reset(const std::vector<Itr>& begin) = 0;

//   virtual void get(std::vector<T>& loop_iteration) = 0;
//   virtual void next() = 0;
//   virtual bool has_more() const = 0;
//   virtual Loop<T, Itr>* clone() const = 0;
//   //virtual begin_clamp(const Loop<T,Itr>& ) const = 0;
  
//   virtual size_t size() const = 0;
//   virtual const std::vector<Itr>& begin() const = 0;
//   virtual const std::vector<Itr>& end() const = 0;
//   virtual const std::vector<Itr>& itr_vec() const = 0;

// }; // class Loop

// template<typename T, typename Itr>
// class SimpleLoop : public Loop<T, Itr> {
//  public:
//   SimpleLoop(size_t index, Itr begin, Itr end)
//       : index_{index},
//         begin_{begin},
//         end_{end},
//         step_{begin<=end ? 1 : -1},
//         beginv_{begin},
//         endv_{end},
//         it_{begin},
//         done_{false} {
//           reset();
//           assert(has_more());
//         }

//   SimpleLoop()
//       : SimpleLoop{0, Itr{}, Itr{}} {}

//   SimpleLoop(const SimpleLoop<T, Itr>& sloop)
//       : SimpleLoop{sloop.index_, sloop.begin_, sloop.end_} {
//     *this = sloop;
//   }

//   SimpleLoop<T, Itr>& operator = (const SimpleLoop<T, Itr>& sloop) {
//     index_ = sloop.index_;
//     begin_ = sloop.begin_;
//     end_ = sloop.end_;
//     beginv_ = sloop.beginv_;
//     endv_ = sloop.endv_;
//     itr_vec_ = sloop.itr_vec_;
//     it_ = sloop.it_;
//     done_ = sloop.done_;
//     return *this;
//   }

//   Loop<T, Itr>* clone() const override {
//     return new SimpleLoop<T, Itr>{*this};
//   }
  
//   void reset() override {
//     resey(begin_);
//   }

//   const std::vector<Itr>& begin() const override {
//     return beginv_;
//   }

//   const std::vector<Itr>& end() const override {
//     return endv_;
//   }

//   const std::vector<Itr>& itr_vec() const override {
//     itr_vec_.clear();
//     itr_vec_.push_back(it_);
//     return itr_vec_;
//   }

//   virtual size_t size() const override {
//     return 1;
//   }

//   void reset(const std::vector<Itr>& begin) override {
//     assert(begin.size() == 1);
//     reset(begin[0]);
//   }

//   void reset(Itr cbegin) {
//     if(step_ > 0) {
//       EXPECTS(cbegin < end_);
//     } else {
//       EXPECTS(cbegin > end_);
//     }
//     it_ = cbegin;
//     done_ = false;
//   }

//   bool has_more() const override {
//     return !done_;
//   }

//   void get(std::vector<T>& loop_iteration) override {
//     assert(index_ < loop_iteration.size());
//     loop_iteration[index_] = *it_;
//   }

//   void next() override {
//     if(it_+step_ == end_) {
//       it_ = begin_;
//       done_ = true;
//     } else {
//       it_ += step_;
//     }
//   }
//  protected:
//   size_t index_;
//   Itr begin_, end_;
//   ssize_t step_;
//   std::vector<Itr> beginv_, endv_;
//   mutable std::vector<Itr> itr_vec_;
//   Itr it_;
//   bool done_;  
// };  // SimpleLoop


// template<typename T, typename Itr>
// class CartesianLoop : public Loop<T, Itr> {
//  public:
//   CartesianLoop(std::vector<std::unique_ptr<Loop<T, Itr>>>&& itrs)
//       : itrs_{std::move(itrs)},
//         done_{false} {
//           size_ = 0;
//           offs_.push_back(0);
//           for(const auto& itr: itrs_) {
//             size_ += itr->size();
//             offs_.push_back(size_);
//             begin_.insert(begin_.end(), itr->begin().begin(), itr->begin().end());
//             end_.insert(end_.end(), itr->end().begin(), itr->end().end());
//           }
//           reset();
//         }
  
//   CartesianLoop()
//       : CartesianLoop{std::vector<std::unique_ptr<Loop<T, Itr>>>{}} {}
  
//   CartesianLoop(const CartesianLoop<T, Itr>& cloop) {
//     *this = cloop;
//   }

//   CartesianLoop<T, Itr>& operator = (const CartesianLoop<T, Itr>& cloop) {
//     itrs_.clear();
//     for(const auto& it: cloop.itrs_) {
//       itrs_.push_back(std::move(std::unique_ptr<Loop<T, Itr>>{it->clone()}));
//     }
//     offs_ = cloop.offs_;
//     begin_ = cloop.begin_;
//     end_ = cloop.end_;
//     itr_vec_ = cloop.itr_vec_;
//     size_ = cloop.size_;
//     done_ = cloop.done_;
//     return *this;
//   }

//   virtual ~CartesianLoop() {}
  
//   Loop<T, Itr>* clone() const override {
//     return new CartesianLoop<T, Itr>{*this};
//   }

//   void reset() override {
//     reset(begin_);
//   }

//   size_t size() const override {
//     return size_;
//   }

//   const std::vector<Itr>& begin() const override {
//     return begin_;
//   }

//   const std::vector<Itr>& end() const override {
//     return end_;
//   }

//   const std::vector<Itr>& itr_vec() const override {
//     itr_vec_.clear();
//     for(const auto& itr: itrs_) {
//       const auto& vec = itr->itr_vec();
//       itr_vec_.insert(itr_vec_.end(),
//                       vec.begin(), vec.end());
//     }
//     return itr_vec_;
//   }

//   void reset(const std::vector<Itr>& clo) override {
//     assert(clo.size() == size_);
//     size_t sz = 0;
//     for(size_t i=0; i<itrs_.size(); i++) {
//       std::vector<Itr> ilo(clo.begin()+offs_[i], clo.begin()+offs_[i+1]);
//       itrs_[i]->reset(ilo);
//       sz += itrs_[i]->size();
//     }
//     done_ = false;
//   }

//   bool has_more() const override {
//     return !done_;
//   }

//   void get(std::vector<T>& itv) override {
//     for(auto& itr: itrs_) {
//       itr->get(itv);
//     }
//   }

//   void next() override {
//     int i = itrs_.size()-1;
//     for(; i>=0; i--) {
//       itrs_[i]->next();
//       if (itrs_[i]->has_more()) {
//         break;
//       }
//       itrs_[i]->reset();
//     }
//     if (i<0) {
//       done_ = true;
//     }
//   }
//  protected:
//   std::vector<std::unique_ptr<Loop<T, Itr>>> itrs_;
//   std::vector<size_t> offs_;
//   std::vector<Itr> begin_;
//   std::vector<Itr> end_;
//   mutable std::vector<Itr> itr_vec_;
//   size_t size_;
//   bool done_;
// };  // CartesianLoop

// template<typename T, typename Itr>
// class TriangleLoop : public CartesianLoop<T, Itr> {
//  public:
//   TriangleLoop(std::vector<std::unique_ptr<Loop<T, Itr>>>&& itrs)
//       : CartesianLoop<T, Itr>(std::move(itrs)) {
//     CartesianLoop<T, Itr>::done_ = false;
//     const auto& itrs_ = this->itrs_;
//     if(itrs_.size() > 0) {
//       size_t sz = itrs_[0]->size();
//       for(const auto& itr: itrs_) {
//         assert(sz == itr->size());
//       }
//     }
//   }

//   TriangleLoop():
//       TriangleLoop{std::vector<std::unique_ptr<Loop<T, Itr>>>{}} {}
  
//   TriangleLoop(const TriangleLoop<T, Itr>& tloop)
//       : CartesianLoop<T, Itr>{tloop} {
//     *this = tloop;
//   }

//   TriangleLoop<T, Itr>& operator = (const TriangleLoop<T, Itr>& tloop) {
//     CartesianLoop<T, Itr>::operator = (tloop);
//     clo_ = tloop.clo_;
//     return *this;
//   }

//   Loop<T, Itr>* clone() const override {
//     return new TriangleLoop<T, Itr>{*this};
//   }

//   void reset(const std::vector<Itr>& clo) override {
//     assert(clo.size() == this->size_);
//     clo_ = clo;
//     this->done_ = false;
//     auto& itrs = this->itrs_;
//     if(itrs.size() > 0) {
//       size_t sz = itrs[0]->size();
//       for(size_t i = 0; i<itrs.size(); i++) {
//         std::vector<Itr> rv{clo.begin() + i*sz,
//               clo.begin() + (i+1)*sz};
//         assert(rv.size() == itrs[i]->size());
//         itrs[i]->reset(rv);
//       }
//     }
//   }

//   // @todo @bug std::max cannot be used with the loop has a negative
//   // step (i.e., hi is lesser than lo
//   void next() override {
//     auto& itrs = this->itrs_;
//     int i = itrs.size()-1;
//     for(; i>=0; i--) {
//       itrs[i]->next();
//       if (itrs[i]->has_more()) {
//         break;
//       }
//     }
//     if (i<0) {
//       this->done_ = true;
//     } else {
//       const auto& ivec = itrs[i]->itr_vec();
//       size_t sz = itrs[0]->size();
//       for(size_t j=i+1; j<itrs.size(); j++) {
//         std::vector<Itr> rv{ivec.begin(), ivec.end()};
//         if(clo_.size() > 0) {
//           for(size_t k = 0; k<sz; k++) {
//             rv[k] = std::max(rv[k], clo_[j*sz+k]);
//           }
//         }
//         assert(rv.size() == itrs[j]->size());
//         itrs[j]->reset(rv);
//       }
//     }
//   }
//  protected:
//   std::vector<Itr> clo_;
// };  // TriangleLoop

  
struct LBCondition {
  size_t pos;
  std::vector<size_t> lhs;
  std::vector<size_t> rhs;
};  

template<typename Itr>
class LBLoopNest {
 public:
  using T = typename Itr::value_type;
  using Iterator = Itr;
  
  LBLoopNest(const std::vector<Itr>& begin,
             const std::vector<Itr>& end,
             const std::vector<std::vector<LBCondition>>& lbs)
      : begin_{begin},
        end_{end},
        itr_{begin},
        lbs_{lbs},
        size_{begin.size()} {
          EXPECTS(begin.size() == size_);
          EXPECTS(end.size() == size_);
          EXPECTS(lbs.size() == size_);
          reset();
          for(size_t i = 0; i < size_; i++) {
            EXPECTS(itr_[i] <= end_[i]);
          }
          for(const auto& lb_v : lbs_) {
            for(const auto& lb_el : lb_v) {
              EXPECTS(lb_el.pos < size_);
              EXPECTS(lb_el.lhs.size() == lb_el.rhs.size());
              for(size_t i = 0; i < lb_el.lhs.size(); i++) {
                EXPECTS(lb_el.lhs[i] < lb_el.pos);
                EXPECTS(lb_el.rhs[i] < lb_el.pos);
              }
            }
          }
        }
  
  LBLoopNest():
      LBLoopNest{{}, {}, {}} {}

  LBLoopNest(const LBLoopNest&) = default;
  
  void reset() {
    for(size_t i=0; i<size_; i++) {
      reset(i);
    }
  }

  void get(std::vector<T>& loop_iteration) const {
    loop_iteration.clear();
    auto bi = std::back_inserter(loop_iteration);
    for(const auto& it: itr_) {
      *bi++ = *it;
    }
  }

  std::vector<T> get() const {
    std::vector<T> loop_iteration;
    get(loop_iteration);
    return loop_iteration;
  }

  template<typename OutputIterator>
  void get(OutputIterator oitr) const {
    for(const auto& it: itr_) {
      *oitr++ = *it;
    }
  }

  void next() {
    ssize_t i = static_cast<ssize_t>(size_) - 1;
    for(; i >= 0; i--) {
      itr_[i]++;
      if(itr_[i] <= end_[i]) {
        break;
      }
    }
    if(i < 0) {
      done_ = true;
    } else {
      for(size_t j = i+1; j < size_; j++) {
        reset(j);
      }
    }
  }
  
  bool has_more() const {
    return !done_;
  }  
  
  size_t size() const {
    return size_;
  }  

  void shift_lbs(size_t off) {
    //@todo implement
  }
  
  friend LBLoopNest<Itr>
  loop_nest_concat(TensorVec<LBLoopNest<Itr>> loop_nest_groups,
                   bool lexicographic_concat) {
    if(loop_nest_groups.size() == 1) {
      return loop_nest_groups[0];
    }
    size_t off = 0;
    std::vector<Itr> begin, end;
    std::vector<std::vector<LBCondition>> lbs;
    for(auto& lng : loop_nest_groups) {
      lng.shift_lbs(off);
      begin.insert(begin.end(), lng.begin().begin(), lng.begin().end());
      end.insert(end.end(), lng.end().begin(), lng.end().end());
      if (lexicographic_concat == false) {
        lbs.insert(lbs.end(), lng.lbs().begin(), lng.lbs().end());
      } else {
        if (off == 0) {
          lbs.insert(lbs.end(), lng.lbs().begin(), lng.lbs().end());
        } else {
          for(size_t i = 0; i<lng.size(); i++) {
            size_t pos = off + i - lng.size();
            std::vector<size_t> lhs, rhs;
            for(size_t j = 0; j> i; j++) {
              lhs.push_back(off + j - lng.size());
              rhs.push_back(off + j);
            }
            std::vector<LBCondition> lb{lng.lbs()[i]};
            lb.push_back({pos, lhs, rhs});
            lbs.push_back(lb);
          }
        }
      }
      off += lng.size();
    }
    return {begin, end, lbs};
  }

  
 protected:
  const std::vector<std::vector<LBCondition>>& lbs() const {
    return lbs_;
  }
  
  void reset(size_t j) {
    EXPECTS(j >= 0 && j < size_);
    itr_[j] = begin_[j];
    for(const auto& lb_el : lbs_[j]) {
      bool consider = true;
      for(size_t i = 0; i < lb_el.lhs.size(); i++) {
        if(itr_[lb_el.lhs[i]] < itr_[lb_el.rhs[i]]) {
          consider = false;
          break;
        }
      }
      if(consider) {
        itr_[j] = std::max(itr_[j], itr_[lb_el.pos]);
      }
    }
  }

  std::vector<Itr> begin_;
  std::vector<Itr> end_;
  std::vector<Itr> itr_;
  std::vector<std::vector<LBCondition>> lbs_;
  
  bool done_;
  size_t size_;
}; // LBLoopNest

}  // namespace tammy


#include "index_space.h"
#include "types.h"

namespace tammy {

class LabeledLoop : public LBLoopNest<IndexSpace::Iterator> {
 public:
  LabeledLoop() = default;
  
  LabeledLoop(const IndexLabelVec& ilv,
              const std::vector<Iterator>& begin,
              const std::vector<Iterator>& end,
              const std::vector<std::vector<LBCondition>>& lbs)
      : LBLoopNest{begin, end, lbs},
        ilv_{ilv} {}

  LabeledLoop(const IndexLabelVec& ilv,
              const LBLoopNest<IndexSpace::Iterator>& lb_loop) 
      : LBLoopNest{lb_loop},
        ilv_{ilv} {}

  const IndexLabelVec& labels() const {
    return ilv_;
  }
  
 protected:
  IndexLabelVec ilv_;
};

class InnerLabeledLoop : public LabeledLoop {
 public:
  using LabeledLoop::LabeledLoop;
};

class OuterLabeledLoop : public LabeledLoop {
 public:
  using LabeledLoop::LabeledLoop;
};

class SymmLoop {
 public:
  SymmLoop() = default;

  SymmLoop(const std::vector<std::pair<int, IndexLabelVec>>& relabel_list)
      : relabel_list_{relabel_list} {
    reset();
  }

  void reset() {
    it_ = relabel_list_.begin();
  }

  bool has_more() const {
    return it_ != relabel_list_.end();
  }

  void next() {
    it_++;
  }

  std::pair<int, IndexLabelVec>
  get() const {
    return *it_;
  }
  
 protected:
  std::vector<std::pair<int, IndexLabelVec>> relabel_list_;
  std::vector<std::pair<int, IndexLabelVec>>::const_iterator it_;
};

class SymmFactor {
 public:
  SymmFactor(const std::vector<IndexLabelVec>& ilv_vec)
      : ilv_vec_{ilv_vec} {}

  SymmFactor() = default;
  SymmFactor(const SymmFactor&) = default;
  
  int factor(const std::map<IndexLabel, BlockIndex>& label_map) {
    int ret = 1;
    for(const auto& ilv : ilv_vec_){
      std::map<BlockIndex, int> indices;
      for(const auto& il: ilv) {
        BlockIndex bid = label_map.find(il)->second;
        indices[bid] += 1;
      }
      int v = 1;
      for(const auto &ind: indices) {
        v *= factorial(ind.second);
      }
      ret *= factorial(ilv.size()) / v;
    }
    return ret;
  }
 private:
  static int factorial(size_t x) {
    EXPECTS(x < 6); //check for overflow
    int ret = 1;
    for(size_t i = 1; i <= x; i++) {
      ret *= i;
    }
    return ret;
  }
  std::vector<IndexLabelVec> ilv_vec_;
};  // class SymmFactor

inline SymmFactor
symm_factor(const std::vector<IndexLabelVec>& perm_symm_groups) {
  return SymmFactor{perm_symm_groups};
}



class OpTemplate {
 protected:
  LabeledLoop outer_loops;
  LabeledLoop inner_loops;
  SymmLoop symm_loop;
  SymmFactor symm_facctor;

  void execute() {
    std::map<IndexLabel, BlockIndex> label_map;
    for (; outer_loops.has_more(); outer_loops.next()) {
      auto outer_indices = outer_loops.get();

      for(; symm_loop.has_more(); symm_loop.next()) {
        auto outer_symm_loop =  symm_loop.get();
      
        //label_map.update(outer_symm_loop, outer_indices);      
      
        //alocate C block
        for (; inner_loops.has_more(); inner_loops.next()) {
          auto inner_indices = inner_loops.get();
          //label_map.update(inner_loops.labels(), inner_indices);
          
          //Get A (and B) block(s)
          //update C block with symm_factor      
        }

        //update overall C block
        //Cbuf[outer_loop.labels()] = outer_sym_sign * Csym_buf[outer_symm_loop];
      }
      //put/acc C block to outer_indices
    }
  }
};


}  // namespace tammy


#endif  // TAMMY_LOOPS_H_
