#ifndef TAMM_INDEX_LOOP_NEST_HPP_
#define TAMM_INDEX_LOOP_NEST_HPP_

#include "tamm/index_space.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

namespace tamm {

class IndexBoundCondition {
 public:
  IndexBoundCondition(TiledIndexLabel this_label,
                      const std::vector<TiledIndexLabel>& lb_labels = {},
                      const std::vector<TiledIndexLabel>& ub_labels = {})
      : this_label_{this_label},
        lb_labels_{lb_labels},
        ub_labels_{ub_labels} {}

  TiledIndexLabel this_label() const {
    return this_label_;
  }

  std::vector<TiledIndexLabel>& lb_labels() {
    return lb_labels_;
  }

  std::vector<TiledIndexLabel>& ub_labels() {
    return ub_labels_;
  }

  const std::vector<TiledIndexLabel>& lb_labels() const {
    return lb_labels_;
  }

  const std::vector<TiledIndexLabel>& ub_labels() const {
    return ub_labels_;
  }
  
 private:
  TiledIndexLabel this_label_;
  std::vector<TiledIndexLabel> lb_labels_;
  std::vector<TiledIndexLabel> ub_labels_;
};


inline IndexBoundCondition
operator <= (const IndexBoundCondition& ref, const IndexBoundCondition& ub) {
  assert(ref.lb_labels().size() == 0);
  assert(ref.ub_labels().size() == 0);
  assert(ub.lb_labels().size() == 0);
  assert(ub.ub_labels().size() == 0);
  return {ref.this_label(), {}, {ub.this_label()}};
}

inline IndexBoundCondition
operator >= (const IndexBoundCondition& ref, const IndexBoundCondition& lb) {
  assert(ref.lb_labels().size() == 0);
  assert(ref.ub_labels().size() == 0);
  assert(lb.lb_labels().size() == 0);
  assert(lb.ub_labels().size() == 0);
  return {ref.this_label(), {lb.this_label()}, {}};
}

inline IndexBoundCondition
operator + (IndexBoundCondition ibc1, const IndexBoundCondition& ibc2) {
  assert(ibc1.this_label() == ibc2.this_label());
  ibc1.lb_labels().insert(ibc1.lb_labels().end(),
                          ibc2.lb_labels().begin(),
                          ibc2.lb_labels().end());
  ibc1.ub_labels().insert(ibc1.ub_labels().end(),
                          ibc2.ub_labels().begin(),
                          ibc2.ub_labels().end());
  return ibc1;
}

inline IndexBoundCondition
operator == (const IndexBoundCondition& lhs, const IndexBoundCondition& rhs) {
  return (lhs <= rhs) + (lhs >= rhs);
}



class IndexLoopNest {
 public:
  IndexLoopNest() {
    reset();
  }
  
  IndexLoopNest(const IndexLoopNest&) = default;
  IndexLoopNest(IndexLoopNest&&) = default;
  ~IndexLoopNest() = default;
  IndexLoopNest& operator = (const IndexLoopNest&) = default;
  IndexLoopNest& operator = (IndexLoopNest&&) = default;

  IndexLoopNest(const std::vector<TiledIndexSpace>& iss,
                const std::vector<IndexVector>& lb_indices,
                const std::vector<IndexVector>& ub_indices,
                const std::vector<IndexVector>& indep_indices)
      : iss_{iss},
        lb_indices_{lb_indices},
        ub_indices_{ub_indices},
        indep_indices_{indep_indices} {
        lb_indices_.resize(iss_.size());
        ub_indices_.resize(iss_.size());
        indep_indices_.resize(iss_.size());
        reset();
        }  
  
  template<typename... Args>
  IndexLoopNest(const IndexBoundCondition& ibc, Args&&... args)
      : IndexLoopNest{std::vector<IndexBoundCondition>{ibc, std::forward<Args>(args)...}} {
  }

  IndexLoopNest(const std::vector<IndexBoundCondition>& ibcs) {
    std::vector<TiledIndexLabel> labels;

    for(const auto& ibc: ibcs) {
      //every label is unique
      assert(std::find(labels.begin(), labels.end(), ibc.this_label()) == labels.end());
      labels.push_back(ibc.this_label());
      iss_.push_back(ibc.this_label().tiled_index_space());
      indep_indices_.push_back({});
      /*
        //indep labels
        for(const auto& is : ibc.this_label().is().indep_is()) {
        //check that thay are already in existing list of labels
        }
      */
      ub_indices_.push_back({});
      for(const auto& lbl: ibc.ub_labels()) {
        auto itr = std::find(labels.begin(), labels.end(), lbl);
        //upper bound label exists
        assert(itr != labels.end());
        ub_indices_.back().push_back(itr - labels.begin());
      }
      lb_indices_.push_back({});
      for(const auto& lbl: ibc.lb_labels()) {
        auto itr = std::find(labels.begin(), labels.end(), lbl);
        //lower bound label exists
        assert(itr != labels.end());
        lb_indices_.back().push_back(itr - labels.begin());
      }
    }
    reset();
  }
  
  class Iterator {
   public:
    Iterator() = default;
    Iterator(const Iterator&) = default;    
    Iterator(Iterator&&) = default;
    ~Iterator() = default;
    Iterator& operator = (const Iterator&) = default;
    Iterator& operator = (Iterator&&) = default;

    Iterator(IndexLoopNest* loop_nest)
        : loop_nest_{loop_nest} {
      bases_.resize(size());
      itrs_.resize(size());
      begins_.resize(size());
      ends_.resize(size());
      done_ = false;
      reset_forward(0);
    }
     
    bool operator == (const Iterator& rhs) const {
      return loop_nest_ == rhs.loop_nest_ &&
          done_ == rhs.done_ &&
          itrs_ == rhs.itrs_;
    }

    bool operator != (const Iterator& rhs) const {
      return !(*this == rhs);
    }

    IndexVector operator * () const {
      assert(!done_);
      assert(itrs_.size() == bases_.size());
      
      IndexVector ret;
      for(int i=0; i<(int)itrs_.size(); i++) {
        ret.push_back(*(bases_[i]+itrs_[i]));
      }
      return ret;
    }
    
    Iterator operator ++ () {
      int i = rollback(size()-1);
      if (i<0) {
        set_end();
      } else {
        itrs_[i]++;
        reset_forward(i+1);
      }
      return *this;
    }

    Iterator operator ++ (int) {
      Iterator ret{*this};
      ++(*this);
      return ret;
    }
    
   private:
    int rollback(int index) {
      int i;
      for(i = index; i >= 0 && itrs_[i]+1 == ends_[i]; i--) {
        //no-op
      }
      return i;
    }

    int size() const {
      return loop_nest_->size();
    }
    
    void reset_forward(int index) {
      assert(index >= 0);
      
      int i = index;
      while (i >=0 && i < size()) {
        std::vector<Index> indep_vals;
        for (const auto& id : loop_nest_->indep_indices_[i]) {
          indep_vals.push_back(bases_[id][itrs_[id]]);
        }
        IndexIterator cbeg, cend;
        assert(indep_vals.size()==0); //@bug no support for dependent index spaces yet
#if 0
        std::tie(cbeg, cend) =  loop_nest_->iss_[i].construct_iterators(indep_vals);
#else
        cbeg = loop_nest_->iss_[i].begin();
        cend = loop_nest_->iss_[i].end();
#endif
        bases_[i] = cbeg;
        ends_[i] = std::distance(cbeg, cend);
        for (const auto& id: loop_nest_->lb_indices_[i]) {
          assert(id < i);
          begins_[i] = std::max(begins_[i], itrs_[id]);
        }
        for (const auto& id: loop_nest_->ub_indices_[i]) {
          assert(id < i);
          ends_[i] = std::min(ends_[i], itrs_[id]+1);
        }
        if (begins_[i] < ends_[i]) {
          itrs_[i] = begins_[i];
          i++;
        } else { 
          i = rollback(i-1);
          if(i>=0) {
            itrs_[i]++;
            i++;
          }
        }
      }
      if (i < 0) {
        set_end();
      }
    }

    void set_end() {
      itrs_.clear();
      done_ = true;
    }

    std::vector<IndexIterator> bases_;
    IndexVector itrs_;
    IndexVector begins_; //current begin
    IndexVector ends_; //current end
    IndexLoopNest* loop_nest_;
    bool done_;
    friend class IndexLoopNest;
  };  

  const Iterator& begin() const {
    return itbegin_;
  }

  const Iterator& end() const {
    return itend_;
  }  

  bool is_valid() const {
    bool ret = true;
    ret = ret && iss_.size() != lb_indices_.size();
    ret = ret && iss_.size() != ub_indices_.size();
    ret = ret && iss_.size() != indep_indices_.size();
    /*
        //indep spaces
        for(const auto& is : ibc.this_label().is().indep_is()) {
        //check that thay are already in existing list of labels
        }
    */
    for(int i=0; i<(int)ub_indices_.size(); i++) {
      for(const auto uid : ub_indices_[i]) {
        ret = ret && uid >=0 && uid < i;
      }
    }
    for(int i=0; i<(int)lb_indices_.size(); i++) {
      for(const auto lid : lb_indices_[i]) {
        ret = ret && lid >=0 && lid < i;
      }
    }
    return ret;
  }

  int size() const {
    return iss_.size();
  }

  void reset() {
    itbegin_ = Iterator{this};
    itend_ = Iterator{this};
    itend_.set_end();
  }
  
  std::vector<TiledIndexSpace> iss_;
  std::vector<IndexVector> lb_indices_;
  std::vector<IndexVector> ub_indices_;
  std::vector<IndexVector> indep_indices_;
  Iterator itbegin_;
  Iterator itend_;
};  // class IndexLoopNest

template<typename... Args>
inline IndexLoopNest loop_spec(Args... args) {
  IndexLoopNest iln{args...};
  assert(iln.is_valid());
  return iln;
}

} //namespace tamm



#endif // TAMM_INDEX_LOOP_NEST_HPP_

