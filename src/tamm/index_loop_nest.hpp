#ifndef TAMM_INDEX_LOOP_NEST_HPP_
#define TAMM_INDEX_LOOP_NEST_HPP_

#include "tamm/tiled_index_space.hpp"
#include "tamm/errors.hpp"
#include "tamm/utils.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

namespace tamm {

/**
 * @brief Bound class for tiled index loop nest.
 * The bounds are defined in terms of index labels: the label for this 
 * loop index (say i) and labels for lower bounds (say lo1, lo2, ..) and 
 * upper bounds (say hi1, hi2, ..). A loop bound specifies the range:
 * 
 * max(lb, lo1, lo2, ...) <= i < min(ub, hi1, h2, ..)
 * 
 * where lb and ub are the lower and upper bouhd for the range of indices 
 * for index label i's index space.
 */
class IndexLoopBound {
 public:
  /**
   * @brief Index loop bound constructor
   * 
   * @paramn [in] this_label Label for which the bound is being specified
   * @param [in] lb_labels List of lower bound labels
   * @param [in] ub_labels List of upper bound labels
   */
  IndexLoopBound(TiledIndexLabel this_label,
                const std::vector<TiledIndexLabel>& lb_labels = {},
                const std::vector<TiledIndexLabel>& ub_labels = {})
      : this_label_{this_label},
        lb_labels_{lb_labels},
        ub_labels_{ub_labels} {}

  /**
   * @brief Access this index loop bound's label
   * 
   * @return this loop bound's label
   */
  TiledIndexLabel this_label() const noexcept {
    return this_label_;
  }

  /**
   * @brief Access this index loop bound's label
   * 
   * @return this loop bound's label
   */
  constexpr const std::vector<TiledIndexLabel>& lb_labels() const noexcept {
    return lb_labels_;
  }

  /**
   * @brief Access this index loop bound's label
   * 
   * @return this loop bound's label
   */
  constexpr const std::vector<TiledIndexLabel>& ub_labels() const noexcept {
    return ub_labels_;
  }
  
 private:
  TiledIndexLabel this_label_; /**< Index label for this loop*/
  std::vector<TiledIndexLabel> lb_labels_; /**< Index labels of lower bounds*/
  std::vector<TiledIndexLabel> ub_labels_; /**<Index labels of upper bounds*/

  std::vector<TiledIndexLabel>& lb_labels() {
     return lb_labels_;
  }

  std::vector<TiledIndexLabel>& ub_labels() {
    return ub_labels_;
  }

  friend IndexLoopBound operator + (IndexLoopBound ilb1,  const IndexLoopBound& ilb2);
}; // class IndexLoopBound


/**
 * @brief Specify an upper bound
 * 
 * @param [in] ref the index label for which upper bound is being specified
 * @param [in] ub the upper bound index label
 * 
 * @return index loop bound corresponding to ref < ub
 */
inline IndexLoopBound
operator <= (const IndexLoopBound& ref, const IndexLoopBound& ub) {
  EXPECTS(ref.lb_labels().size() == 0);
  EXPECTS(ref.ub_labels().size() == 0);
  EXPECTS(ub.lb_labels().size() == 0);
  EXPECTS(ub.ub_labels().size() == 0);
  return {ref.this_label(), {}, {ub.this_label()}};
}

/**
 * @brief Specify an lower bound
 * 
 * @param [in] ref the index label for which lower bound is being specified
 * @param [in] ub the lower bound index label
 * 
 * @return index loop bound corresponding to ref >= lb
 */
inline IndexLoopBound
operator >= (const IndexLoopBound& ref, const IndexLoopBound& lb) {
  EXPECTS(ref.lb_labels().size() == 0);
  EXPECTS(ref.ub_labels().size() == 0);
  EXPECTS(lb.lb_labels().size() == 0);
  EXPECTS(lb.ub_labels().size() == 0);
  return {ref.this_label(), {lb.this_label()}, {}};
}

/**
 * @brief Combine index loop bounds. Both bounds are to be on the same index 
 * label. The lower and upper bounds from both conditions are combined to create
 * a new index loop bound.
 * 
 * @param [in] ilb1 first loop bound 
 * @param [in] ilb2 second loop bound
 * 
 * @pre @code ilb1.this_label() == ilb2.this_label() @endcode
 * 
 * @return index loop bound that satisfies the loop conditions in both @param 
 * ilb1 and @param ilb2
 */
inline IndexLoopBound
operator + (IndexLoopBound ilb1, const IndexLoopBound& ilb2) {
  EXPECTS(ilb1.this_label() == ilb2.this_label());
  ilb1.lb_labels().insert(ilb1.lb_labels().end(),
                          ilb2.lb_labels().begin(),
                          ilb2.lb_labels().end());
  ilb1.ub_labels().insert(ilb1.ub_labels().end(),
                          ilb2.ub_labels().begin(),
                          ilb2.ub_labels().end());
  return ilb1;
}

/**
 * @brief Equality comparison operator
 * 
 * @param [in] lhs Left-hand side
 * @param [in] rhs Right-hand side
 * 
 * @return true if lhs == rhs
 */
inline IndexLoopBound
operator == (const IndexLoopBound& lhs, const IndexLoopBound& rhs) {
  return (lhs <= rhs) + (lhs >= rhs);
}


class IndexLoopNest {
 public:
  /**
   * @brief Construct a new Index Loop Nest object
   *
   */
  IndexLoopNest() {
    reset();
  }
  
  IndexLoopNest(const IndexLoopNest&) = default;
  IndexLoopNest(IndexLoopNest&&) = default;
  ~IndexLoopNest() = default;
  IndexLoopNest& operator = (const IndexLoopNest&) = default;
  IndexLoopNest& operator = (IndexLoopNest&&) = default;

  IndexLoopNest(const std::vector<TiledIndexSpace>& iss,
                const std::vector<std::vector<size_t>>& lb_indices,
                const std::vector<std::vector<size_t>>& ub_indices,
                const std::vector<std::vector<size_t>>& indep_indices)
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
  IndexLoopNest(const IndexLoopBound& ibc, Args&&... args)
      : IndexLoopNest{std::vector<IndexLoopBound>{ibc, std::forward<Args>(args)...}} {
  }

  IndexLoopNest(const std::vector<IndexLoopBound>& ibcs) {
    std::vector<TiledIndexLabel> labels;

    for(const auto& ibc: ibcs) {
      //every label is unique
      EXPECTS(std::find(labels.begin(), labels.end(), ibc.this_label()) == labels.end());
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
        EXPECTS(itr != labels.end());
        ub_indices_.back().push_back(itr - labels.begin());
      }
      lb_indices_.push_back({});
      for(const auto& lbl: ibc.lb_labels()) {
        auto itr = std::find(labels.begin(), labels.end(), lbl);
        //lower bound label exists
        EXPECTS(itr != labels.end());
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

    virtual IndexVector operator * () const {
      EXPECTS(!done_);
      EXPECTS(itrs_.size() == bases_.size());
      
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
    
    void set_end() {
      itrs_.clear();
      done_ = true;
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
      EXPECTS(index >= 0);

      int i = index;
      while (i >=0 && i < size()) {
        std::vector<Index> indep_vals;
        EXPECTS(i< loop_nest_->indep_indices_.size());
        for (const auto& id : loop_nest_->indep_indices_[i]) {
          indep_vals.push_back(*(bases_[id]+itrs_[id]));
        }
        IndexIterator cbeg, cend;
        //EXPECTS(indep_vals.size()==0); //@bug no support for dependent index spaces yet
#if 0
        std::tie(cbeg, cend) =  loop_nest_->iss_[i].construct_iterators(indep_vals);
#else
        cbeg = loop_nest_->iss_[i](indep_vals).begin();
        cend = loop_nest_->iss_[i](indep_vals).end();
#endif
        bases_[i] = cbeg;
        begins_[i] = 0;
        ends_[i] = std::distance(cbeg, cend);
        for (const auto& id: loop_nest_->lb_indices_[i]) {
          EXPECTS(static_cast<int>(id) < i);
          begins_[i] = std::max(begins_[i], itrs_[id]);
        }
        for (const auto& id: loop_nest_->ub_indices_[i]) {
          EXPECTS(static_cast<int>(id) < i);
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
    for(size_t i=0; i<ub_indices_.size(); i++) {
      for(const auto uid : ub_indices_[i]) {
        ret = ret && uid >=0 && uid < i;
      }
    }
    for(size_t i=0; i<lb_indices_.size(); i++) {
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
  std::vector<std::vector<size_t>> lb_indices_;
  std::vector<std::vector<size_t>> ub_indices_;
  std::vector<std::vector<size_t>> indep_indices_;
  Iterator itbegin_;
  Iterator itend_;
};  // class IndexLoopNest

class LabelLoopNest {
public:
    LabelLoopNest(const LabelLoopNest&) = default;
    LabelLoopNest(LabelLoopNest&&)      = default;
    ~LabelLoopNest()                    = default;
    LabelLoopNest& operator=(const LabelLoopNest&) = default;
    LabelLoopNest& operator=(LabelLoopNest&&) = default;

    LabelLoopNest(const IndexLabelVec& input_labels) :
      input_labels_{input_labels} {
        const IndexLabelVec& unique_labels =
          internal::unique_entries(input_labels_);
        sorted_unique_labels_ = internal::sort_on_dependence(unique_labels);

        std::vector<TiledIndexSpace> iss;
        for(const auto& lbl : sorted_unique_labels_) {
            iss.push_back(lbl.tiled_index_space());
        }
        std::vector<std::vector<size_t>> indep_indices =
          construct_dep_map(sorted_unique_labels_);
        index_loop_nest_ = IndexLoopNest{iss, {}, {}, indep_indices};
        reset();
    }

    class Iterator {
    public:
        Iterator()                = default;
        Iterator(const Iterator&) = default;
        Iterator(Iterator&&)      = default;
        ~Iterator()               = default;
        Iterator& operator=(const Iterator&) = default;
        Iterator& operator=(Iterator&&) = default;

        Iterator(LabelLoopNest& label_loop_nest) :
          label_loop_nest_{&label_loop_nest},
          index_loop_itr_{&label_loop_nest.index_loop_nest_} {}

        bool operator==(const Iterator& rhs) const {
            return label_loop_nest_ == rhs.label_loop_nest_ &&
                   index_loop_itr_ == rhs.index_loop_itr_;
        }

        bool operator!=(const Iterator& rhs) const { return !(*this == rhs); }

        virtual IndexVector operator*() const {
            IndexVector itval = *index_loop_itr_;
            return internal::perm_map_apply(
              itval, label_loop_nest_->perm_map_sorted_to_input_labels_);
        }

        Iterator operator++() {
            ++index_loop_itr_;
            return *this;
        }

        Iterator operator++(int) {
            Iterator ret{*this};
            ++(*this);
            return ret;
        }

        void set_end() { index_loop_itr_.set_end(); }

    private:
        LabelLoopNest* label_loop_nest_;
        IndexLoopNest::Iterator index_loop_itr_;
    }; // class LabelLoopNest::Iterator

    const Iterator& begin() const { return itbegin_; }

    const Iterator& end() const { return itend_; }

private:
    std::vector<std::vector<size_t>> construct_dep_map(
      const IndexLabelVec& labels) {
        std::vector<std::vector<size_t>> dep_map(labels.size());
        size_t til = labels.size();
        // std::vector<TiledIndexSpace> iss;
        // for(const auto& lbl: labels) {
        //   iss.push_back(lbl.tiled_index_space());
        // }
        for(size_t i = 0; i < til; i++) {
            auto il  = labels[i];
            auto tis = labels[i].tiled_index_space();
            if(tis.is_dependent()) {
                /// @todo do we need this check here?
                EXPECTS(il.dep_labels().size() ==
                        il.tiled_index_space()
                          .index_space()
                          .num_key_tiled_index_spaces());
                for(auto& dep : il.dep_labels()) {
                    size_t pos = 0;
                    for(pos = 0; pos < labels.size(); pos++) {
                        if(labels[pos].tiled_index_space() ==
                             dep.tiled_index_space() &&
                           dep.get_label() == labels[pos].get_label()) {
                            dep_map[i].push_back(pos);
                            break;
                        }
                    }
                    EXPECTS(pos < labels.size());
                }
            }
        }
        return dep_map;
    }

    void reset() {
        itbegin_ = Iterator{*this};
        itend_   = Iterator{*this};
        itend_.set_end();
    }

    IndexLabelVec input_labels_;
    IndexLoopNest index_loop_nest_;
    IndexLabelVec sorted_unique_labels_;
    std::vector<size_t> perm_map_input_to_sorted_labels_;
    std::vector<size_t> perm_map_sorted_to_input_labels_;
    Iterator itbegin_;
    Iterator itend_;
}; // class LabelLoopNest

template<typename... Args>
inline IndexLoopNest loop_spec(Args... args) {
  IndexLoopNest iln{args...};
  EXPECTS(iln.is_valid());
  return iln;
}

} //namespace tamm

#endif // TAMM_INDEX_LOOP_NEST_HPP_

