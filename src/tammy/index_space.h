// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMY_INDEX_SPACE_H_
#define TAMMY_INDEX_SPACE_H_

#include <vector>
#include <algorithm>

#include "boundvec.h"
#include "types.h"
#include "errors.h"

namespace tammy {

class IndexRange;
class IndexLabel;

class IndexSpace {
 public:
  using Iterator = std::vector<BlockIndex>::const_iterator;  
  
  virtual bool has_spatial() const {
    return false;
  }
  virtual Irrep spatial(BlockIndex) const {
    return Irrep{0};
  }
  virtual bool has_spin() const {
    return false;
  }
  virtual Spin spin(BlockIndex) const {
    return Spin{1};
  }
  virtual Size size(BlockIndex) const = 0;
  virtual Offset offset(BlockIndex) const = 0;
  virtual Iterator begin(RangeValue rv,
                         const TensorVec<Iterator>& indep_indices={}) const = 0;
  virtual Iterator end(RangeValue rv,
                       const TensorVec<Iterator>& indep_indices={}) const = 0;

  virtual bool is_superset_of(RangeValue rv1,
                              RangeValue rv2) const = 0;
  
  virtual IndexRange ER() const = 0;
  virtual IndexRange NR() const = 0;

  virtual int num_indep_indices() const = 0;
  
  template<typename... LabelArgs>
  auto N(LabelArgs... labels) const;

  template<typename... LabelArgs>
  auto E(LabelArgs... labels) const;

  bool is_identical_to(const IndexSpace& is) const {
    return this == &is;
  }

  //strict definition of compatbility
  bool is_compatible_with(const IndexSpace& is) const {
    return is_identical_to(is);
  }
};  // IndexSpace

inline bool
operator == (const IndexSpace& lhs, const IndexSpace& rhs) {
  return lhs.is_identical_to(rhs);
}

inline bool
operator < (const IndexSpace& lhs, const IndexSpace& rhs) {
  return &lhs < &rhs;
}

inline bool
operator != (const IndexSpace& lhs, const IndexSpace& rhs) {
  return !(lhs == rhs);
}

inline bool
operator > (const IndexSpace& lhs, const IndexSpace& rhs) {
  return !(lhs < rhs) && (lhs != rhs);
}

inline bool
operator <= (const IndexSpace& lhs, const IndexSpace& rhs) {
  return (lhs < rhs) || (lhs == rhs);
}

inline bool
operator >= (const IndexSpace& lhs, const IndexSpace& rhs) {
  return (lhs > rhs) || (lhs == rhs);
}

///////////////////////////////////////////////////////////

class IndexRange {
 public:
  using BeginEndFn = IndexSpace::Iterator  (*) (const IndexRange& ir,
                                                const TensorVec<IndexSpace::Iterator>& indep_indices);
  IndexRange()
      : is_{nullptr},
        rv_{0},
        begin_fn_{default_begin_},
        end_fn_{default_end_} {}
  
  IndexRange(const IndexSpace& is,
             RangeValue rv)
      : is_{&is},
        rv_{rv},
        begin_fn_{default_begin_},
        end_fn_{default_end_} {}

  IndexRange(const IndexRange& ir) = default;
  IndexRange& operator = (const IndexRange& ir) = default;

  const IndexSpace& is() const {
    EXPECTS(is_);
    return *is_;
  }

  RangeValue rv() const {
    return rv_;
  }

  int num_indep_indices() const {
    return is_->num_indep_indices();
  }

  IndexLabel labels(Label label) const;

  template<typename... LabelArgs>
  auto labels(Label label, LabelArgs... labels) const;

  IndexSpace::Iterator begin(const TensorVec<IndexSpace::Iterator>& indep_indices = {}) const {
    return begin_fn_(*this, indep_indices);
  }

  IndexSpace::Iterator end(const TensorVec<IndexSpace::Iterator>& indep_indices = {}) const {
    return end_fn_(*this, indep_indices);
  }

  IndexRange& set_begin(BeginEndFn fn) {
    begin_fn_ = fn;
    return *this;
  }
  
  IndexRange& set_end(BeginEndFn fn) {
    end_fn_ = fn;
    return *this;
  }

  bool is_superset_of(const IndexRange& ir) const {
    EXPECTS(is_->is_compatible_with(*ir.is_));
    is_->is_superset_of(rv_, ir.rv_);
  }
  
 private:
  const IndexSpace* is_; //non-owning pointer (using pointer to get default constructor)
  RangeValue rv_;

  BeginEndFn begin_fn_;
  BeginEndFn end_fn_;

  static BeginEndFn default_begin_;
  
  static BeginEndFn default_end_;  
};  // IndexRange

IndexRange::BeginEndFn
IndexRange::default_begin_ = [] (const IndexRange& ir,
                                 const TensorVec<IndexSpace::Iterator>& indep_indices) {
  return ir.is().begin(ir.rv(), indep_indices);
};

IndexRange::BeginEndFn
IndexRange::default_end_ = [] (const IndexRange& ir,
                               const TensorVec<IndexSpace::Iterator>& indep_indices) {
  return ir.is().end(ir.rv(), indep_indices);
};

inline bool
operator == (const IndexRange& lhs, const IndexRange& rhs) {
  return lhs.is() == rhs.is() && 
      lhs.rv() == rhs.rv();
}

inline bool
operator < (const IndexRange& lhs, const IndexRange& rhs) {
  return (lhs.is() < rhs.is()) ||
      ((lhs.is() == rhs.is()) && (lhs.rv() < rhs.rv()));
}

inline bool
operator != (const IndexRange& lhs, const IndexRange& rhs) {
  return !(lhs == rhs);
}

inline bool
operator > (const IndexRange& lhs, const IndexRange& rhs) {
  return !(lhs < rhs) && (lhs != rhs);
}

inline bool
operator <= (const IndexRange& lhs, const IndexRange& rhs) {
  return (lhs < rhs) || (lhs == rhs);
}

inline bool
operator >= (const IndexRange& lhs, const IndexRange& rhs) {
  return (lhs > rhs) || (lhs == rhs);
}

///////////////////////////////////////////////////////////

class SubIndexSpace : public IndexSpace {
 public:
  SubIndexSpace(IndexSpace& full_space)
      : full_space_{full_space} {}

  virtual Size size(BlockIndex) const = 0;
  virtual Offset offset(BlockIndex) const = 0;
  virtual Iterator begin(RangeValue rv,
                         const TensorVec<Iterator>& indep_indices={}) const = 0;
  virtual Iterator end(RangeValue rv,
                       const TensorVec<Iterator>& indep_indices={}) const = 0;

  virtual bool is_superset_of(RangeValue rv1,
                              RangeValue rv2) const = 0;
  
  virtual IndexRange ER() const = 0;
  virtual IndexRange NR() const = 0;

  virtual int num_indep_indices() const = 0;
  

  bool is_identical_to(const IndexSpace& is) const{
    return this == &is;
  }

  //strict definition of compatbility
  bool is_compatible_with(const IndexSpace& is) const{
    return full_space_.is_compatible_with(is);
  }

 protected:
  IndexSpace& full_space_;
  std::vector<std::vector<BlockIndex>> block_indices_;
};  // SubIndexSpace


///////////////////////////////////////////////////////////


class DependentIndexLabel;

class IndexLabel {
 public:
  IndexLabel(IndexRange ir,
             Label label)
      : ir_{ir},
        label_{label} {}

  IndexLabel()
      : label_{-1} {}
  
  IndexLabel(const IndexLabel&) = default;
  IndexLabel& operator = (const IndexLabel&) = default;
  
  const IndexRange& ir() const {
    return ir_;
  }
  
  Label label() const {
    return label_;
  }

  DependentIndexLabel operator() (IndexLabel il1) const;
  DependentIndexLabel operator() () const;
  DependentIndexLabel operator() (IndexLabel il1, IndexLabel il2) const;

 protected:
  IndexRange ir_;
  Label label_;
};  // class IndexLabel

inline bool
operator == (const IndexLabel& lhs, const IndexLabel& rhs) {
  return lhs.ir() == rhs.ir() &&
      lhs.label() == rhs.label();
}

inline bool
operator < (const IndexLabel& lhs, const IndexLabel& rhs) {
  return (lhs.ir() < rhs.ir()) ||
      ((lhs.ir() == rhs.ir()) && (lhs.label() < rhs.label()));
}

inline bool
operator != (const IndexLabel& lhs, const IndexLabel& rhs) {
  return !(lhs == rhs);
}

inline bool
operator > (const IndexLabel& lhs, const IndexLabel& rhs) {
  return !(lhs < rhs) && (lhs != rhs);
}

inline bool
operator <= (const IndexLabel& lhs, const IndexLabel& rhs) {
  return (lhs < rhs) || (lhs == rhs);
}

inline bool
operator >= (const IndexLabel& lhs, const IndexLabel& rhs) {
  return (lhs > rhs) || (lhs == rhs);
}


///////////////////////////////////////////////////////////


class DependentIndexLabel {
 public:
  DependentIndexLabel()
      : il_{},
        indep_labels_{} {}
  
  DependentIndexLabel(IndexRange& ir,
                      Label label,
                      const TensorVec<IndexLabel>& indep_labels)
      : il_{ir, label},
        indep_labels_{indep_labels} {}

  DependentIndexLabel(const IndexLabel &il,
                      const TensorVec<IndexLabel>& indep_labels)
      : il_{il},
        indep_labels_{indep_labels} {}

  DependentIndexLabel(const DependentIndexLabel&) = default;
  DependentIndexLabel& operator = (const DependentIndexLabel&) = default;

  const IndexRange& ir() const {
    return il_.ir();
  }
  
  const IndexLabel& il() const {
    return il_;
  }
  
  Label label() const {
    return il_.label();
  }

  const TensorVec<IndexLabel>& indep_labels() const {
    return indep_labels_;
  }

 protected:
  IndexLabel il_;
  TensorVec<IndexLabel> indep_labels_;
};  // class DependentIndexLabel

///////////////////////////////////////////////////////////

template<typename... LabelArgs>
inline auto 
IndexSpace::N(LabelArgs... labels) const {
  // return std::make_tuple(N(label), N(labels)...);
  return NR().labels(labels...);
}

template<typename... LabelArgs>
inline auto 
IndexSpace::E(LabelArgs... labels) const {
  // return std::make_tuple(E(label), E(labels)...);
  return ER().labels(labels...);
}
///////////////////////////////////////////////////////////

inline IndexLabel
IndexRange::labels(Label label) const {
  return IndexLabel{*this, label};
}

template<typename... LabelArgs>
inline auto 
IndexRange::labels(Label label, LabelArgs... rest) const {
  return std::make_tuple(labels(label), labels(rest)...);
}

///////////////////////////////////////////////////////////

inline DependentIndexLabel
IndexLabel::operator() (IndexLabel il1) const {
  // @todo: disabled for testing should be enabled with SubIndexSpaces
  EXPECTS(ir_.num_indep_indices() == 1);
  return {*this, {il1}};
}

inline DependentIndexLabel
IndexLabel::operator() () const {
  return {*this, {}};
}

inline DependentIndexLabel
IndexLabel::operator() (IndexLabel il1, IndexLabel il2) const {
  EXPECTS(ir_.num_indep_indices() == 2);
  return {*this, {il1, il2}};
}

///////////////////////////////////////////////////////////
class IndexInfo {
 public:
  IndexInfo (const IndexInfo& info)
      : labels_{info.labels_},
        group_sizes_{info.group_sizes_} {}

  IndexInfo (const IndexLabel& il) {
    if(il.label() < 0){
      group_sizes_ = {0};
    }
    else {
      labels_ = TensorVec<DependentIndexLabel>{il()};
      group_sizes_ = {1};
    }
  }

  IndexInfo (const DependentIndexLabel& dil)
      : labels_{TensorVec<DependentIndexLabel>{dil}},
        group_sizes_{1} { }


  template<typename... LabelArgs>
  IndexInfo (const IndexLabel& lbl, LabelArgs... labels) {
    if(lbl.label() != -1)
      labels_.push_back(lbl());
    pack(labels_, labels...);

    group_sizes_ = {labels_.size(), 0, 0};  
  }

  template<typename... LabelArgs>
  IndexInfo (const DependentIndexLabel& dlbl, LabelArgs... labels) {
    if(dlbl.il().label() != -1)
      labels_.push_back(dlbl);
    pack(labels_, labels...);

    group_sizes_ = {labels_.size(), 0, 0};
  }

  void add_to_last_group(DependentIndexLabel dil) {
    EXPECTS(group_sizes_.size() > 0);
    labels_.push_back(dil);
    group_sizes_.back() += 1;
  }
  
  const TensorVec<DependentIndexLabel>& labels() const {
    return labels_;
  }
  
  TensorVec<IndexPosition> ipmask() const {
    TensorVec<IndexPosition> ipvec;
    assert(group_sizes_.size() == 3);
    std::fill_n(std::back_inserter(ipvec), group_sizes_[0], IndexPosition::neither);
    std::fill_n(std::back_inserter(ipvec), group_sizes_[1], IndexPosition::upper);
    std::fill_n(std::back_inserter(ipvec), group_sizes_[2], IndexPosition::lower);
    return ipvec;
  }

  IndexInfo& operator |= (const IndexInfo& rhs){
    labels_.insert_back(rhs.labels_.begin(),
                            rhs.labels_.end());
    group_sizes_.insert_back(rhs.group_sizes_.begin(),
                             rhs.group_sizes_.end());    
    return *this;
  }

  bool is_valid() const{
    TensorVec<IndexLabel> i_labels;
    for(auto l : labels_){
      i_labels.push_back(l.il());
    }

    std::sort(i_labels.begin(), i_labels.end());
    //all labels are unique
    EXPECTS(std::adjacent_find(i_labels.begin(), i_labels.end()) == i_labels.end());

    for(auto l : labels_) {
      for(auto indep: l.indep_labels()) {
        auto u = std::find(i_labels.begin(), i_labels.end(), indep);
        // all sub-index spaces should be in the construction
        EXPECTS(u != i_labels.end());
        if(u == i_labels.end())
          return false;
      }
    }

    return true;
  }

  TensorVec<IndexRange> ranges() const {
      
    TensorVec<IndexRange> ranges;
    for(auto l : labels_) {
      ranges.push_back(l.ir());
    }
    return ranges;
  }
  
 protected:
  TensorVec<DependentIndexLabel> labels_;
  TensorVec<size_t> group_sizes_;

  void pack(TensorVec<DependentIndexLabel>& label) {}

  template<typename ...Args>
  void pack(TensorVec<DependentIndexLabel>& label, IndexLabel ilbl, Args... rest) {
    if(ilbl.label() != -1)
      label.push_back(ilbl());
    pack(label, rest...);
  }

  template<typename ...Args>
  void pack(TensorVec<DependentIndexLabel>& label, DependentIndexLabel dilbl, Args... rest) {
    label.push_back(dilbl);
    pack(label, rest...);
  }
};

inline IndexInfo
operator + (IndexInfo lhs, const DependentIndexLabel& rhs) {
  lhs.add_to_last_group(rhs);
  return lhs;
}

inline IndexInfo
operator + (IndexInfo lhs, const IndexLabel& rhs) {
  lhs.add_to_last_group(rhs());
  return lhs;
}

inline IndexInfo 
operator | (IndexInfo lhs, const IndexInfo& rhs) {
  lhs |= rhs;
  return lhs;
}

/////////////////////////////////////////////
using IndexLabelVec = TensorVec<IndexLabel>;


}  // namespace tammy

#endif  // TAMMY_INDEX_SPACE_H_
