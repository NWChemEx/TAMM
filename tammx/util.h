// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_UTIL_H_
#define TAMMX_UTIL_H_

#include <cassert>
#include <iosfwd>
#include <cmath>
#include <map>
#include <algorithm>
#include <string>
#include "tammx/boundvec.h"
#include "tammx/types.h"
#include "tammx/tce.h"
#include "tammx/perm.h"
#include "tammx/tensor_dims.h"
#include "tammx/tensor_labels.h"

/**
 * @todo Check types are convertible to necessary type rather than is_same
 */

namespace tammx {

template<typename T>
TensorVec<T>
flatten(const TensorVec<TensorVec<T>> &vec) {
  TensorVec<T> ret;
  for(auto &v : vec) {
    ret.insert_back(v.begin(), v.end());
  }
  return ret;
}

inline TensorRange
flatten_range(const TensorVec<TensorSymmGroup> &vec) {
  TensorRange ret;
  for(auto &v : vec) {
    for(size_t i=0; i<v.size(); i++) {
      ret.push_back(v.rt());
    }
  }
  return ret;
}

/**
 * @todo @bug Check if this works when a label already presented is
 * updated
 */
template<typename T>
class LabelMap {
 public:
  LabelMap& update(const IndexLabelVec& labels,
                   const TensorVec<T>& ids) {
    // EXPECTS(labels.size() + labels_.size()  <= labels_.max_size());
    // labels_.insert_back(labels.begin(), labels.end());
    // ids_.insert_back(ids.begin(), ids.end());
    EXPECTS(labels.size() == ids.size());
    for(int i=0; i<labels.size(); i++) {
      lmap_[labels[i]] = ids[i];
    }
    return *this;
  }

  TensorVec<T> get_blockid(const IndexLabelVec& labels) const {
    TensorVec<T> ret;
    for(auto l: labels) {
      auto itr = lmap_.find(l);
      EXPECTS(itr != lmap_.end());
      ret.push_back(itr->second);
    }
    return ret;
    // TensorVec<T> ret(labels.size());
    // using size_type = IndexLabelVec::size_type;
    // for(size_type i=0; i<labels.size(); i++) {
    //   auto itr = std::find(begin(labels_), end(labels_), labels[i]);
    //   EXPECTS(itr != end(labels_));
    //   ret[i] = ids_[itr - begin(labels_)];
    // }
    // return ret;
  }

 private:
  // TensorRank rank_{};
  // IndexLabelVec labels_;
  // TensorVec<T> ids_;
  std::map<IndexLabel, T> lmap_;
};


template<typename Container>
auto intersect(const Container &ctr1, const Container &ctr2) {
#if 0
  Container ret;
  for (auto &x : ctr1) {
    for (auto &y : ctr2) {
      if (x == y) {
        ret.push_back(x);
      }
    }
  }
  return ret;
#else
  Container ret;
  Container c1 = ctr1;
  Container c2 = ctr2;
  std::sort(c1.begin(), c1.end());
  std::sort(c2.begin(), c2.end());
  std::set_intersection(c1.begin(), c1.end(),
                        c2.begin(), c2.end(),
                        std::back_inserter(ret));
  return ret;
#endif
}

inline TensorVec<IndexLabelVec>
group_labels(const TensorVec<TensorSymmGroup>& groups, const IndexLabelVec& labels) {
  unsigned sz = 0;
  for(auto v : groups) {
    sz += v.size();
  }
  EXPECTS(sz == labels.size());

  size_t pos = 0;
  TensorVec<IndexLabelVec> ret;
  for(auto &sg : groups) {
    size_t i=0;
    while(i<sg.size()) {
      IndexLabelVec lbl{labels[pos+i]};
      size_t i1;
      for(i1=1; i+i1<sg.size() && labels[pos+i+i1].rt() == labels[pos+i].rt(); i1++) {
        lbl.push_back(labels[pos+i+i1]);
      }
      ret.push_back(lbl);
      i += i1;
    }
    pos += sg.size();
  }
  return ret;
}



inline TensorVec<TensorVec<IndexLabelVec>>
group_partition(const TensorVec<IndexLabelVec>& label_groups_1,
                const TensorVec<IndexLabelVec>& label_groups_2) {
  TensorVec<TensorVec<IndexLabelVec>> ret_labels;
  for (auto &lg1 : label_groups_1) {
    TensorVec<IndexLabelVec> ret_group;
    for (auto &lg2 : label_groups_2) {
      auto lbls = intersect(lg1, lg2);
      if (lbls.size() > 0) {
        ret_group.push_back(lbls);
      }
    }
    ret_labels.push_back(ret_group);
  }
  EXPECTS(ret_labels.size() == label_groups_1.size());
  return ret_labels;
}

inline TensorVec<TensorVec<IndexLabelVec>>
group_partition(const TensorVec<TensorSymmGroup>& indices1,
                const IndexLabelVec& label1,
                const TensorVec<TensorSymmGroup>& indices2,
                const IndexLabelVec& label2) {
  auto label_groups_1 = group_labels(indices1, label1);
  auto label_groups_2 = group_labels(indices2, label2);
  return group_partition(label_groups_1, label_groups_2);
}

inline TensorVec<TensorVec<IndexLabelVec>>
group_partition(const TensorVec<TensorSymmGroup>& indices1,
                const IndexLabelVec& label1,
                const TensorVec<TensorSymmGroup>& indices2,
                const IndexLabelVec& label2,
                const TensorVec<TensorSymmGroup>& indices3,
                const IndexLabelVec& label3) {
  auto label_groups_1 = group_labels(indices1, label1);
  auto label_groups_2 = group_labels(indices2, label2);
  auto label_groups_3 = group_labels(indices3, label3);
  auto grp12 = group_partition(label_groups_1, label_groups_2);
  auto grp13 = group_partition(label_groups_1, label_groups_3);
  EXPECTS(grp12.size() == grp13.size());
  auto grp = grp12;
  for(size_t i=0; i<grp.size(); i++) {
    grp[i].insert_back(grp13[i].begin(), grp13[i].end());
  }
  EXPECTS(grp.size() == indices1.size());
  return grp;
}

using std::to_string;

inline std::string
to_string(const IndexLabel& lbl) {
  return to_string(lbl.rt()) + to_string(lbl.label);
}

template<typename T, int maxsize>
std::string
to_string(const BoundVec<T,maxsize> &vec, const std::string& sep = ",") {
  std::string ret;
  for(int i=0; i<vec.size()-1; i++) {
    ret += to_string(vec[i]) + sep;
  }
  if(vec.size()>0) {
    ret += to_string(vec.back());
  }
  return ret;
}

inline TensorVec<TensorSymmGroup>
slice_indices(const TensorVec<TensorSymmGroup>& indices,
              const IndexLabelVec& label) {
  TensorVec<TensorSymmGroup> ret;
  auto grp_labels = group_labels(indices, label);
  for(auto &gl: grp_labels) {
    EXPECTS(gl.size() > 0);
    ret.push_back(TensorSymmGroup{gl[0].rt(), gl.size()});
  }
  return ret;
}


inline int
factorial(int n) {
  EXPECTS(n >= 0 && n <= maxrank);
  if (n <= 1) return 1;
  if (n == 2) return 2;
  if (n == 3) return 6;
  if (n == 4) return 12;
  if (n == 5) return 60;
  int ret = 1;
  for(int i=1; i<=n; i++) {
    ret *= i;
  }
  return ret;
}


template<typename Itr>
class NestedIterator {
 public:
  NestedIterator(const std::vector<Itr>& itrs)
      : itrs_{itrs},
        done_{false} {
          reset();
        }

  void reset() {
    for(auto& it: itrs_) {
      it.reset();
      EXPECTS(it.has_more());
    }
  }

  size_t itr_size() const {
    size_t ret = 0;
    for(const auto& it: itrs_) {
      ret += it.size();
    }
    return ret;
  }

  bool has_more() {
    return !done_;
  }

  IndexLabelVec get() const {
    IndexLabelVec ret;
    for(const auto& it: itrs_) {
      auto vtmp = it.get();
      ret.insert_back(vtmp.begin(), vtmp.end());
    }
    return ret;
  }

  void next() {
    int i = itrs_.size()-1;
    for(; i>=0; i--) {
      itrs_[i].next();
      if (itrs_[i].has_more()) {
        break;
      }
      itrs_[i].reset();
    }
    if (i<0) {
      done_ = true;
    }
  }

 private:
  std::vector<Itr> itrs_;
  bool done_;
};


} //namespace tammx


#endif  // TAMMX_UTIL_H_

