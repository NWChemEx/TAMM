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
  LabelMap& update(const TensorLabel& labels,
                   const TensorVec<T>& ids) {
    // Expects(labels.size() + labels_.size()  <= labels_.max_size());
    // labels_.insert_back(labels.begin(), labels.end());
    // ids_.insert_back(ids.begin(), ids.end());
    Expects(labels.size() == ids.size());
    for(int i=0; i<labels.size(); i++) {
      lmap_[labels[i]] = ids[i];
    }
    return *this;
  }

  TensorVec<T> get_blockid(const TensorLabel& labels) const {
    TensorVec<T> ret;
    for(auto l: labels) {
      auto itr = lmap_.find(l);
      Expects(itr != lmap_.end());
      ret.push_back(itr->second);
    }
    return ret;
    // TensorVec<T> ret(labels.size());
    // using size_type = TensorLabel::size_type;
    // for(size_type i=0; i<labels.size(); i++) {
    //   auto itr = std::find(begin(labels_), end(labels_), labels[i]);
    //   Expects(itr != end(labels_));
    //   ret[i] = ids_[itr - begin(labels_)];
    // }
    // return ret;
  }

 private:
  // TensorRank rank_{};
  // TensorLabel labels_;
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

inline TensorVec<TensorLabel>
group_labels(const TensorVec<TensorSymmGroup>& groups, const TensorLabel& labels) {
  unsigned sz = 0;
  for(auto v : groups) {
    sz += v.size();
  }
  Expects(sz == labels.size());

  size_t pos = 0;
  TensorVec<TensorLabel> ret;
  for(auto &sg : groups) {
    size_t i=0;
    while(i<sg.size()) {
      TensorLabel lbl{labels[pos+i]};
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
  Expects(ret_labels.size() == label_groups_1.size());
  return ret_labels;
}

inline TensorVec<TensorVec<TensorLabel>>
group_partition(const TensorVec<TensorSymmGroup>& indices1,
                const TensorLabel& label1,
                const TensorVec<TensorSymmGroup>& indices2,
                const TensorLabel& label2) {
  auto label_groups_1 = group_labels(indices1, label1);
  auto label_groups_2 = group_labels(indices2, label2);
  return group_partition(label_groups_1, label_groups_2);
}

inline TensorVec<TensorVec<TensorLabel>>
group_partition(const TensorVec<TensorSymmGroup>& indices1,
                const TensorLabel& label1,
                const TensorVec<TensorSymmGroup>& indices2,
                const TensorLabel& label2,
                const TensorVec<TensorSymmGroup>& indices3,
                const TensorLabel& label3) {
  auto label_groups_1 = group_labels(indices1, label1);
  auto label_groups_2 = group_labels(indices2, label2);
  auto label_groups_3 = group_labels(indices3, label3);
  auto grp12 = group_partition(label_groups_1, label_groups_2);
  auto grp13 = group_partition(label_groups_1, label_groups_3);
  Expects(grp12.size() == grp13.size());
  auto grp = grp12;
  for(size_t i=0; i<grp.size(); i++) {
    grp[i].insert_back(grp13[i].begin(), grp13[i].end());
  }
  Expects(grp.size() == indices1.size());
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
              const TensorLabel& label) {
  TensorVec<TensorSymmGroup> ret;
  auto grp_labels = group_labels(indices, label);
  for(auto &gl: grp_labels) {
    Expects(gl.size() > 0);
    ret.push_back(TensorSymmGroup{gl[0].rt(), gl.size()});
  }
  return ret;
}


inline int
factorial(int n) {
  Expects(n >= 0 && n <= maxrank);
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


namespace tensor_labels {

struct OLabel : public IndexLabel {
  OLabel(int n)
      : IndexLabel{n, RangeType{DimType::o}} {}
};

struct VLabel : public IndexLabel {
  VLabel(int n)
      : IndexLabel{n, RangeType{DimType::v}} {}
};

struct NLabel : public IndexLabel {
  NLabel(int n)
      : IndexLabel{n, RangeType{DimType::n}} {}
};

const OLabel h1{1}, h2{2}, h3{3}, h4{4}, h5{5}, h6{6}, h7{7}, h8{8}, h9{9}, h10{10}, h11{11};
const VLabel p1{1}, p2{2}, p3{3}, p4{4}, p5{5}, p6{6}, p7{7}, p8{8}, p9{9}, p10{10}, p11{11};

const OLabel i{0}, j{1};
const VLabel a{0}, b{1};

} // namespace tensor_labels

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
      Expects(it.has_more());
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

  TensorLabel get() const {
    TensorLabel ret;
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

