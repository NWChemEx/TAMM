// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMX_UTIL_H__
#define TAMMX_UTIL_H__

#include <cassert>
#include <iosfwd>
#include "tammx/strong_int.h"
#include "tammx/boundvec.h"
#include "tammx/types.h"

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

template<typename T>
class LabelMap {
 public:
  LabelMap& update(const TensorLabel& labels,
                   const TensorVec<T>& ids) {
    Expects(labels.size() + labels_.size()  <= labels_.max_size());
    labels_.insert_back(labels.begin(), labels.end());
    ids_.insert_back(ids.begin(), ids.end());
    return *this;
  }

  TensorVec<T> get_blockid(const TensorLabel& labels) const {
    TensorVec<T> ret(labels.size());
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
  TensorVec<T> ids_;
};

inline TensorPerm
perm_compute(const TensorLabel& from, const TensorLabel& to) {
  TensorPerm layout;

  Expects(from.size() == to.size());
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
  for(int i=0; i<perm.size(); i++) {
    auto itr = std::find(perm.begin(), perm.end(), i);
    Expects(itr != perm.end());
    num_inversions += std::abs((itr - perm.begin()) - i);
  }
  return num_inversions / 2;
}

template<typename T>
inline TensorVec<T>
perm_apply(const TensorVec<T>& label, const TensorPerm& perm) {
  TensorVec<T> ret;
  std::cerr<<__FUNCTION__<<":"<<__LINE__<<": label="<<label<<std::endl;
  std::cerr<<__FUNCTION__<<":"<<__LINE__<<": perm="<<perm<<std::endl;
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

inline bool
is_permutation(TensorPerm perm) {
  std::sort(perm.begin(), perm.end());
  for(int i=0 ;i<perm.size(); i++) {
    if(perm[i] != i)
      return false;
  }
  return true;
}

inline TensorPerm
perm_invert(const TensorPerm& perm) {
  TensorPerm ret(perm.size());
  Expects(is_permutation(perm));
  for(unsigned i=0; i<perm.size(); i++) {
    auto itr = std::find(perm.begin(), perm.end(), i);
    Expects(itr != perm.end());
    ret[i] = itr - perm.begin();
  }
  return ret;
}

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



}; //namespace tammx


#endif  // TAMMX_UTIL_H__

