// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMY_PERM_SYMMETRY_H_
#define TAMMY_PERM_SYMMETRY_H_

#include "tammy/boundvec.h"
#include "tammy/types.h"
#include "tammy/errors.h"
// #include "tammy/generator.h"
//#include "tammy_old/generator.h"
#include "tammy/loops.h"

#include <memory>

namespace tammy {
enum class PermRelation { symmetry, antisymmetry, none };

struct PermGroupInfo {
  unsigned int size_;
  TensorVec<TensorVec<size_t>> perm_list_;
  TensorVec<PermRelation> perm_relation_;
};  // PermGroupInfo

class PermGroup {
 private:
  size_t size_; //maximum size of the enclosing permutation group
  size_t index_;
  std::vector<PermGroup> groups_;
  size_t num_groups_;
  PermRelation relation_;
  
 public:
  PermGroup() :
      size_{0},
      index_{0},
      relation_{PermRelation::none} {}

  PermGroup(const PermGroupInfo& pgi)
      : PermGroup{pgi.size_,
        pgi.perm_list_,
        pgi.perm_relation_} { }

  PermGroup(const PermGroup& pg) = default;
  PermGroup(PermGroup&& pg) = default;

  PermGroup& operator = (PermGroup other) {
    swap(*this, other);
    return *this;
  }

  PermGroup(size_t size,
            size_t index)
      : size_{size},
        index_{index},
        relation_{PermRelation::none} {
          EXPECTS(index < size);
        }

  PermGroup(size_t size,
            const std::vector<PermGroup>& groups,
            PermRelation relation)
      : size_{size},
        index_{0},
        groups_{groups},
        relation_{relation} {
          if(groups.size() == 1) {
            *this = groups[0];
          }
        }

  PermGroup(size_t size,
            const TensorVec<size_t>& indices,
            PermRelation relation)
      : size_{size} {
    for(size_t ind : indices) {
      groups_.push_back(PermGroup{size_, ind});
    }
    relation_ = relation;
    if(groups_.size() == 1) {
      relation_ = PermRelation::none;
    }
  }

  PermGroup(unsigned size,
            const TensorVec<TensorVec<size_t>>& perm_list,
            const TensorVec<PermRelation>& perm_relation)
      : size_{size} {

    validate(size, perm_list, perm_relation);

    if(perm_list.size() == 1) {
      *this = PermGroup{size, perm_list[0], perm_relation[0]};
      return;
    }

    size_t ngrps = perm_list.size();
    std::vector<PermGroup> groups(size);
    std::vector<bool> group_exists(size, false);
    EXPECTS(ngrps <= size);
    
    for(int g=0; g<ngrps; g++)  {
      EXPECTS(perm_list[g].size() > 0);
      std::vector<PermGroup> lgroups;
      if(perm_list[g][0] < size) {
        for(size_t i=0; i<perm_list[g].size(); i++) {
          EXPECTS(perm_list[g][i] < size);
          lgroups.push_back(PermGroup{size, perm_list[g][i]});
        }
      } else {
        for(size_t i=0; i<perm_list[g].size(); i++) {
          EXPECTS(perm_list[g][i] >= size);
          EXPECTS(perm_list[g][i] < 2*size);
          size_t idx = perm_list[g][i] - size;
          EXPECTS(idx < g);
          EXPECTS(group_exists[idx] == true);
          lgroups.push_back(groups[idx]);
          group_exists[idx] = false;
        }
      }
      EXPECTS(lgroups.size() > 0);
      if(lgroups.size() == 1) {
        groups[g] = lgroups[0];
      } else {
        groups[g] = PermGroup{size, lgroups, perm_relation[g]};
      }
      group_exists[g] = true;
    }

    relation_ = PermRelation::none;
    for(size_t g = 0; g<groups.size(); g++) {
      if(group_exists[g]) {
        groups_.push_back(groups[g]);
      }
    }
    EXPECTS(groups_.size() > 0);
    if(groups_.size() == 1) {
      *this = groups_[0];
    }
  }

  bool empty() const {
    return size() == 0;
  }

  size_t size() const {
    return size_;
  }

  size_t num_groups() const {
    if(empty()) {
      return 0;
    } else if (groups_.size() == 0) {
      return 1;
    } else {
      return groups_.size();
    }
  }
  
  PermRelation relation() const {
    return relation_;
  }

  size_t index() const {
    return index_;
  }

  const std::vector<PermGroup>& groups() const {
    return groups_;
  }

  template<typename T>
  std::pair<int,TensorVec<T>>
  find_unique(const TensorVec<T>& vec) const {
    TensorVec<T> ivec{vec};
    int ninv = permute_to_unique(ivec);
    return {ninv, ivec};
  }

  template<typename T>
  int permute_to_unique(TensorVec<T>& vec) const {
    if(num_groups() == 0 || num_groups() == 1) {
      return 0;
    }
    int nsub_inversions = 0;
    for(auto& grp: groups_) {
      nsub_inversions += grp.permute_to_unique(vec);
    }
    if(relation_ == PermRelation::none) {
      return nsub_inversions;
    }

    int ninversions;
    for(size_t i=1; i < num_groups(); i++) {
      for(int j=i-1; j>=0; j--) {
        if(greater_than(vec, groups_[j], groups_[i])) {
          swap(vec, groups_[j], groups_[i]);
          ninversions += i-j;
        }
      }
    }
    if(relation_ == PermRelation::antisymmetry) {
      return nsub_inversions + ninversions;
    } else {
      return nsub_inversions;
    }
  }

  // template<typename T>
  // std::unique_ptr<Generator<T>>
  // unique_generator(const TensorVec<T>& lo,
  //                  const TensorVec<T>& hi) const {
  //   //@todo validate hi and lo
  //   assert(lo.size() == size_);
  //   assert(hi.size() == size_);

  //   TensorVec<std::unique_ptr<Generator<T>>> itrs;
  //   for(const auto& idx: indices_) {
  //     itrs.push_back(std::make_unique<SimpleGenerator<T>>(idx, lo[idx], hi[idx]));
  //   }
  //   for(const auto& grp: groups_) {
  //     itrs.push_back(grp->unique_generator<T>(lo, hi));
  //   }
  //   assert(itrs.size() > 0);
  //   if(itrs.size() == 1) {
  //     return std::move(itrs[0]);
  //   }

  //   if(relation_ == PermRelation::symmetry ||
  //      relation_ == PermRelation::antisymmetry) {
  //     return std::make_unique<TriangleGenerator<T>>(std::move(itrs));
  //   } else {
  //     return std::make_unique<CartesianGenerator<T>>(std::move(itrs));
  //   }
  // }

  // template<typename Itr, typename T>
  // std::unique_ptr<LBLoopNest<Itr>>
  // unique_loop_nest(const TensorVec<IndexRange>& vec) const {

  //   TensorVec<std::unique_ptr<LBLoopNest<Itr>>> loops;

  //   std::vector<Itr> begin;
  //   std::vector<Itr> end;
  //   std::vector<std::vector<LBCondition>> lbs;
  //   size_t pos = -1;
  //   for(const auto& idx: indices_){
  //     begin.push_back(vec[idx].begin());
  //     end.push_back(vec[idx].end());
  //     if(pos < 0){
  //       lbs.push_back({});
  //     }
  //     else {
  //       lbs.push_back({pos++, {}, {}});
  //     }
  //   }
  //   loops.push_back(std::make_unique<LBLoopNest<Itr>>(begin, end, lbs));

  //   for(const auto& grp: groups_) {
  //     loops.push_back(grp->unique_loop_nest<Itr, T>(vec));
  //   }

  //   EXPECTS(loops.size() > 0);
  //   if(loops.size() == 1) {
  //     return std::move(loops[0]);
  //   }

  //   return combine(loops);
  // }

  // LBLoopNest<IndexSpace::Iterator>
  // sliced_loop_nest(const IndexLabelVec& ilv) const {
  //   PermGroup perm_group = slice(ilv);
  //   TensorVec<IndexRange> irs;
  //   for(const auto& lbl: ilv) {
  //     irs.push_back(lbl.ir());
  //   }
  //   return unique_loop_nest(irs);
  // }

  LBLoopNest<IndexSpace::Iterator>
  unique_loop_nest(const IndexRangeVec& irv) const {
    EXPECTS(size() == irv.size());
    if(num_groups() == 0) {
      return {};
    }
    if(num_groups() == 1) {
      return {{irv[index()].begin()}, {irv[index()].end()}, {}};
    }
    TensorVec<LBLoopNest<IndexSpace::Iterator>> loop_nest_groups;
    for(size_t i=0; i<num_groups(); i++) {
      loop_nest_groups.push_back(groups_[i].unique_loop_nest(irv));
    }
    // return combine(size_, loop_nest_groups, perm_relation_);
    return loop_nest_concat(loop_nest_groups, relation_!=PermRelation::none);
  }

  LBLoopNest<IndexSpace::Iterator>
  unique_loop_nest(const IndexLabelVec& ilv) const {
    IndexRangeVec irv;
    for(const auto& lbl: ilv) {
      irv.push_back(lbl.ir());
    }
    return unique_loop_nest(irv);
  }

  // static LBLoopNest<IndexSpace::Iterator>
  // combine(size_t size,
  //         TensorVec<LBLoopNest<IndexSpace::Iterator>> loop_nest_groups,
  //         PermRelation relation) {
  //   if(loop_nest_groups.size() == 1) {
  //     return loop_nest_groups[0];
  //   }
  //   size_t off = 0;
  //   using Itr = IndexSpace::Iterator;
  //   std::vector<Itr> begin, end;
  //   std::vector<std::vector<LBCondition>> lbs;
  //   for(auto& lng : loop_nest_groups) {
  //     lng.shift_lbs(off);
  //     begin.insert(begin.end(), lng.begin().begin(), lng.begin().end());
  //     end.insert(end.end(), lng.end().begin(), lng.end().end());
  //     if (relation == PermRelation::none) {
  //       lbs.insert(lbs.end(), lng.lbs().begin(), lng.lbs().end());
  //     } if(relation == PermRelation::symm || relation == PermRelation::anti) {
  //       for(size_t i = 0; i<lng.size(); i++) {
  //         size_t pos = off + i - lng.size();
  //         std::vector<size_t> lhs, rhs;
  //         for(size_t j = 0; j> i; j++) {
  //           lhs.push_back(off + j - lng.size());
  //           rhs.push_back(off + j);
  //         }          
  //         lng.lbs()[i].push_back({pos, lhs, rhs});
  //         lbs.push_back(lng.lbs()[i]);
  //       }
  //     } else {
  //       NOT_IMPLEMENTED();
  //     }
  //     off += lng.size();
  //   }
  // }
  
  PermGroup
  slice(const IndexLabelVec& ilv) const {
    if(num_groups() == 0 || num_groups() == 1) {
      return *this;
    }

    TensorVec<TensorVec<size_t>> nested_grp_ids;
    TensorVec<bool> considered(num_groups(), false);
    TensorVec<IndexRangeVec> group_ranges;
    for(size_t i=0; i<num_groups(); i++) {
      group_ranges.push_back(gather_ranges(ilv, i));
    }

    for(auto itr = considered.begin();
        itr != considered.end();
        itr = std::find(itr+1, considered.end(), false)) {
      unsigned i = itr - considered.begin();
      nested_grp_ids.push_back({i});
      for(size_t j = i+1; j<num_groups(); j++) {
        if (std::equal(group_ranges[j].begin(),
                       group_ranges[j].end(),
                       group_ranges[i].begin())) {
          EXPECTS(considered[j] == false);
          nested_grp_ids.back().push_back(j);
          considered[j] = true;
        }
      }
    }

    std::vector<PermGroup> nested_groups;
    for(size_t i=0; i<nested_grp_ids.size(); i++) {
      EXPECTS(nested_grp_ids[i].size() > 0);
      if(nested_grp_ids[i].size() == 1) {
        nested_groups.push_back(groups_[nested_grp_ids[i][0]].slice(ilv));
      } else {
        std::vector<PermGroup> lgroups;
        for(size_t grp_id : nested_grp_ids[i]) {
          lgroups.push_back(groups_[grp_id].slice(ilv));
        }
        nested_groups.push_back(PermGroup{size_, lgroups, relation_});
      }
    }
    if(nested_grp_ids.size() == 1) {
      return nested_groups[0];
    } else {
      return PermGroup{size_, nested_groups, PermRelation::none};
    }
  }

 private:
  friend void swap(PermGroup& first, PermGroup& second) {
    using std::swap;
    swap(first.size_, second.size_);
    swap(first.index_, second.index_);
    swap(first.groups_, second.groups_);
    swap(first.relation_, second.relation_);
  }
  

  IndexRangeVec gather_ranges(const IndexLabelVec& ilv) const {
    IndexRangeVec ret;
    for(size_t i=0; i<num_groups(); i++) {
      TensorVec<IndexRange> girv{gather_ranges(ilv, i)};
      ret.insert_back(girv.begin(), girv.end());
    }
    return ret;
  }

  IndexRangeVec gather_ranges(const IndexLabelVec& ilv, size_t grp_id) const {
    IndexLabelVec gilv = gather_labels(ilv, grp_id);
    IndexRangeVec ret;
    for(const auto& gil: gilv) {
      ret.push_back(gil.ir());
    }
    return ret;
  }

  IndexLabelVec gather_labels(const IndexLabelVec& ilv) const {
    IndexLabelVec ret;
    for(size_t i=0; i<num_groups(); i++) {
      IndexLabelVec gilv{gather_labels(ilv, i)};
      ret.insert_back(gilv.begin(), gilv.end());
    }
    return ret;
  }

  IndexLabelVec gather_labels(const IndexLabelVec& ilv, size_t grp_id) const {
    EXPECTS(grp_id >= 0 && grp_id < num_groups());
    if(num_groups() == 1) {
      return {ilv[index_]};
    } else {
      return groups_[grp_id].gather_labels(ilv);
    }
  }

  //@todo Not sure this is checking all cases, especially recursive ones.
  static void validate(unsigned int size,
                       const TensorVec<TensorVec<size_t>>& perm_list,
                       const TensorVec<PermRelation>& perm_relation) {
    assert(perm_list.size() == perm_relation.size());

    std::vector<bool> flags(2*size, false);
    for(size_t i=0; i<perm_list.size(); i++) {
      assert(perm_list.size() > 0);
      bool is_grp = false;
      size_t grp_sz = 0;
      PermRelation grp_rel = PermRelation::none;
      assert(perm_list[i].size() > 0);
      if(perm_list[i][0] >= size) {
        is_grp = true;
        assert(perm_list[i].size() > 1);
        size_t grp_id = perm_list[i][0] - size;
        assert(grp_id < perm_list.size());
        grp_sz = perm_list[grp_id].size();
        grp_rel = perm_relation[grp_id];
      }
      for(size_t j=0; j<perm_list[i].size(); j++) {
        assert(flags[perm_list[i][j]] == false);
        flags[perm_list[i][j]] = true;
        if(perm_list[i][j] < size) {
          assert(!is_grp);
        } else if(perm_list[i][j] < 2*size) {
          assert(is_grp);
          size_t gid = perm_list[i][j] - size;
          assert(gid >=0 && gid+1 < perm_list.size());
          assert(perm_list[gid].size() == grp_sz);
          assert(perm_relation[gid] == grp_rel);
        } else {
          assert(0); //invalid input
        }
      }
    }
    for(unsigned int i=0; i<size; i++) {
      assert(flags[i] == true);
    }
  }

  // template<typename Itr>
  // std::unique_ptr<LBLoopNest<Itr>>
  // combine(TensorVec<std::unique_ptr<LBLoopNest<Itr>>> loops) {
  //   // @todo: Combine LBLoopNest

  //   std::vector<Itr> begin;
  //   std::vector<Itr> end;
  //   std::vector<std::vector<LBCondition>> lbs;

  //   for(const auto& loop: loops) {

  //   }

  //   return std::make_unique<LBLoopNest<Itr>>(begin, end, lbs);
  // }

  template<typename T>
  static void
  vec_swap(TensorVec<T>& vec,
       const PermGroup& pg1,
       const PermGroup& pg2) {
    assert(pg1.size() == pg2.size());
    assert(vec.size() == pg1.size());
    assert(pg1.num_groups() == pg2.num_groups());

    if(pg1.num_groups() == 0) {
      //no-op
    }
    if(pg1.num_groups() == 1) {
      return std::swap(vec[pg1.index()], vec[pg2.index()]);
    }
    for(size_t i=0; i<pg1.groups_.size(); i++) {
      vec_swap(vec, pg1.groups_[i], pg2.groups_[i]);
    }
  }

  template<typename T>
  static bool
  less_than_or_equal(const TensorVec<T>& vec,
                     const PermGroup& pg1,
                     const PermGroup& pg2) {
    assert(pg1.size() == pg2.size());
    assert(vec.size() == pg1.size());
    assert(pg1.num_groups() == pg2.num_groups());

    if(pg1.num_groups() == 0) {
      UNREACHABLE();
      return true;
    }
    if(pg1.num_groups() == 1) {
      return vec[pg1.index()] <= vec[pg2.index()];
    }

    bool ret = true;
    for(size_t i=0; i<pg1.groups_.size(); i++) {
      ret = ret & less_than_or_equal(vec,
                                     pg1.groups_[i],
                                     pg2.groups_[i]);
    }
    return ret;
  }

  template<typename T>
  static bool
  equal(const TensorVec<T>& vec,
        const PermGroup& pg1,
        const PermGroup& pg2) {
    assert(pg1.size() == pg2.size());
    assert(vec.size() == pg1.size());
    assert(pg1.num_groups() == pg2.num_groups());

    if(pg1.num_groups() == 0) {
      return true;
    }
    if(pg1.num_groups() == 1) {
      return vec[pg1.index()] == vec[pg2.index()];
    }
    
    bool ret = true;
    for(size_t i=0; i<pg1.groups_.size(); i++) {
      ret = ret & equal(vec, pg1.groups_[i], pg2.groups_[i]);
    }
    return ret;
  }

  template<typename T>
  static bool
  greater_than(const TensorVec<T>& vec,
               const PermGroup& pg1,
               const PermGroup& pg2) {
    return !less_than_or_equal(vec, pg1, pg2);
  }

};  // PermGroup

// inline PermGroupInfo
// antisymm(unsigned int nupper, unsigned int nlower) {
//   TensorVec<TensorVec<unsigned int>> perm_list;
//   TensorVec<PermRelation> perm_relation;

//   if(nupper > 0) {
//     TensorVec<unsigned int> upper;
//     for(unsigned int i=0; i<nupper; i++) {
//       upper.push_back(i);
//     }
//     perm_list.push_back(upper);
//     perm_relation.push_back(PermRelation::antisymmetry);
//   }

//   if(nlower > 0) {
//     TensorVec<unsigned int> lower;
//     for(unsigned int i=0; i<nlower; i++) {
//       lower.push_back(nupper + i);
//     }
//     perm_list.push_back(lower);
//     perm_relation.push_back(PermRelation::antisymmetry);
//   }
//   return {nupper+nlower, perm_list, perm_relation};
// }

// inline PermGroupInfo
// symm(unsigned int nupper, unsigned int nlower) {
//   assert(nupper == nlower);
//   TensorVec<PermRelation> perm_relation(nupper+1, PermRelation::symmetry);

//   TensorVec<TensorVec<unsigned int>> perm_groups;
//   for(unsigned int i=0; i<nupper; i++) {
//     perm_groups.push_back({i, i+nupper});
//   }
//   TensorVec<unsigned int> last;
//   for(unsigned int i=0; i<nupper; i++) {
//     last.push_back(nupper + nlower + i);
//   }
//   perm_groups.push_back(last);
//   return {nupper+nlower, perm_groups, perm_relation};
// }

#if 0
template<typename T>
inline PermGroupInfo
operator | (const T&, PermGroupInfo pgi) {
  pgi.size_ += 1;
  pgi.perm_relation_.insert(pgi.perm_relation_.begin(),
                            PermRelation::none);
  auto& pl = pgi.perm_list_;
  for(size_t i=0; i<pl.size(); i++) {
    for(size_t j=0; j<pl[i].size(); j++) {
      pl[i][j] += 1;
    }
  }
  pl.insert(pl.begin(), TensorVec<unsigned int>{0});
  return pgi;
}
#endif

inline PermGroupInfo
operator | (PermGroupInfo lhs,
            PermGroupInfo rhs) {
  lhs.size_ += rhs.size_;
  lhs.perm_list_.insert_back(rhs.perm_list_.begin(), rhs.perm_list_.end());
  lhs.perm_relation_.insert_back(rhs.perm_relation_.begin(), rhs.perm_relation_.end());
  return lhs;
}

inline PermGroupInfo
operator - (const IndexLabel& lhs,
            const IndexLabel& rhs){

  return {2, {{static_cast<unsigned int>(lhs.label()), static_cast<unsigned int>(rhs.label())}}, {}};
}

// class OuterLabeledLoop : public LabeledLoop {
//   public:
//     OuterLabeledLoop() = default;
// };

// class InnerLabeledLoop : public LabeledLoop {
//   public:
//     InnerLabeledLoop() = default;

//     InnerLabeledLoop(const IndexLabelVec& ilv,
//                      const std::vector<Iterator>& begin,
//                      const std::vector<Iterator>& end,
//                      const std::vector<std::vector<LBCondition>>& lbs)
//           : LabeledLoop {ilv, begin, end, lbs} {}
// };

inline OuterLabeledLoop
outer(const PermGroupInfo& info) {
  PermGroup pg(info);

  //return pg.unique_loop_nest();
}

inline OuterLabeledLoop
outer(const IndexLabel& il) {
  return outer({1, {{static_cast<unsigned int>(il.label())}}, {} });
}

inline InnerLabeledLoop
inner(const PermGroupInfo& info) {
  PermGroup pg(info);

}

inline InnerLabeledLoop
inner(const IndexLabel& il) {
  return inner({1, {{static_cast<unsigned int>(il.label())}}, {}});
}

}  // namespace tammy

#endif // TAMMY_PERM_SYMMETRY_H_
