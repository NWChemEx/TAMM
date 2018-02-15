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
  TensorVec<TensorVec<unsigned int>> perm_list_;
  TensorVec<PermRelation> perm_relation_;
};  // PermGroupInfo

class PermGroup {
  public:
  size_t size_; //maximum size of the enclosing permutation group
  TensorVec<unsigned> indices_;
  TensorVec<std::unique_ptr<PermGroup>> groups_;
  PermRelation relation_;

  PermGroup() {}

  PermGroup(const PermGroupInfo& pgi)
      : PermGroup{pgi.size_,
        pgi.perm_list_,
        pgi.perm_relation_} { }

  PermGroup(const PermGroup& pg)
      : size_{pg.size_},
        indices_{pg.indices_},
        relation_{pg.relation_} {
          for(const auto& grp: pg.groups_) {
            auto ptr = std::make_unique<PermGroup>(*grp);
            groups_.push_back(std::move(ptr));
          }
        }

  PermGroup(size_t size,
            const TensorVec<unsigned>& indices,
            TensorVec<std::unique_ptr<PermGroup>>&& groups,
            PermRelation relation)
      : size_{size},
        indices_{indices},
        groups_{std::move(groups)},
        relation_{relation} {}

  PermGroup(size_t size,
            const TensorVec<unsigned>& indices,
            PermRelation relation)
      : size_{size},
        indices_{indices},
        relation_{relation} {}

  PermGroup(PermGroup&& pg)
      : size_{pg.size_},
        indices_{std::move(pg.indices_)},
        groups_{std::move(pg.groups_)},
        relation_{std::move(pg.relation_)} {}

  PermGroup(unsigned size,
            const TensorVec<TensorVec<unsigned>>& perm_list,
            const TensorVec<PermRelation>& perm_relation)
      : size_{size} {

    validate(size, perm_list, perm_relation);

    if(perm_list.size() == 1) {
      size_ = size;
      indices_ = perm_list[0];
      relation_ = perm_relation[0];
      return;
    }

    int ngrps = perm_list.size();
    std::vector<std::unique_ptr<PermGroup>> groups(2*size);
    std::vector<bool> group_exists(2*size, false);

    for(int g=0; g<ngrps; g++)  {
      assert(perm_list[g].size() > 0);

      TensorVec<unsigned> indices;
      TensorVec<std::unique_ptr<PermGroup>> itrs_g;

      if(perm_list[g][0] < size) {
        for(size_t i=0; i<perm_list[g].size(); i++) {
          assert(perm_list[g][i]<size);
        }
        indices.insert_back(perm_list[g].begin(),
                            perm_list[g].end());
      } else {
        for(size_t i=0; i<perm_list[g].size(); i++) {
          unsigned idx = perm_list[g][i];
          assert(idx >= size);
          group_exists[idx] = true;
          itrs_g.push_back(std::move(groups[idx]));
          group_exists[idx] = false;
        }
      }
      groups[g] = std::make_unique<PermGroup>(size, indices, std::move(itrs_g), perm_relation[g]);
      group_exists[g] = true;
    }

    for(size_t g = 0; g<groups.size(); g++) {
      if(group_exists[g]) {
        groups_.push_back(std::move(groups[g]));
      }
    }
    assert(groups_.size() > 0);
    if(groups_.size() > 1) {
      relation_ = PermRelation::none;
    } else {
      indices_ = groups_[0]->indices_;
      relation_ = groups_[0]->relation_;
      groups_ = std::move(groups_[0]->groups_);
    }
  }

  PermGroup& operator = (PermGroup&&) = default;

  bool empty() const {
    return indices_.size() == 0 && groups_.size() == 0;
  }

  size_t size() const {
    return size_;
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
    int nsub_inversions = 0;
    for(auto& grp: groups_) {
      nsub_inversions += grp->permute_to_unique(vec);
    }
    if(relation_ == PermRelation::none) {
      return nsub_inversions;
    }
    
    int ninversions = 0;
    for(size_t i=1; i < indices_.size(); i++) {
      for(int j=i-1; j>=0; j--) {
        if(vec[indices_[j]] > vec[indices_[i]]) {
          std::swap(vec[indices_[j]], vec[indices_[i]]);
          ninversions += 1;
        }
      }
    }
    for(size_t i=1; i < groups_.size(); i++) {
      for(int j=i-1; j>=0; j--) {
        if(greater_than(vec, *groups_[j], *groups_[i])) {
          swap(vec, *groups_[j], *groups_[i]);
          ninversions += 1;
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

  template<typename Itr, typename T> 
  std::unique_ptr<LBLoopNest<Itr>>
  unique_loop_nest(const TensorVec<IndexRange>& vec) const {

    TensorVec<std::unique_ptr<LBLoopNest<Itr>>> loops;

    std::vector<Itr> begin;
    std::vector<Itr> end;
    std::vector<std::vector<LBCondition>> lbs;
    size_t pos = -1;
    for(const auto& idx: indices_){
      begin.push_back(vec[idx].begin());
      end.push_back(vec[idx].end());
      if(pos < 0){
        lbs.push_back({});
      }
      else {
        lbs.push_back({pos++, {}, {}});
      }
    }
    loops.push_back(std::make_unique<LBLoopNest<Itr>>(begin, end, lbs));

    for(const auto& grp: groups_) {
      loops.push_back(grp->unique_loop_nest<Itr, T>(vec));
    }

    EXPECTS(loops.size() > 0);
    if(loops.size() == 1) {
      return std::move(loops[0]);
    }
    
    return combine(loops);
  }

  template<typename Itr> 
  LBLoopNest<Itr> unique_loop_nest() const {

  }

 private:
  //@todo Not sure this is checking all cases, especially recursive ones.
  static void validate(unsigned int size,
                       const TensorVec<TensorVec<unsigned int>>& perm_list,
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

  template<typename Itr>
  std::unique_ptr<LBLoopNest<Itr>> 
  combine(TensorVec<std::unique_ptr<LBLoopNest<Itr>>> loops) {
    // @todo: Combine LBLoopNest
    
    std::vector<Itr> begin;
    std::vector<Itr> end;
    std::vector<std::vector<LBCondition>> lbs;

    for(const auto& loop: loops) {

    }
    
    return std::make_unique<LBLoopNest<Itr>>(begin, end, lbs);
  }
  
  template<typename T>
  static void
  swap(TensorVec<T>& vec,
       const PermGroup& pg1,
       const PermGroup& pg2) {
    assert(pg1.size() == pg2.size());
    assert(vec.size() == pg1.size());
    assert(pg1.groups_.size() == pg2.groups_.size());
    assert(pg1.indices_.size() == pg2.indices_.size());

    for(size_t i=0; i<pg1.indices_.size(); i++) {
      std::swap(vec[pg1.indices_[i]],
                vec[pg2.indices_[i]]);
    }
    for(size_t i=0; i<pg1.groups_.size(); i++) {
      swap(vec, *pg1.groups_[i], *pg2.groups_[i]);
    }
  }

  template<typename T>
  static bool
  less_than_or_equal(const TensorVec<T>& vec,
                     const PermGroup& pg1,
                     const PermGroup& pg2) {
    assert(pg1.size() == pg2.size());
    assert(vec.size() == pg1.size());
    assert(pg1.groups_.size() == pg2.groups_.size());
    assert(pg1.indices_.size() == pg2.indices_.size());

    bool ret = true;
    for(size_t i=0; i<pg1.indices_.size(); i++) {
      ret = ret & (vec[pg1.indices_[i]] <=
                   vec[pg2.indices_[i]]);
    }
    for(size_t i=0; i<pg1.groups_.size(); i++) {
      ret = ret & less_than_or_equal(vec,
                                     *pg1.groups_[i],
                                     *pg2.groups_[i]);
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
    assert(pg1.groups_.size() == pg2.groups_.size());
    assert(pg1.indices_.size() == pg2.indices_.size());

    bool ret = true;
    for(size_t i=0; i<pg1.indices_.size(); i++) {
      ret = ret & (vec[pg1.indices_[i]] ==
                   vec[pg2.indices_[i]]);
    }
    for(size_t i=0; i<pg1.groups_.size(); i++) {
      ret = ret & equal(vec, *pg1.groups_[i], *pg2.groups_[i]);
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

inline PermGroupInfo
antisymm(unsigned int nupper, unsigned int nlower) {
  TensorVec<TensorVec<unsigned int>> perm_list;
  TensorVec<PermRelation> perm_relation;

  if(nupper > 0) {
    TensorVec<unsigned int> upper;
    for(unsigned int i=0; i<nupper; i++) {
      upper.push_back(i);
    }
    perm_list.push_back(upper);
    perm_relation.push_back(PermRelation::antisymmetry);
  }

  if(nlower > 0) {
    TensorVec<unsigned int> lower;
    for(unsigned int i=0; i<nlower; i++) {
      lower.push_back(nupper + i);
    }
    perm_list.push_back(lower);
    perm_relation.push_back(PermRelation::antisymmetry);
  }
  return {nupper+nlower, perm_list, perm_relation};
}

inline PermGroupInfo
symm(unsigned int nupper, unsigned int nlower) {
  assert(nupper == nlower);
  TensorVec<PermRelation> perm_relation(nupper+1, PermRelation::symmetry);

  TensorVec<TensorVec<unsigned int>> perm_groups;
  for(unsigned int i=0; i<nupper; i++) {
    perm_groups.push_back({i, i+nupper});
  }
  TensorVec<unsigned int> last;
  for(unsigned int i=0; i<nupper; i++) {
    last.push_back(nupper + nlower + i);
  }
  perm_groups.push_back(last);
  return {nupper+nlower, perm_groups, perm_relation};
}

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

class OuterLabeledLoop : public LabeledLoop {
  public: 
    OuterLabeledLoop() = default;
};

class InnerLabeledLoop : public LabeledLoop {
  public:
    InnerLabeledLoop() = default;

    InnerLabeledLoop(const IndexLabelVec& ilv,
                     const std::vector<Iterator>& begin,
                     const std::vector<Iterator>& end,
                     const std::vector<std::vector<LBCondition>>& lbs)
          : LabeledLoop {ilv, begin, end, lbs} {}
};

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
