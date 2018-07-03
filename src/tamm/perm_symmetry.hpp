// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMM_PERM_SYMMETRY_HPP_
#define TAMM_PERM_SYMMETRY_HPP_

#include "tamm/boundvec.h"
#include "tamm/errors.h"
#include "tamm/loops.h"
#include "tamm/types.h"

#include <memory>

namespace tamm {
enum class PermRelation { symmetry, antisymmetry, none };

struct PermGroupInfo {
    unsigned int size_;
    TensorVec<TensorVec<size_t>> perm_list_;
    TensorVec<PermRelation> perm_relation_;
}; // PermGroupInfo

class PermGroup {
    private:
    size_t size_; // maximum size of the enclosing permutation group
    size_t index_;
    std::vector<PermGroup> groups_;
    size_t num_groups_;
    PermRelation relation_;

    public:
    PermGroup() : size_{0}, index_{0}, relation_{PermRelation::none} {}

    PermGroup(const PermGroupInfo& pgi) :
      PermGroup{pgi.size_, pgi.perm_list_, pgi.perm_relation_} {}

    PermGroup(const PermGroup& pg) = default;
    PermGroup(PermGroup&& pg)      = default;

    PermGroup& operator=(PermGroup other) {
        swap(*this, other);
        return *this;
    }

    PermGroup(size_t size, size_t index = 0) :
      size_{size},
      index_{index},
      relation_{PermRelation::none} {
        EXPECTS(index < size);
    }

    PermGroup(size_t size, const std::vector<PermGroup>& groups,
              PermRelation relation) :
      size_{size},
      index_{0},
      groups_{groups},
      relation_{relation} {
        if(groups.size() == 1) { *this = groups[0]; }
    }

    PermGroup(size_t size, const TensorVec<size_t>& indices,
              PermRelation relation) :
      size_{size} {
        for(size_t ind : indices) { groups_.push_back(PermGroup{size_, ind}); }
        relation_ = relation;
        if(groups_.size() == 1) { relation_ = PermRelation::none; }
    }

    PermGroup(unsigned size, const TensorVec<TensorVec<size_t>>& perm_list,
              const TensorVec<PermRelation>& perm_relation) :
      size_{size} {
        validate(size, perm_list, perm_relation);

        if(perm_list.size() == 1) {
            *this = PermGroup{size, perm_list[0], perm_relation[0]};
            return;
        }

        size_t ngrps = perm_list.size();
        std::vector<PermGroup> groups(size);
        std::vector<bool> group_exists(size, false);
        EXPECTS(ngrps <= size);

        for(int g = 0; g < ngrps; g++) {
            EXPECTS(perm_list[g].size() > 0);
            std::vector<PermGroup> lgroups;
            if(perm_list[g][0] < size) {
                for(size_t i = 0; i < perm_list[g].size(); i++) {
                    EXPECTS(perm_list[g][i] < size);
                    lgroups.push_back(PermGroup{size, perm_list[g][i]});
                }
            } else {
                for(size_t i = 0; i < perm_list[g].size(); i++) {
                    EXPECTS(perm_list[g][i] >= size);
                    EXPECTS(perm_list[g][i] < 2 * size);
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
        for(size_t g = 0; g < groups.size(); g++) {
            if(group_exists[g]) { groups_.push_back(groups[g]); }
        }
        EXPECTS(groups_.size() > 0);
        if(groups_.size() == 1) { *this = groups_[0]; }
    }

    bool empty() const { return size() == 0; }

    size_t size() const { return size_; }

    size_t num_groups() const {
        if(empty()) {
            return 0;
        } else if(groups_.size() == 0) {
            return 1;
        } else {
            return groups_.size();
        }
    }

    PermRelation relation() const { return relation_; }

    size_t index() const { return index_; }

    const std::vector<PermGroup>& groups() const { return groups_; }

    template<typename T>
    std::pair<int, TensorVec<T>> find_unique(const TensorVec<T>& vec) const {
        TensorVec<T> ivec{vec};
        int ninv = permute_to_unique(ivec);
        return {ninv, ivec};
    }

    template<typename T>
    int permute_to_unique(TensorVec<T>& vec) const {
        if(num_groups() == 0 || num_groups() == 1) { return 0; }
        int nsub_inversions = 0;
        for(auto& grp : groups_) {
            nsub_inversions += grp.permute_to_unique(vec);
        }
        if(relation_ == PermRelation::none) { return nsub_inversions; }

        int ninversions;
        for(size_t i = 1; i < num_groups(); i++) {
            for(int j = i - 1; j >= 0; j--) {
                if(greater_than(vec, groups_[j], groups_[i])) {
                    swap(vec, groups_[j], groups_[i]);
                    ninversions += i - j;
                }
            }
        }
        if(relation_ == PermRelation::antisymmetry) {
            return nsub_inversions + ninversions;
        } else {
            return nsub_inversions;
        }
    }

    LBLoopNest<IndexSpace::Iterator> unique_loop_nest(
      const IndexRangeVec& irv) const {
        EXPECTS(size() == irv.size());
        if(num_groups() == 0) { return {}; }
        if(num_groups() == 1) {
            return {{irv[index()].begin()}, {irv[index()].end()}, {}};
        }
        TensorVec<LBLoopNest<IndexSpace::Iterator>> loop_nest_groups;
        for(size_t i = 0; i < num_groups(); i++) {
            loop_nest_groups.push_back(groups_[i].unique_loop_nest(irv));
        }
        // return combine(size_, loop_nest_groups, perm_relation_);
        return loop_nest_concat(loop_nest_groups,
                                relation_ != PermRelation::none);
    }

    LBLoopNest<IndexSpace::Iterator> unique_loop_nest(
      const IndexLabelVec& ilv) const {
        IndexRangeVec irv;
        for(const auto& lbl : ilv) { irv.push_back(lbl.ir()); }
        return unique_loop_nest(irv);
    }

    template<typename T>
    PermGroup slice(const TensorVec<T>& tv) const {
        if(num_groups() == 0 || num_groups() == 1) { return *this; }

        TensorVec<TensorVec<T>> group_tvs;
        for(size_t i = 0; i < num_groups(); i++) {
            group_tvs.push_back(gather(tv, i));
        }
        TensorVec<TensorVec<size_t>> nested_grp_ids;
        TensorVec<bool> considered(num_groups(), false);
        for(auto itr = considered.begin(); itr != considered.end();
            itr      = std::find(itr + 1, considered.end(), false)) {
            unsigned i = itr - considered.begin();
            nested_grp_ids.push_back({i});
            for(size_t j = i + 1; j < num_groups(); j++) {
                if(std::equal(group_tvs[j].begin(), group_tvs[j].end(),
                              group_tvs[i].begin())) {
                    EXPECTS(considered[j] == false);
                    nested_grp_ids.back().push_back(j);
                    considered[j] = true;
                }
            }
        }

        std::vector<PermGroup> nested_groups;
        for(size_t i = 0; i < nested_grp_ids.size(); i++) {
            EXPECTS(nested_grp_ids[i].size() > 0);
            if(nested_grp_ids[i].size() == 1) {
                nested_groups.push_back(
                  groups_[nested_grp_ids[i][0]].slice(tv));
            } else {
                std::vector<PermGroup> lgroups;
                for(size_t grp_id : nested_grp_ids[i]) {
                    lgroups.push_back(groups_[grp_id].slice(tv));
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

    PermGroup slice(const IndexLabelVec& ilv) const {
        IndexRangeVec irv;
        for(const auto& lbl : ilv) { irv.push_back(lbl.ir()); }
        return slice(irv);
    }

    static PermGroup antisymm(size_t nupper, size_t nlower) {
        TensorVec<size_t> upper(nupper), lower(nlower);
        std::iota(upper.begin(), upper.end(), 0);
        std::iota(lower.begin(), lower.end(), nupper);
        size_t size = nupper + nlower;
        return {size,
                std::vector<PermGroup>{
                  PermGroup{size, upper, PermRelation::antisymmetry},
                  PermGroup{size, lower, PermRelation::antisymmetry},
                },
                PermRelation::none};
    }

    static PermGroup symm(size_t size) {
        EXPECTS(size % 2 == 0);
        size_t nupper = size / 2;
        std::vector<PermGroup> groups;
        for(size_t i = 0; i < nupper; i++) {
            groups.push_back(
              {size, TensorVec<size_t>{i, i + nupper}, PermRelation::symmetry});
        }
        return {size, groups, PermRelation::symmetry};
    }

    //@todo Do not depend on MSO
    PermGroup remove_index(size_t ind) const {
        EXPECTS(ind >= 0 && ind < size_);
        TensorVec<bool> flags(size_, true);
        flags[ind] = false;
        return slice(flags);
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
        IndexLabelVec retl = gather(ilv);
        for(const auto& lbl : retl) { ret.push_back(lbl.ir()); }
        return ret;
    }

    template<typename T>
    TensorVec<T> gather(const TensorVec<T>& tv) const {
        TensorVec<T> ret;
        for(size_t i = 0; i < num_groups(); i++) {
            TensorVec<T> tvg{gather(tv, i)};
            ret.insert_back(tvg.begin(), tvg.end());
        }
        return ret;
    }

    template<typename T>
    TensorVec<T> gather(const TensorVec<T>& tv, size_t grp_id) const {
        EXPECTS(grp_id >= 0 && grp_id < num_groups());
        if(num_groups() == 1) {
            return {tv[index_]};
        } else {
            return groups_[grp_id].gather(tv);
        }
    }

    //@todo Not sure this is checking all cases, especially recursive ones.
    static void validate(unsigned int size,
                         const TensorVec<TensorVec<size_t>>& perm_list,
                         const TensorVec<PermRelation>& perm_relation) {
        assert(perm_list.size() == perm_relation.size());

        std::vector<bool> flags(2 * size, false);
        for(size_t i = 0; i < perm_list.size(); i++) {
            assert(perm_list.size() > 0);
            bool is_grp          = false;
            size_t grp_sz        = 0;
            PermRelation grp_rel = PermRelation::none;
            assert(perm_list[i].size() > 0);
            if(perm_list[i][0] >= size) {
                is_grp = true;
                assert(perm_list[i].size() > 1);
                size_t grp_id = perm_list[i][0] - size;
                assert(grp_id < perm_list.size());
                grp_sz  = perm_list[grp_id].size();
                grp_rel = perm_relation[grp_id];
            }
            for(size_t j = 0; j < perm_list[i].size(); j++) {
                assert(flags[perm_list[i][j]] == false);
                flags[perm_list[i][j]] = true;
                if(perm_list[i][j] < size) {
                    assert(!is_grp);
                } else if(perm_list[i][j] < 2 * size) {
                    assert(is_grp);
                    size_t gid = perm_list[i][j] - size;
                    assert(gid >= 0 && gid + 1 < perm_list.size());
                    assert(perm_list[gid].size() == grp_sz);
                    assert(perm_relation[gid] == grp_rel);
                } else {
                    assert(0); // invalid input
                }
            }
        }
        for(unsigned int i = 0; i < size; i++) { assert(flags[i] == true); }
    }

    template<typename T>
    static void vec_swap(TensorVec<T>& vec, const PermGroup& pg1,
                         const PermGroup& pg2) {
        assert(pg1.size() == pg2.size());
        assert(vec.size() == pg1.size());
        assert(pg1.num_groups() == pg2.num_groups());

        if(pg1.num_groups() == 0) {
            // no-op
        }
        if(pg1.num_groups() == 1) {
            return std::swap(vec[pg1.index()], vec[pg2.index()]);
        }
        for(size_t i = 0; i < pg1.groups_.size(); i++) {
            vec_swap(vec, pg1.groups_[i], pg2.groups_[i]);
        }
    }

    template<typename T>
    static bool less_than_or_equal(const TensorVec<T>& vec,
                                   const PermGroup& pg1, const PermGroup& pg2) {
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
        for(size_t i = 0; i < pg1.groups_.size(); i++) {
            ret = ret & less_than_or_equal(vec, pg1.groups_[i], pg2.groups_[i]);
        }
        return ret;
    }

    template<typename T>
    static bool equal(const TensorVec<T>& vec, const PermGroup& pg1,
                      const PermGroup& pg2) {
        assert(pg1.size() == pg2.size());
        assert(vec.size() == pg1.size());
        assert(pg1.num_groups() == pg2.num_groups());

        if(pg1.num_groups() == 0) { return true; }
        if(pg1.num_groups() == 1) {
            return vec[pg1.index()] == vec[pg2.index()];
        }

        bool ret = true;
        for(size_t i = 0; i < pg1.groups_.size(); i++) {
            ret = ret & equal(vec, pg1.groups_[i], pg2.groups_[i]);
        }
        return ret;
    }

    template<typename T>
    static bool greater_than(const TensorVec<T>& vec, const PermGroup& pg1,
                             const PermGroup& pg2) {
        return !less_than_or_equal(vec, pg1, pg2);
    }

}; // PermGroup

//@todo check compatibility
inline PermGroup operator+(const PermGroup& perm_group_1,
                           const PermGroup& perm_group_2) {
    return {perm_group_1.size(),
            std::vector<PermGroup>{perm_group_1, perm_group_2},
            PermRelation::none};
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

inline PermGroupInfo operator|(PermGroupInfo lhs, PermGroupInfo rhs) {
    lhs.size_ += rhs.size_;
    lhs.perm_list_.insert_back(rhs.perm_list_.begin(), rhs.perm_list_.end());
    lhs.perm_relation_.insert_back(rhs.perm_relation_.begin(),
                                   rhs.perm_relation_.end());
    return lhs;
}

inline PermGroupInfo operator-(const IndexLabel& lhs, const IndexLabel& rhs) {
    return {2,
            {{static_cast<unsigned int>(lhs.label()),
              static_cast<unsigned int>(rhs.label())}},
            {}};
}

inline OuterLabeledLoop outer(const PermGroupInfo& info) {
    PermGroup pg(info);

    // return pg.unique_loop_nest();
}

inline OuterLabeledLoop outer(const IndexLabel& il) {
    return outer({1, {{static_cast<unsigned int>(il.label())}}, {}});
}

inline InnerLabeledLoop inner(const PermGroupInfo& info) { PermGroup pg(info); }

inline InnerLabeledLoop inner(const IndexLabel& il) {
    return inner({1, {{static_cast<unsigned int>(il.label())}}, {}});
}

} // namespace tamm

#endif // TAMM_PERM_SYMMETRY_H_
