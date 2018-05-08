// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMM_LOOPS_HPP_
#define TAMM_LOOPS_HPP_

#include <algorithm>
#include <memory>
#include <vector>

#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/index_space.hpp"
#include "tamm/types.hpp"

namespace tamm {

using IndexLabelVec = std::vector<TiledIndexLabel>;

struct LBCondition {
    size_t pos;
    std::vector<size_t> lhs;
    std::vector<size_t> rhs;
};

template<typename Itr>
class LBLoopNest {
public:
    using value_type = typename Itr::value_type;
    using Iterator   = Itr;

    LBLoopNest(const std::vector<Itr>& begin, const std::vector<Itr>& end,
               const std::vector<std::vector<LBCondition>>& lbs) :
      begin_{begin},
      end_{end},
      itr_{begin},
      lbs_{lbs},
      size_{begin.size()} {
        // EXPECTS(begin.size() == size_);
        // EXPECTS(end.size() == size_);
        // EXPECTS(lbs.size() == size_);
        // reset();
        // for(size_t i = 0; i < size_; i++) { EXPECTS(itr_[i] <= end_[i]); }
        // for(const auto& lb_v : lbs_) {
        //     for(const auto& lb_el : lb_v) {
        //         EXPECTS(lb_el.pos < size_);
        //         EXPECTS(lb_el.lhs.size() == lb_el.rhs.size());
        //         for(size_t i = 0; i < lb_el.lhs.size(); i++) {
        //             EXPECTS(lb_el.lhs[i] < lb_el.pos);
        //             EXPECTS(lb_el.rhs[i] < lb_el.pos);
        //         }
        //     }
        // }
    }

    LBLoopNest() : LBLoopNest{{}, {}, {}} {}

    LBLoopNest(const LBLoopNest&) = default;

    void reset() {
        // if(size_ == 0) { return; }
        // reset_starting_from(0);
    }

    void reset_starting_from(size_t start) {
        // EXPECTS(start < size_);
        // reset_with_lb(start);
        // for(size_t i = start + 1; i < size_ && itr_[0] < end_[0]; i++) {
        //     reset_with_lb(i);
        //     for(; itr_[i] == end_[i] && i >= 1; i--) {
        //         itr_[i - 1]++;
        //         EXPECTS(itr_[i - 1] <= end_[i - 1]);
        //     }
        // }
        // if(itr_[0] == end_[0]) {
        //     itr_  = end_;
        //     done_ = true;
        // }
    }

    void get(std::vector<value_type>& loop_iteration) const {
        // loop_iteration.clear();
        // auto bi = std::back_inserter(loop_iteration);
        // for(const auto& it : itr_) { *bi++ = *it; }
    }

    std::vector<value_type> get() const {
        // std::vector<value_type> loop_iteration;
        // get(loop_iteration);
        // return loop_iteration;
    }

    template<typename OutputIterator>
    void get(OutputIterator oitr) const {
        // for(const auto& it : itr_) { *oitr++ = *it; }
    }

    void next() {
        // ssize_t i = static_cast<ssize_t>(size_) - 1;
        // for(; i >= 0; i--) {
        //     itr_[i]++;
        //     if(itr_[i] < end_[i]) { break; }
        // }
        // if(i < 0) {
        //     itr_  = end_;
        //     done_ = true;
        // } else if(i + 1 < size_) {
        //     reset_starting_from(i + 1);
        // }
    }

    bool has_more() const { return !done_; }

    size_t size() const { return size_; }

    void shift_lbs(size_t off) {
        // for(auto& lb_v : lbs_) {
        //     for(auto& lb_el : lb_v) {
        //         lb_el.pos += off;
        //         for(auto& l : lb_el.lhs) { l += off; }
        //         for(auto& r : lb_el.rhs) { r += off; }
        //     }
        // }
    }

    friend LBLoopNest<Itr> loop_nest_concat(
      TensorVec<LBLoopNest<Itr>> loop_nest_groups, bool lexicographic_concat) {
        if(loop_nest_groups.size() == 1) { return loop_nest_groups[0]; }
        size_t off = 0;
        std::vector<Itr> begin, end;
        std::vector<std::vector<LBCondition>> lbs;
        for(auto& lng : loop_nest_groups) {
            lng.shift_lbs(off);
            begin.insert(begin.end(), lng.begin_.begin(), lng.begin_.end());
            end.insert(end.end(), lng.end_.begin(), lng.end_.end());
            if(lexicographic_concat == false) {
                lbs.insert(lbs.end(), lng.lbs().begin(), lng.lbs().end());
            } else {
                if(off == 0) {
                    lbs.insert(lbs.end(), lng.lbs().begin(), lng.lbs().end());
                } else {
                    for(size_t i = 0; i < lng.size(); i++) {
                        size_t pos = off + i - lng.size();
                        std::vector<size_t> lhs, rhs;
                        for(size_t j = 0; j > i; j++) {
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
    const std::vector<std::vector<LBCondition>>& lbs() const { return lbs_; }

    //@post itr_[j]  <= end_[j]
    void reset_with_lb(size_t j) {
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
            if(consider) { itr_[j] = std::max(itr_[j], itr_[lb_el.pos]); }
        }
        if(itr_[j] >= end_[j]) { itr_[j] = end_[j]; }
    }

    std::vector<Itr> begin_;
    std::vector<Itr> end_;
    std::vector<Itr> itr_;
    std::vector<std::vector<LBCondition>> lbs_;

    bool done_;
    size_t size_;
}; // LBLoopNest

class LabeledLoop : public LBLoopNest<IndexSpace::Iterator> {
public:
    LabeledLoop() = default;

    LabeledLoop(const IndexLabelVec& ilv, const std::vector<Iterator>& begin,
                const std::vector<Iterator>& end,
                const std::vector<std::vector<LBCondition>>& lbs) :
      LBLoopNest{begin, end, lbs},
      ilv_{ilv} {}

    LabeledLoop(const IndexLabelVec& ilv,
                const LBLoopNest<IndexSpace::Iterator>& lb_loop) :
      LBLoopNest{lb_loop},
      ilv_{ilv} {}

    const IndexLabelVec& labels() const { return ilv_; }

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

    SymmLoop(const std::vector<std::pair<int, IndexLabelVec>>& relabel_list) :
      relabel_list_{relabel_list} {
        reset();
    }

    void reset() { it_ = relabel_list_.begin(); }

    bool has_more() const { return it_ != relabel_list_.end(); }

    void next() { it_++; }

    std::pair<int, IndexLabelVec> get() const { return *it_; }

protected:
    std::vector<std::pair<int, IndexLabelVec>> relabel_list_;
    std::vector<std::pair<int, IndexLabelVec>>::const_iterator it_;
};

class SymmFactor {
public:
    SymmFactor(const std::vector<IndexLabelVec>& ilv_vec) : ilv_vec_{ilv_vec} {}

    SymmFactor()                  = default;
    SymmFactor(const SymmFactor&) = default;

    // int factor(const std::map<IndexLabel, BlockIndex>& label_map) {
    //     int ret = 1;
    //     for(const auto& ilv : ilv_vec_) {
    //         std::map<BlockIndex, int> indices;
    //         for(const auto& il : ilv) {
    //             BlockIndex bid = label_map.find(il)->second;
    //             indices[bid] += 1;
    //         }
    //         int v = 1;
    //         for(const auto& ind : indices) { v *= factorial(ind.second); }
    //         ret *= factorial(ilv.size()) / v;
    //     }
    //     return ret;
    // }

private:
    static int factorial(size_t x) {
        EXPECTS(x < 6); // check for overflow
        int ret = 1;
        for(size_t i = 1; i <= x; i++) { ret *= i; }
        return ret;
    }
    std::vector<IndexLabelVec> ilv_vec_;
}; // class SymmFactor

inline SymmFactor symm_factor(
  const std::vector<IndexLabelVec>& perm_symm_groups) {
    return SymmFactor{perm_symm_groups};
}

class OpTemplate {
protected:
    LabeledLoop outer_loops;
    LabeledLoop inner_loops;
    SymmLoop symm_loop;
    SymmFactor symm_facctor;

    void execute() {
        // std::map<IndexLabel, BlockIndex> label_map;
        // for(; outer_loops.has_more(); outer_loops.next()) {
        //     auto outer_indices = outer_loops.get();

        //     for(; symm_loop.has_more(); symm_loop.next()) {
        //         auto outer_symm_loop = symm_loop.get();

        //         // label_map.update(outer_symm_loop, outer_indices);

        //         // alocate C block
        //         for(; inner_loops.has_more(); inner_loops.next()) {
        //             auto inner_indices = inner_loops.get();
        //             // label_map.update(inner_loops.labels(), inner_indices);

        //             // Get A (and B) block(s)
        //             // update C block with symm_factor
        //         }

        //         // update overall C block
        //         // Cbuf[outer_loop.labels()] = outer_sym_sign *
        //         // Csym_buf[outer_symm_loop];
        //     }
        //     // put/acc C block to outer_indices
        // }
    }
};

} // namespace tamm

#endif // TAMM_LOOPS_HPP_
