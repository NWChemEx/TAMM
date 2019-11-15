#ifndef TAMM_OPS_HPP_
#define TAMM_OPS_HPP_

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>


#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"

#include "tamm/tensor.hpp"
#include "tamm/types.hpp"
#include "tamm/work.hpp"
#include "tamm/runtime_engine.hpp"
#include "tamm/utils.hpp"
#include "tamm/kernels/assign.hpp"
#include "tamm/kernels/multiply.hpp"

#define DO_NB 0

namespace tamm {

extern double multOpTime;
extern double multOpGetTime;
extern double multOpWaitTime;
extern double multOpAddTime;
extern double multOpDgemmTime;

class TimerGuard {
public:
    TimerGuard(double *refptr)
    : refptr_{refptr} {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    ~TimerGuard() {
        std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
        *refptr_ += std::chrono::duration_cast<std::chrono::duration<double>>((end_time - start_time_)).count();
    }
private:
    double *refptr_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};  // TimeGuard

enum class ResultMode { update, set };

namespace internal {

template<typename T>
class LabelMap {
public:
    LabelMap()                = default;
    LabelMap(const LabelMap&) = default;
    LabelMap(LabelMap&&)      = default;
    LabelMap& operator=(const LabelMap&) = default;
    LabelMap& operator=(LabelMap&&) = default;
    ~LabelMap()                     = default;

    LabelMap& update(const IndexLabelVec& labels, const std::vector<T>& vals) {
        EXPECTS(labels.size() == vals.size());
        for(size_t i = 0; i < vals.size(); i++) { map_[labels[i]] = vals[i]; }
        return *this;
    }

    std::vector<T> get(const IndexLabelVec& labels) {
        std::vector<T> ret;
        for(const auto& lbl : labels) {
            auto itr = map_.find(lbl);
            EXPECTS(itr != map_.end());
            ret.push_back(itr->second);
        }
        return ret;
    }

private:
    std::map<TiledIndexLabel, T> map_;
};

inline size_t idx(int n, const size_t* id, const size_t* sz,
                  const PermVector& p) {
    size_t idx = 0;
    for(int i = 0; i < n - 1; i++) { idx = (idx + id[p[i]]) * sz[p[i + 1]]; }
    if(n > 0) { idx += id[p[n - 1]]; }
    return idx;
}

template<typename T>
inline void index_permute(T* dbuf, const T* sbuf,
                          const PermVector& perm_to_dest,
                          const std::vector<size_t>& ddims, T scale) {
    EXPECTS(dbuf != nullptr && sbuf != nullptr);
    EXPECTS(perm_to_dest.size() == ddims.size());

    const size_t ndim = perm_to_dest.size();
    EXPECTS(ddims.size() == ndim);

    if(ndim == 0) {
        dbuf[0] = scale * sbuf[0];
    } else if(ndim == 1) {
        for(size_t i = 0; i < ddims[0]; i++) { dbuf[i] = scale * sbuf[i]; }
    } else if(ndim == 2) {
        size_t sz[] = {ddims[0], ddims[1]};
        size_t i[2], c;
        for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
            for(i[1] = 0; i[1] < sz[1]; i[1]++, c++) {
                dbuf[c] = scale * sbuf[idx(2, i, sz, perm_to_dest)];
            }
        }
    } else if(ndim == 3) {
        size_t sz[] = {ddims[0], ddims[1], ddims[2]};
        size_t i[3], c;
        for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
            for(i[1] = 0; i[1] < sz[1]; i[1]++) {
                for(i[2] = 0; i[2] < sz[2]; i[2]++, c++) {
                    dbuf[c] = scale * sbuf[idx(3, i, sz, perm_to_dest)];
                }
            }
        }
    } else if(ndim == 4) {
        size_t sz[] = {ddims[0], ddims[1], ddims[2], ddims[3]};
        size_t i[4], c;
        for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
            for(i[1] = 0; i[1] < sz[1]; i[1]++) {
                for(i[2] = 0; i[2] < sz[2]; i[2]++) {
                    for(i[3] = 0; i[3] < sz[3]; i[3]++, c++) {
                        dbuf[c] = scale * sbuf[idx(4, i, sz, perm_to_dest)];
                    }
                }
            }
        }
    } else {
        NOT_IMPLEMENTED();
    }

}

template<typename T>
inline void index_permute_acc(T* dbuf, const T* sbuf,
                              const PermVector& perm_to_dest,
                              const std::vector<size_t>& ddims, T scale) {
    EXPECTS(dbuf != nullptr && sbuf != nullptr);
    EXPECTS(perm_to_dest.size() == ddims.size());

    const size_t ndim = perm_to_dest.size();
    EXPECTS(ddims.size() == ndim);

    if(ndim == 0) {
        dbuf[0] = scale * sbuf[0];
    } else if(ndim == 1) {
        for(size_t i = 0; i < ddims[0]; i++) { dbuf[i] += scale * sbuf[i]; }
    } else if(ndim == 2) {
        size_t sz[] = {ddims[0], ddims[1]};
        size_t i[2], c;
        for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
            for(i[1] = 0; i[1] < sz[1]; i[1]++, c++) {
                dbuf[c] += scale * sbuf[idx(2, i, sz, perm_to_dest)];
            }
        }
    } else if(ndim == 3) {
        size_t sz[] = {ddims[0], ddims[1], ddims[2]};
        size_t i[3], c;
        for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
            for(i[1] = 0; i[1] < sz[1]; i[1]++) {
                for(i[2] = 0; i[2] < sz[2]; i[2]++, c++) {
                    dbuf[c] += scale * sbuf[idx(3, i, sz, perm_to_dest)];
                }
            }
        }
    } else if(ndim == 4) {
        size_t sz[] = {ddims[0], ddims[1], ddims[2], ddims[3]};
        size_t i[4], c;
        for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
            for(i[1] = 0; i[1] < sz[1]; i[1]++) {
                for(i[2] = 0; i[2] < sz[2]; i[2]++) {
                    for(i[3] = 0; i[3] < sz[3]; i[3]++, c++) {
                        dbuf[c] += scale * sbuf[idx(4, i, sz, perm_to_dest)];
                    }
                }
            }
        }
    } else {
        NOT_IMPLEMENTED();
    }
}

/**
 * @brief
 *
 * @todo add support for triangular arrays
 *
 * @tparam T
 * @param dbuf
 * @param ddims
 * @param dlabel
 * @param sbuf
 * @param sdims
 * @param slabel
 * @param scale
 * @param update
 */
template<typename T>
inline void block_add(T* dbuf, const std::vector<size_t>& ddims,
                      const IndexLabelVec& dlabel, T* sbuf,
                      const std::vector<size_t>& sdims,
                      const IndexLabelVec& slabel, T scale, bool update) {
    if(are_permutations(dlabel, slabel)) {
        EXPECTS(slabel.size() == dlabel.size());
        EXPECTS(sdims.size() == slabel.size());
        EXPECTS(ddims.size() == dlabel.size());
        auto label_perm = perm_compute(dlabel, slabel);
        for(unsigned i = 0; i < label_perm.size(); i++) {
            EXPECTS(ddims[i] == sdims[label_perm[i]]);
        }
        if(!update) {
            index_permute(dbuf, sbuf, label_perm, ddims, scale);
        } else {
            index_permute_acc(dbuf, sbuf, label_perm, ddims, scale);
        }
    } else {
        IndexLabelVec unique_labels = unique_entries(dlabel);
        unique_labels               = sort_on_dependence(unique_labels);

        const auto& dperm_map = perm_map_compute(unique_labels, dlabel);
        const auto& sperm_map = perm_map_compute(unique_labels, slabel);
        const auto& dinv_pm   = perm_map_compute(dlabel, unique_labels);

        auto idx = [](const auto& index_vec, const auto& dims_vec) {
            size_t ret = 0, ld = 1;
            EXPECTS(index_vec.size() == dims_vec.size());
            for(int i = index_vec.size(); i >= 0; i--) {
                ret += ld * index_vec[i];
                ld *= dims_vec[i];
            }
            return ret;
        };

        std::vector<size_t> itrv(unique_labels.size(), 0);
        std::vector<size_t> endv(unique_labels.size());
        endv = internal::perm_map_apply(ddims, dinv_pm);
        do {
            const auto& itval  = itrv;
            const auto& sindex = perm_map_apply(itval, sperm_map);
            const auto& dindex = perm_map_apply(itval, dperm_map);
            if(!update) {
                dbuf[idx(dindex, ddims)] = scale * sbuf[idx(sindex, sdims)];
            } else {
                dbuf[idx(dindex, ddims)] += scale * sbuf[idx(sindex, sdims)];
            }
        } while(internal::cartesian_iteration(itrv, endv));
    }
}

template<typename T>
inline void block_mult(T cscale, T* cbuf, const std::vector<size_t>& cdims,
                       const IndexLabelVec& clabel, T abscale, T* abuf,
                       const std::vector<size_t>& adims,
                       const IndexLabelVec& alabel, T* bbuf,
                       const std::vector<size_t>& bdims,
                       const IndexLabelVec& blabel) {
    for(const auto& d : cdims) {
        if(d == 0) { return; }
    }
    for(const auto& d : adims) {
        if(d == 0) { return; }
    }
    for(const auto& d : bdims) {
        if(d == 0) { return; }
    }

    IndexLabelVec all_labels{clabel};
    all_labels.insert(all_labels.end(), alabel.begin(), alabel.end());
    all_labels.insert(all_labels.end(), blabel.begin(), blabel.end());

    IndexLabelVec unique_labels = unique_entries(all_labels);
    IndexLabelVec sorted_labels = sort_on_dependence(unique_labels);

    const auto& cperm_map  = perm_map_compute(sorted_labels, clabel);
    const auto& aperm_map  = perm_map_compute(sorted_labels, alabel);
    const auto& bperm_map  = perm_map_compute(sorted_labels, blabel);
    const auto& all_inv_pm = perm_map_compute(all_labels, sorted_labels);

    auto idx = [](const auto& index_vec, const auto& dims_vec) {
        size_t ret = 0, ld = 1;
        EXPECTS(index_vec.size() == dims_vec.size());
        for(int i = -1 + index_vec.size(); i >= 0; i--) {
            ret += ld * index_vec[i];
            ld *= dims_vec[i];
        }
        return ret;
    };

    std::vector<size_t> itrv(sorted_labels.size(), 0);
    std::vector<size_t> endv(sorted_labels.size());

    std::vector<size_t> all_dims{cdims};
    all_dims.insert(all_dims.end(), adims.begin(), adims.end());
    all_dims.insert(all_dims.end(), bdims.begin(), bdims.end());
    endv = internal::perm_map_apply(all_dims, all_inv_pm);

    if(std::fabs(cscale) > 1e-11) { NOT_IMPLEMENTED(); }
    do {
        const auto& itval  = itrv;
        const auto& cindex = perm_map_apply(itval, cperm_map);
        const auto& aindex = perm_map_apply(itval, aperm_map);
        const auto& bindex = perm_map_apply(itval, bperm_map);
        size_t cidx        = idx(cindex, cdims);
        cbuf[cidx] +=
          abscale * abuf[idx(aindex, adims)] * bbuf[idx(bindex, bdims)];
    } while(internal::cartesian_iteration(itrv, endv));
}

} // namespace internal

class OpList;

class Op {
public:
    virtual TensorBase* writes() const                      = 0;
    virtual TensorBase* accumulates() const                 = 0;
    virtual std::vector<TensorBase*> reads() const          = 0;
    virtual bool is_memory_barrier() const                  = 0;
    virtual std::shared_ptr<Op> clone() const               = 0;
    virtual void execute(ExecutionContext& ec,
                         ExecutionHW hw = ExecutionHW::CPU) = 0;
    virtual OpList canonicalize() const                     = 0;
    virtual ~Op() {}
};

class OpList : public std::vector<std::shared_ptr<Op>> {
public:
    // Ctors
    OpList() {}

    template<typename T, typename... Args>
    OpList(T l_op, Args... args) : OpList(args...) {
        insert(begin(), l_op.clone());
    }
}; // OpList

template<typename T, typename LabeledTensorT>
class SetOp : public Op {
public:
    SetOp() = default;

    SetOp(LabeledTensorT lhs, T alpha, bool is_assign) :
      lhs_{lhs},
      alpha_{alpha},
      is_assign_{is_assign} {
        if(!lhs.has_str_lbl() && !lhs.labels().empty()) {
            
            auto lbls = lhs.labels();
            internal::update_labels(lbls);
            lhs_.set_labels(lbls);
        } 
        
        if(lhs.has_str_lbl()){
            fillin_labels();
        }

        validate();
    }

    SetOp(const SetOp<T, LabeledTensorT>&) = default;

    T alpha() const { return alpha_; }

    LabeledTensorT lhs() const { return lhs_; }

    bool is_assign() const { return is_assign_; }

    OpList canonicalize() const override { return OpList{(*this)}; }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new SetOp<T, LabeledTensorT>{*this});
    }

    void execute(ExecutionContext& ec, ExecutionHW hw = ExecutionHW::CPU) override {
        
#if 0
//previous implementation
        using TensorElType = typename LabeledTensorT::element_type;
        LabelLoopNest loop_nest{lhs_.labels()};

        auto lambda = [&](const IndexVector& blockid) {
            auto tensor = lhs_.tensor();
            EXPECTS(blockid.size() == lhs_.labels().size());
            EXPECTS(blockid.size() == tensor.num_modes());
            const auto& translated_blockid =
              internal::translate_blockid(blockid, lhs_);

            const size_t size = tensor.block_size(translated_blockid);
            
            std::vector<TensorElType> buf(size, static_cast<TensorElType>(alpha()));

            if(is_assign_) {
                tensor.put(translated_blockid, buf);
            } else {
                tensor.add(translated_blockid, buf);
            }
        };
        do_work(ec, loop_nest, lambda);
#endif
#if 1
        LabelLoopNest loop_nest{lhs_.labels()};

        auto lambda = [=,&loop_nest](const IndexVector& blockid) {
            auto tensor = lhs_.tensor();
            EXPECTS(blockid.size() == lhs_.labels().size());
            EXPECTS(blockid.size() == tensor.num_modes());
#if 1      
            auto [extracted_llabels, lblockid] = internal::extract_blockid_and_label(
                loop_nest.sorted_unique_labels(), blockid, lhs_.labels());
            
            EXPECTS(lhs_.labels().size() == extracted_llabels.size());

            for(int i = 0; i < extracted_llabels.size(); i++) {
                EXPECTS(
                  extracted_llabels[i].tiled_index_space().is_compatible_with(
                    lhs_.labels()[i].tiled_index_space()));
            }

            for(int i = 0; i < extracted_llabels.size(); i++) {
                EXPECTS(
                  extracted_llabels[i].tiled_index_space().is_compatible_with(
                    lhs_.tensor()().labels()[i].tiled_index_space()));
            }

            IndexVector translated_blockid;
            bool tlb_valid;
            std::tie(translated_blockid, tlb_valid) =
              internal::translate_blockid_if_possible(
                lblockid, extracted_llabels, lhs_.tensor()().labels());

            // if(translated_blockid.empty()) {
            //     return;
            // }

            for(const auto id : translated_blockid) {
                if(id == -1) return;
            }

            if(!tensor.is_non_zero(translated_blockid)) { return; }
#else            
            const auto translated_blockid = internal::translate_blockid(blockid, lhs_);
#endif
            if(is_assign_) {
                ec.re()->submitTask([=](RuntimeEngine::RuntimeContext rc) {
                        BlockBuffer bf = rc.get_buf_tmp(tensor, translated_blockid);
                        std::fill(bf.begin(), bf.end(), alpha_);
                        bf.release_put();  // goes through runtime (may be lazy)
                        }, TempAccess{IndexedTensor{tensor, translated_blockid}},
                        WritePermission{IndexedTensor{tensor, translated_blockid}});
            } else {
                ec.re()->submitTask([=](RuntimeEngine::RuntimeContext rc) {
                        BlockBuffer bf = rc.get_buf_tmp(tensor, translated_blockid);
                        std::fill(bf.begin(), bf.end(), alpha_);
                        bf.release_add();
                        }, TempAccess{IndexedTensor{tensor, translated_blockid}},
                        AccumPermission{IndexedTensor{tensor, translated_blockid}});
            }
        };
        do_work(ec, loop_nest, lambda);
#endif
    }

    TensorBase* writes() const {
        if(is_assign()) {
            return lhs_.base_ptr();
        } else {
            return nullptr;
        }
    }

    std::vector<TensorBase*> reads() const {
        std::vector<TensorBase*> res;
        return res;
    }

    TensorBase* accumulates() const {
        if(is_assign()) {
            return nullptr;
        } else {
            return lhs_.base_ptr();
        }
    }
    bool is_memory_barrier() const {
        return false;
    }

protected:
    void fillin_labels() {
        using internal::fillin_tensor_label_from_map;
        using internal::update_fillin_map;
        std::map<std::string, Label> str_to_labels;
        update_fillin_map(str_to_labels, lhs_.str_map(), lhs_.str_labels(), 0);
        fillin_tensor_label_from_map(lhs_, str_to_labels);
    }
    
    /**
     * @brief Check if the parameters form a valid operation. The parameters
     * form a valid operation if:
     *
     * 1. Every label depended on by another label (i.e., all 'd' such that
     * there exists label 'l(d)') is bound at least once
     *
     * 2. There are no conflicting dependent label specifications. That if
     * 'a(i)' is a label in either lta or ltc, there is no label 'a(j)' (i!=j)
     * in either lta or ltc.
     *
     * @pre lhs_.validate(), rhs1_.validate() and rhs2_.validate() have been
     *  invoked
     */
    void validate() {
        IndexLabelVec ilv{lhs_.labels()};

        for(size_t i = 0; i < ilv.size(); i++) {
            for(const auto& dl : ilv[i].secondary_labels()) {
                size_t j;
                for(j = 0; j < ilv.size(); j++) {
                    if(dl.tiled_index_space() == ilv[j].tiled_index_space() &&
                       dl.label() == ilv[j].label()) {
                        break;
                    }
                }
                EXPECTS(j < ilv.size());
            }
        }

        for(size_t i = 0; i < ilv.size(); i++) {
            const auto& ilbl = ilv[i];
            for(size_t j = i + 1; j < ilv.size(); j++) {
                const auto& jlbl = ilv[j];
                if(ilbl.tiled_index_space() == jlbl.tiled_index_space() &&
                   ilbl.label() == jlbl.label()) {
                    EXPECTS(ilbl == jlbl);
                }
            }
        }
    }

    LabeledTensorT lhs_;
    T alpha_;
    bool is_assign_;
}; // class SetOp

template<typename LabeledTensorT, typename Func>
class ScanOp : public Op {
public:
    ScanOp(const LabeledTensorT& lhs, Func func) : lhs_{lhs}, func_{func} {
        fillin_labels();
    }

    OpList canonicalize() const override { return OpList{(*this)}; }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new ScanOp<LabeledTensorT, Func>{*this});
    }

    void execute(ExecutionContext& ec, ExecutionHW hw = ExecutionHW::CPU) override {
        using TensorElType = typename LabeledTensorT::element_type;
        // the iterator to generate the tasks
        const auto& tensor = lhs_.tensor();
        LabelLoopNest loop_nest{lhs_.labels()};
        // const IndexLabelVec& iter_labels =
        // internal::sort_on_dependence(lhs_.labels());
        // std::vector<IndexLoopBound> ilbs;
        // for(const auto& lbl : iter_labels) { ilbs.push_back({lbl}); }
        // IndexLoopNest loop_nest { ilbs };
        // const std::vector<size_t>& lhs_pm =
        //   internal::perm_map_compute(iter_labels, lhs_.labels());
        // auto loop_nest = lhs_.tensor().loop_nest();
        // function to compute one block
        auto lambda = [&](const IndexVector& blockid) {
            auto tensor = lhs_.tensor();
            EXPECTS(blockid.size() == lhs_.labels().size());
            EXPECTS(blockid.size() == tensor.num_modes());
            const auto& translated_blockid =
              internal::translate_blockid(blockid, lhs_);
            // const IndexVector& blockid =
            //   internal::perm_map_apply(itval, lhs_pm);
            const size_t size = tensor.block_size(translated_blockid);
            std::vector<TensorElType> buf(size);
            tensor.get(translated_blockid, buf);
            func_(tensor, translated_blockid, buf);
        };
        // ec->...(loop_nest, lambda);
        //@todo use a scheduler
        do_work(ec, loop_nest, lambda);
    }

    TensorBase* writes() const {
        return nullptr;
    }

    std::vector<TensorBase*> reads() const {
        return {lhs_.base_ptr()};
    }

    TensorBase* accumulates() const {
        return nullptr;
    }

    bool is_memory_barrier() const {
        return false;
    }

protected:
    void fillin_labels() {
        using internal::fillin_tensor_label_from_map;
        using internal::update_fillin_map;
        std::map<std::string, Label> str_to_labels;
        update_fillin_map(str_to_labels, lhs_.str_map(), lhs_.str_labels(), 0);
        fillin_tensor_label_from_map(lhs_, str_to_labels);
    }

    /**
     * @brief Check if the parameters form a valid operation. The parameters
     * form a valid operation if:
     *
     * 1. Every label depended on by another label (i.e., all 'd' such that
     * there exists label 'l(d)') is bound at least once
     *
     * 2. There are no conflicting dependent label specifications. That if
     * 'a(i)' is a label in either lta or ltc, there is no label 'a(j)' (i!=j)
     * in either lta or ltc.
     *
     * @pre lhs_.validate(), rhs1_.validate() and rhs2_.validate() have been
     *  invoked
     */
    void validate() {
        IndexLabelVec ilv{lhs_.labels()};

        for(size_t i = 0; i < ilv.size(); i++) {
            for(const auto& dl : ilv[i].secondary_labels()) {
                size_t j;
                for(j = 0; j < ilv.size(); j++) {
                    if(dl.tiled_index_space() == ilv[j].tiled_index_space() &&
                       dl.label() == ilv[j].label()) {
                        break;
                    }
                }
                EXPECTS(j < ilv.size());
            }
        }

        for(size_t i = 0; i < ilv.size(); i++) {
            const auto& ilbl = ilv[i];
            for(size_t j = i + 1; j < ilv.size(); j++) {
                const auto& jlbl = ilv[j];
                if(ilbl.tiled_index_space() == jlbl.tiled_index_space() &&
                   ilbl.label() == jlbl.label()) {
                    EXPECTS(ilbl == jlbl);
                }
            }
        }
    }

    LabeledTensorT lhs_;
    Func func_;
};

/**
 * @ingroup operations
 * @brief Map operation. Invoke a function on each block of a tensor to set it.
 * @tparam LabeledTensorType
 * @tparam Func
 * @tparam N
 */
template<typename LabeledTensorT, typename Func, int N>
class MapOp : public Op {
public:
    using RHS = std::array<LabeledTensorT, N>;
    using T   = typename LabeledTensorT::element_type;

    MapOp(LabeledTensorT& lhs, Func func, RHS& rhs,
          ResultMode mode = ResultMode::set) :
      lhs_{lhs},
      func_{func},
      rhs_{rhs} {
        fillin_labels();
        validate();
    }

    OpList canonicalize() const override { return OpList{(*this)}; }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new MapOp<LabeledTensorT, Func, N>{*this});
    }

    void execute(ExecutionContext& ec, ExecutionHW hw = ExecutionHW::CPU) override {
        using TensorElType = typename LabeledTensorT::element_type;

        IndexLabelVec merged_labels{lhs_.labels()};
        for(const auto& rlt : rhs_) {
            merged_labels.insert(merged_labels.end(), rlt.labels().begin(),
                                 rlt.labels().end());
        }
        LabelLoopNest loop_nest{merged_labels};

        auto lambda = [&](const IndexVector& itval) {
            auto ltensor = lhs_.tensor();
            IndexVector lblockid, rblockid[N];
            auto it = itval.begin();
            lblockid.insert(lblockid.end(), it, it + lhs_.labels().size());
            it += lhs_.labels().size();
            for(size_t i = 0; i < N; i++) {
                rblockid[i].insert(rblockid[i].end(), it,
                                   it + rhs_[i].labels().size());
                it += rhs_[i].labels().size();
                // Translate each rhs blockid
                rblockid[i] = internal::translate_blockid(rblockid[i], rhs_[i]);
            }
            // Translate lhs blockid
            lblockid = internal::translate_blockid(lblockid, lhs_);

            const size_t lsize = ltensor.block_size(lblockid);
            std::vector<TensorElType> lbuf(lsize);
            std::vector<TensorElType> rbuf[N];
            for(size_t i = 0; i < N; i++) {
                const auto& rtensor_i = rhs_[i].tensor();
                const size_t isz      = rtensor_i.block_size(rblockid[i]);
                rbuf[i].resize(isz);
                rtensor_i.get(rblockid[i], rbuf[i]);
            }
            func_(ltensor, lblockid, lbuf, rblockid, rbuf);
            ltensor.put(lblockid, lbuf);
        };
        //@todo use a scheduler
        do_work(ec, loop_nest, lambda);
    }

    TensorBase* writes() const {
        return lhs_.base_ptr();
    }

    TensorBase* accumulates() const {
        return nullptr;
    }

    std::vector<TensorBase*> reads() const {
        std::vector<TensorBase*> res; 
        for(const auto& lt : rhs_) {
            res.push_back(lt.base_ptr());
        }
        return res;
    }

    bool is_memory_barrier() const {
        return false;
    }

protected:
    void fillin_labels() {
        using internal::fillin_tensor_label_from_map;
        using internal::update_fillin_map;
        std::map<std::string, Label> str_to_labels;
        update_fillin_map(str_to_labels, lhs_.str_map(), lhs_.str_labels(), 0);
        size_t off = lhs_.str_labels().size();
        update_fillin_map(str_to_labels, lhs_.str_map(), lhs_.str_labels(), 0);
        for(size_t i = 0; i < N; i++) {
            update_fillin_map(str_to_labels, rhs_[i].str_map(),
                              rhs_[i].str_labels(), off);
            off += rhs_[i].str_labels().size();
        }
        fillin_tensor_label_from_map(lhs_, str_to_labels);
        for(size_t i = 0; i < N; i++) {
            fillin_tensor_label_from_map(rhs_[i], str_to_labels);
        }
    }

    void validate() {
        for(auto& rhs : rhs_) {
            EXPECTS_STR((lhs_.tensor().base_ptr()!= rhs.tensor().base_ptr()), 
                      "Self assignment is not supported in tensor operations!");
        }

        IndexLabelVec ilv{lhs_.labels()};
        for(size_t i = 0; i < N; i++) {
            ilv.insert(ilv.end(), rhs_[i].labels().begin(),
                       rhs_[i].labels().end());
        }

        for(size_t i = 0; i < ilv.size(); i++) {
            for(const auto& dl : ilv[i].secondary_labels()) {
                size_t j;
                for(j = 0; j < ilv.size(); j++) {
                    if(dl.tiled_index_space() == ilv[j].tiled_index_space() &&
                       dl.label() == ilv[j].label()) {
                        break;
                    }
                }
                EXPECTS(j < ilv.size());
            }
        }

        for(size_t i = 0; i < ilv.size(); i++) {
            const auto& ilbl = ilv[i];
            for(size_t j = i + 1; j < ilv.size(); j++) {
                const auto& jlbl = ilv[j];
                if(ilbl.tiled_index_space() == jlbl.tiled_index_space() &&
                   ilbl.label() == jlbl.label()) {
                    EXPECTS(ilbl == jlbl);
                }
            }
        }
    }

    LabeledTensorT& lhs_;
    Func func_;
    std::array<LabeledTensorT, N> rhs_;
};

template<typename T>
std::ostream& operator << (std::ostream& os, const std::vector<T>& vec) {
    os<<"<";
    for(const auto& v: vec) {
        os<<v<<", ";
    }
    os<<">";
    return os;
}

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2>
class AddOp : public Op {
public:
    AddOp() = default;
    AddOp(LabeledTensorT1 lhs, T alpha, LabeledTensorT2 rhs, bool is_assign) :
      lhs_{lhs},
      alpha_{alpha},
      rhs_{rhs},
      is_assign_{is_assign} {
        EXPECTS(lhs.has_str_lbl() == rhs.has_str_lbl());

        if(!lhs.has_str_lbl() && !lhs.labels().empty()) {
            auto lhs_lbls = lhs.labels();
            auto rhs_lbls = rhs.labels();

            auto labels{lhs_lbls};
            labels.insert(labels.end(), rhs_lbls.begin(), rhs_lbls.end());
            internal::update_labels(labels);

            lhs_lbls = IndexLabelVec(labels.begin(),
                                     labels.begin() + lhs.labels().size());
            rhs_lbls = IndexLabelVec(labels.begin() + lhs.labels().size(),
                                     labels.begin() + lhs.labels().size() +
                                       rhs.labels().size());

            lhs_.set_labels(lhs_lbls);
            rhs_.set_labels(rhs_lbls);
        }

        if(lhs.has_str_lbl()){
            fillin_labels();
        }

        fillin_int_labels();
        validate();
    }

    AddOp(const AddOp<T, LabeledTensorT1, LabeledTensorT2>&) = default;

    T alpha() const { return alpha_; }

    LabeledTensorT1 lhs() const { return lhs_; }

    LabeledTensorT2 rhs() const { return rhs_; }

    bool is_assign() const { return is_assign_; }

    OpList canonicalize() const override {
        OpList result{};

        if(is_assign_) {
            auto lhs{lhs_};
            auto assign_op = (lhs = 0);
            result.push_back(assign_op.clone());
            AddOp n_op{lhs_, alpha_, rhs_, false};
            result.push_back(n_op.clone());
        } else {
            result.push_back((*this).clone());
        }

        return result;
    }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new AddOp<T, LabeledTensorT1, LabeledTensorT2>{*this});
    }

    void execute(ExecutionContext& ec, ExecutionHW hw = ExecutionHW::CPU) override {
    #if 1
        // std::cerr<<"DOING ADDOP\n";

        IndexLabelVec merged_labels{lhs_.labels()};
        merged_labels.insert(merged_labels.end(), rhs_.labels().begin(),
                             rhs_.labels().end());

        LabelLoopNest loop_nest{merged_labels};

        auto lambda = [=,&loop_nest](const IndexVector& blockid) {
            auto ltensor = lhs_.tensor();
            auto rtensor = rhs_.tensor();
            IndexVector lblockid, rblockid;
            IndexLabelVec extracted_llabels, extracted_rlabels;
#if 1
            split_block_id(lblockid, rblockid, lhs_.labels().size(), rhs_.labels().size(), blockid);
            
            std::tie(extracted_llabels, lblockid) =
              internal::extract_blockid_and_label(
                loop_nest.sorted_unique_labels(), lblockid, lhs_.labels());
            
            std::tie(extracted_rlabels, rblockid) =
              internal::extract_blockid_and_label(
                loop_nest.sorted_unique_labels(), rblockid, rhs_.labels());
#endif
#if 0
            const auto translated_lblockid = internal::translate_blockid(lblockid, lhs_);
            const auto translated_rblockid = internal::translate_blockid(rblockid, rhs_);
#else
            EXPECTS(lhs_.labels().size() == extracted_llabels.size());
            EXPECTS(rhs_.labels().size() == extracted_rlabels.size());
            for(size_t i = 0; i < extracted_llabels.size(); i++) {
                EXPECTS(
                  extracted_llabels[i].tiled_index_space().is_compatible_with(
                    lhs_.labels()[i].tiled_index_space()));
            }
            for(size_t i = 0; i < extracted_rlabels.size(); i++) {
                EXPECTS(
                  extracted_rlabels[i].tiled_index_space().is_compatible_with(
                    rhs_.labels()[i].tiled_index_space()));
            }
            for(size_t i = 0; i < extracted_llabels.size(); i++) {
                EXPECTS(
                  extracted_llabels[i].tiled_index_space().is_compatible_with(
                    lhs_.tensor()().labels()[i].tiled_index_space()));
            }
            for(size_t i = 0; i < extracted_rlabels.size(); i++) {
                EXPECTS(
                  extracted_rlabels[i].tiled_index_space().is_compatible_with(
                    rhs_.tensor()().labels()[i].tiled_index_space()));
            }
            IndexVector translated_lblockid, translated_rblockid;
            // bool tlb_valid, trb_valid;
            // std::tie(translated_lblockid, tlb_valid) =
            //   internal::translate_blockid_if_possible(
            //     lblockid, extracted_llabels, lhs_.tensor()().labels());
    
            // std::tie(translated_rblockid, trb_valid) =
            //   internal::translate_blockid_if_possible(
            //     rblockid, extracted_rlabels, rhs_.tensor()().labels());

            IndexVector translated_blockid;
            bool lbl_valid; 
            
            IndexVector full_blk_id{lblockid};
            full_blk_id.insert(full_blk_id.end(), rblockid.begin(), rblockid.end());

            IndexLabelVec extracted_lbls{extracted_llabels};
            extracted_lbls.insert(extracted_lbls.end(), extracted_rlabels.begin(), extracted_rlabels.end());

            auto lt_lbls = lhs_.tensor()().labels();
            auto rt_lbls = rhs_.tensor()().labels();
            IndexLabelVec tensor_lbls{lt_lbls};
            tensor_lbls.insert(tensor_lbls.end(), rt_lbls.begin(), rt_lbls.end());


             std::tie(translated_blockid, lbl_valid) =
              internal::translate_blockid_if_possible(
                full_blk_id, extracted_lbls, tensor_lbls);

             split_block_id(translated_lblockid, translated_rblockid,
                            lhs_.labels().size(), rhs_.labels().size(),
                            translated_blockid);

#endif

            for(const auto id : translated_lblockid) {
                if (id == -1) return;
            }
            for(const auto id : translated_rblockid) {
                if (id == -1) return;
            }

            // std::cerr<<"Executing AddOp for lhs="<<translated_lblockid<<" rhs="<<translated_rblockid<<"\n";
            // Check if lhs is non-zero
            if(!ltensor.is_non_zero(translated_lblockid) ||
               !rtensor.is_non_zero(translated_rblockid)) {
                return;
            }

        using TensorElType1 = typename LabeledTensorT1::element_type;
        using TensorElType2 = typename LabeledTensorT2::element_type;
        if constexpr(std::is_same_v<TensorElType1,TensorElType2>){
            // std::cerr<<"Executing NONZERO AddOp for lhs="<<translated_lblockid<<" rhs="<<translated_rblockid<<"\n";
            if(is_assign_) {

                ec.re()->submitTask([=](RuntimeEngine::RuntimeContext rc) {
                        BlockBuffer lbf = rc.get_buf_tmp(ltensor, translated_lblockid);
                        BlockBuffer rbf = rc.get_buf_read(rtensor, translated_rblockid);

                        SizeVec ldims_sz, rdims_sz;
                        for(const auto v : lbf.block_dims()) { ldims_sz.push_back(v); }
                        for(const auto v : rbf.block_dims()) { rdims_sz.push_back(v); }
                        kernels::assign(lbf.data(), ldims_sz, lhs_int_labels_, alpha_, rbf.data(),
                                rdims_sz, rhs_int_labels_, is_assign_);
                        lbf.release_put();
                        },
                        TempAccess(IndexedTensor{ltensor, translated_lblockid}),
                        WritePermission(IndexedTensor{ltensor, translated_lblockid}),
                        ReadAccess(IndexedTensor{rtensor, translated_rblockid}));
            } else {
                ec.re()->submitTask([=](RuntimeEngine::RuntimeContext rc) {
                        BlockBuffer lbf = rc.get_buf_tmp(ltensor, translated_lblockid, 0);
                        BlockBuffer rbf = rc.get_buf_read(rtensor, translated_rblockid);

                        SizeVec ldims_sz, rdims_sz;
                        for(const auto v : lbf.block_dims()) { ldims_sz.push_back(v); }
                        for(const auto v : rbf.block_dims()) { rdims_sz.push_back(v); }
                        kernels::assign(lbf.data(), ldims_sz, lhs_int_labels_, alpha_, rbf.data(),
                                rdims_sz, rhs_int_labels_, is_assign_);
                        lbf.release_add();
                        },
                        TempAccess(IndexedTensor{ltensor, translated_lblockid}),
                        AccumPermission(IndexedTensor{ltensor, translated_lblockid}),
                        ReadAccess(IndexedTensor{rtensor, translated_rblockid}));
            }
        }
        #ifdef USE_BLIS
        else {
            const size_t l_size = ltensor.block_size(translated_lblockid);
            const size_t r_size = rtensor.block_size(translated_rblockid);
            std::vector<TensorElType1> rbuf(l_size); //C=R, make rbuf complex
            std::vector<TensorElType1> lbuf(l_size);

            if constexpr(!std::is_same_v<TensorElType1,TensorElType2>){
                //TODO: this should happen only if lhs = complex
                if constexpr(internal::is_complex_v<TensorElType1>){
                    std::vector<TensorElType2> rbuf_real(r_size);
                    rtensor.get(translated_rblockid, rbuf_real);
                    TensorElType2* rbuf_ptr = reinterpret_cast<TensorElType2*>(&rbuf[0]);
                    if constexpr(std::is_same_v<TensorElType2,double>)
                        bli_dcopyv(BLIS_NO_CONJUGATE,r_size,&rbuf_real[0],1,rbuf_ptr,2);
                    else if constexpr(std::is_same_v<TensorElType2,float>)
                        bli_scopyv(BLIS_NO_CONJUGATE,r_size,&rbuf_real[0],1,rbuf_ptr,2);
                }
                //real = complex
                else if constexpr(internal::is_complex_v<TensorElType2>){
                    std::vector<TensorElType2> rbuf_comp(r_size);
                    rtensor.get(translated_rblockid, rbuf_comp);
                    TensorElType1* rbuf_ptr = reinterpret_cast<TensorElType1*>(&rbuf_comp[0]);
                    if constexpr(std::is_same_v<TensorElType1,double>)
                        bli_dcopyv(BLIS_NO_CONJUGATE,r_size,rbuf_ptr+1,2,&rbuf[0],1);
                    else if constexpr(std::is_same_v<TensorElType1,float>)
                        bli_scopyv(BLIS_NO_CONJUGATE,r_size,rbuf_ptr+1,2,&rbuf[0],1);                    
                }

            }
            else rtensor.get(translated_rblockid, rbuf);
            
            const auto& ldims = lhs_.tensor().block_dims(translated_lblockid);
            const auto& rdims = rhs_.tensor().block_dims(translated_rblockid);

            SizeVec ldims_sz, rdims_sz;
            for(const auto v : ldims) { ldims_sz.push_back(v); }
            for(const auto v : rdims) { rdims_sz.push_back(v); }
            kernels::assign<TensorElType1>(&lbuf[0], ldims_sz, lhs_int_labels_, alpha_,
                            &rbuf[0], rdims_sz, rhs_int_labels_, is_assign_);
            if(is_assign_) {
                ltensor.put(translated_lblockid, lbuf);
            } else {
                ltensor.add(translated_lblockid, lbuf);
            }
        }
        #endif

        };

        //@todo use a scheduler
        do_work(ec, loop_nest, lambda);
    #endif
        
    }

    TensorBase* writes() const {
        if(is_assign()) {
            return lhs_.base_ptr();
        } else {
            return nullptr;
        }
    }

    TensorBase* accumulates() const {
        if(!is_assign()) {
            return lhs_.base_ptr();
        } else {
            return nullptr;
        }
    }

    std::vector<TensorBase*> reads() const {
        std::vector<TensorBase*> res; 
        res.push_back(rhs_.base_ptr());

        return res;
    }

    bool is_memory_barrier() const {
        return false;
    }

protected:
    void fillin_labels() {
        using internal::fillin_tensor_label_from_map;
        using internal::update_fillin_map;
        // every string in RHS is also in LHS. So number only LHS strings
        std::map<std::string, Label> str_to_labels;
        update_fillin_map(str_to_labels, lhs_.str_map(), lhs_.str_labels(), 0);
        fillin_tensor_label_from_map(lhs_, str_to_labels);
        fillin_tensor_label_from_map(rhs_, str_to_labels);
    }

    void split_block_id(IndexVector& lhs_lbls, IndexVector& rhs_lbls,
                        size_t lhs_size, size_t rhs_size,
                        const IndexVector& full_blockid) const {
        IndexVector new_lhs, new_rhs;

        new_lhs.insert(new_lhs.end(), full_blockid.begin(),
                       full_blockid.begin() + lhs_size);
        new_rhs.insert(new_rhs.end(), full_blockid.begin() + lhs_size,
                       full_blockid.end());

        lhs_lbls = new_lhs;
        rhs_lbls = new_rhs;
    }

    /**
     * @brief Check if the parameters forma valid add operation. The parameters
     * (ltc, tuple(alpha,lta)) form a valid add operation if:
     *
     * 1. Every label depended on by another label (i.e., all 'd' such that
     * there exists label 'l(d)') is bound at least once
     *
     * 2. There are no conflicting dependent label specifications. That if
     * 'a(i)' is a label in either lta or ltc, there is no label 'a(j)' (i!=j)
     * in either lta or ltc.
     *
     * @tparam LabeledTensorType Type RHS labeled tensor
     * @tparam T Type of scaling factor (alpha)
     * @param ltc LHS tensor being added to
     * @param rhs RHS (scaling factor and labeled tensor)
     *
     * @pre ltc.validate() has been invoked
     * @pre lta.validate() has been invoked
     */
    void validate() {
        EXPECTS_STR((lhs_.tensor().base_ptr()!= rhs_.tensor().base_ptr()), 
                      "Self assignment is not supported in tensor operations!");

        IndexLabelVec ilv{lhs_.labels()};
        ilv.insert(ilv.end(), rhs_.labels().begin(), rhs_.labels().end());

        for(size_t i = 0; i < ilv.size(); i++) {
            for(const auto& dl : ilv[i].secondary_labels()) {
                size_t j;
                for(j = 0; j < ilv.size(); j++) {
                    if(dl.tiled_index_space() == ilv[j].tiled_index_space() &&
                       dl.label() == ilv[j].label()) {
                        break;
                    }
                }
                EXPECTS(j < ilv.size());
            }
        }

        for(size_t i = 0; i < ilv.size(); i++) {
            const auto& ilbl = ilv[i];
            for(size_t j = i + 1; j < ilv.size(); j++) {
                const auto& jlbl = ilv[j];
                if(ilbl.tiled_index_space() == jlbl.tiled_index_space() &&
                   ilbl.label() == jlbl.label()) {
                    EXPECTS(ilbl == jlbl);
                }
            }
        }
    }

    void fillin_int_labels() {
        std::map<TileLabelElement, int> primary_labels_map;
        int cnt = -1;
        for(const auto& lbl : lhs_.labels()) {
            primary_labels_map[lbl.primary_label()] = --cnt;
        }
        for(const auto& lbl : rhs_.labels()) {
            primary_labels_map[lbl.primary_label()] = --cnt;
        }
        for(const auto& lbl : lhs_.labels()) {
            lhs_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
        }
        for(const auto& lbl : rhs_.labels()) {
            rhs_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
        }
    }

    LabeledTensorT1 lhs_;
    T alpha_;
    LabeledTensorT2 rhs_;
    IntLabelVec lhs_int_labels_, rhs_int_labels_;
    bool is_assign_;
}; // class AddOp

 #ifdef USE_TALSH
template<typename T>
struct AddBuf {
    //AddBuf() = default;
    AddBuf(bool isgpu, talsh_task_t* tt, tensor_handle* tc, tensor_handle*ta, tensor_handle* tb, Tensor<T> tensor, 
        std::vector<T>&& cbuf, const IndexVector& blockid)
    : tensor_{tensor}, blockid_{blockid}, cbuf_{cbuf}, tt_{tt}, tc_{tc}, ta_{ta}, tb_{tb}, isgpu_{isgpu} {
        // tensor.nb_add(blockid, buf_, &nbhdl_);
        //tensor.add(blockid, buf_);
    }
    ~AddBuf() {
        assert(nbhdl_.getCompletionStatus() == true);
    }
    bool is_done() {
        return true;
        // return nbhdl_.getCompletionStatus();
    }
    void wait() {
        if(!nbhdl_.getCompletionStatus()) {
            nbhdl_.waitForCompletion();
        }
    }

    std::vector<T> cbuf_;
    IndexVector blockid_;
    bool isgpu_;
    Tensor<T> tensor_;
    talsh_task_t* tt_;
    tensor_handle* tc_;
    tensor_handle* ta_;
    tensor_handle* tb_;
    DataCommunicationHandle nbhdl_;
};
#else
template<typename T>
struct AddBuf {
    //AddBuf() = default;
    AddBuf(bool isgpu, Tensor<T> tensor, 
        std::vector<T>&& cbuf, const IndexVector& blockid)
    : tensor_{tensor}, blockid_{blockid}, cbuf_{cbuf}, isgpu_{isgpu} {
    }
    ~AddBuf() {
        assert(nbhdl_.getCompletionStatus() == true);
    }
    bool is_done() {
        return true;
    }
    void wait() {
        if(!nbhdl_.getCompletionStatus()) {
            nbhdl_.waitForCompletion();
        }
    }

    std::vector<T> cbuf_;
    IndexVector blockid_;
    bool isgpu_;
    Tensor<T> tensor_;
    DataCommunicationHandle nbhdl_;
};
#endif

template<typename T, typename LabeledTensorT1, typename LabeledTensorT2, typename LabeledTensorT3>
class MultOp : public Op {
public:
    MultOp() = default;
    MultOp(LabeledTensorT1 lhs, T alpha, LabeledTensorT2 rhs1,
           LabeledTensorT3 rhs2, bool is_assign) :
      lhs_{lhs},
      alpha_{alpha},
      rhs1_{rhs1},
      rhs2_{rhs2},
      is_assign_{is_assign} {
        EXPECTS(lhs.has_str_lbl() == rhs1.has_str_lbl()
                && rhs1.has_str_lbl() == rhs2.has_str_lbl());
        if(!lhs.has_str_lbl() && !lhs.labels().empty()) {
            auto lhs_lbls  = lhs.labels();
            auto rhs1_lbls = rhs1.labels();
            auto rhs2_lbls = rhs2.labels();

            auto labels{lhs_lbls};
            labels.insert(labels.end(), rhs1_lbls.begin(), rhs1_lbls.end());
            labels.insert(labels.end(), rhs2_lbls.begin(), rhs2_lbls.end());

            internal::update_labels(labels);

            lhs_lbls  = IndexLabelVec(labels.begin(),
                                     labels.begin() + lhs.labels().size());
            rhs1_lbls = IndexLabelVec(labels.begin() + lhs.labels().size(),
                                      labels.begin() + lhs.labels().size() +
                                        rhs1.labels().size());
            rhs2_lbls = IndexLabelVec(
              labels.begin() + lhs.labels().size() + rhs1.labels().size(),
              labels.begin() + lhs.labels().size() + rhs1.labels().size() +
                rhs2.labels().size());
            lhs_.set_labels(lhs_lbls);
            rhs1_.set_labels(rhs1_lbls);
            rhs2_.set_labels(rhs2_lbls);
        }

        if(lhs.has_str_lbl()){
            fillin_labels();
        }

        fillin_int_labels();
        validate();
    }

    MultOp(const MultOp<T, LabeledTensorT1, LabeledTensorT2, LabeledTensorT3>&) = default;

    LabeledTensorT1 lhs() const { return lhs_; }

    T alpha() const { return alpha_; }

    LabeledTensorT2 rhs1() const { return rhs1_; }

    LabeledTensorT3 rhs2() const { return rhs2_; }

    bool is_assign() const { return is_assign_; }

    OpList canonicalize() const override {
        OpList result{};

        if(is_assign_) {
            auto lhs{lhs_};
            auto assign_op = (lhs = 0);
            result.push_back(assign_op.clone());
            MultOp n_op{lhs_, alpha_, rhs1_, rhs2_, false};
            result.push_back(n_op.clone());
        } else {
            result.push_back((*this).clone());
        }

        return result;
    }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new MultOp{*this});
    }

    void execute(ExecutionContext& ec, ExecutionHW hw = ExecutionHW::CPU) override {
        EXPECTS(!is_assign_);
        #if 1
        using TensorElType = typename LabeledTensorT1::element_type;
        // determine set of all labels
        IndexLabelVec all_labels{lhs_.labels()};
        all_labels.insert(all_labels.end(), rhs1_.labels().begin(),
                          rhs1_.labels().end());
        all_labels.insert(all_labels.end(), rhs2_.labels().begin(),
                          rhs2_.labels().end());
        LabelLoopNest loop_nest{all_labels};

        std::vector<AddBuf<T>*> add_bufs;

        // function to compute one block
        auto lambda = [=,&add_bufs,&loop_nest](const IndexVector itval) {
            auto ctensor = lhs_.tensor();
            auto atensor = rhs1_.tensor();
            auto btensor = rhs2_.tensor();
            // compute blockids from the loop indices. itval is the loop index

#if 1      
            auto it = itval.begin();
            IndexVector citval{it, it + lhs_.labels().size()};
            it += lhs_.labels().size();
            IndexVector aitval{it, it + rhs1_.labels().size()};
            it += rhs1_.labels().size();
            IndexVector bitval{it, it + rhs2_.labels().size()};

            auto [extracted_clabels, cblockid] = internal::extract_blockid_and_label(
                loop_nest.sorted_unique_labels(), citval, lhs_.labels());

            auto [extracted_alabels, ablockid] = internal::extract_blockid_and_label(
                loop_nest.sorted_unique_labels(), aitval, rhs1_.labels());
            
            auto [extracted_blabels, bblockid] = internal::extract_blockid_and_label(
                loop_nest.sorted_unique_labels(), bitval, rhs2_.labels());
            
            EXPECTS(lhs_.labels().size() == extracted_clabels.size());
            EXPECTS(rhs1_.labels().size() == extracted_alabels.size());
            EXPECTS(rhs2_.labels().size() == extracted_blabels.size());

            for(size_t i = 0; i < extracted_clabels.size(); i++) {
                EXPECTS(
                    extracted_clabels[i].tiled_index_space().is_compatible_with(
                    lhs_.labels()[i].tiled_index_space()));
            }

            for(size_t i = 0; i < extracted_alabels.size(); i++) {
                EXPECTS(
                    extracted_alabels[i].tiled_index_space().is_compatible_with(
                    rhs1_.labels()[i].tiled_index_space()));
            }

            for(size_t i = 0; i < extracted_blabels.size(); i++) {
                EXPECTS(
                    extracted_blabels[i].tiled_index_space().is_compatible_with(
                    rhs2_.labels()[i].tiled_index_space()));
            }

            for(size_t i = 0; i < extracted_clabels.size(); i++) {
                EXPECTS(
                    extracted_clabels[i].tiled_index_space().is_compatible_with(
                    lhs_.tensor()().labels()[i].tiled_index_space()));
            }

            for(size_t i = 0; i < extracted_alabels.size(); i++) {
                EXPECTS(
                    extracted_alabels[i].tiled_index_space().is_compatible_with(
                    rhs1_.tensor()().labels()[i].tiled_index_space()));
            }

            for(size_t i = 0; i < extracted_blabels.size(); i++) {
                EXPECTS(
                    extracted_blabels[i].tiled_index_space().is_compatible_with(
                    rhs2_.tensor()().labels()[i].tiled_index_space()));
            }

            IndexVector blockid{cblockid};
            blockid.insert(blockid.end(), ablockid.begin(), ablockid.end());
            blockid.insert(blockid.end(), bblockid.begin(), bblockid.end());

            IndexLabelVec extracted_lbls{extracted_clabels};
            extracted_lbls.insert(extracted_lbls.end(), extracted_alabels.begin(), extracted_alabels.end());
            extracted_lbls.insert(extracted_lbls.end(), extracted_blabels.begin(), extracted_blabels.end());

            auto tc_lbls = lhs_.tensor()().labels();
            auto ta_lbls = rhs1_.tensor()().labels();
            auto tb_lbls = rhs2_.tensor()().labels();

            IndexLabelVec tensor_lbls{tc_lbls};
            tensor_lbls.insert(tensor_lbls.end(), ta_lbls.begin(), ta_lbls.end());
            tensor_lbls.insert(tensor_lbls.end(), tb_lbls.begin(), tb_lbls.end());
            
            IndexVector translated_blockid;
            bool tb_valid;

            std::tie(translated_blockid, tb_valid) =
              internal::translate_blockid_if_possible(
                blockid, extracted_lbls, tensor_lbls);

            auto id_it = translated_blockid.begin();
            IndexVector translated_cblockid{id_it, id_it + lhs_.labels().size()};
            id_it += lhs_.labels().size();
            IndexVector translated_ablockid{id_it, id_it + rhs1_.labels().size()};
            id_it += rhs1_.labels().size();
            IndexVector translated_bblockid{id_it, id_it + rhs2_.labels().size()};
            
            for(const auto id : translated_cblockid) {
                if (id == -1) return;
            }
            for(const auto id : translated_ablockid) {
                if (id == -1) return;
            }
            for(const auto id : translated_bblockid) {
                if (id == -1) return;
            }
        
#else
            const auto translated_cblockid = internal::translate_blockid(cblockid, lhs_);
            const auto translated_ablockid = internal::translate_blockid(ablockid, rhs1_);
            const auto translated_bblockid = internal::translate_blockid(bblockid, rhs2_);

#endif
            if( !ctensor.is_non_zero(translated_cblockid) || 
                !atensor.is_non_zero(translated_ablockid) ||
                !btensor.is_non_zero(translated_bblockid)) 
                return;

            using TensorElType1 = typename LabeledTensorT1::element_type;
            using TensorElType2 = typename LabeledTensorT2::element_type;
            using TensorElType3 = typename LabeledTensorT3::element_type;

#if 0
            if constexpr(std::is_same_v<TensorElType1,TensorElType2> 
                         && std::is_same_v<TensorElType1,TensorElType3>) {
                ec.re()->submitTask([=](RuntimeEngine::RuntimeContext rc){
                        BlockBuffer cbuf = rc.get_buf_tmp(ctensor, translated_cblockid);
                        BlockBuffer abuf = rc.get_buf_read(atensor, translated_ablockid);
                        BlockBuffer bbuf = rc.get_buf_read(btensor, translated_bblockid);
                        // double cscale = is_assign_ ? 0 : 1;
                        TensorElType cscale{0};

                        SizeVec adims_sz, bdims_sz, cdims_sz;
                        for(const auto v : abuf.block_dims()) { adims_sz.push_back(v); }
                        for(const auto v : bbuf.block_dims()) { bdims_sz.push_back(v); }
                        for(const auto v : cbuf.block_dims()) { cdims_sz.push_back(v); }
                        kernels::block_multiply(ec.pg().rank().value(),alpha_, abuf.data(), adims_sz,
                                rhs1_int_labels_, bbuf.data(), bdims_sz,
                                rhs2_int_labels_, cscale, cbuf.data(),
                                cdims_sz, lhs_int_labels_, hw, ec.has_gpu());

                        // add the computed update to the tensor
                        cbuf.release_add();

                        }, TempAccess{IndexedTensor{ctensor, translated_cblockid}},
                        AccumPermission{IndexedTensor{ctensor, translated_cblockid}}, 
                        ReadAccess{IndexedTensor{atensor, translated_ablockid}}, 
                        ReadAccess{IndexedTensor{btensor, translated_bblockid}}); 
            }
            else
#endif            
            {
                TimerGuard tg_total{&multOpTime};
                const int dev_id = ec.gpu_devid();
                // determine set of all labels

                // compute block size and allocate buffers
                const size_t csize = ctensor.block_size(translated_cblockid);
                const size_t asize = atensor.block_size(translated_ablockid);
                const size_t bsize = btensor.block_size(translated_bblockid);

                std::vector<TensorElType1> cbuf(csize, 0);
                std::vector<TensorElType2> abuf(asize);
                std::vector<TensorElType3> bbuf(bsize);
                // get inputs
#if DO_NB
                DataCommunicationHandle a_nbhandle,b_nbhandle,c_nbhandle;

            {
                TimerGuard tg_get{&multOpGetTime};
                atensor.nb_get(translated_ablockid, abuf,&a_nbhandle);
                btensor.nb_get(translated_bblockid, bbuf,&b_nbhandle);
            }
            {
                TimerGuard tg_wait{&multOpWaitTime};
                if(!a_nbhandle.getCompletionStatus()) a_nbhandle.waitForCompletion();
                if(!b_nbhandle.getCompletionStatus()) b_nbhandle.waitForCompletion();
            }
#else
                { 
                    TimerGuard tg_get{&multOpGetTime};
                    atensor.get(translated_ablockid, abuf);
                }
                { 
                    TimerGuard tg_get{&multOpGetTime};
                    btensor.get(translated_bblockid, bbuf);
                }
#endif
                const auto& cdims = ctensor.block_dims(translated_cblockid);
                const auto& adims = atensor.block_dims(translated_ablockid);
                const auto& bdims = btensor.block_dims(translated_bblockid);
                // double cscale = is_assign_ ? 0 : 1;
                T cscale{0};

                SizeVec adims_sz, bdims_sz, cdims_sz;
                for(const auto v : adims) { adims_sz.push_back(v); }
                for(const auto v : bdims) { bdims_sz.push_back(v); }
                for(const auto v : cdims) { cdims_sz.push_back(v); }
                #ifdef USE_TALSH
                talsh_task_t* talsh_task = new talsh_task_t();
                TALSH *gpu_mult = new TALSH{ec.num_gpu()};

                tensor_handle *th_a = new tensor_handle();
                tensor_handle *th_b = new tensor_handle();
                tensor_handle *th_c = new tensor_handle(); 
                talshTaskClean(talsh_task);
                #endif
                bool isgpu = false;

                {
                    #ifdef USE_TALSH
                    AddBuf<TensorElType1> *ab = new AddBuf<TensorElType1>{isgpu, talsh_task, th_c, th_a, th_b, 
                        ctensor, std::move(cbuf),translated_cblockid};
                        #else
                    AddBuf<TensorElType1> *ab = new AddBuf<TensorElType1>{isgpu, 
                        ctensor, std::move(cbuf),translated_cblockid};                        
                    #endif
                    add_bufs.push_back(ab);

                    TimerGuard tg_dgemm{&multOpDgemmTime};                    
                    kernels::block_multiply<T,TensorElType1,TensorElType2,TensorElType3>
                                        (ab->isgpu_, 
                                        #ifdef USE_TALSH
                                        *gpu_mult, *talsh_task, *th_c, *th_a, *th_b, 
                                        #endif
                                        dev_id, alpha_, 
                                        abuf.data(), adims_sz,
                                        rhs1_int_labels_, bbuf.data(), bdims_sz,
                                        rhs2_int_labels_, cscale, (ab->cbuf_).data(),
                                        cdims_sz, lhs_int_labels_, hw, ec.has_gpu());

                    #ifndef DO_NB
                    // add the computed update to the tensor
                    ctensor.add(translated_cblockid, ab->cbuf_);
                    add_bufs.clear();                   
                    #endif
                }

                // add the computed update to the tensor
                // { 
                //     TimerGuard tg_add{&multOpAddTime};
                //     //ctensor.add(translated_cblockid, cbuf);
                //     const int k = 20;
                //     if(add_bufs.size() >= k) {
                //         for(auto& ab: add_bufs) {
                //             #ifdef USE_TALSH
                //             {
                //                 if(ab->isgpu_) {
                //                     int done = NOPE;
                //                     int sts, errc = TALSH_SUCCESS;
                //                     while(done != YEP && errc == TALSH_SUCCESS) {
                //                     done=talshTaskComplete(ab->tt_, &sts, &errc);
                //                     }
                //                     assert(errc == TALSH_SUCCESS);
                //                     errc = talshTaskDestruct(ab->tt_);
                //                     assert(errc == TALSH_SUCCESS);
                //                     talshTensorDestruct(ab->ta_);
                //                     talshTensorDestruct(ab->tb_);
                //                     talshTensorDestruct(ab->tc_);  
                //                 }                             
                //             }
                //             #endif
                //         }
                //          for(auto& ab: add_bufs) {
                //             (ab->tensor_).nb_add(ab->blockid_, ab->cbuf_, &(ab->nbhdl_));
                //             ab->wait();
                //             delete ab;
                //          }

                //         add_bufs.clear();
                //     }
                // }
            }

            };
            //@todo use a scheduler
            //@todo make parallel
            do_work(ec, loop_nest, lambda);

            #ifdef DO_NB
                { 
                    TimerGuard tg_add{&multOpAddTime};
                    for(auto& ab: add_bufs) {
                            #ifdef USE_TALSH
                            {
                                if(ab->isgpu_) {
                                    int done = NOPE;
                                    int sts, errc = TALSH_SUCCESS;
                                    while(done != YEP && errc == TALSH_SUCCESS) {
                                    done=talshTaskComplete(ab->tt_, &sts, &errc);
                                    }
                                    assert(errc == TALSH_SUCCESS);
                                    errc = talshTaskDestruct(ab->tt_);
                                    assert(errc == TALSH_SUCCESS);
                                    talshTensorDestruct(ab->ta_);
                                    talshTensorDestruct(ab->tb_);
                                    talshTensorDestruct(ab->tc_);  
                                }                             
                            }
                            #endif
                    }
                     for(auto& ab: add_bufs) {
                            (ab->tensor_).nb_add(ab->blockid_, ab->cbuf_, &(ab->nbhdl_));
                            ab->wait();
                            delete ab;                            
                    }
                    add_bufs.clear();
                }
            #endif

#endif

    }

    TensorBase* writes() const {
        if(is_assign()) {
            return lhs_.base_ptr();
        } else {
            return nullptr;
        }
    }

    TensorBase* accumulates() const {
        if(!is_assign()) {
            return lhs_.base_ptr();
        } else {
            return nullptr;
        }
    }

    std::vector<TensorBase*> reads() const {
        std::vector<TensorBase*> res; 
        res.push_back(rhs1_.base_ptr());
        res.push_back(rhs2_.base_ptr());

        return res;
    }

    bool is_memory_barrier() const {
        return false;
    }

protected:
    void fillin_labels() {
        using internal::fillin_tensor_label_from_map;
        using internal::update_fillin_map;
        std::map<std::string, Label> str_to_labels;
        const size_t lsize  = lhs_.labels().size();
        const size_t r1size = rhs1_.labels().size();
        const size_t r2size = rhs2_.labels().size();

        update_fillin_map(str_to_labels, lhs_.str_map(), lhs_.str_labels(), 0);
        update_fillin_map(str_to_labels, rhs1_.str_map(), rhs1_.str_labels(),
                          lsize);
        update_fillin_map(str_to_labels, rhs2_.str_map(), rhs2_.str_labels(),
                          lsize + r1size);
        fillin_tensor_label_from_map(lhs_, str_to_labels);
        fillin_tensor_label_from_map(rhs1_, str_to_labels);
        fillin_tensor_label_from_map(rhs2_, str_to_labels);
    }

    void fillin_int_labels() {
        std::map<TileLabelElement, int> primary_labels_map;
        int cnt = -1;
        for(const auto& lbl : lhs_.labels()) {
            primary_labels_map[lbl.primary_label()] = --cnt;
        }
        for(const auto& lbl : rhs1_.labels()) {
            primary_labels_map[lbl.primary_label()] = --cnt;
        }
        for(const auto& lbl : rhs2_.labels()) {
            primary_labels_map[lbl.primary_label()] = --cnt;
        }
        for(const auto& lbl : lhs_.labels()) {
            lhs_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
        }
        for(const auto& lbl : rhs1_.labels()) {
            rhs1_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
        }
        for(const auto& lbl : rhs2_.labels()) {
            rhs2_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
        }
    }

    /**
     * @brief Check if the parameters forma valid add operation. The parameters
     * (ltc, tuple(alpha,lta)) form a valid add operation if:
     *
     * 1. Every label depended on by another label (i.e., all 'd' such that
     * there exists label 'l(d)') is bound at least once
     *
     * 2. There are no conflicting dependent label specifications. That if
     * 'a(i)' is a label in either lta or ltc, there is no label 'a(j)' (i!=j)
     * in either lta or ltc.
     *
     * @pre lhs_.validate(), rhs1_.validate() and rhs2_.validate() have been
     *  invoked
     */
    void validate() {
        EXPECTS_STR((lhs_.tensor().base_ptr()!= rhs1_.tensor().base_ptr() &&
                       lhs_.tensor().base_ptr()!= rhs2_.tensor().base_ptr()), 
                     "Self assignment is not supported in tensor operations!");

        IndexLabelVec ilv{lhs_.labels()};
        ilv.insert(ilv.end(), rhs1_.labels().begin(), rhs1_.labels().end());
        ilv.insert(ilv.end(), rhs2_.labels().begin(), rhs2_.labels().end());

        for(size_t i = 0; i < ilv.size(); i++) {
            for(const auto& dl : ilv[i].secondary_labels()) {
                size_t j;
                for(j = 0; j < ilv.size(); j++) {
                    if(dl.tiled_index_space() == ilv[j].tiled_index_space() &&
                       dl.label() == ilv[j].label()) {
                        break;
                    }
                }
                EXPECTS(j < ilv.size());
            }
        }

        for(size_t i = 0; i < ilv.size(); i++) {
            const auto& ilbl = ilv[i];
            for(size_t j = i + 1; j < ilv.size(); j++) {
                const auto& jlbl = ilv[j];
                if(ilbl.tiled_index_space() == jlbl.tiled_index_space() &&
                   ilbl.label() == jlbl.label()) {
                    EXPECTS(ilbl == jlbl);
                }
            }
        }
    }

    LabeledTensorT1 lhs_;
    T alpha_;
    LabeledTensorT2 rhs1_;
    LabeledTensorT3 rhs2_;
    IntLabelVec lhs_int_labels_;
    IntLabelVec rhs1_int_labels_;
    IntLabelVec rhs2_int_labels_;
    bool is_assign_;
}; // class MultOp

template<typename TensorType>
class AllocOp : public Op {
public:
    AllocOp(TensorType tensor, ExecutionContext& ec) :
      tensor_{tensor},
      ec_{ec} {}

    AllocOp(const AllocOp<TensorType>&) = default;

    TensorType tensor() const { return tensor_; }

    OpList canonicalize() const override { return OpList{(*this)}; }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new AllocOp{*this});
    }

    void execute(ExecutionContext& ec, ExecutionHW hw = ExecutionHW::CPU) override { tensor_.allocate(&ec_); }

    TensorBase* writes() const {
        return tensor_.base_ptr();
    }

    TensorBase* accumulates() const {
        return nullptr;
    }

    std::vector<TensorBase*> reads() const {
        return {};
    }

    bool is_memory_barrier() const {
        return false;
    }

protected:
    TensorType tensor_;
    ExecutionContext& ec_;
}; // class AllocOp

template<typename TensorType>
class DeallocOp : public Op {
public:
    DeallocOp(TensorType tensor) : tensor_{tensor} {}

    DeallocOp(const DeallocOp<TensorType>&) = default;

    TensorType tensor() const { return tensor_; }

    OpList canonicalize() const override { return OpList{(*this)}; }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new DeallocOp{*this});
    }

    void execute(ExecutionContext& ec, ExecutionHW hw = ExecutionHW::CPU) override { tensor_.deallocate(); }

    TensorBase* writes() const {
        return tensor_.base_ptr();
    }

    std::vector<TensorBase*> reads() const {
        return {};
    }

    TensorBase* accumulates() const {
        return {};
    }

    bool is_memory_barrier() const {
        return false;
    }

protected:
    TensorType tensor_;
}; // class AllocOp

} // namespace tamm

#endif // TAMM_OPS_HPP_

