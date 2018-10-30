#ifndef TAMM_OPS_HPP_
#define TAMM_OPS_HPP_

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/kernels/assign.hpp"
#include "tamm/kernels/multiply.hpp"
#include "tamm/tensor.hpp"
#include "tamm/types.hpp"
#include "tamm/utils.hpp"
#include "tamm/work.hpp"

namespace tamm {

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
    // //auto inv_perm = perm_invert(perm);
    // auto inv_sizes = perm_apply(ddims, inv_perm);
    // TensorVec<size_t> sizes;
    // TensorVec<int> iperm;
    // for(unsigned i=0; i<ddims.size(); i++) {
    //   sizes.push_back(inv_sizes[i].value());
    //   iperm.push_back(perm[i]+1);
    // }
    // index_sort(sbuf, dbuf,
    //            sizes.size(), &sizes[0], &iperm[0], scale);
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
    virtual std::shared_ptr<Op> clone() const = 0;
    virtual void execute(ExecutionContext& ec)     = 0;
    virtual OpList canonicalize() const       = 0;
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
        fillin_labels();
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

    void execute(ExecutionContext& ec) override {
        using TensorElType = typename LabeledTensorT::element_type;
        LabelLoopNest loop_nest{lhs_.labels()};

//previous implementation
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
// New task-based implementation
//TODO: parameter list would be buffer pointer and other parameters
// Buffer pointer ---> tensor(translated_blockid)
       auto cpulambda = [&](const IndexVector& translated_blockid, RuntimeSummary& rs) {
            auto tensor = lhs_.tensor();

            const size_t size = tensor.block_size(translated_blockid);
            auto buf = static_cast<TensorElType*> (rs.get_buffer(size* sizeof(TensorElType)));
            //TODO: something similar to the following line
            //std::vector<TensorElType> buf(size, static_cast<TensorElType>(alpha()));
            //std::fill(std::begin(buf), std::end(buf), alpha());
            std::fill(buf, buf+size, alpha());

            if(is_assign_) {
                //copy buf to tensor buffer
                tensor.put(translated_blockid, buf);
            } else {
                // add buf to tensor buffer
                tensor.add(translated_blockid, buf);
            }
            rs.release(buf);
        };
        auto gpulambda = nullptr;
        FunctionDefinition function_definition(cpulambda, gpulambda);

        auto tensor = lhs_.tensor();

        auto first = loop_nest.begin();
        auto last = loop_nst.end();

        TaskEngine te;
        for(; first!= last; ++first) {
            auto blockid =  *first;
            EXPECTS(blockid.size() == lhs_.labels().size());
            EXPECTS(blockid.size() == tensor.num_modes());
            const auto& translated_blockid =
              internal::translate_blockid(blockid, lhs_);
            te.submitTask(function_definition, {WRITE(tensor(translated_blockid))}, alpha());
        }
        RuntimeEngine& runtime_engine = ec.re();
        runtime_engine.executeAllThreads(te);
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

    void execute(ExecutionContext& ec) override {
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

    void execute(ExecutionContext& ec) override {
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

template<typename T, typename LabeledTensorT>
class AddOp : public Op {
public:
    AddOp() = default;
    AddOp(LabeledTensorT lhs, T alpha, LabeledTensorT rhs, bool is_assign) :
      lhs_{lhs},
      alpha_{alpha},
      rhs_{rhs},
      is_assign_{is_assign} {
        fillin_labels();
        fillin_int_labels();
        validate();
    }

    AddOp(const AddOp<T, LabeledTensorT>&) = default;

    T alpha() const { return alpha_; }

    LabeledTensorT lhs() const { return lhs_; }

    LabeledTensorT rhs() const { return rhs_; }

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
        return std::shared_ptr<Op>(new AddOp<T, LabeledTensorT>{*this});
    }

    void execute(ExecutionContext& ec) override {
        using TensorElType = typename LabeledTensorT::element_type;

        IndexLabelVec merged_labels{lhs_.labels()};
        merged_labels.insert(merged_labels.end(), rhs_.labels().begin(),
                             rhs_.labels().end());

        LabelLoopNest loop_nest{merged_labels};

        auto lambda = [&](const IndexVector& blockid) {
            auto ltensor = lhs_.tensor();
            auto rtensor = rhs_.tensor();
            IndexVector lblockid, rblockid;
            split_block_id(lblockid, rblockid, lhs_.labels().size(),
                           rhs_.labels().size(), blockid);
            lblockid = internal::translate_blockid(lblockid, lhs_);
            rblockid = internal::translate_blockid(rblockid, rhs_);

            // Check if lhs is non-zero
            if(!ltensor.is_non_zero(lblockid) ||
               !rtensor.is_non_zero(rblockid)) {
                return;
            }

            const size_t size = ltensor.block_size(lblockid);
            std::vector<TensorElType> rbuf(size);
            std::vector<TensorElType> lbuf(size);
            rtensor.get(rblockid, rbuf);
            const auto& ldims = lhs_.tensor().block_dims(lblockid);
            const auto& rdims = rhs_.tensor().block_dims(rblockid);

            SizeVec ldims_sz, rdims_sz;
            for(const auto v : ldims) { ldims_sz.push_back(v); }
            for(const auto v : rdims) { rdims_sz.push_back(v); }
            kernels::assign(&lbuf[0], ldims_sz, lhs_int_labels_, alpha_,
                            &rbuf[0], rdims_sz, rhs_int_labels_, is_assign_);
            if(is_assign_) {
                ltensor.put(lblockid, lbuf);
            } else {
                ltensor.add(lblockid, lbuf);
            }
        };

        //@todo use a scheduler
        //@todo make parallel
        do_work(ec, loop_nest, lambda);
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

    LabeledTensorT lhs_;
    T alpha_;
    LabeledTensorT rhs_;
    IntLabelVec lhs_int_labels_, rhs_int_labels_;
    bool is_assign_;
}; // class AddOp

template<typename T, typename LabeledTensorT>
class MultOp : public Op {
public:
    MultOp() = default;
    MultOp(LabeledTensorT lhs, T alpha, LabeledTensorT rhs1,
           LabeledTensorT rhs2, bool is_assign) :
      lhs_{lhs},
      alpha_{alpha},
      rhs1_{rhs1},
      rhs2_{rhs2},
      is_assign_{is_assign} {
        fillin_labels();
        fillin_int_labels();
        validate();
        // if(is_assign_) {
        //     NOT_IMPLEMENTED(); // C=A*B not implemented
        // }
    }

    MultOp(const MultOp<T, LabeledTensorT>&) = default;

    LabeledTensorT lhs() const { return lhs_; }

    T alpha() const { return alpha_; }

    LabeledTensorT rhs1() const { return rhs1_; }

    LabeledTensorT rhs2() const { return rhs2_; }

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

    void execute(ExecutionContext& ec) override {
        EXPECTS(!is_assign_);
        using TensorElType = typename LabeledTensorT::element_type;
        // determine set of all labels
        IndexLabelVec all_labels{lhs_.labels()};
        all_labels.insert(all_labels.end(), rhs1_.labels().begin(),
                          rhs1_.labels().end());
        all_labels.insert(all_labels.end(), rhs2_.labels().begin(),
                          rhs2_.labels().end());
        LabelLoopNest loop_nest{all_labels};
        // function to compute one block
        auto lambda = [this](const IndexVector itval) {
            auto ctensor = lhs_.tensor();
            auto atensor = rhs1_.tensor();
            auto btensor = rhs2_.tensor();
            // compute blockids from the loop indices. itval is the loop index
            auto it = itval.begin();
            IndexVector cblockid{it, it + lhs_.labels().size()};
            it += lhs_.labels().size();
            IndexVector ablockid{it, it + rhs1_.labels().size()};
            it += rhs1_.labels().size();
            IndexVector bblockid{it, it + rhs2_.labels().size()};

            cblockid = internal::translate_blockid(cblockid, lhs_);
            ablockid = internal::translate_blockid(ablockid, rhs1_);
            bblockid = internal::translate_blockid(bblockid, rhs2_);

            // Check if lhs is non-zero
            if(!ctensor.is_non_zero(cblockid)) {
                return;
            } else if(!atensor.is_non_zero(ablockid) ||
                      !btensor.is_non_zero(bblockid)) {
                return;
            }

            // compute block size and allocate buffers
            const size_t csize = ctensor.block_size(cblockid);
            const size_t asize = atensor.block_size(ablockid);
            const size_t bsize = btensor.block_size(bblockid);

            std::vector<TensorElType> cbuf(csize, 0);
            std::vector<TensorElType> abuf(asize);
            std::vector<TensorElType> bbuf(bsize);
            // get inputs
            atensor.get(ablockid, abuf);
            btensor.get(bblockid, bbuf);
            const auto& cdims = ctensor.block_dims(cblockid);
            const auto& adims = atensor.block_dims(ablockid);
            const auto& bdims = btensor.block_dims(bblockid);
            // double cscale = is_assign_ ? 0 : 1;
            TensorElType cscale{0};

            SizeVec adims_sz, bdims_sz, cdims_sz;
            for(const auto v : adims) { adims_sz.push_back(v); }
            for(const auto v : bdims) { bdims_sz.push_back(v); }
            for(const auto v : cdims) { cdims_sz.push_back(v); }
            kernels::block_multiply(alpha_, abuf.data(), adims_sz,
                                    rhs1_int_labels_, bbuf.data(), bdims_sz,
                                    rhs2_int_labels_, cscale, cbuf.data(),
                                    cdims_sz, lhs_int_labels_);

            // add the computed update to the tensor
            ctensor.add(cblockid, cbuf);
        };
        //@todo use a scheduler
        //@todo make parallel
        do_work(ec, loop_nest, lambda);
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

    LabeledTensorT lhs_;
    T alpha_;
    LabeledTensorT rhs1_;
    LabeledTensorT rhs2_;
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

    void execute(ExecutionContext& ec) override { tensor_.allocate(&ec_); }

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

    void execute(ExecutionContext& ec) override { tensor_.deallocate(); }

protected:
    TensorType tensor_;
}; // class AllocOp

} // namespace tamm

#endif // TAMM_OPS_HPP_
