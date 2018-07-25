#ifndef TAMM_OPS_HPP_
#define TAMM_OPS_HPP_

#include <memory>
#include <algorithm>

#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/tensor.hpp"
#include "tamm/types.hpp"
#include "tamm/work.hpp"

namespace tamm {

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

inline void update_fillin_map(std::map<std::string, Label>& str_to_labels,
                              const std::vector<bool>& str_map,
                              const std::vector<std::string>& str_labels,
                              int initial_off) {
    const size_t sz = str_labels.size();
    for(size_t i = 0; i < sz; i++) {
        if(str_map[i]) {
            str_to_labels[str_labels[i]] = -initial_off - i - 1;
        }
    }
}

template<typename LabelTensorT>
inline void fillin_tensor_label_from_map(
  LabelTensorT& ltensor, const std::map<std::string, Label>& str_to_labels) {
    IndexLabelVec new_labels = ltensor.labels();
    const size_t sz          = ltensor.labels().size();
    for(size_t i = 0; i < sz; i++) {
        if(ltensor.str_map()[i]) {
          EXPECTS(str_to_labels.find(ltensor.str_labels()[i]) != str_to_labels.end());
            new_labels[i] = ltensor.tensor().tiled_index_spaces()[i].label(
              str_to_labels.find(ltensor.str_labels()[i])->second);
        }
    }
    ltensor.set_labels(new_labels);
}

inline size_t
idx(int n, const size_t *id, const size_t *sz, const PermVector& p) {
  size_t idx = 0;
  for (int i = 0; i < n - 1; i++) {
    idx = (idx + id[p[i]]) * sz[p[i + 1]];
  }
  if (n > 0) {
    idx += id[p[n - 1]];
  }
  // std::cerr<<"idx return = "<<idx<<std::endl;
  return idx;
}

template<typename T>
inline void
index_permute(T* dbuf, const T* sbuf, const PermVector& perm_to_dest, 
              const std::vector<size_t>& ddims, T scale) {
  // static_assert(std::is_same<T1, double>(), "index_permute only works with doubles");
  // static_assert(std::is_convertible<T2, double>(), "index_permute only works with scale convertible to double");
  EXPECTS(dbuf!=nullptr && sbuf!=nullptr);
  EXPECTS(perm_to_dest.size() == ddims.size());

  const size_t ndim = perm_to_dest.size();
  EXPECTS(ddims.size() == ndim);

  if(ndim == 0) {
    dbuf[0] = scale * sbuf[0];
  } else if(ndim == 1) {
    for(size_t i=0; i<ddims[0]; i++) {
      dbuf[i] = scale * sbuf[i];
    }
  } else if(ndim == 2) {
    size_t sz[] = {ddims[0], ddims[1]};
    size_t i[2], c;
    for(c=0, i[0]=0; i[0]<sz[0]; i[0]++) {
      for(i[1]=0; i[1]<sz[1]; i[1]++, c++) {
        dbuf[c] = scale * sbuf[idx(2, i, sz, perm_to_dest)];
      }
    }
  } else if(ndim == 3) {
    size_t sz[] = {ddims[0], ddims[1], ddims[2]};
    size_t i[3], c;
    for(c=0, i[0]=0; i[0]<sz[0]; i[0]++) {
      for(i[1]=0; i[1]<sz[1]; i[1]++) {
        for(i[2]=0; i[2]<sz[1]; i[2]++, c++) {
          dbuf[c] = scale * sbuf[idx(3, i, sz, perm_to_dest)];
        }
      }
    }
  } else if(ndim == 4) {
    size_t sz[] = {ddims[0], ddims[1], ddims[2], ddims[3]};
    size_t i[4], c;
    for(c=0, i[0]=0; i[0]<sz[0]; i[0]++) {
      for(i[1]=0; i[1]<sz[1]; i[1]++) {
        for(i[2]=0; i[2]<sz[2]; i[2]++) {
          for(i[3]=0; i[3]<sz[3]; i[3]++, c++) {
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
inline void
index_permute_acc(T* dbuf, const T* sbuf, const PermVector& perm_to_dest, 
              const std::vector<size_t>& ddims, T scale) {
  // static_assert(std::is_same<T1, double>(), "index_permute only works with doubles");
  // static_assert(std::is_convertible<T2, double>(), "index_permute only works with scale convertible to double");
  EXPECTS(dbuf!=nullptr && sbuf!=nullptr);
  EXPECTS(perm_to_dest.size() == ddims.size());

  const size_t ndim = perm_to_dest.size();
  EXPECTS(ddims.size() == ndim);

  if(ndim == 0) {
    dbuf[0] = scale * sbuf[0];
  } else if(ndim == 1) {
    for(size_t i=0; i<ddims[0]; i++) {
      dbuf[i] += scale * sbuf[i];
    }
  } else if(ndim == 2) {
    size_t sz[] = {ddims[0], ddims[1]};
    size_t i[2], c;
    for(c=0, i[0]=0; i[0]<sz[0]; i[0]++) {
      for(i[1]=0; i[1]<sz[1]; i[1]++, c++) {
        dbuf[c] += scale * sbuf[idx(2, i, sz, perm_to_dest)];
      }
    }
  } else if(ndim == 3) {
    size_t sz[] = {ddims[0], ddims[1], ddims[2]};
    size_t i[3], c;
    for(c=0, i[0]=0; i[0]<sz[0]; i[0]++) {
      for(i[1]=0; i[1]<sz[1]; i[1]++) {
        for(i[2]=0; i[2]<sz[1]; i[2]++, c++) {
          dbuf[c] += scale * sbuf[idx(3, i, sz, perm_to_dest)];
        }
      }
    }
  } else if(ndim == 4) {
    size_t sz[] = {ddims[0], ddims[1], ddims[2], ddims[3]};
    size_t i[4], c;
    for(c=0, i[0]=0; i[0]<sz[0]; i[0]++) {
      for(i[1]=0; i[1]<sz[1]; i[1]++) {
        for(i[2]=0; i[2]<sz[2]; i[2]++) {
          for(i[3]=0; i[3]<sz[3]; i[3]++, c++) {
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
 * @ingroup perm
 * @brief Compute permutation to be performed to permute vector @p from to vector @p to.
 * @param from Source vector for the permutation
 * @param to Target vector for the permutation
 * @pre @p from and @p to are permutations of each other
 * @pre from.size() == to.size()
 * @return Vector to permute @p from to @p to.
 * @post Return ret such that:
 * ensures 0<=i<from.size(): to[i] = from[ret[i]]
 */
inline PermVector
perm_compute(const IndexLabelVec& from, const IndexLabelVec& to) {
  PermVector layout;

  EXPECTS(from.size() == to.size());
  for(auto p : to) {
    auto itr = std::find(from.begin(), from.end(), p);
    EXPECTS(itr != from.end());
    layout.push_back(itr - from.begin());
  }
  return layout;
}

template<typename T>
inline bool are_permutations(const std::vector<T>& vec1,
const std::vector<T>& vec2) {
  if(vec1.size() != vec2.size()) {
    return false;
  }
  std::vector<bool> taken(vec1.size(), false);
  for(size_t i=0; i<vec1.size(); i++) {
    auto it = std::find(vec2.begin(), vec2.end(), vec1[i]);
    if(it == vec2.end()) {
      return false;
    }
    if(taken[std::distance(vec2.begin(), it)] == true) {
      return false;
    }
  }
  return true;
}

template<typename T>
std::vector<T> unique_entries(const std::vector<T>& input_vec) {
    std::vector<T> ret;
#if 1
    for(const auto& val : input_vec) {
        auto it = std::find(ret.begin(), ret.end(), val);
        if(it == ret.end()) { ret.push_back(val); }
    }
#else
    ret = input_vec;
    std::sort(ret.begin(), ret.end());
    std::unique(ret.begin(), ret.end());
#endif
    return ret;
}

template<typename T>
std::vector<size_t> perm_map_compute(const std::vector<T>& unique_vec,
                                     const std::vector<T>& vec_required) {
    std::vector<size_t> ret;
    for(const auto& val : unique_vec) {
        auto it = std::find(unique_vec.begin(), unique_vec.end(), val);
        EXPECTS(it != unique_vec.end());
        ret.push_back(it - unique_vec.begin());
    }
    return ret;
}

template<typename T, typename Integer>
std::vector<T> perm_map_apply(const std::vector<T>& input_vec,
                              const std::vector<Integer>& perm_map) {
    std::vector<T> ret;
    for(const auto& pm : perm_map) {
        EXPECTS(pm < input_vec.size());
        ret.push_back(input_vec[pm]);
    }
    return ret;
}

inline IndexLabelVec sort_on_dependence(const IndexLabelVec& labels) {
    IndexLabelVec ret;
    for(const auto& lbl : labels) {
        for(const auto& dlbl : lbl.dep_labels()) {
            const auto it = std::find(ret.begin(), ret.end(), dlbl);
            if(it == ret.end()) { ret.push_back(dlbl); }
        }
        const auto it = std::find(ret.begin(), ret.end(), lbl);
        if(it == ret.end()) { ret.push_back(lbl); }
    }
    return ret;
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
        unique_labels = sort_on_dependence(unique_labels);
        // std::sort(unique_labels.begin(), unique_labels.end());
        // std::unique(unique_labels.begin(), unique_labels.end());
        const auto& dperm_map = perm_map_compute(unique_labels, dlabel);
        const auto& sperm_map = perm_map_compute(unique_labels, slabel);

        auto idx = [](const auto& index_vec, const auto& dims_vec) {
            size_t ret = 0, ld = 1;
            EXPECTS(index_vec.size() == dims_vec.size());
            for(size_t i = index_vec.size(); i >= 0; i--) {
                ret += ld * index_vec[i];
                ld *= dims_vec[i];
            }
            return ret;
        };

        std::vector<IndexLoopBound> ilbs;
        for(const auto& lbl : unique_labels) { ilbs.push_back({lbl}); }
        IndexLoopNest iln = IndexLoopNest{ilbs};
        for(const auto& itval : iln) {
            const auto& sindex = perm_map_apply(sperm_map, itval);
            const auto& dindex = perm_map_apply(dperm_map, itval);
            if(!update) {
                dbuf[idx(dindex, ddims)] = scale * sbuf[idx(sindex, sdims)];
            } else {
                dbuf[idx(dindex, ddims)] += scale * sbuf[idx(sindex, sdims)];
            }
        }
    }
}

template<typename T>
inline void block_mult(T cscale, T* cbuf, const std::vector<size_t>& cdims,
                       const IndexLabelVec& clabel, T abscale, T* abuf,
                       const std::vector<size_t>& adims,
                       const IndexLabelVec& alabel, T* bbuf,
                       const std::vector<size_t>& bdims,
                       const IndexLabelVec& blabel) {
    IndexLabelVec all_labels{clabel};
    all_labels.insert(all_labels.end(), alabel.begin(), alabel.end());
    all_labels.insert(all_labels.end(), blabel.begin(), blabel.end());
    IndexLabelVec unique_labels = unique_entries(all_labels);
    unique_labels = sort_on_dependence(unique_labels);
    // std::sort(unique_labels.begin(), unique_labels.end());
    // std::unique(unique_labels.begin(), unique_labels.end());
    const auto& cperm_map = perm_map_compute(unique_labels, clabel);
    const auto& bperm_map = perm_map_compute(unique_labels, alabel);
    const auto& aperm_map = perm_map_compute(unique_labels, blabel);

    auto idx = [](const auto& index_vec, const auto& dims_vec) {
        size_t ret = 0, ld = 1;
        EXPECTS(index_vec.size() == dims_vec.size());
        for(size_t i = index_vec.size(); i >= 0; i--) {
            ret += ld * index_vec[i];
            ld *= dims_vec[i];
        }
        return ret;
    };

    std::vector<IndexLoopBound> ilbs;
    for(const auto& lbl : unique_labels) { ilbs.push_back({lbl}); }
    IndexLoopNest iln = IndexLoopNest{ilbs};
    for(const auto& itval : iln) {
        const auto& cindex = perm_map_apply(cperm_map, itval);
        const auto& aindex = perm_map_apply(aperm_map, itval);
        const auto& bindex = perm_map_apply(bperm_map, itval);
        size_t cidx        = idx(cindex, cdims);
        cbuf[cidx] = cscale * cbuf[cidx] + abscale * abuf[idx(aindex, adims)] *
                                             bbuf[idx(bindex, bdims)];
    }
}

} // namespace internal

class Op {
public:
    virtual std::shared_ptr<Op> clone() const    = 0;
    virtual void execute(ProcGroup ec_pg) = 0;
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

#if 0
    SetOp(LabeledTensorT lhs, T alpha, const LabeledLoop& loop_nest,
          bool is_assign) :
      lhs_{lhs},
      alpha_{alpha},
      loop_nest_{loop_nest},
      is_assign_{is_assign} {}
#endif

    SetOp(LabeledTensorT lhs, T alpha,
          /*          const LabeledLoop& loop_nest,*/
          bool is_assign) :
      lhs_{lhs},
      alpha_{alpha},
      // loop_nest_{loop_nest},
      is_assign_{is_assign} {
        fillin_labels();
        validate();
    }

    SetOp(const SetOp<T, LabeledTensorT>&) = default;

    T alpha() const { return alpha_; }

    LabeledTensorT lhs() const { return lhs_; }

    bool is_assign() const { return is_assign_; }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new SetOp<T, LabeledTensorT>{*this});
    }

    void execute(ProcGroup ec_pg) override {
        using TensorElType = typename LabeledTensorT::element_type;
        // the iterator to generate the tasks
        const IndexLabelVec& sorted_labels =
          internal::sort_on_dependence(lhs_.labels());
        std::vector<IndexLoopBound> ilbs;
        for(const auto& lbl : sorted_labels) { ilbs.push_back({lbl}); }
        IndexLoopNest loop_nest { ilbs };
        const std::vector<size_t>& lhs_pm =
          internal::perm_map_compute(sorted_labels, lhs_.labels());
        // auto loop_nest = lhs_.tensor().loop_nest();
        // function to compute one block
        auto lambda = [&](const IndexVector itval) {
            auto tensor = lhs_.tensor();
            const IndexVector& blockid =
              internal::perm_map_apply(itval, lhs_pm);
            size_t size = tensor.block_size(blockid);
            std::vector<TensorElType> buf(size,
                                          static_cast<TensorElType>(alpha()));
            if(is_assign_) {
                tensor.put(blockid, span<TensorElType>(&buf[0], size));
            } else {
                tensor.add(blockid, span<TensorElType>(&buf[0], size));
            }
        };
        // ec->...(loop_nest, lambda);
        //@todo use a scheduler
        do_work(ec_pg, loop_nest, lambda);
    }

protected:
    void fillin_labels() {
        using internal::update_fillin_map;
        using internal::fillin_tensor_label_from_map;
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
            for(const auto& dl : ilv[i].dep_labels()) {
                size_t j;
                for(j = 0; j < ilv.size(); j++) {
                    if(dl.tiled_index_space() == ilv[j].tiled_index_space() &&
                       dl.get_label() == ilv[j].get_label()) {
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
                   ilbl.get_label() == jlbl.get_label()) {
                    EXPECTS(ilbl == jlbl);
                }
            }
        }
    }

    LabeledTensorT lhs_;
    T alpha_;
    // LabeledLoop loop_nest_;
    bool is_assign_;
}; // class SetOp

template<typename Func, typename LabeledTensorT>
class ScanOp : public Op {
    public:
  ScanOp(const LabeledTensorT& lhs, Func func)
      : lhs_{lhs},
        func_{func} {
    EXPECTS(lhs.tensor_ != nullptr);
    fillin_labels();
  }

//   TensorImpl* writes() const override {
//     return ltensor_.tensor_;
//   }

//   std::vector<TensorImpl*> reads() const {
//     return {};
//   }

  void execute(const ProcGroup& ec_pg) override {
        using TensorElType = typename LabeledTensorT::element_type;
        // the iterator to generate the tasks
        const auto& tensor = lhs_.tensor();
        const IndexLabelVec& iter_labels = internal::sort_on_dependence(lhs_.labels());
        std::vector<IndexLoopBound> ilbs;
        for(const auto& lbl : iter_labels) { ilbs.push_back({lbl}); }
        IndexLoopNest loop_nest { ilbs };
        const std::vector<size_t>& lhs_pm =
          internal::perm_map_compute(iter_labels, lhs_.labels());
        // auto loop_nest = lhs_.tensor().loop_nest();
        // function to compute one block
        auto lambda = [&](const IndexVector itval) {
            auto tensor = lhs_.tensor();
            const IndexVector& blockid =
              internal::perm_map_apply(itval, lhs_pm);
            size_t size = tensor.block_size(blockid);
            std::vector<TensorElType> buf(size);
            tensor.get(blockid, span<TensorElType>(&buf[0], size));
            func_(tensor, blockid, buf);
        };
        // ec->...(loop_nest, lambda);
        //@todo use a scheduler
        do_work(ec_pg, loop_nest, lambda);
  }

protected:
    void fillin_labels() {    
        using internal::update_fillin_map;
        using internal::fillin_tensor_label_from_map;
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
            for(const auto& dl : ilv[i].dep_labels()) {
                size_t j;
                for(j = 0; j < ilv.size(); j++) {
                    if(dl.tiled_index_space() == ilv[j].tiled_index_space() &&
                       dl.get_label() == ilv[j].get_label()) {
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
                   ilbl.get_label() == jlbl.get_label()) {
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
    // using RHS_Blocks = std::array<Block<T>, N>;

    MapOp(LabeledTensorT& lhs, Func func,
          RHS& rhs) //, ResultMode mode = ResultMode::set)
      :
      lhs_{lhs},
      func_{func},
      rhs_{rhs} {
        fillin_labels();
        validate();
    }

    //   TensorImpl* writes() const override {
    //     return lhs_.tensor_;
    //   }

    //   std::vector<TensorImpl*> reads() const {
    //     std::vector<TensorImpl*> ret;
    //     for(auto& lt: rhs_) {
    //       ret.push_back(lt.tensor_);
    //     }
    //     return ret;
    //   }

    void execute(const ProcGroup& ec_pg) override {
        using TensorElType = typename LabeledTensorT::element_type;
        // the iterator to generate the tasks
        const auto& tensor = lhs_.tensor();
        const IndexLabelVec& iter_labels = internal::sort_on_dependence(lhs_.labels());
        std::vector<IndexLoopBound> ilbs;
        for(const auto& lbl : iter_labels) { ilbs.push_back({lbl}); }
        IndexLoopNest loop_nest { ilbs };
        const std::vector<size_t>& lhs_pm =
          internal::perm_map_compute(iter_labels, lhs_.labels());
        std::vector<size_t> rhs_pm[N];
        for(size_t i=0; i<N; i++) {
            rhs_pm[i] = internal::perm_map_compute(iter_labels, rhs_[i].labels());
        }
        // auto loop_nest = lhs_.tensor().loop_nest();
        // function to compute one block
        auto lambda = [&](const IndexVector itval) {
            auto ltensor = lhs_.tensor();
            const IndexVector& lblockid =
              internal::perm_map_apply(itval, lhs_pm);
            IndexVector rblockid[N];
            for(size_t i=0; i<N; i++) {
              rblockid[i] = internal::perm_map_apply(itval, rhs_pm[i]);
            }
            const size_t lsize = ltensor.block_size(lblockid);
            std::vector<TensorElType> lbuf(lsize);
            std::vector<TensorElType> rbuf[N];
            for(size_t i=0; i<N; i++) {
                const auto& rtensor_i = rhs_[i].tensor();
                size_t isz = rtensor_i.block_size(rblockid[i]);
                rbuf[i].resize(isz);
                rtensor_i.get(rblockid[i], span<TensorElType>(&rbuf[i], isz));
            }
            func_(tensor, lblockid, lbuf, rblockid, rbuf);
            ltensor.put(lblockid, span<TensorElType>(&lbuf, lsize));
        };
        // ec->...(loop_nest, lambda);
        //@todo use a scheduler
        do_work(ec_pg, loop_nest, lambda);
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
            for(const auto& dl : ilv[i].dep_labels()) {
                size_t j;
                for(j = 0; j < ilv.size(); j++) {
                    if(dl.tiled_index_space() == ilv[j].tiled_index_space() &&
                       dl.get_label() == ilv[j].get_label()) {
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
                   ilbl.get_label() == jlbl.get_label()) {
                    EXPECTS(ilbl == jlbl);
                }
            }
        }
    }

    LabeledTensorT& lhs_;
    Func func_;
    std::array<LabeledTensorT, N> rhs_;
    // ResultMode mode_;
};

template<typename T, typename LabeledTensorT>
class AddOp : public Op {
public:
    AddOp() = default;
    AddOp(LabeledTensorT lhs, T alpha, LabeledTensorT rhs,
          /*const LabeledLoop& loop_nest,*/ bool is_assign) :
      lhs_{lhs},
      alpha_{alpha},
      rhs_{rhs},
      // loop_nest_{loop_nest},
      is_assign_{is_assign} {
        fillin_labels();
        validate();
    }

    AddOp(const AddOp<T, LabeledTensorT>&) = default;

    T alpha() const { return alpha_; }

    LabeledTensorT lhs() const { return lhs_; }

    LabeledTensorT rhs() const { return rhs_; }

    bool is_assign() const { return is_assign_; }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new AddOp<T, LabeledTensorT>{*this});
    }

    void execute(ProcGroup ec_pg) override {
        using TensorElType = typename LabeledTensorT::element_type;
        // the iterator to generate the tasks
        const IndexLabelVec& sorted_labels =
          internal::sort_on_dependence(lhs_.labels());
        std::vector<IndexLoopBound> ilbs;
        for(const auto& lbl : sorted_labels) { ilbs.push_back({lbl}); }
        IndexLoopNest loop_nest{ilbs};
        const std::vector<size_t>& lhs_pm =
          internal::perm_map_compute(sorted_labels, lhs_.labels());
        const std::vector<size_t>& rhs_pm =
          internal::perm_map_compute(sorted_labels, rhs_.labels());
        // auto loop_nest = lhs_.tensor().loop_nest();
        // function to compute one block
        auto lambda = [this, lhs_pm, rhs_pm](const IndexVector itval) {
            auto ltensor = lhs_.tensor();
            auto rtensor = rhs_.tensor();
            const IndexVector& lblockid =
              internal::perm_map_apply(itval, lhs_pm);
            const IndexVector& rblockid =
              internal::perm_map_apply(itval, rhs_pm);
            size_t size = ltensor.block_size(lblockid);
            // IndexVector rblockid = internal::LabelMap<Index>()
            //                          .update(lhs_.labels(), lblockid)
            //                          .get(rhs_.labels());
            std::vector<TensorElType> rbuf(size);
            std::vector<TensorElType> lbuf(size);
            rtensor.get(rblockid, span<TensorElType>(&rbuf[0], size));
            const auto& ldims = lhs_.tensor().block_dims(lblockid);
            const auto& rdims = rhs_.tensor().block_dims(rblockid);
            internal::block_add(&lbuf[0], ldims, lhs_.labels(), &rbuf[0], rdims,
                                rhs_.labels(), alpha_, !is_assign_);
            if(is_assign_) {
                ltensor.put(lblockid, span<TensorElType>(&lbuf[0], size));
            } else {
                ltensor.add(lblockid, span<TensorElType>(&lbuf[0], size));
            }
        };
        // ec->...(loop_nest, lambda);
        //@todo use a scheduler
        //@todo make parallel
        do_work(ec_pg, loop_nest, lambda);
    }

protected:
    void fillin_labels() {
        using internal::update_fillin_map;
        using internal::fillin_tensor_label_from_map;
        // every string in RHS is also in LHS. So number only LHS strings
        std::map<std::string, Label> str_to_labels;
        update_fillin_map(str_to_labels, lhs_.str_map(), lhs_.str_labels(), 0);
        fillin_tensor_label_from_map(lhs_, str_to_labels);
        fillin_tensor_label_from_map(rhs_, str_to_labels);
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
            for(const auto& dl : ilv[i].dep_labels()) {
                size_t j;
                for(j = 0; j < ilv.size(); j++) {
                    if(dl.tiled_index_space() == ilv[j].tiled_index_space() &&
                       dl.get_label() == ilv[j].get_label()) {
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
                   ilbl.get_label() == jlbl.get_label()) {
                    EXPECTS(ilbl == jlbl);
                }
            }
        }
    }

    LabeledTensorT lhs_;
    T alpha_;
    LabeledTensorT rhs_;
    // LabeledLoop loop_nest_;
    bool is_assign_;
}; // class AddOp

template<typename T, typename LabeledTensorT>
class MultOp : public Op {
public:
    MultOp() = default;
    MultOp(LabeledTensorT lhs, T alpha, LabeledTensorT rhs1,
           LabeledTensorT rhs2, // LabeledLoop outer_loop_nest,
           /*LabeledLoop inner_loop_nest, SymmFactor symm_factor,*/
           bool is_assign) :
      lhs_{lhs},
      alpha_{alpha},
      rhs1_{rhs1},
      rhs2_{rhs2},
      //   outer_loop_nest_{outer_loop_nest},
      //   inner_loop_nest_{inner_loop_nest},
      //   symm_factor_{symm_factor},
      is_assign_{is_assign} {
        fillin_labels();
        validate();
        if(!is_assign_) {
            NOT_IMPLEMENTED(); //C+=A*B not implemented
        }    
    }

    MultOp(const MultOp<T, LabeledTensorT>&) = default;

    LabeledTensorT lhs() const { return lhs_; }

    T alpha() const { return alpha_; }

    LabeledTensorT rhs1() const { return rhs1_; }

    LabeledTensorT rhs2() const { return rhs2_; }

    bool is_assign() const { return is_assign_; }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new MultOp{*this});
    }

    void execute(ProcGroup ec_pg) override {
        EXPECTS(!is_assign_);
        using TensorElType = typename LabeledTensorT::element_type;
        // the iterator to generate the tasks
        IndexLabelVec sorted_labels;
        const std::vector<size_t>& cpm =
          internal::perm_map_compute(sorted_labels, lhs_.labels());
        const std::vector<size_t>& apm =
          internal::perm_map_compute(sorted_labels, rhs1_.labels());
        const std::vector<size_t>& bpm =
          internal::perm_map_compute(sorted_labels, rhs2_.labels());
#if 1
        IndexLabelVec all_labels{lhs_.labels()};
        all_labels.insert(all_labels.end(), rhs1_.labels().begin(), rhs1_.labels().end());
        all_labels.insert(all_labels.end(), rhs2_.labels().begin(), rhs2_.labels().end());
        IndexLabelVec unique_labels = internal::unique_entries(all_labels);
        unique_labels = internal::sort_on_dependence(unique_labels);

        std::vector<IndexLoopBound> ilbs;
        for(const auto& lbl : unique_labels) { ilbs.push_back({lbl}); }
        IndexLoopNest loop_nest{ilbs};

        //auto loop_nest = lhs_.tensor().loop_nest();
        //auto inner_loop_nest = lhs_.tensor().loop_nest();
        // function to compute one block
        auto lambda = [this,cpm,apm,bpm](const IndexVector itval) {
            auto ctensor         = lhs_.tensor();
            auto atensor         = rhs1_.tensor();
            auto btensor         = rhs2_.tensor();
            const IndexVector& cblockid =
              internal::perm_map_apply(itval, cpm);
            const IndexVector& ablockid =
              internal::perm_map_apply(itval, apm);
            const IndexVector& bblockid =
              internal::perm_map_apply(itval, bpm);
            size_t csize          = ctensor.block_size(cblockid);
            size_t asize          = atensor.block_size(ablockid);
            size_t bsize          = btensor.block_size(bblockid);
            std::vector<TensorElType> cbuf(csize);
            std::vector<TensorElType> abuf(asize);
            std::vector<TensorElType> bbuf(bsize);
            atensor.get(ablockid, span<TensorElType>(&abuf[0], asize));
            btensor.get(bblockid, span<TensorElType>(&bbuf[0], bsize));
            const auto& cdims = ctensor.block_dims(cblockid);
            const auto& adims = atensor.block_dims(ablockid);
            const auto& bdims = btensor.block_dims(bblockid);
            double cscale = is_assign_ ? 0 : 1;
            internal::block_mult(0.0, &cbuf[0], cdims, lhs_.labels(),
            alpha_, &abuf[0], adims, rhs1_.labels(), &bbuf[0], bdims, rhs2_.labels());
            // if(is_assign_) {
            //     ctensor.put(cblockid, span<TensorElType>(&cbuf[0], csize));
            // } else {
                ctensor.add(cblockid, span<TensorElType>(&cbuf[0], csize));
            // }
        };
        // ec->...(loop_nest, lambda);
        //@todo use a scheduler
        //@todo make parallel
        do_work(ec_pg, loop_nest, lambda);
#endif
    }

protected:
    void fillin_labels() {
        using internal::update_fillin_map;
        using internal::fillin_tensor_label_from_map;
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
            for(const auto& dl : ilv[i].dep_labels()) {
                size_t j;
                for(j = 0; j < ilv.size(); j++) {
                    if(dl.tiled_index_space() == ilv[j].tiled_index_space() &&
                       dl.get_label() == ilv[j].get_label()) {
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
                   ilbl.get_label() == jlbl.get_label()) {
                    EXPECTS(ilbl == jlbl);
                }
            }
        }
    }

    LabeledTensorT lhs_;
    T alpha_;
    LabeledTensorT rhs1_;
    LabeledTensorT rhs2_;
    // LabeledLoop outer_loop_nest_;
    // LabeledLoop inner_loop_nest_;
    // SymmFactor symm_factor_;
    bool is_assign_;
}; // class MultOp

template<typename TensorType>
class AllocOp : public Op {
public:
    AllocOp(TensorType tensor, ExecutionContext* ec) : tensor_{tensor}, ec_{ec} {}

    AllocOp(const AllocOp<TensorType>&) = default;

    TensorType tensor() const { return tensor_; }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new AllocOp{*this});
    }

    void execute(ProcGroup ec_pg) override { tensor_.alloc(ec_); }

protected:
    TensorType tensor_;
    ExecutionContext* ec_;
}; // class AllocOp

template<typename TensorType>
class DeallocOp : public Op {
public:
    DeallocOp(TensorType tensor) : tensor_{tensor} {}

    DeallocOp(const DeallocOp<TensorType>&) = default;

    TensorType tensor() const { return tensor_; }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new DeallocOp{*this});
    }

    void execute(ProcGroup ec_pg) override { tensor_.dealloc(); }

protected:
    TensorType tensor_;
}; // class AllocOp

} // namespace tamm

#endif // TAMM_OPS_HPP_
