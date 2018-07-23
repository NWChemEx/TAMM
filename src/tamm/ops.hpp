#ifndef TAMM_OPS_HPP_
#define TAMM_OPS_HPP_

#include <memory>

#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/labeled_tensor.hpp"
#include "tamm/tensor_impl.hpp"
#include "tamm/types.hpp"

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
inline void
block_add (T* dbuf, const std::vector<size_t>& ddims, 
          const IndexLabelVec& dlabel, 
          T* sbuf, const std::vector<size_t>& sdims,
          const IndexLabelVec& slabel,
          T scale, bool update) {
  EXPECTS(slabel.size() == dlabel.size());
  EXPECTS(sdims.size() == slabel.size());
  EXPECTS(ddims.size() == dlabel.size());
  auto label_perm = perm_compute(dlabel, slabel);
  for(unsigned i=0; i<label_perm.size(); i++) {
    EXPECTS(ddims[i] == sdims[label_perm[i]]);
  }
  if(!update) {
    index_permute(dbuf, sbuf, label_perm, ddims, scale);
  } else {
    index_permute_acc(dbuf, sbuf, label_perm, ddims, scale);
  }
}


} // namespace internal

class Op {
public:
    virtual std::shared_ptr<Op> clone() const = 0;
    virtual void execute()                    = 0;
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
    }

    SetOp(const SetOp<T, LabeledTensorT>&) = default;

    T alpha() const { return alpha_; }

    LabeledTensorT lhs() const { return lhs_; }

    bool is_assign() const { return is_assign_; }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new SetOp<T, LabeledTensorT>{*this});
    }

    void execute() override {
        using TensorElType = typename LabeledTensorT::element_type;
        // the iterator to generate the tasks
        auto loop_nest = lhs_.tensor().loop_nest();
        // function to compute one block
        auto lambda = [&](const IndexVector blockid) {
            auto tensor = lhs_.tensor();
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
        //@todo make parallel
        for(const auto& blockid : loop_nest) { lambda(blockid); }
    }

protected:
    void fillin_labels() {
        using internal::update_fillin_map;
        using internal::fillin_tensor_label_from_map;
        std::map<std::string, Label> str_to_labels;
        update_fillin_map(str_to_labels, lhs_.str_map(), lhs_.str_labels(), 0);
        fillin_tensor_label_from_map(lhs_, str_to_labels);
    }

    LabeledTensorT lhs_;
    T alpha_;
    // LabeledLoop loop_nest_;
    bool is_assign_;
}; // class SetOp

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
    }

    AddOp(const AddOp<T, LabeledTensorT>&) = default;

    T alpha() const { return alpha_; }

    LabeledTensorT lhs() const { return lhs_; }

    LabeledTensorT rhs() const { return rhs_; }

    bool is_assign() const { return is_assign_; }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new AddOp<T, LabeledTensorT>{*this});
    }

    void execute() override {
        using TensorElType = typename LabeledTensorT::element_type;
        // the iterator to generate the tasks
        auto loop_nest = lhs_.tensor().loop_nest();
        // function to compute one block
        auto lambda = [this](const IndexVector lblockid) {
            auto ltensor         = lhs_.tensor();
            auto rtensor         = rhs_.tensor();
            size_t size          = ltensor.block_size(lblockid);
            IndexVector rblockid = internal::LabelMap<Index>()
                                     .update(lhs_.labels(), lblockid)
                                     .get(rhs_.labels());
            std::vector<TensorElType> rbuf(size);
            std::vector<TensorElType> lbuf(size);
            rtensor.get(rblockid, span<TensorElType>(&rbuf[0], size));
            const auto& ldims = lhs_.tensor().block_dims(lblockid);
            const auto& rdims = rhs_.tensor().block_dims(rblockid);
            internal::block_add(&lbuf[0], ldims, lhs_.labels(), &rbuf[0], 
                              rdims, rhs_.labels(), alpha_, !is_assign_);
            if(is_assign_) {
                ltensor.put(lblockid, span<TensorElType>(&lbuf[0], size));
            } else {
                ltensor.add(lblockid, span<TensorElType>(&lbuf[0], size));
            }
        };
        // ec->...(loop_nest, lambda);
        //@todo use a scheduler
        //@todo make parallel
        for(const auto& lblockid : loop_nest) { lambda(lblockid); }
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

    void execute() override {}

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
    AllocOp(TensorType tensor) : tensor_{tensor} {}

    AllocOp(const AllocOp<TensorType>&) = default;

    TensorType tensor() const { return tensor_; }

    std::shared_ptr<Op> clone() const override {
        return std::shared_ptr<Op>(new AllocOp{*this});
    }

    void execute() override { tensor_.allocate(); }

protected:
    TensorType tensor_;
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

    void execute() override { tensor_.deallocate(); }

protected:
    TensorType tensor_;
}; // class AllocOp

} // namespace tamm

#endif // TAMM_OPS_HPP_
