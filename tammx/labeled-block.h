#ifndef TAMMX_LABELEDBLOCK_H_
#define TAMMX_LABELEDBLOCK_H_

#include "tammx/types.h"
#include "tammx/block.h"

namespace tammx {


template<typename T>
struct LabeledBlock;

template<typename T1,
         typename T2>
inline std::tuple<T1, LabeledBlock<T2>>
operator * (T1 alpha, LabeledBlock<T2> block) {
  return {alpha, block};
}

template<typename T1,
         typename T2>
inline std::tuple<T1, LabeledBlock<T2>>
operator * (LabeledBlock<T2> block, T1 alpha) {
  return {alpha, block};
}

template<typename T1, typename T2>
inline std::tuple<T1, LabeledBlock<T2>, LabeledBlock<T2>>
operator * (const std::tuple<T1, LabeledBlock<T2>>& rhs1, LabeledBlock<T2> rhs2)  {
  return std::tuple_cat(rhs1, std::make_tuple(rhs2));
}

template<typename T>
inline std::tuple<LabeledBlock<T>, LabeledBlock<T>>
operator * (LabeledBlock<T> rhs1, LabeledBlock<T> rhs2)  {
  return std::make_tuple(rhs1, rhs2);
}

template<typename T1,
         typename T2>
inline std::tuple<T1, LabeledBlock<T2>, LabeledBlock<T2>>
operator * (T1 alpha, std::tuple<LabeledBlock<T2>, LabeledBlock<T2>> rhs) {
  return std::tuple_cat(std::make_tuple(alpha), rhs);
}


template<typename T>
struct LabeledBlock {
  Block<T> *block_;
  TensorLabel label_;

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  void operator = (T1 value) {
    auto buf = reinterpret_cast<T*>(block_->buf());
    auto rval = static_cast<T>(value);
    for(int i=0; i<block_->block_size(); i++) {
      buf[i] = rval;
    }
  }

  void operator = (LabeledBlock<T> rhs) {
    *this = 1 * rhs;
  }

  template<typename T1>
  void operator = (std::tuple<T1, LabeledBlock<T>> rhs);

  template<typename T1>
  void operator = (std::tuple<T1, LabeledBlock<T>, LabeledBlock<T>> rhs);

  void operator = (std::tuple<LabeledBlock<T>, LabeledBlock<T>> rhs) {
    *this = 1 * std::get<0>(rhs) * std::get<1>(rhs);
  }

  void operator += (LabeledBlock<T> rhs) {
    *this += 1 * rhs;
  }

  template<typename T1>
  void operator += (std::tuple<T1, LabeledBlock<T>> rhs);

  template<typename T1>
  void operator += (std::tuple<T1, LabeledBlock<T>, LabeledBlock<T>> rhs);

  void operator += (std::tuple<LabeledBlock<T>, LabeledBlock<T>> rhs) {
    *this += 1 * std::get<0>(rhs) * std::get<1>(rhs);
  }
};


namespace impl {
/**
 * performs: cbuf[dims] = scale *abuf[perm(dims)]
 *
 * @todo unsafe. passing 1 instead of 1.0 might lead to unexpected results.
 */
template<typename T>
inline void
index_permute_acc(uint8_t* dbuf, uint8_t* sbuf, const TensorPerm& perm, const TensorIndex& ddims, T scale) {
  Expects(dbuf!=nullptr && sbuf!=nullptr);
  Expects(perm.size() == ddims.size());

  auto inv_perm = perm_invert(perm);
  auto inv_sizes = perm_apply(ddims, inv_perm);
  TensorVec<size_t> sizes;
  TensorVec<int> iperm;
  for(unsigned i=0; i<ddims.size(); i++) {
    sizes.push_back(inv_sizes[i].value());
    iperm.push_back(inv_perm[i]+1);
  }
  index_sortacc(sbuf, dbuf,
                sizes.size(), &sizes[0], &iperm[0], scale);
}

/**
 *  @todo unsafe. passing 1 instead of 1.0 might lead to unexpected results.
 */
template<typename T>
inline void
index_permute(uint8_t* dbuf, uint8_t* sbuf, const TensorPerm& perm, const TensorIndex& ddims, T scale) {
  Expects(dbuf!=nullptr && sbuf!=nullptr);
  Expects(perm.size() == ddims.size());

  auto inv_perm = perm_invert(perm);
  auto inv_sizes = perm_apply(ddims, inv_perm);
  TensorVec<size_t> sizes;
  TensorVec<int> iperm;
  for(unsigned i=0; i<ddims.size(); i++) {
    sizes.push_back(inv_sizes[i].value());
    iperm.push_back(inv_perm[i]+1);
  }

  index_sort(sbuf, dbuf,
             sizes.size(), &sizes[0], &iperm[0], scale);
}



template<typename T>
// C storage order: A[m,k], B[k,n], C[m,n]
inline void
matmul(int m, int n, int k, T *A, int lda, T *B, int ldb, T *C, int ldc, T alpha, T beta) {
  Expects(m>0 && n>0 && k>0);
  Expects(A!=nullptr && B!=nullptr && C!=nullptr);

  for(int x=0; x<m; x++) {
    for(int y=0; y<n; y++) {
      T value = 0;
      for(int z=0; z<k; z++) {
        value += A[x*lda + z] * B[z*ldb + y];
      }
      C[x*ldc + y] = beta * C[x*ldc + y] + alpha * value;
    }
  }
}

template<typename T>
inline TensorPerm
perm_compute(const LabeledBlock<T>& lblock_from, const TensorLabel& label_to) {
  auto store = perm_apply(lblock_from.label_,
                          perm_invert(lblock_from.block_->layout()));
  return perm_compute(store, label_to);
}

template<typename T, typename T1>
void multiply(LabeledBlock<T>& clb, std::tuple<T1, LabeledBlock<T>, LabeledBlock<T>> rhs, T beta) {
  const LabeledBlock<T>& alb = std::get<1>(rhs);
  const LabeledBlock<T>& blb = std::get<2>(rhs);

  auto &ablock = *alb.block_;
  auto &bblock = *blb.block_;
  auto &cblock = *clb.block_;

  auto &alabel = alb.label_;
  auto &blabel = blb.label_;
  auto &clabel = clb.label_;

  auto aext_labels = intersect(clabel, alabel);
  auto bext_labels = intersect(clabel, blabel);
  auto sum_labels = intersect(alabel, blabel);

  auto alabel_sort = aext_labels;
  alabel_sort.insert_back(sum_labels.begin(), sum_labels.end());
  auto blabel_sort = sum_labels;
  blabel_sort.insert_back(bext_labels.begin(), bext_labels.end());
  auto clabel_sort = aext_labels;
  clabel_sort.insert_back(bext_labels.begin(), bext_labels.end());

  //TTGT
  //TT
  auto elsize = sizeof(T);
  auto abuf_sort = std::make_unique<uint8_t[]>(ablock.size() * elsize);
  auto bbuf_sort = std::make_unique<uint8_t[]>(bblock.size() * elsize);
  auto cbuf_sort = std::make_unique<uint8_t[]>(cblock.size() * elsize);

  auto aperm = perm_compute(ablock(alabel), alabel_sort);
  auto bperm = perm_compute(bblock(blabel), blabel_sort);

  /**
   * @@todo 1.0 is a bug. Make this work with.
   */
  index_permute(abuf_sort.get(), ablock.buf(), aperm,
                perm_apply(ablock.block_dims(), aperm), 1.0);
  index_permute(bbuf_sort.get(), bblock.buf(), bperm,
                perm_apply(bblock.block_dims(), bperm), 1.0);

  // G
  auto alpha = std::get<0>(rhs);
  auto lmap = LabelMap<BlockDim>()
      .update(alabel, ablock.block_dims())
      .update(blabel, bblock.block_dims());
  auto aext_dims = lmap.get_blockid(aext_labels);
  auto bext_dims = lmap.get_blockid(bext_labels);
  auto sum_dims = lmap.get_blockid(sum_labels);
  int m = std::accumulate(aext_dims.begin(), aext_dims.end(), BlockDim{1}, std::multiplies<>()).value();
  int n = std::accumulate(bext_dims.begin(), bext_dims.end(), BlockDim{1}, std::multiplies<>()).value();
  int k = std::accumulate(sum_dims.begin(), sum_dims.end(), BlockDim{1}, std::multiplies<>()).value();

  matmul<T>(m, n, k, reinterpret_cast<T*>(abuf_sort.get()), k,
            reinterpret_cast<T*>(bbuf_sort.get()), n,
            reinterpret_cast<T*>(cbuf_sort.get()), n,
            static_cast<T>(alpha), static_cast<T>(beta));
  auto cperm = perm_invert(perm_compute(cblock(clabel), clabel_sort));
  //T
  index_permute(cblock.buf(), reinterpret_cast<uint8_t*>(cbuf_sort.get()),
                cperm, cblock.block_dims(), 1.0);
}

template<typename T, typename T1>
inline void
block_add (LabeledBlock<T>& clb, std::tuple<T1, LabeledBlock<T>> rhs, bool update) {
  const LabeledBlock<T>& alb = std::get<1>(rhs);

  auto &ablock = *alb.block_;
  auto &cblock = *clb.block_;

  auto &clabel = clb.label_;
  auto &alabel = alb.label_;

  auto label_perm = perm_compute(alabel, clabel);
  for(unsigned i=0; i<label_perm.size(); i++) {
    Expects(cblock.block_dims()[i] == ablock.block_dims()[label_perm[i]]);
  }

  auto &alayout = ablock.layout();
  auto &clayout = cblock.layout();

  std::cerr<<__FUNCTION__<<":"<<__LINE__<<": alabel="<<alabel<<std::endl;
  std::cerr<<__FUNCTION__<<":"<<__LINE__<<": alayout="<<alayout<<std::endl;

  Expects(clayout.size() == cblock.tensor().rank());
  Expects(clabel.size() == perm_invert(clayout).size());
  Expects(alabel.size() == perm_invert(alayout).size());
  auto cstore = perm_apply(clabel, perm_invert(clayout));
  auto astore = perm_apply(alabel, perm_invert(alayout));

  auto store_perm = perm_compute(astore, cstore);
  auto alpha = std::get<0>(rhs);
  if(!update) {
    index_permute(cblock.buf(), ablock.buf(), store_perm, cblock.block_dims(), alpha);
  } else {
    index_permute_acc(cblock.buf(), ablock.buf(), store_perm, cblock.block_dims(), alpha);
  }
}

} // namespace tammx::impl

template<typename T>
template<typename T1>
inline void
LabeledBlock<T>::operator = (std::tuple<T1, LabeledBlock<T>> rhs) {
  impl::block_add(*this, rhs, false);
}

template<typename T>
template<typename T1>
inline void
LabeledBlock<T>::operator += (std::tuple<T1, LabeledBlock<T>> rhs) {
  impl::block_add(*this, rhs, true);
}

template<typename T>
template<typename T1>
inline void
LabeledBlock<T>::operator += (std::tuple<T1, LabeledBlock<T>, LabeledBlock<T>> rhs) {
  impl::multiply(*this, rhs, T(1));
}


template<typename T>
template<typename T1>
inline void
LabeledBlock<T>::operator = (std::tuple<T1, LabeledBlock<T>, LabeledBlock<T>> rhs) {
  impl::multiply(*this, rhs, T(0));
}

} // namespace tammx

#endif  // TAMMX_LABELEDBLOCK_H_
