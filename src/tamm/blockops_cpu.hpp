#pragma once

#include <array>
#include <set>
#include <vector>

#include "tamm/block_span.hpp"
#include "tamm/iteration.hpp"
#include "tamm/perm.hpp"
//#include "tamm/scalar.hpp"
#include "tamm/types.hpp"

namespace tamm::blockops::cpu {

///////////////////////////////////////////////////////////////////////////////
//
//                                 set routines
//
///////////////////////////////////////////////////////////////////////////////

template<typename T1, typename T2>
void flat_set(BlockSpan<T1>& lhs, const T2& value_) {
  auto   buf          = lhs.buf();
  size_t num_elements = lhs.num_elements();
  for(size_t i = 0; i < num_elements; ++i) { buf[i] = value_; }
}

template<typename T1, typename T2>
void flat_update(BlockSpan<T1>& lhs, const T2& value_) {
  auto   buf          = lhs.buf();
  size_t num_elements = lhs.num_elements();
  for(size_t i = 0; i < num_elements; ++i) { buf[i] += value_; }
}

template<typename T1, typename T2, typename T3>
void flat_update(const T1& scale, BlockSpan<T2>& lhs, const T3& value_) {
  auto   buf          = lhs.buf();
  size_t num_elements = lhs.num_elements();
  for(size_t i = 0; i < num_elements; ++i) { buf[i] = scale * buf[i] + value_; }
}

template<typename TL>
void ipgen_loop_set(TL* lbuf, const std::vector<int>& lld, TL value,
                    const std::vector<int>& unique_label_dims) {
  EXPECTS(lbuf != nullptr);
  EXPECTS(lld.size() == unique_label_dims.size());
  const size_t ndim = unique_label_dims.size();
  if(ndim == 0) { *lbuf = value; }
  else if(ndim == 1) {
    for(int i0 = 0, i0doff = 0; i0 < unique_label_dims[0]; ++i0, i0doff += lld[0]) {
      lbuf[i0doff] = value;
    }
  }
  else if(ndim == 2) {
    for(int i0 = 0, i0doff = 0; i0 < unique_label_dims[0]; ++i0, i0doff += lld[0]) {
      for(int i1 = 0, i1doff = i0doff; i1 < unique_label_dims[1]; ++i1, i1doff += lld[1]) {
        lbuf[i1doff] = value;
      }
    }
  }
  else if(ndim == 3) {
    for(int i0 = 0, i0doff = 0; i0 < unique_label_dims[0]; ++i0, i0doff += lld[0]) {
      for(int i1 = 0, i1doff = i0doff; i1 < unique_label_dims[1]; ++i1, i1doff += lld[1]) {
        for(int i2 = 0, i2doff = i1doff; i2 < unique_label_dims[2]; ++i2, i2doff += lld[2]) {
          lbuf[i2doff] = value;
        }
      }
    }
  }
  else if(ndim == 4) {
    for(int i0 = 0, i0doff = 0; i0 < unique_label_dims[0]; ++i0, i0doff += lld[0]) {
      for(int i1 = 0, i1doff = i0doff; i1 < unique_label_dims[1]; ++i1, i1doff += lld[1]) {
        for(int i2 = 0, i2doff = i1doff; i2 < unique_label_dims[2]; ++i2, i2doff += lld[2]) {
          for(int i3 = 0, i3doff = i2doff; i3 < unique_label_dims[3]; ++i3, i3doff += lld[3]) {
            lbuf[i3doff] = value;
          }
        }
      }
    }
  }
  else { NOT_IMPLEMENTED(); }
}

template<typename TL>
void ipgen_loop_update(TL* lbuf, const std::vector<int>& lld, TL value,
                       const std::vector<int>& unique_label_dims) {
  EXPECTS(lbuf != nullptr);
  EXPECTS(lld.size() == unique_label_dims.size());
  const size_t ndim = unique_label_dims.size();
  if(ndim == 0) { *lbuf += value; }
  else if(ndim == 1) {
    for(int i0 = 0, i0doff = 0; i0 < unique_label_dims[0]; ++i0, i0doff += lld[0]) {
      lbuf[i0doff] += value;
    }
  }
  else if(ndim == 2) {
    for(int i0 = 0, i0doff = 0; i0 < unique_label_dims[0]; ++i0, i0doff += lld[0]) {
      for(int i1 = 0, i1doff = i0doff; i1 < unique_label_dims[1]; ++i1, i1doff += lld[1]) {
        lbuf[i1doff] += value;
      }
    }
  }
  else if(ndim == 3) {
    for(int i0 = 0, i0doff = 0; i0 < unique_label_dims[0]; ++i0, i0doff += lld[0]) {
      for(int i1 = 0, i1doff = i0doff; i1 < unique_label_dims[1]; ++i1, i1doff += lld[1]) {
        for(int i2 = 0, i2doff = i1doff; i2 < unique_label_dims[2]; ++i2, i2doff += lld[2]) {
          lbuf[i2doff] += value;
        }
      }
    }
  }
  else if(ndim == 4) {
    for(int i0 = 0, i0doff = 0; i0 < unique_label_dims[0]; ++i0, i0doff += lld[0]) {
      for(int i1 = 0, i1doff = i0doff; i1 < unique_label_dims[1]; ++i1, i1doff += lld[1]) {
        for(int i2 = 0, i2doff = i1doff; i2 < unique_label_dims[2]; ++i2, i2doff += lld[2]) {
          for(int i3 = 0, i3doff = i2doff; i3 < unique_label_dims[3]; ++i3, i3doff += lld[3]) {
            lbuf[i3doff] += value;
          }
        }
      }
    }
  }
  else { NOT_IMPLEMENTED(); }
}

template<typename TL>
void ipgen_loop_update(TL lscale, TL* lbuf, const std::vector<int>& lld, TL value,
                       const std::vector<size_t>& unique_label_dims) {
  EXPECTS(lbuf != nullptr);
  EXPECTS(lld.size() == unique_label_dims.size());
  const size_t ndim = unique_label_dims.size();
  if(ndim == 0) { *lbuf = lscale * *lbuf + value; }
  else if(ndim == 1) {
    for(int i0 = 0, i0doff = 0; i0 < unique_label_dims[0]; ++i0, i0doff += lld[0]) {
      lbuf[i0doff] = lscale * lbuf[i0doff] + value;
    }
  }
  else if(ndim == 2) {
    for(int i0 = 0, i0doff = 0; i0 < unique_label_dims[0]; ++i0, i0doff += lld[0]) {
      for(int i1 = 0, i1doff = i0doff; i1 < unique_label_dims[1]; ++i1, i1doff += lld[1]) {
        lbuf[i1doff] = lscale * lbuf[i1doff] + value;
      }
    }
  }
  else if(ndim == 3) {
    for(int i0 = 0, i0doff = 0; i0 < unique_label_dims[0]; ++i0, i0doff += lld[0]) {
      for(int i1 = 0, i1doff = i0doff; i1 < unique_label_dims[1]; ++i1, i1doff += lld[1]) {
        for(int i2 = 0, i2doff = i1doff; i2 < unique_label_dims[2]; ++i2, i2doff += lld[2]) {
          lbuf[i2doff] = lscale * lbuf[i2doff] + value;
        }
      }
    }
  }
  else if(ndim == 4) {
    for(int i0 = 0, i0doff = 0; i0 < unique_label_dims[0]; ++i0, i0doff += lld[0]) {
      for(int i1 = 0, i1doff = i0doff; i1 < unique_label_dims[1]; ++i1, i1doff += lld[1]) {
        for(int i2 = 0, i2doff = i1doff; i2 < unique_label_dims[2]; ++i2, i2doff += lld[2]) {
          for(int i3 = 0, i3doff = i2doff; i3 < unique_label_dims[3]; ++i3, i3doff += lld[3]) {
            lbuf[i3doff] = lscale * lbuf[i3doff] + value;
          }
        }
      }
    }
  }
  else { NOT_IMPLEMENTED(); }
}

///////////////////////////////////////////////////////////////////////////////
//
//                               flat_assign routines
//
///////////////////////////////////////////////////////////////////////////////

template<typename TL, typename TR>
void flat_assign(BlockSpan<TL>& lhs, const BlockSpan<TR>& rhs) {
  TL*          lbuf         = lhs.buf();
  const TR*    rbuf         = rhs.buf();
  const size_t num_elements = lhs.num_elements();
  for(int i = 0; i < num_elements; i++) { *lbuf++ = *rbuf++; }
}

template<typename TL, typename TR>
void flat_assign(BlockSpan<TL>& lhs, TL scale, const BlockSpan<TR>& rhs) {
  TL*          lbuf         = lhs.buf();
  const TR*    rbuf         = rhs.buf();
  const size_t num_elements = lhs.num_elements();
  for(size_t i = 0; i < num_elements; i++) { *lbuf++ = scale * *rbuf++; }
}

template<typename TL, typename TR>
void flat_update(BlockSpan<TL>& lhs, const BlockSpan<TR>& rhs) {
  TL*          lbuf         = lhs.buf();
  const TR*    rbuf         = rhs.buf();
  const size_t num_elements = lhs.num_elements();
  for(size_t i = 0; i < num_elements; i++) { *lbuf++ += *rbuf++; }
}

template<typename TL, typename TR>
void flat_update(BlockSpan<TL>& lhs, TL rscale, const BlockSpan<TR>& rhs) {
  TL*          lbuf         = lhs.buf();
  const TR*    rbuf         = rhs.buf();
  const size_t num_elements = lhs.num_elements();
  for(size_t i = 0; i < num_elements; i++) { *lbuf++ += rscale * *rbuf++; }
}

template<typename TL, typename TR>
void flat_update(TL lscale, BlockSpan<TL>& lhs, TL rscale, const BlockSpan<TR>& rhs) {
  TL*          lbuf         = lhs.buf();
  const TR*    rbuf         = rhs.buf();
  const size_t num_elements = lhs.num_elements();
  for(size_t i = 0; i < num_elements; i++) { lbuf[i] = lscale * lbuf[i] + rscale * rbuf[i]; }
}

template<typename TL, typename TR>
void flat_assign(BlockSpan<TL>& lhs, const Scalar& scale, const BlockSpan<TR>& rhs) {
  std::visit(overloaded{[&](auto e) {
               if constexpr(std::is_assignable<TL&, decltype(e)>::value) {
                 flat_assign(lhs, static_cast<TL>(e), rhs);
               }
               else { NOT_ALLOWED(); }
             }},
             scale.value());
}

template<typename TL, typename TR>
void flat_update(BlockSpan<TL>& lhs, const Scalar& scale, const BlockSpan<TR>& rhs) {
  std::visit(overloaded{[&](auto e) {
               if constexpr(std::is_assignable<TL&, decltype(e)>::value) {
                 flat_update(lhs, static_cast<TL>(e), rhs);
               }
               else { NOT_ALLOWED(); }
             }},
             scale.value());
}

template<typename TL, typename TR>
void flat_update(const Scalar& lscale, BlockSpan<TL>& lhs, const Scalar& rscale,
                 const BlockSpan<TR>& rhs) {
  std::visit(overloaded{[&](auto el) {
               std::visit(overloaded{[&](auto er) {
                            if constexpr(std::is_assignable<TL&, decltype(el)>::value &&
                                         std::is_assignable<TL&, decltype(er)>::value) {
                              flat_update(el, lhs, er, rhs);
                            }
                            else { NOT_ALLOWED(); }
                          }},
                          rscale.value());
             }},
             lscale.value());
}

template<typename Func, typename T, typename... BlockSpans>
void flat_lambda(Func&& func, BlockSpan<T>& block, BlockSpans&&... rest) {
  EXPECTS(((block.buf() != nullptr) && ... && (rest.buf() != nullptr)));
  int num_elements = block.tensor().block_size(block.blockid());
  for(int i = 0; i < num_elements; i++) {
    std::forward<Func>(func)(block.buf()[i], (std::forward<BlockSpans>(rest).buf()[i])...);
  }
}

///////////////////////////////////////////////////////////////////////////////
//
//                index_permute routines
//
///////////////////////////////////////////////////////////////////////////////

inline size_t idx(int n, const size_t* id, const std::vector<size_t>& ldims, const PermVector& p) {
  size_t idx = 0;
  for(int i = 0; i < n - 1; i++) { idx = (idx + id[p[i]]) * ldims[p[i + 1]]; }
  if(n > 0) { idx += id[p[n - 1]]; }
  return idx;
}

template<typename TL, typename TR>
void index_permute_assign(TL* lbuf, const TR* rbuf, const PermVector& perm_to_dest,
                          const std::vector<size_t>& ldims) {
  EXPECTS(lbuf != nullptr && rbuf != nullptr);
  EXPECTS(perm_to_dest.size() == ldims.size());

  const size_t ndim = perm_to_dest.size();
  EXPECTS(ldims.size() == ndim);

  if(ndim == 0) { lbuf[0] = rbuf[0]; }
  else if(ndim == 1) {
    for(size_t i = 0; i < ldims[0]; i++) { lbuf[i] = rbuf[i]; }
  }
  else if(ndim == 2) {
    size_t i[2], c;
    for(c = 0, i[0] = 0; i[0] < ldims[0]; i[0]++) {
      for(i[1] = 0; i[1] < ldims[1]; i[1]++, c++) {
        lbuf[c] = rbuf[idx(2, i, ldims, perm_to_dest)];
      }
    }
  }
  else if(ndim == 3) {
    size_t i[3], c;
    for(c = 0, i[0] = 0; i[0] < ldims[0]; i[0]++) {
      for(i[1] = 0; i[1] < ldims[1]; i[1]++) {
        for(i[2] = 0; i[2] < ldims[2]; i[2]++, c++) {
          lbuf[c] = rbuf[idx(3, i, ldims, perm_to_dest)];
        }
      }
    }
  }
  else if(ndim == 4) {
    size_t i[4], c;
    for(c = 0, i[0] = 0; i[0] < ldims[0]; i[0]++) {
      for(i[1] = 0; i[1] < ldims[1]; i[1]++) {
        for(i[2] = 0; i[2] < ldims[2]; i[2]++) {
          for(i[3] = 0; i[3] < ldims[3]; i[3]++, c++) {
            lbuf[c] = rbuf[idx(4, i, ldims, perm_to_dest)];
          }
        }
      }
    }
  }
  else { NOT_IMPLEMENTED(); }
}

template<typename TL, typename TR>
void index_permute_assign(TL* lbuf, TL rscale, const TR* rbuf, const PermVector& perm_to_dest,
                          const std::vector<size_t>& ldims) {
  EXPECTS(lbuf != nullptr && rbuf != nullptr);
  EXPECTS(perm_to_dest.size() == ldims.size());

  const size_t ndim = perm_to_dest.size();
  EXPECTS(ldims.size() == ndim);

  if(ndim == 0) { lbuf[0] = rscale * rbuf[0]; }
  else if(ndim == 1) {
    for(size_t i = 0; i < ldims[0]; i++) { lbuf[i] = rscale * rbuf[i]; }
  }
  else if(ndim == 2) {
    size_t i[2], c;
    for(c = 0, i[0] = 0; i[0] < ldims[0]; i[0]++) {
      for(i[1] = 0; i[1] < ldims[1]; i[1]++, c++) {
        lbuf[c] = rscale * rbuf[idx(2, i, ldims, perm_to_dest)];
      }
    }
  }
  else if(ndim == 3) {
    size_t i[3], c;
    for(c = 0, i[0] = 0; i[0] < ldims[0]; i[0]++) {
      for(i[1] = 0; i[1] < ldims[1]; i[1]++) {
        for(i[2] = 0; i[2] < ldims[2]; i[2]++, c++) {
          lbuf[c] = rscale * rbuf[idx(3, i, ldims, perm_to_dest)];
        }
      }
    }
  }
  else if(ndim == 4) {
    size_t i[4], c;
    for(c = 0, i[0] = 0; i[0] < ldims[0]; i[0]++) {
      for(i[1] = 0; i[1] < ldims[1]; i[1]++) {
        for(i[2] = 0; i[2] < ldims[2]; i[2]++) {
          for(i[3] = 0; i[3] < ldims[3]; i[3]++, c++) {
            lbuf[c] = rscale * rbuf[idx(4, i, ldims, perm_to_dest)];
          }
        }
      }
    }
  }
  else { NOT_IMPLEMENTED(); }
}

template<typename TL, typename TR>
void index_permute_update(TL* lbuf, const TR* rbuf, const PermVector& perm_to_dest,
                          const std::vector<size_t>& ldims) {
  EXPECTS(lbuf != nullptr && rbuf != nullptr);
  EXPECTS(perm_to_dest.size() == ldims.size());

  const size_t ndim = perm_to_dest.size();
  EXPECTS(ldims.size() == ndim);

  if(ndim == 0) { lbuf[0] += rbuf[0]; }
  else if(ndim == 1) {
    for(size_t i = 0; i < ldims[0]; i++) { lbuf[i] += rbuf[i]; }
  }
  else if(ndim == 2) {
    size_t i[2], c;
    for(c = 0, i[0] = 0; i[0] < ldims[0]; i[0]++) {
      for(i[1] = 0; i[1] < ldims[1]; i[1]++, c++) {
        lbuf[c] += rbuf[idx(2, i, ldims, perm_to_dest)];
      }
    }
  }
  else if(ndim == 3) {
    size_t i[3], c;
    for(c = 0, i[0] = 0; i[0] < ldims[0]; i[0]++) {
      for(i[1] = 0; i[1] < ldims[1]; i[1]++) {
        for(i[2] = 0; i[2] < ldims[2]; i[2]++, c++) {
          lbuf[c] += rbuf[idx(3, i, ldims, perm_to_dest)];
        }
      }
    }
  }
  else if(ndim == 4) {
    size_t i[4], c;
    for(c = 0, i[0] = 0; i[0] < ldims[0]; i[0]++) {
      for(i[1] = 0; i[1] < ldims[1]; i[1]++) {
        for(i[2] = 0; i[2] < ldims[2]; i[2]++) {
          for(i[3] = 0; i[3] < ldims[3]; i[3]++, c++) {
            lbuf[c] += rbuf[idx(4, i, ldims, perm_to_dest)];
          }
        }
      }
    }
  }
  else { NOT_IMPLEMENTED(); }
}

template<typename TL, typename TR>
void index_permute_update(TL* lbuf, TL rscale, const TR* rbuf, const PermVector& perm_to_dest,
                          const std::vector<size_t>& ldims) {
  EXPECTS(lbuf != nullptr && rbuf != nullptr);
  EXPECTS(perm_to_dest.size() == ldims.size());

  const size_t ndim = perm_to_dest.size();
  EXPECTS(ldims.size() == ndim);

  if(ndim == 0) { lbuf[0] += rscale * rbuf[0]; }
  else if(ndim == 1) {
    for(size_t i = 0; i < ldims[0]; i++) { lbuf[i] += rscale * rbuf[i]; }
  }
  else if(ndim == 2) {
    size_t i[2], c;
    for(c = 0, i[0] = 0; i[0] < ldims[0]; i[0]++) {
      for(i[1] = 0; i[1] < ldims[1]; i[1]++, c++) {
        lbuf[c] += rscale * rbuf[idx(2, i, ldims, perm_to_dest)];
      }
    }
  }
  else if(ndim == 3) {
    size_t i[3], c;
    for(c = 0, i[0] = 0; i[0] < ldims[0]; i[0]++) {
      for(i[1] = 0; i[1] < ldims[1]; i[1]++) {
        for(i[2] = 0; i[2] < ldims[2]; i[2]++, c++) {
          lbuf[c] += rscale * rbuf[idx(3, i, ldims, perm_to_dest)];
        }
      }
    }
  }
  else if(ndim == 4) {
    size_t i[4], c;
    for(c = 0, i[0] = 0; i[0] < ldims[0]; i[0]++) {
      for(i[1] = 0; i[1] < ldims[1]; i[1]++) {
        for(i[2] = 0; i[2] < ldims[2]; i[2]++) {
          for(i[3] = 0; i[3] < ldims[3]; i[3]++, c++) {
            lbuf[c] += rscale * rbuf[idx(4, i, ldims, perm_to_dest)];
          }
        }
      }
    }
  }
  else { NOT_IMPLEMENTED(); }
}

template<typename TL, typename TR>
void index_permute_update(TL lscale, TL* lbuf, TL rscale, const TR* rbuf,
                          const PermVector& perm_to_dest, const std::vector<size_t>& ldims) {
  EXPECTS(lbuf != nullptr && rbuf != nullptr);
  EXPECTS(perm_to_dest.size() == ldims.size());

  const size_t ndim = perm_to_dest.size();
  EXPECTS(ldims.size() == ndim);

  if(ndim == 0) { lbuf[0] = lscale * lbuf[0] + rscale * rbuf[0]; }
  else if(ndim == 1) {
    for(size_t i = 0; i < ldims[0]; i++) { lbuf[i] = lscale * lbuf[i] + rscale * rbuf[i]; }
  }
  else if(ndim == 2) {
    size_t i[2], c;
    for(c = 0, i[0] = 0; i[0] < ldims[0]; i[0]++) {
      for(i[1] = 0; i[1] < ldims[1]; i[1]++, c++) {
        lbuf[c] = lscale * lbuf[i] + rscale * rbuf[idx(2, i, ldims, perm_to_dest)];
      }
    }
  }
  else if(ndim == 3) {
    size_t i[3], c;
    for(c = 0, i[0] = 0; i[0] < ldims[0]; i[0]++) {
      for(i[1] = 0; i[1] < ldims[1]; i[1]++) {
        for(i[2] = 0; i[2] < ldims[2]; i[2]++, c++) {
          lbuf[c] = lscale * lbuf[i] + rscale * rbuf[idx(3, i, ldims, perm_to_dest)];
        }
      }
    }
  }
  else if(ndim == 4) {
    size_t i[4], c;
    for(c = 0, i[0] = 0; i[0] < ldims[0]; i[0]++) {
      for(i[1] = 0; i[1] < ldims[1]; i[1]++) {
        for(i[2] = 0; i[2] < ldims[2]; i[2]++) {
          for(i[3] = 0; i[3] < ldims[3]; i[3]++, c++) {
            lbuf[c] = lscale * lbuf[i] + rscale * rbuf[idx(4, i, ldims, perm_to_dest)];
          }
        }
      }
    }
  }
  else { NOT_IMPLEMENTED(); }
}

///////////////////////////////////////////////////////////////////////////////
//
//                General index permutation
//
///////////////////////////////////////////////////////////////////////////////

inline size_t ipgen_idx(const std::vector<size_t>& index_vec, const std::vector<size_t>& dims_vec) {
  size_t ret = 0, ld = 1;
  EXPECTS(index_vec.size() == dims_vec.size());
  for(int i = index_vec.size(); i >= 0; i--) {
    ret += ld * index_vec[i];
    ld *= dims_vec[i];
  }
  return ret;
}

template<typename TL, typename TR>
void ipgen_assign(TL* lbuf, const std::vector<size_t>& ddims, const PermVector& dperm_map,
                  const PermVector& dinv_perm_map, TR* rbuf, const std::vector<size_t>& sdims,
                  const PermVector& sperm_map, int unique_label_count) {
  std::vector<size_t> itrv(unique_label_count, 0);
  std::vector<size_t> endv(unique_label_count);
  endv = internal::perm_map_apply(ddims, dinv_perm_map);
  do {
    const auto& itval              = itrv;
    const auto& sindex             = internal::perm_map_apply(itval, sperm_map);
    const auto& dindex             = internal::perm_map_apply(itval, dperm_map);
    lbuf[ipgen_idx(dindex, ddims)] = rbuf[ipgen_idx(sindex, sdims)];
  } while(internal::cartesian_iteration(itrv, endv));
}

template<typename TL, typename TR>
void ipgen_assign(TL* lbuf, const std::vector<size_t>& ddims, const PermVector& dperm_map,
                  const PermVector& dinv_perm_map, TL rscale, TR* rbuf,
                  const std::vector<size_t>& sdims, const PermVector& sperm_map,
                  int unique_label_count) {
  std::vector<size_t> itrv(unique_label_count, 0);
  std::vector<size_t> endv(unique_label_count);
  endv = internal::perm_map_apply(ddims, dinv_perm_map);
  do {
    const auto& itval              = itrv;
    const auto& sindex             = internal::perm_map_apply(itval, sperm_map);
    const auto& dindex             = internal::perm_map_apply(itval, dperm_map);
    lbuf[ipgen_idx(dindex, ddims)] = rscale * rbuf[ipgen_idx(sindex, sdims)];
  } while(internal::cartesian_iteration(itrv, endv));
}
template<typename TL, typename TR>
void ipgen_update(TL* lbuf, const std::vector<size_t>& ddims, const PermVector& dperm_map,
                  const PermVector& dinv_perm_map, TR* rbuf, const std::vector<size_t>& sdims,
                  const PermVector& sperm_map, int unique_label_count) {
  std::vector<size_t> itrv(unique_label_count, 0);
  std::vector<size_t> endv(unique_label_count);
  endv = internal::perm_map_apply(ddims, dinv_perm_map);
  do {
    const auto& itval  = itrv;
    const auto& sindex = internal::perm_map_apply(itval, sperm_map);
    const auto& dindex = internal::perm_map_apply(itval, dperm_map);
    lbuf[ipgen_idx(dindex, ddims)] += rbuf[ipgen_idx(sindex, sdims)];
  } while(internal::cartesian_iteration(itrv, endv));
}

template<typename TL, typename TR>
void ipgen_update(TL* lbuf, const std::vector<size_t>& ddims, const PermVector& dperm_map,
                  const PermVector& dinv_perm_map, TL rscale, TR* rbuf,
                  const std::vector<size_t>& sdims, const PermVector& sperm_map,
                  int unique_label_count) {
  std::vector<size_t> itrv(unique_label_count, 0);
  std::vector<size_t> endv(unique_label_count);
  endv = internal::perm_map_apply(ddims, dinv_perm_map);
  do {
    const auto& itval  = itrv;
    const auto& sindex = internal::perm_map_apply(itval, sperm_map);
    const auto& dindex = internal::perm_map_apply(itval, dperm_map);
    lbuf[ipgen_idx(dindex, ddims)] += rscale * rbuf[ipgen_idx(sindex, sdims)];
  } while(internal::cartesian_iteration(itrv, endv));
}

template<typename TL, typename TR>
void ipgen_update(TL lscale, TL* lbuf, const std::vector<size_t>& ddims,
                  const PermVector& dperm_map, const PermVector& dinv_perm_map, TL rscale, TR* rbuf,
                  const std::vector<size_t>& sdims, const PermVector& sperm_map,
                  int unique_label_count) {
  std::vector<size_t> itrv(unique_label_count, 0);
  std::vector<size_t> endv(unique_label_count);
  endv = internal::perm_map_apply(ddims, dinv_perm_map);
  do {
    const auto& itval  = itrv;
    const auto& sindex = internal::perm_map_apply(itval, sperm_map);
    const auto& dindex = internal::perm_map_apply(itval, dperm_map);
    auto        doff   = ipgen_idx(dindex, ddims);
    lbuf[doff]         = lscale * lbuf[doff] + rscale * rbuf[ipgen_idx(sindex, sdims)];
  } while(internal::cartesian_iteration(itrv, endv));
}

///////////////////////////////////////////////////////////////////////////////
//
//                                ipgen_loop
//
///////////////////////////////////////////////////////////////////////////////

template<typename TL, typename TR>
void ipgen_loop_assign(TL* lbuf, const std::vector<int>& lld, TR* rbuf, const std::vector<int>& rld,
                       const std::vector<int>& unique_label_dims) {
  EXPECTS(lbuf != nullptr);
  EXPECTS(rbuf != nullptr);
  EXPECTS(lld.size() == rld.size());
  EXPECTS(lld.size() == unique_label_dims.size());
  const size_t ndim = unique_label_dims.size();
  if(ndim == 0) { *lbuf = *rbuf; }
  else if(ndim == 1) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      lbuf[i0doff] = rbuf[i0soff];
    }
  }
  else if(ndim == 2) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        lbuf[i1doff] = rbuf[i1soff];
      }
    }
  }
  else if(ndim == 3) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        for(int i2 = 0, i2doff = i1doff, i2soff = i1soff; i2 < unique_label_dims[2];
            ++i2, i2doff += lld[2], i2soff += rld[2]) {
          lbuf[i2doff] = rbuf[i2soff];
        }
      }
    }
  }
  else if(ndim == 4) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        for(int i2 = 0, i2doff = i1doff, i2soff = i1soff; i2 < unique_label_dims[2];
            ++i2, i2doff += lld[2], i2soff += rld[2]) {
          for(int i3 = 0, i3doff = i2doff, i3soff = i2soff; i3 < unique_label_dims[3];
              ++i3, i3doff += lld[3], i3soff += rld[3]) {
            lbuf[i3doff] = rbuf[i3soff];
          }
        }
      }
    }
  }
  else { NOT_IMPLEMENTED(); }
}

template<typename TL, typename TR>
void ipgen_loop_assign(TL* lbuf, const std::vector<int>& lld, TL rscale, TR* rbuf,
                       const std::vector<int>& rld, const std::vector<int>& unique_label_dims) {
  EXPECTS(lbuf != nullptr);
  EXPECTS(rbuf != nullptr);
  EXPECTS(lld.size() == rld.size());
  EXPECTS(lld.size() == unique_label_dims.size());
  const size_t ndim = unique_label_dims.size();
  if(ndim == 0) { *lbuf = rscale * *rbuf; }
  else if(ndim == 1) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      lbuf[i0doff] = rscale * rbuf[i0soff];
    }
  }
  else if(ndim == 2) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        lbuf[i1doff] = rscale * rbuf[i1soff];
      }
    }
  }
  else if(ndim == 3) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        for(int i2 = 0, i2doff = i1doff, i2soff = i1soff; i2 < unique_label_dims[2];
            ++i2, i2doff += lld[2], i2soff += rld[2]) {
          lbuf[i2doff] = rscale * rbuf[i2soff];
        }
      }
    }
  }
  else if(ndim == 4) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        for(int i2 = 0, i2doff = i1doff, i2soff = i1soff; i2 < unique_label_dims[2];
            ++i2, i2doff += lld[2], i2soff += rld[2]) {
          for(int i3 = 0, i3doff = i2doff, i3soff = i2soff; i3 < unique_label_dims[3];
              ++i3, i3doff += lld[3], i3soff += rld[3]) {
            lbuf[i3doff] = rscale * rbuf[i3soff];
          }
        }
      }
    }
  }
  else { NOT_IMPLEMENTED(); }
}

template<typename TL, typename TR>
void ipgen_loop_update(TL* lbuf, const std::vector<int>& lld, TR* rbuf, const std::vector<int>& rld,
                       const std::vector<int>& unique_label_dims) {
  EXPECTS(lbuf != nullptr);
  EXPECTS(rbuf != nullptr);
  EXPECTS(lld.size() == rld.size());
  EXPECTS(lld.size() == unique_label_dims.size());
  const size_t ndim = unique_label_dims.size();
  if(ndim == 0) { *lbuf += *rbuf; }
  else if(ndim == 1) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      lbuf[i0doff] += rbuf[i0soff];
    }
  }
  else if(ndim == 2) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        lbuf[i1doff] += rbuf[i1soff];
      }
    }
  }
  else if(ndim == 3) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        for(int i2 = 0, i2doff = i1doff, i2soff = i1soff; i2 < unique_label_dims[2];
            ++i2, i2doff += lld[2], i2soff += rld[2]) {
          lbuf[i2doff] += rbuf[i2soff];
        }
      }
    }
  }
  else if(ndim == 4) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        for(int i2 = 0, i2doff = i1doff, i2soff = i1soff; i2 < unique_label_dims[2];
            ++i2, i2doff += lld[2], i2soff += rld[2]) {
          for(int i3 = 0, i3doff = i2doff, i3soff = i2soff; i3 < unique_label_dims[3];
              ++i3, i3doff += lld[3], i3soff += rld[3]) {
            lbuf[i3doff] += rbuf[i3soff];
          }
        }
      }
    }
  }
  else { NOT_IMPLEMENTED(); }
}

template<typename TL, typename TR>
void ipgen_loop_update(TL* lbuf, const std::vector<int>& lld, TR rscale, const TR* rbuf,
                       const std::vector<int>& rld, const std::vector<int>& unique_label_dims) {
  EXPECTS(lbuf != nullptr);
  EXPECTS(rbuf != nullptr);
  EXPECTS(lld.size() == rld.size());
  EXPECTS(lld.size() == unique_label_dims.size());
  const size_t ndim = unique_label_dims.size();
  if(ndim == 0) { *lbuf += rscale * *rbuf; }
  else if(ndim == 1) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      lbuf[i0doff] += rscale * rbuf[i0soff];
    }
  }
  else if(ndim == 2) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        lbuf[i1doff] += rscale * rbuf[i1soff];
      }
    }
  }
  else if(ndim == 3) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        for(int i2 = 0, i2doff = i1doff, i2soff = i1soff; i2 < unique_label_dims[2];
            ++i2, i2doff += lld[2], i2soff += rld[2]) {
          lbuf[i2doff] += rscale * rbuf[i2soff];
        }
      }
    }
  }
  else if(ndim == 4) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        for(int i2 = 0, i2doff = i1doff, i2soff = i1soff; i2 < unique_label_dims[2];
            ++i2, i2doff += lld[2], i2soff += rld[2]) {
          for(int i3 = 0, i3doff = i2doff, i3soff = i2soff; i3 < unique_label_dims[3];
              ++i3, i3doff += lld[3], i3soff += rld[3]) {
            lbuf[i3doff] += rscale * rbuf[i3soff];
          }
        }
      }
    }
  }
  else if(ndim == 5) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        for(int i2 = 0, i2doff = i1doff, i2soff = i1soff; i2 < unique_label_dims[2];
            ++i2, i2doff += lld[2], i2soff += rld[2]) {
          for(int i3 = 0, i3doff = i2doff, i3soff = i2soff; i3 < unique_label_dims[3];
              ++i3, i3doff += lld[3], i3soff += rld[3]) {
            for(int i4 = 0, i4doff = i3doff, i4soff = i3soff; i4 < unique_label_dims[4];
                ++i4, i4doff += lld[4], i4soff += rld[4]) {
              lbuf[i4doff] += rscale * rbuf[i4soff];
            }
          }
        }
      }
    }
  }
  else { NOT_IMPLEMENTED(); }
}

template<typename TL, typename TR>
void ipgen_loop_update(TL lscale, TL* lbuf, const std::vector<int>& lld, TL rscale, const TR* rbuf,
                       const std::vector<int>& rld, const std::vector<int>& unique_label_dims) {
  EXPECTS(lbuf != nullptr);
  EXPECTS(rbuf != nullptr);
  EXPECTS(lld.size() == rld.size());
  EXPECTS(lld.size() == unique_label_dims.size());
  const size_t ndim = unique_label_dims.size();
  if(ndim == 0) { *lbuf = lscale * *lbuf + *rbuf; }
  else if(ndim == 1) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      lbuf[i0doff] = lscale * lbuf[i0doff] + rscale * rbuf[i0soff];
    }
  }
  else if(ndim == 2) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        lbuf[i1doff] = lscale * lbuf[i1doff] + rscale * rbuf[i1soff];
      }
    }
  }
  else if(ndim == 3) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        for(int i2 = 0, i2doff = i1doff, i2soff = i1soff; i2 < unique_label_dims[2];
            ++i2, i2doff += lld[2], i2soff += rld[2]) {
          lbuf[i2doff] = lscale * lbuf[i2doff] + rscale * rbuf[i2soff];
        }
      }
    }
  }
  else if(ndim == 4) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        for(int i2 = 0, i2doff = i1doff, i2soff = i1soff; i2 < unique_label_dims[2];
            ++i2, i2doff += lld[2], i2soff += rld[2]) {
          for(int i3 = 0, i3doff = i2doff, i3soff = i2soff; i3 < unique_label_dims[3];
              ++i3, i3doff += lld[3], i3soff += rld[3]) {
            lbuf[i3doff] = lscale * lbuf[i3doff] + rscale * rbuf[i3soff];
          }
        }
      }
    }
  }
  else if(ndim == 5) {
    for(int i0 = 0, i0doff = 0, i0soff = 0; i0 < unique_label_dims[0];
        ++i0, i0doff += lld[0], i0soff += rld[0]) {
      for(int i1 = 0, i1doff = i0doff, i1soff = i0soff; i1 < unique_label_dims[1];
          ++i1, i1doff += lld[1], i1soff += rld[1]) {
        for(int i2 = 0, i2doff = i1doff, i2soff = i1soff; i2 < unique_label_dims[2];
            ++i2, i2doff += lld[2], i2soff += rld[2]) {
          for(int i3 = 0, i3doff = i2doff, i3soff = i2soff; i3 < unique_label_dims[3];
              ++i3, i3doff += lld[3], i3soff += rld[3]) {
            for(int i4 = 0, i4doff = i3doff, i4soff = i3soff; i4 < unique_label_dims[4];
                ++i4, i4doff += lld[4], i4soff += rld[4]) {
              lbuf[i4doff] = lscale * lbuf[i4doff] + rscale * rbuf[i4soff];
            }
          }
        }
      }
    }
  }
  else { NOT_IMPLEMENTED(); }
}

///////////////////////////////////////////////////////////////////////////////
//
//                          ipgen_loop lambda
//
///////////////////////////////////////////////////////////////////////////////

template<typename Func, size_t N, typename... Bufs>
void ipgen_loop_lambda(Func&& func, const std::vector<size_t>& unique_label_dims,
                       const std::array<std::vector<size_t>, N>& blds, Bufs... bufs) {
  static_assert(N == sizeof...(Bufs));
  const size_t ndim = unique_label_dims.size();
  int          pos;
  if(ndim == 0) { func((*bufs)...); }
  else if(ndim == 1) {
    std::array<int, N> i0off = {0};
    for(int i0 = 0; i0 < unique_label_dims[0]; ++i0) {
      pos = 0, (void) func(bufs[i0off[pos++]]...);
      for(int apos = 0; apos < N; apos++) { i0off[apos] += blds[apos][0]; }
    }
  }
  else if(ndim == 2) {
    std::array<int, N> i0off = {0};
    for(int i0 = 0; i0 < unique_label_dims[0]; ++i0) {
      std::array<int, N> i1off = i0off;
      for(int i1 = 0; i1 < unique_label_dims[1]; ++i1) {
        pos = 0, (void) func(bufs[i1off[pos++]]...);
        for(int apos = 0; apos < N; apos++) { i1off[apos] += blds[apos][1]; }
      }
      for(int apos = 0; apos < N; apos++) { i0off[apos] += blds[apos][0]; }
    }
  }
  else if(ndim == 3) {
    std::array<int, N> i0off = {0};
    for(int i0 = 0; i0 < unique_label_dims[0]; ++i0) {
      std::array<int, N> i1off = i0off;
      for(int i1 = 0; i1 < unique_label_dims[1]; ++i1) {
        std::array<int, N> i2off = i1off;
        for(int i2 = 0; i2 < unique_label_dims[2]; ++i2) {
          pos = 0, (void) func(bufs[i2off[pos++]]...);
          for(int apos = 0; apos < N; apos++) { i2off[apos] += blds[apos][2]; }
        }
        for(int apos = 0; apos < N; apos++) { i1off[apos] += blds[apos][1]; }
      }
      for(int apos = 0; apos < N; apos++) { i0off[apos] += blds[apos][0]; }
    }
  }
  else if(ndim == 4) {
    std::array<int, N> i0off = {0};
    for(int i0 = 0; i0 < unique_label_dims[0]; ++i0) {
      std::array<int, N> i1off = i0off;
      for(int i1 = 0; i1 < unique_label_dims[1]; ++i1) {
        std::array<int, N> i2off = i1off;
        for(int i2 = 0; i2 < unique_label_dims[2]; ++i2) {
          std::array<int, N> i3off = i2off;
          for(int i3 = 0; i3 < unique_label_dims[3]; ++i3) {
            pos = 0, (void) func(bufs[i3off[pos++]]...);
            for(int apos = 0; apos < N; apos++) { i3off[apos] += blds[apos][3]; }
          }
          for(int apos = 0; apos < N; apos++) { i2off[apos] += blds[apos][2]; }
        }
        for(int apos = 0; apos < N; apos++) { i1off[apos] += blds[apos][1]; }
      }
      for(int apos = 0; apos < N; apos++) { i0off[apos] += blds[apos][0]; }
    }
  }
  else { NOT_IMPLEMENTED(); }
}

///////////////////////////////////////////////////////////////////////////////
//
//                         ipgen_loop plan builder
//
///////////////////////////////////////////////////////////////////////////////

template<int N>
class IpGenLoopBuilder {
public:
  IpGenLoopBuilder(): unique_label_count_{0} {}

  IpGenLoopBuilder(const IpGenLoopBuilder<N>&) = default;

  template<typename T>
  IpGenLoopBuilder(const std::array<std::vector<T>, N>& labels_arr) {
    std::vector<T> merged_labels;
    for(const auto& labels: labels_arr) {
      merged_labels.insert(merged_labels.end(), labels.begin(), labels.end());
    }
    std::vector<T> unique_labels = internal::unique_entries(merged_labels);
    unique_label_count_          = unique_labels.size();

    auto perm_dup_map_compute = [](const std::vector<T>& from, const std::vector<T>& to) {
      std::vector<int> ret(from.size(), -1);
      for(size_t i = 0; i < from.size(); i++) {
        auto it = std::find(to.begin(), to.end(), from[i]);
        if(it != to.end()) { ret[i] = it - to.begin(); }
      }
      return ret;
    };

    for(int i = 0; i < N; i++) {
      a2u_perm_map_[i] = perm_dup_map_compute(labels_arr[i], unique_labels);
    }
  }

  const std::array<std::vector<int>, N>& a2u_perm_map() const { return a2u_perm_map_; }
  int                                    unique_label_count() const { return unique_label_count_; }
  const std::vector<int>&                unique_label_dims() const { return unique_label_dims_; }
  const std::array<std::vector<int>, N>& u2ald() const { return u2ald_; }

  template<typename... BlockSpans>
  void update_plan(BlockSpans&&... blocks) {
    static_assert(sizeof...(BlockSpans) == N);
    unique_label_dims_.clear();
    unique_label_dims_.resize(unique_label_count_);
    for(int a = 0; a < N; a++) {
      u2ald_[a].clear();
      u2ald_[a].resize(unique_label_count_, 0);
    }
    ///@todo EM: changed int to size_t
    std::array<std::vector<size_t>, N> bdims = {blocks.block_dims()...};
    std::array<std::vector<int>, N>    blds;

    auto exclusive_product = [](const std::vector<size_t>& vec) {
      std::vector<int> ret(vec.size(), 1);
      std::partial_sum(vec.rbegin(), vec.rend(), ret.rbegin(), std::multiplies<int>{});
      for(size_t i = 0; i < vec.size(); i++) {
        if(vec[i] != 0) { ret[i] /= vec[i]; }
      }
      return ret;
    };
    for(int i = 0; i < N; i++) { blds[i] = exclusive_product(bdims[i]); }
    // std::exclusive_scan(ddims.rbegin(), ddims.rend(), dld.rbegin(),
    //                     std::multiplies<int>{});
    for(int a = 0; a < N; a++) {
      for(size_t i = 0; i < blds[a].size(); i++) {
        if(a2u_perm_map_[a][i] >= 0) {
          // EXPECTS(a2u_perm_map_[a][i] < unique_label_count_);
          unique_label_dims_[a2u_perm_map_[a][i]] = bdims[a][i];
          u2ald_[a][a2u_perm_map_[a][i]] += blds[a][i];
        }
      }
    }
  }

private:
  std::array<std::vector<int>, N> a2u_perm_map_;
  int                             unique_label_count_;
  std::vector<int>                unique_label_dims_;
  std::array<std::vector<int>, N> u2ald_;
}; // struct IpGenLoopBuilder

} // namespace tamm::blockops::cpu
