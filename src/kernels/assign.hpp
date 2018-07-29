
#include <vector>
#include <cassert>
#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <chrono>

#define NOT_IMPLEMENTED() assert(0)
#define EXPECTS(cond) assert(cond)

namespace internal {

template<typename T>
void ip0(const std::vector<size_t>& /*loop_dims*/, T* dst, const std::vector<size_t>& /*loop_dld*/,
         T scale, const T* src, const std::vector<size_t>& /*loop_sld*/) {
  dst[0] = scale * src[0];
}

template<typename T>
void ip1(const std::vector<size_t>& loop_dims, T* dst, const std::vector<size_t>& loop_dld,
         T scale, const T* src, const std::vector<size_t>& loop_sld) {
  const size_t ndim = 1;
  size_t soff[ndim], doff[ndim];
  size_t i[ndim];

  for(i[0] = 0, soff[0]=0, doff[0]=0; i[0] < loop_dims[0]; i[0]++, soff[0]+=loop_sld[0], doff[0]+=loop_dld[0]) {
    dst[doff[0]] = scale * src[soff[0]];
  }
}

template<typename T>
void ip2(const std::vector<size_t>& loop_dims, T* dst, const std::vector<size_t>& loop_dld,
         T scale, const T* src, const std::vector<size_t>& loop_sld) {
  const size_t ndim = 2;
  size_t soff[ndim], doff[ndim];
  size_t i[ndim];

  for(i[0] = 0, soff[0]=0, doff[0]=0; i[0] < loop_dims[0]; i[0]++, soff[0]+=loop_sld[0], doff[0]+=loop_dld[0]) {
    for(i[1] = 0, soff[1]=soff[0], doff[1]=doff[0]; i[1] < loop_dims[1]; i[1]++, soff[1]+=loop_sld[1], doff[1]+=loop_dld[1]) {
      dst[doff[1]] = scale * src[soff[1]];
    }
  }
}

template<typename T>
void ip3(const std::vector<size_t>& loop_dims, T* dst, const std::vector<size_t>& loop_dld,
         T scale, const T* src, const std::vector<size_t>& loop_sld) {
  const size_t ndim = 3;
  size_t soff[ndim], doff[ndim];
  size_t i[ndim];

  for(i[0] = 0, soff[0]=0, doff[0]=0; i[0] < loop_dims[0]; i[0]++, soff[0]+=loop_sld[0], doff[0]+=loop_dld[0]) {
    for(i[1] = 0, soff[1]=soff[0], doff[1]=doff[0]; i[1] < loop_dims[1]; i[1]++, soff[1]+=loop_sld[1], doff[1]+=loop_dld[1]) {
      for(i[2] = 0, soff[2]=soff[1], doff[2]=doff[1]; i[2] < loop_dims[2]; i[2]++, soff[2]+=loop_sld[2], doff[2]+=loop_dld[2]) {
        dst[doff[2]] = scale * src[soff[2]];
      }
    }
  }
}

template<typename T>
void ip4(const std::vector<size_t>& loop_dims, T* dst, const std::vector<size_t>& loop_dld,
         T scale, const T* src, const std::vector<size_t>& loop_sld) {
  const size_t ndim = 4;
  size_t soff[ndim], doff[ndim];
  size_t i[ndim];

  for(i[0] = 0, soff[0]=0, doff[0]=0; i[0] < loop_dims[0]; i[0]++, soff[0]+=loop_sld[0], doff[0]+=loop_dld[0]) {
    for(i[1] = 0, soff[1]=soff[0], doff[1]=doff[0]; i[1] < loop_dims[1]; i[1]++, soff[1]+=loop_sld[1], doff[1]+=loop_dld[1]) {
      for(i[2] = 0, soff[2]=soff[1], doff[2]=doff[1]; i[2] < loop_dims[2]; i[2]++, soff[2]+=loop_sld[2], doff[2]+=loop_dld[2]) {
        for(i[3] = 0, soff[3]=soff[2], doff[3]=doff[2]; i[3] < loop_dims[3]; i[3]++, soff[3]+=loop_sld[3], doff[3]+=loop_dld[3]) {
          dst[doff[3]] = scale * src[soff[3]];
        }
      }
    }
  }
}

template<typename T>
void ipacc0(const std::vector<size_t>& /*loop_dims*/, T* dst, const std::vector<size_t>& /*loop_dld*/,
            T scale, const T* src, const std::vector<size_t>& /*loop_sld*/) {
  dst[0] += scale * src[0];
}

template<typename T>
void ipacc1(const std::vector<size_t>& loop_dims, T* dst, const std::vector<size_t>& loop_dld,
            T scale, const T* src, const std::vector<size_t>& loop_sld) {
  const size_t ndim = 1;
  size_t soff[ndim], doff[ndim];
  size_t i[ndim];

  for(i[0] = 0, soff[0]=0, doff[0]=0; i[0] < loop_dims[0]; i[0]++, soff[0]+=loop_sld[0], doff[0]+=loop_dld[0]) {
    dst[doff[0]] += scale * src[soff[0]];
  }
}

template<typename T>
void ipacc2(const std::vector<size_t>& loop_dims, T* dst, const std::vector<size_t>& loop_dld,
            T scale, const T* src, const std::vector<size_t>& loop_sld) {
  const size_t ndim = 2;
  size_t soff[ndim], doff[ndim];
  size_t i[ndim];

  for(i[0] = 0, soff[0]=0, doff[0]=0; i[0] < loop_dims[0]; i[0]++, soff[0]+=loop_sld[0], doff[0]+=loop_dld[0]) {
    for(i[1] = 0, soff[1]=soff[0], doff[1]=doff[0]; i[1] < loop_dims[1]; i[1]++, soff[1]+=loop_sld[1], doff[1]+=loop_dld[1]) {
      dst[doff[1]] += scale * src[soff[1]];
    }
  }
}

template<typename T>
void ipacc3(const std::vector<size_t>& loop_dims, T* dst, const std::vector<size_t>& loop_dld,
            T scale, const T* src, const std::vector<size_t>& loop_sld) {
  const size_t ndim = 3;
  size_t soff[ndim], doff[ndim];
  size_t i[ndim];

  for(i[0] = 0, soff[0]=0, doff[0]=0; i[0] < loop_dims[0]; i[0]++, soff[0]+=loop_sld[0], doff[0]+=loop_dld[0]) {
    for(i[1] = 0, soff[1]=soff[0], doff[1]=doff[0]; i[1] < loop_dims[1]; i[1]++, soff[1]+=loop_sld[1], doff[1]+=loop_dld[1]) {
      for(i[2] = 0, soff[2]=soff[1], doff[2]=doff[1]; i[2] < loop_dims[2]; i[2]++, soff[2]+=loop_sld[2], doff[2]+=loop_dld[2]) {
        dst[doff[2]] += scale * src[soff[2]];
      }
    }
  }
}

template<typename T>
void ipacc4(const std::vector<size_t>& loop_dims, T* dst, const std::vector<size_t>& loop_dld,
            T scale, const T* src, const std::vector<size_t>& loop_sld) {
  const size_t ndim = 4;
  size_t soff[ndim], doff[ndim];
  size_t i[ndim];

  for(i[0] = 0, soff[0]=0, doff[0]=0; i[0] < loop_dims[0]; i[0]++, soff[0]+=loop_sld[0], doff[0]+=loop_dld[0]) {
    for(i[1] = 0, soff[1]=soff[0], doff[1]=doff[0]; i[1] < loop_dims[1]; i[1]++, soff[1]+=loop_sld[1], doff[1]+=loop_dld[1]) {
      for(i[2] = 0, soff[2]=soff[1], doff[2]=doff[1]; i[2] < loop_dims[2]; i[2]++, soff[2]+=loop_sld[2], doff[2]+=loop_dld[2]) {
        for(i[3] = 0, soff[3]=soff[2], doff[3]=doff[2]; i[3] < loop_dims[3]; i[3]++, soff[3]+=loop_sld[3], doff[3]+=loop_dld[3]) {
          dst[doff[3]] += scale * src[soff[3]];
        }
      }
    }
  }
}

} // namaepsce internal


//////////////////////////////////////////////////////////////////////////

namespace internal {
using PermVector = std::vector<size_t>;
using IndexLabelVec = std::vector<size_t>;

inline size_t
idx(int n, const size_t *id, const size_t *sz, const PermVector& p) {
  size_t idx = 0;
  for (int i = 0; i < n - 1; i++) {
    idx = (idx + id[p[i]]) * sz[p[i + 1]];
  }
  if (n > 0) {
    idx += id[p[n - 1]];
  }
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
        for(i[2]=0; i[2]<sz[2]; i[2]++, c++) {
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
std::vector<size_t> perm_map_compute(const std::vector<T>& unique_vec,
                                     const std::vector<T>& vec_required) {
  std::vector<size_t> ret;
  for(const auto& val : vec_required) {
    auto it = std::find(unique_vec.begin(), unique_vec.end(), val);
    EXPECTS(it >= unique_vec.begin());
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

template<typename T, typename Integer>
void perm_map_apply(std::vector<T>& out_vec, const std::vector<T>& input_vec,
                    const std::vector<Integer>& perm_map) {
  out_vec.resize(perm_map.size());
  for(size_t i=0; i<perm_map.size(); i++) {
    EXPECTS(perm_map[i] < input_vec.size());
    out_vec[i] = input_vec[perm_map[i]];
  }
}


template<typename T>
bool cartesian_iteration(std::vector<T>& itr, const std::vector<T>& end) {
  EXPECTS(itr.size() == end.size());
  // if(!std::lexicographical_compare(itr.begin(), itr.end(), end.begin(),
  //                                  end.end())) {
  //     return false;
  // }
  int i;
  for(i = -1 + itr.size(); i>=0 && itr[i]+1 == end[i]; i--) {
    itr[i] = T{0};        
  }
  // EXPECTS(itr.size() == 0 || i>=0);
  if(i>=0) {
    ++itr[i];
    return true;
  }
  return false;
}


template<typename T>
std::vector<T> unique_entries(const std::vector<T>& input_vec) {
    std::vector<T> ret;
    for(const auto& val : input_vec) {
        auto it = std::find(ret.begin(), ret.end(), val);
        if(it == ret.end()) { ret.push_back(val); }
    }
    return ret;
}

template<typename T>
void ip_gen(T* dst, const std::vector<size_t>& ddims, const std::vector<size_t>& dlabels,
            T scale, const T* src, const std::vector<size_t>& sdims, const std::vector<size_t>& slabels,
            bool is_assign=true) {
  IndexLabelVec unique_labels = unique_entries(dlabels);
  //unique_labels = sort_on_dependence(unique_labels);
  // std::sort(unique_labels.begin(), unique_labels.end());
  // std::unique(unique_labels.begin(), unique_labels.end());
  const auto& dperm_map = perm_map_compute(unique_labels, dlabels);
  const auto& sperm_map = perm_map_compute(unique_labels, slabels);
  const auto& dinv_pm = perm_map_compute(dlabels, unique_labels);

  auto idx = [](const auto& index_vec, const auto& dims_vec) {
    size_t ret = 0, ld = 1;
    EXPECTS(index_vec.size() == dims_vec.size());
    for(int i = index_vec.size(); i >= 0; i--) {
      ret += ld * index_vec[i];
      ld *= dims_vec[i];
    }
    return ret;
  };


  // std::vector<IndexLoopBound> ilbs;
  // for(const auto& lbl : unique_labels) { ilbs.push_back({lbl}); }
  // IndexLoopNest iln = IndexLoopNest{ilbs};
  std::vector<size_t> itrv(unique_labels.size(), 0);
  std::vector<size_t> endv(unique_labels.size());
  endv = internal::perm_map_apply(ddims, dinv_pm);
  do {
    const auto& itval = itrv;
    const auto& sindex = perm_map_apply(itval, sperm_map);
    const auto& dindex = perm_map_apply(itval, dperm_map);
    //if(!update) {
#if 0
    dst[idx(dindex, ddims)] = scale * src[idx(sindex, sdims)];
#endif
    //} else {
      //dbuf[idx(dindex, ddims)] += scale * sbuf[idx(sindex, sdims)];
      //}
  } while(internal::cartesian_iteration(itrv, endv));
}

} // namespace internal

//////////////////////////////////////////////////////////////////////////

template<typename T>
void ip(T* dst, const std::vector<size_t>& ddims, const std::vector<size_t>& dlabels,
        T scale, const T* src, const std::vector<size_t>& sdims, const std::vector<size_t>& slabels,
        bool is_assign=true) {
  const size_t ndim = ddims.size();

  assert(ddims.size() == sdims.size());
  assert(ddims.size() == dlabels.size());
  assert(sdims.size() == slabels.size());

  std::vector<size_t> sld{sdims}, dld{ddims};
  sld.insert(sld.end(), 1);
  dld.insert(dld.end(), 1);
  std::partial_sum(sld.rbegin(), sld.rend(), sld.rbegin(), std::multiplies<T>());
  std::partial_sum(dld.rbegin(), dld.rend(), dld.rbegin(), std::multiplies<T>());
    
  std::vector<size_t> loop_labels;
  for(const auto& lbl: dlabels) {
    if(std::find(loop_labels.begin(), loop_labels.end(), lbl) == loop_labels.end()) {
      loop_labels.push_back(lbl);
    }
  }
  for(const auto& lbl: slabels) {
    if(std::find(loop_labels.begin(), loop_labels.end(), lbl) == loop_labels.end()) {
      loop_labels.push_back(lbl);
    }
  }
  std::vector<size_t> loop_dims(loop_labels.size()), loop_sld(loop_labels.size()), loop_dld(loop_labels.size());
  for(size_t i=0; i<loop_labels.size(); i++) {
    const auto& lbl = loop_labels[i];
    auto sit = std::find(slabels.begin(), slabels.end(), lbl);
    if(sit != slabels.end()) {
      loop_sld[i] = sld[sit - slabels.begin()+1];
      loop_dims[i] = sdims[sit - slabels.begin()];
    }
  }

  for(size_t i=0; i<loop_labels.size(); i++) {
    const auto& lbl = loop_labels[i];
    auto dit = std::find(dlabels.begin(), dlabels.end(), lbl);
    if(dit != dlabels.end()) {
      loop_dld[i] = dld[dit - dlabels.begin()+1];
      loop_dims[i] = ddims[dit - dlabels.begin()];
    }
  }

  // std::cerr<<"loop labels =[";
  // for(const auto v : loop_labels) {
  //   std::cerr<<v<<" ";
  // }
  // std::cerr<<"]\n";

  // std::cerr<<"loop dims =[";
  // for(const auto v : loop_dims) {
  //   std::cerr<<v<<" ";
  // }
  // std::cerr<<"]\n";
  
  // std::cerr<<"loop_dld =[";
  // for(const auto v : loop_dld) {
  //   std::cerr<<v<<" ";
  // }
  // std::cerr<<"]\n";

  // std::cerr<<"loop_sld =[";
  // for(const auto v : loop_sld) {
  //   std::cerr<<v<<" ";
  // }
  // std::cerr<<"]\n";

#if 0
  auto perm_to_dest = internal::perm_compute(dlabels, slabels);
  if(is_assign) {
    internal::index_permute(dst, src, perm_to_dest, ddims, scale);
  } else {
    NOT_IMPLEMENTED();
    //internal::index_permute_acc(dst, src, perm_to_dest, ddims, scale);
  }
#elif 1
  internal::ip_gen(dst, ddims, dlabels, scale, src, sdims, slabels, is_assign);
#else
  if(is_assign) {
    if(ndim == 0) {
      internal::ip0(loop_dims, dst, loop_dld, scale, src, loop_sld);
    } else if(ndim == 1) {
      internal::ip1(loop_dims, dst, loop_dld, scale, src, loop_sld);
    } else if(ndim == 2) {
      internal::ip2(loop_dims, dst, loop_dld, scale, src, loop_sld);
    } else if(ndim == 3) {
      internal::ip3(loop_dims, dst, loop_dld, scale, src, loop_sld);
    } else if(ndim == 4) {
      internal::ip4(loop_dims, dst, loop_dld, scale, src, loop_sld);
    } else {
      NOT_IMPLEMENTED();
    }
  } else {
    if(ndim == 0) {
      internal::ipacc0(loop_dims, dst, loop_dld, scale, src, loop_sld);
    } else if(ndim == 1) {
      internal::ipacc1(loop_dims, dst, loop_dld, scale, src, loop_sld);
    } else if(ndim == 2) {
      internal::ipacc2(loop_dims, dst, loop_dld, scale, src, loop_sld);
    } else if(ndim == 3) {
      internal::ipacc3(loop_dims, dst, loop_dld, scale, src, loop_sld);
    } else if(ndim == 4) {
      internal::ipacc4(loop_dims, dst, loop_dld, scale, src, loop_sld);
    } else {
      NOT_IMPLEMENTED();
    }
  }
#endif
}

int main() {
  int ret = 0;
  std::vector<double> src(N*N*N*N, 1.4), dst(N*N*N*N, 3.2);

  {
    auto start = std::chrono::high_resolution_clock::now();
    //ip(dst.data(),{N*N*N*N}, {0,1,2,3}, 1.0, src.data(), {N*N*N*N}, {0,1,2,3});
    ip(dst.data(),{N*N*N*N}, {0}, 8.1, src.data(), {N*N*N*N}, {0});
    auto end = std::chrono::high_resolution_clock::now();
    std::cerr<<"Time = "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<< "ms.\n";
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    //ip(dst.data(),{N*N*N*N}, {0,1,2,3}, 1.0, src.data(), {N*N*N*N}, {0,1,2,3});
    ip(dst.data(),{N*N,N*N}, {0,1}, 8.1, src.data(), {N*N,N*N}, {0,1});
    auto end = std::chrono::high_resolution_clock::now();
    std::cerr<<"Time = "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<< "ms.\n";
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    //ip(dst.data(),{N*N*N*N}, {0,1,2,3}, 1.0, src.data(), {N*N*N*N}, {0,1,2,3});
    ip(dst.data(),{N*N,N*N}, {1,0}, 8.1, src.data(), {N*N,N*N}, {1,0});
    auto end = std::chrono::high_resolution_clock::now();
    std::cerr<<"Time = "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<< "ms.\n";
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    //ip(dst.data(),{N*N*N*N}, {0,1,2,3}, 1.0, src.data(), {N*N*N*N}, {0,1,2,3});
    ip(dst.data(),{N*N,N*N}, {0,1}, 8.1, src.data(), {N*N,N*N}, {1,0});
    auto end = std::chrono::high_resolution_clock::now();
    std::cerr<<"Time = "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<< "ms.\n";
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    //ip(dst.data(),{N*N*N*N}, {0,1,2,3}, 1.0, src.data(), {N*N*N*N}, {0,1,2,3});
    ip(dst.data(),{N*N,N*N}, {1,0}, 8.1, src.data(), {N*N,N*N}, {0,1});
    auto end = std::chrono::high_resolution_clock::now();
    std::cerr<<"Time = "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<< "ms.\n";
  }

  return int(dst[0]);
}

#undef N

#if 0
template<typename T>
void gemm_wrapper(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                  const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                  const int K, T alpha, const T* A,
                  const int lda, const T* B, const int ldb,
                  T beta, T* C, const int ldc);

template<>
void gemm_wrapper<double>(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                          const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                          const int K, double alpha, const double* A,
                          const int lda, const double* B, const int ldb,
                          double beta, double* C, const int ldc) {
  cblas_dgemm(Order, TransA, TransB,
              M, N, K,
              alpha, A, lda,
              B, ldb,
              beta, C, ldc);
}

template<>
void gemm_wrapper<float>(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                         const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                         const int K, double alpha, const double* A,
                         const int lda, const double* B, const int ldb,
                         double beta, double* C, const int ldc) {
  cblas_sgemm(Order, TransA, TransB,
              M, N, K,
              alpha, A, lda,
              B, ldb,
              beta, C, ldc);
}

template<>
void gemm_wrapper<std::complex<float>>(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                                       const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                                       const int K, double alpha, const double* A,
                                       const int lda, const double* B, const int ldb,
                                       double beta, double* C, const int ldc) {
  cblas_zgemm(Order, TransA, TransB,
              M, N, K,
              alpha, A, lda,
              B, ldb,
              beta, C, ldc);
}
#endif

/*
void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
const int K, const double alpha, const double *A,
const int lda, const double *B, const int ldb,
const double beta, double *C, const int ldc);
*/
template<typename T>
void block_mult(T alpha, const T* abuf, const std::vector<size_t>& adims,
                const std::vector<size_t>& alabels, T beta, const T* bbuf, const std::vector<size_t>& bdims,
                const std::vector<size_t>& blabels, T* cbuf, const std::vector<T>& cdims, const std::vector<size_t>& clabels) {
  const size_t asize = std::accumulate(adims.begin(), adims.end(), 1, std::multiplies<size_t>());
  const size_t bsize = std::accumulate(bdims.begin(), bdims.end(), 1, std::multiplies<size_t>());
  const size_t csize = std::accumulate(cdims.begin(), cdims.end(), 11, std::multiplies<size_t>());

  EXPECTS(abuf!=nullptr && bbuf!=nullptr && cbuf!=nullptr);
  
  std::vector<size_t> asorted_labels{alabels}, bsorted_labels{blabels}, csorted_labels{clabels};
  std::sort(asorted_labels.begin(), asorted_labels.end());
  std::sort(bsorted_labels.begin(), bsorted_labels.end());
  std::sort(csorted_labels.begin(), csorted_labels.end());

  std::vector<size_t> inner_labels, aouter_labels, bouter_labels, batch_labels;
  std::vector<size_t> inner_dims, aouter_dims, bouter_dims, batch_dims;

  int B=1, M=1, N=1, K=1;
  for(size_t i=0; i<cdims.size(); i++) {
    const auto& lbl = clabels[i];
    bool is_in_a = std::binary_search(asorted_labels.begin(), asorted_labels.end(), lbl);
    bool is_in_b = std::binary_search(bsorted_labels.begin(), bsorted_labels.end(), lbl);
    if(is_in_a && is_in_b) {
      batch_labels.push_back(lbl);
      batch_dims.push_back(cdims[i]);
      B *= cdims[i];
    } else if(is_in_a) {
      aouter_labels.push_back(lbl);
      aouter_dims.push_back(cdims[i]);
      M *= cdims[i];
    } else if (is_in_b) {
      bouter_labels.push_back(lbl);
      bouter_dims.push_back(cdims[i]);
      N *= cdims[i];
    } else {
      assert(0); //should not be reachable
    }
  }

  for(size_t i=0; i<adims.size(); i++) {
    const auto& lbl = alabels[i];
    bool is_in_b = std::binary_search(bsorted_labels.begin(), bsorted_labels.end(), lbl);
    bool is_in_c = std::binary_search(csorted_labels.begin(), csorted_labels.end(), lbl);
    if(is_in_b && is_in_c) {
      //already added in batch_labels
    } else if(is_in_b) {
      inner_labels.push_back(lbl);
      inner_dims.push_back(adims[i]);
      K *= adims[i];
    } else if (is_in_c) {
      //already added to aouter
    } else {
      assert(0); //should not be reachable
    }
  }

  std::vector<size_t> ainter_labels{batch_labels};
  ainter_labels.insert(ainter_labels.end(), aouter_labels.begin(), aouter_labels.end());
  ainter_labels.insert(ainter_labels.end(), inner_labels.begin(), inner_labels.end());

  std::vector<size_t> binter_labels{batch_labels};
  binter_labels.insert(binter_labels.end(), inner_labels.begin(), inner_labels.end());
  binter_labels.insert(binter_labels.end(), bouter_labels.begin(), bouter_labels.end());

  std::vector<size_t> cinter_labels{batch_labels};
  cinter_labels.insert(cinter_labels.end(), aouter_labels.begin(), aouter_labels.end());
  cinter_labels.insert(cinter_labels.end(), bouter_labels.begin(), bouter_labels.end());

  std::vector<size_t> ainter_dims{batch_dims};
  ainter_dims.insert(ainter_dims.end(), aouter_dims.begin(), aouter_dims.end());
  ainter_dims.insert(ainter_dims.end(), inner_dims.begin(), inner_dims.end());

  std::vector<size_t> binter_dims{batch_dims};
  binter_dims.insert(binter_dims.end(), inner_dims.begin(), inner_dims.end());
  binter_dims.insert(binter_dims.end(), bouter_dims.begin(), bouter_dims.end());

  std::vector<size_t> cinter_dims{batch_dims};
  cinter_dims.insert(cinter_dims.end(), aouter_dims.begin(), aouter_dims.end());
  cinter_dims.insert(cinter_dims.end(), bouter_dims.begin(), bouter_dims.end());
  
  std::vector<T> ainter_buf(asize), binter_buf(bsize), cinter_buf(csize);  
  ip(ainter_buf, ainter_dims, ainter_labels, 1.0, abuf, adims, alabels, true);
  ip(binter_buf, binter_dims, binter_labels, 1.0, bbuf, bdims, blabels, true);


#if 0
  auto transA = CblasNoTrans;
  auto transB = CblasNoTrans;
#endif
  int ainter_ld = K;
  int binter_ld = N;
  int cinter_ld = N;
  int batch_ld = M*N*K;
  
  //dgemm
  for(size_t i=0; i<B; i++) {
#if 0
    cblas_dgemm(CblasRowMajor, transA, transB,
                M, N, K, 
                alpha, ainter_buf.get.() + i *batch_ld,
                ainter_ld, binter_buf.get() + i*batch_ld, binter_ld,
                beta, cbuf.get() + i*batch_ld, cinter_ld);
#endif
  }
  ip(cbuf, cdims, clabels, 1.0, cinter_buf, cinter_dims, cinter_labels, true);
}

