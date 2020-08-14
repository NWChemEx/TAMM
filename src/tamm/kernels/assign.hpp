#ifndef TAMM_KERNELS_ASSIGN_H_
#define TAMM_KERNELS_ASSIGN_H_

#include "tamm/errors.hpp"
#include "tamm/types.hpp"
#include "tamm/utils.hpp"

#include "hptt/hptt.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

namespace tamm {

namespace internal {

template<typename T>
void ip0(const SizeVec& /*loop_dims*/, T* dst, const SizeVec& /*loop_dld*/,
         T scale, const T* src, const SizeVec& /*loop_sld*/) {
    dst[0] = scale * src[0];
}

template<typename T>
void ip1(const SizeVec& loop_dims, T* dst, const SizeVec& loop_dld, T scale,
         const T* src, const SizeVec& loop_sld) {
    const size_t ndim = 1;
    Size soff[ndim], doff[ndim];
    size_t i[ndim];

    for(i[0] = 0, soff[0] = 0, doff[0] = 0; i[0] < loop_dims[0];
        i[0]++, soff[0] += loop_sld[0], doff[0] += loop_dld[0]) {
        dst[doff[0].value()] = scale * src[soff[0].value()];
    }
}

template<typename T>
void ip2(const SizeVec& loop_dims, T* dst, const SizeVec& loop_dld, T scale,
         const T* src, const SizeVec& loop_sld) {
    const size_t ndim = 2;
    Size soff[ndim], doff[ndim];
    size_t i[ndim];

    for(i[0] = 0, soff[0] = 0, doff[0] = 0; i[0] < loop_dims[0];
        i[0]++, soff[0] += loop_sld[0], doff[0] += loop_dld[0]) {
        for(i[1] = 0, soff[1] = soff[0], doff[1] = doff[0]; i[1] < loop_dims[1];
            i[1]++, soff[1] += loop_sld[1], doff[1] += loop_dld[1]) {
            dst[doff[1].value()] = scale * src[soff[1].value()];
        }
    }
}

template<typename T>
void ip3(const SizeVec& loop_dims, T* dst, const SizeVec& loop_dld, T scale,
         const T* src, const SizeVec& loop_sld) {
    const size_t ndim = 3;
    Size soff[ndim], doff[ndim];
    size_t i[ndim];

    for(i[0] = 0, soff[0] = 0, doff[0] = 0; i[0] < loop_dims[0];
        i[0]++, soff[0] += loop_sld[0], doff[0] += loop_dld[0]) {
        for(i[1] = 0, soff[1] = soff[0], doff[1] = doff[0]; i[1] < loop_dims[1];
            i[1]++, soff[1] += loop_sld[1], doff[1] += loop_dld[1]) {
            for(i[2] = 0, soff[2] = soff[1], doff[2] = doff[1];
                i[2] < loop_dims[2];
                i[2]++, soff[2] += loop_sld[2], doff[2] += loop_dld[2]) {
                dst[doff[2].value()] = scale * src[soff[2].value()];
            }
        }
    }
}

template<typename T>
void ip4(const SizeVec& loop_dims, T* dst, const SizeVec& loop_dld, T scale,
         const T* src, const SizeVec& loop_sld) {
    const size_t ndim = 4;
    Size soff[ndim], doff[ndim];
    size_t i[ndim];

    for(i[0] = 0, soff[0] = 0, doff[0] = 0; i[0] < loop_dims[0];
        i[0]++, soff[0] += loop_sld[0], doff[0] += loop_dld[0]) {
        for(i[1] = 0, soff[1] = soff[0], doff[1] = doff[0]; i[1] < loop_dims[1];
            i[1]++, soff[1] += loop_sld[1], doff[1] += loop_dld[1]) {
            for(i[2] = 0, soff[2] = soff[1], doff[2] = doff[1];
                i[2] < loop_dims[2];
                i[2]++, soff[2] += loop_sld[2], doff[2] += loop_dld[2]) {
                for(i[3] = 0, soff[3] = soff[2], doff[3] = doff[2];
                    i[3] < loop_dims[3];
                    i[3]++, soff[3] += loop_sld[3], doff[3] += loop_dld[3]) {
                    dst[doff[3].value()] = scale * src[soff[3].value()];
                }
            }
        }
    }
}

template<typename T>
void ipacc0(const SizeVec& /*loop_dims*/, T* dst, const SizeVec& /*loop_dld*/,
            T scale, const T* src, const SizeVec& /*loop_sld*/) {
    dst[0] += scale * src[0];
}

template<typename T>
void ipacc1(const SizeVec& loop_dims, T* dst, const SizeVec& loop_dld, T scale,
            const T* src, const SizeVec& loop_sld) {
    const size_t ndim = 1;
    Size soff[ndim], doff[ndim];
    size_t i[ndim];

    for(i[0] = 0, soff[0] = 0, doff[0] = 0; i[0] < loop_dims[0];
        i[0]++, soff[0] += loop_sld[0], doff[0] += loop_dld[0]) {
        dst[doff[0].value()] += scale * src[soff[0].value()];
    }
}

template<typename T>
void ipacc2(const SizeVec& loop_dims, T* dst, const SizeVec& loop_dld, T scale,
            const T* src, const SizeVec& loop_sld) {
    const size_t ndim = 2;
    Size soff[ndim], doff[ndim];
    size_t i[ndim];

    for(i[0] = 0, soff[0] = 0, doff[0] = 0; i[0] < loop_dims[0];
        i[0]++, soff[0] += loop_sld[0], doff[0] += loop_dld[0]) {
        for(i[1] = 0, soff[1] = soff[0], doff[1] = doff[0]; i[1] < loop_dims[1];
            i[1]++, soff[1] += loop_sld[1], doff[1] += loop_dld[1]) {
            dst[doff[1].value()] += scale * src[soff[1].value()];
        }
    }
}

template<typename T>
void ipacc3(const SizeVec& loop_dims, T* dst, const SizeVec& loop_dld, T scale,
            const T* src, const SizeVec& loop_sld) {
    const size_t ndim = 3;
    Size soff[ndim], doff[ndim];
    size_t i[ndim];

    for(i[0] = 0, soff[0] = 0, doff[0] = 0; i[0] < loop_dims[0];
        i[0]++, soff[0] += loop_sld[0], doff[0] += loop_dld[0]) {
        for(i[1] = 0, soff[1] = soff[0], doff[1] = doff[0]; i[1] < loop_dims[1];
            i[1]++, soff[1] += loop_sld[1], doff[1] += loop_dld[1]) {
            for(i[2] = 0, soff[2] = soff[1], doff[2] = doff[1];
                i[2] < loop_dims[2];
                i[2]++, soff[2] += loop_sld[2], doff[2] += loop_dld[2]) {
                dst[doff[2].value()] += scale * src[soff[2].value()];
            }
        }
    }
}

template<typename T>
void ipacc4(const SizeVec& loop_dims, T* dst, const SizeVec& loop_dld, T scale,
            const T* src, const SizeVec& loop_sld) {
    const size_t ndim = 4;
    Size soff[ndim], doff[ndim];
    size_t i[ndim];

    for(i[0] = 0, soff[0] = 0, doff[0] = 0; i[0] < loop_dims[0];
        i[0]++, soff[0] += loop_sld[0], doff[0] += loop_dld[0]) {
        for(i[1] = 0, soff[1] = soff[0], doff[1] = doff[0]; i[1] < loop_dims[1];
            i[1]++, soff[1] += loop_sld[1], doff[1] += loop_dld[1]) {
            for(i[2] = 0, soff[2] = soff[1], doff[2] = doff[1];
                i[2] < loop_dims[2];
                i[2]++, soff[2] += loop_sld[2], doff[2] += loop_dld[2]) {
                for(i[3] = 0, soff[3] = soff[2], doff[3] = doff[2];
                    i[3] < loop_dims[3];
                    i[3]++, soff[3] += loop_sld[3], doff[3] += loop_dld[3]) {
                    dst[doff[3].value()] += scale * src[soff[3].value()];
                }
            }
        }
    }
}

inline size_t idx(int n, const size_t* id, const Size* sz,
                  const PermVector& p) {
    Size idx = 0;
    for(int i = 0; i < n - 1; i++) { idx = (idx + id[p[i]]) * sz[p[i + 1]]; }
    if(n > 0) { idx += id[p[n - 1]]; }
    return idx.value();
}

template<typename T>
void index_permute_acc(T* dbuf, const T* sbuf, const PermVector& perm_to_dest,
                       const SizeVec& ddims, T scale) {
    EXPECTS(dbuf != nullptr && sbuf != nullptr);
    EXPECTS(perm_to_dest.size() == ddims.size());

    const size_t ndim = perm_to_dest.size();
    EXPECTS(ddims.size() == ndim);

    if(ndim == 0) {
        dbuf[0] += scale * sbuf[0];
    } else if(ndim == 1) {
        for(size_t i = 0; i < ddims[0]; i++) { dbuf[i] = scale * sbuf[i]; }
    } else if(ndim == 2) {
        Size sz[] = {ddims[0], ddims[1]};
        size_t i[2];
        size_t c;
        for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
            for(i[1] = 0; i[1] < sz[1]; i[1]++, c++) {
                dbuf[c] += scale * sbuf[idx(2, i, sz, perm_to_dest)];
            }
        }
    } else if(ndim == 3) {
        Size sz[] = {ddims[0], ddims[1], ddims[2]};
        size_t i[3];
        size_t c;
        for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
            for(i[1] = 0; i[1] < sz[1]; i[1]++) {
                for(i[2] = 0; i[2] < sz[2]; i[2]++, c++) {
                    dbuf[c] += scale * sbuf[idx(3, i, sz, perm_to_dest)];
                }
            }
        }
    } else if(ndim == 4) {
        Size sz[] = {ddims[0], ddims[1], ddims[2], ddims[3]};
        size_t i[4];
        size_t c;
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

template<typename T>
void index_permute(T* dbuf, const T* sbuf, const PermVector& perm_to_dest,
                   const SizeVec& ddims, T scale) {
    EXPECTS(dbuf != nullptr && sbuf != nullptr);
    EXPECTS(perm_to_dest.size() == ddims.size());

    const size_t ndim = perm_to_dest.size();
    EXPECTS(ddims.size() == ndim);

    if(ndim == 0) {
        dbuf[0] = scale * sbuf[0];
    } else if(ndim == 1) {
        for(size_t i = 0; i < ddims[0]; i++) { dbuf[i] = scale * sbuf[i]; }
    } else if(ndim == 2) {
        Size sz[] = {ddims[0], ddims[1]};
        size_t i[2];
        size_t c;
        for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
            for(i[1] = 0; i[1] < sz[1]; i[1]++, c++) {
                dbuf[c] = scale * sbuf[idx(2, i, sz, perm_to_dest)];
            }
        }
    } else if(ndim == 3) {
        Size sz[] = {ddims[0], ddims[1], ddims[2]};
        size_t i[3];
        size_t c;
        for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
            for(i[1] = 0; i[1] < sz[1]; i[1]++) {
                for(i[2] = 0; i[2] < sz[2]; i[2]++, c++) {
                    dbuf[c] = scale * sbuf[idx(3, i, sz, perm_to_dest)];
                }
            }
        }
    } else if(ndim == 4) {
        Size sz[] = {ddims[0], ddims[1], ddims[2], ddims[3]};
        size_t i[4];
        size_t c;
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
void ip_gen(T* dst, const SizeVec& ddims, const IntLabelVec& dlabels, T scale,
            const T* src, const SizeVec& sdims, const IntLabelVec& slabels,
            bool is_assign = true) {
    IntLabelVec unique_labels = unique_entries(dlabels);
    const auto& dperm_map     = perm_map_compute(unique_labels, dlabels);
    const auto& sperm_map     = perm_map_compute(unique_labels, slabels);
    const auto& dinv_pm       = perm_map_compute(dlabels, unique_labels);

    auto idx = [](const auto& index_vec, const auto& dims_vec) {
        Size ret = 0, ld = 1;
        EXPECTS(index_vec.size() == dims_vec.size());
        for(int i = index_vec.size(); i >= 0; i--) {
            ret += ld * index_vec[i];
            ld *= dims_vec[i];
        }
        return ret;
    };

    SizeVec itrv(unique_labels.size(), 0);
    SizeVec endv(unique_labels.size());
    endv = internal::perm_map_apply(ddims, dinv_pm);
    do {
        const auto& itval  = itrv;
        const auto& sindex = perm_map_apply(itval, sperm_map);
        const auto& dindex = perm_map_apply(itval, dperm_map);
        if(is_assign) {
            dst[idx(dindex, ddims)] = scale * src[idx(sindex, sdims)];
        } else {
            dst[idx(dindex, ddims)] += scale * src[idx(sindex, sdims)];
        }
    } while(internal::cartesian_iteration(itrv, endv));
}

template<typename T>
void ip_gen_loop(T* dst, const SizeVec& ddims, const IntLabelVec& dlabels,
                 T scale, const T* src, const SizeVec& sdims,
                 const IntLabelVec& slabels, bool is_assign = true) {
    const size_t ndim = ddims.size();

    // assert(ddims.size() == sdims.size());
    // assert(ddims.size() == dlabels.size());
    // assert(sdims.size() == slabels.size());

    SizeVec sld{sdims}, dld{ddims};
    sld.insert(sld.end(), 1);
    dld.insert(dld.end(), 1);
    std::partial_sum(sld.rbegin(), sld.rend(), sld.rbegin(),
                     std::multiplies<Size>());
    std::partial_sum(dld.rbegin(), dld.rend(), dld.rbegin(),
                     std::multiplies<Size>());

    IntLabelVec loop_labels;
    for(const auto& lbl : dlabels) {
        if(std::find(loop_labels.begin(), loop_labels.end(), lbl) ==
           loop_labels.end()) {
            loop_labels.push_back(lbl);
        }
    }
    for(const auto& lbl : slabels) {
        if(std::find(loop_labels.begin(), loop_labels.end(), lbl) ==
           loop_labels.end()) {
            loop_labels.push_back(lbl);
        }
    }
    SizeVec loop_dims(loop_labels.size()), loop_sld(loop_labels.size()),
      loop_dld(loop_labels.size());
    for(size_t i = 0; i < loop_labels.size(); i++) {
        const auto& lbl = loop_labels[i];
        auto sit        = std::find(slabels.begin(), slabels.end(), lbl);
        if(sit != slabels.end()) {
            loop_sld[i]  = sld[sit - slabels.begin() + 1];
            loop_dims[i] = sdims[sit - slabels.begin()];
        }
    }

    for(size_t i = 0; i < loop_labels.size(); i++) {
        const auto& lbl = loop_labels[i];
        auto dit        = std::find(dlabels.begin(), dlabels.end(), lbl);
        if(dit != dlabels.end()) {
            loop_dld[i]  = dld[dit - dlabels.begin() + 1];
            loop_dims[i] = ddims[dit - dlabels.begin()];
        }
    }

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
}

template<typename T>
void ip_hptt(T* dst, const SizeVec& ddims, const IntLabelVec& dlabels, T scale,
             const T* src, const SizeVec& sdims, const IntLabelVec& slabels,
             bool is_assign = true) {
    // EXPECTS(ddims.size() == dlabels.size());
    // EXPECTS(sdims.size() == slabels.size());
    // EXPECTS(ddims.size() == sdims.size());
    // EXPECTS(src != nullptr);
    // EXPECTS(dst != nullptr);

    const int ndim = ddims.size();
    int perm[ndim];
    int size[ndim];
    T beta         = is_assign ? 0 : 1;
    int numThreads = 1;
    for(size_t i = 0; i < sdims.size(); i++) { size[i] = sdims[i].value(); }
    for(size_t i = 0; i < dlabels.size(); i++) {
        auto it = std::find(slabels.begin(), slabels.end(), dlabels[i]);
        EXPECTS(it != slabels.end());
        perm[i] = it - slabels.begin();
    }
    // create a plan (shared_ptr)
    auto plan = hptt::create_plan(perm, ndim, scale, src, size, NULL, beta, dst,
                                  NULL, hptt::ESTIMATE, numThreads, NULL, true);

    // execute the transposition
    plan->execute();
}

} // namespace internal

//////////////////////////////////////////////////////////////////////////

namespace kernels {
template<typename T>
void assign(T* dst, const SizeVec& ddims, const IntLabelVec& dlabels, T scale,
            const T* src, const SizeVec& sdims, const IntLabelVec& slabels,
            bool is_assign = true) {
    const size_t ndim = ddims.size();

    // assert(ddims.size() == sdims.size());
    // assert(ddims.size() == dlabels.size());
    // assert(sdims.size() == slabels.size());

    if(internal::are_permutations(slabels, dlabels)) {
        if(ndim == 0) {
            auto perm_to_dest = internal::perm_compute(dlabels, slabels);
            if(is_assign) {
                internal::index_permute(dst, src, perm_to_dest, ddims, scale);
            } else {
                internal::index_permute_acc(dst, src, perm_to_dest, ddims,
                                            scale);
            }
        } else
            internal::ip_hptt(dst, ddims, dlabels, scale, src, sdims, slabels,
                              is_assign);
    } else {
        internal::ip_gen_loop(dst, ddims, dlabels, scale, src, sdims, slabels,
                              is_assign);
    }
}

} // namespace kernels

} // namespace tamm

#endif // TAMM_KERNELS_ASSIGN_H_
