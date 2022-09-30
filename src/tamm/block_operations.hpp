#pragma once

#include <algorithm>
// // #include <chrono>
// // #include <iostream>
// #include <memory>
#include <vector>

#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/iteration.hpp"
#include "tamm/kernels/assign.hpp"
#include "tamm/kernels/multiply.hpp"
#include "tamm/utils.hpp"

namespace tamm::internal {

// inline size_t idx(int n, const size_t* id, const size_t* sz,
//                   const PermVector& p) {
//     size_t idx = 0;
//     for(int i = 0; i < n - 1; i++) { idx = (idx + id[p[i]]) * sz[p[i + 1]]; }
//     if(n > 0) { idx += id[p[n - 1]]; }
//     return idx;
// }

// template<typename T>
// inline void index_permute(T* dbuf, const T* sbuf,
//                           const PermVector& perm_to_dest,
//                           const std::vector<size_t>& ddims, T scale) {
//     EXPECTS(dbuf != nullptr && sbuf != nullptr);
//     EXPECTS(perm_to_dest.size() == ddims.size());

//     const size_t ndim = perm_to_dest.size();
//     EXPECTS(ddims.size() == ndim);

//     if(ndim == 0) {
//         dbuf[0] = scale * sbuf[0];
//     } else if(ndim == 1) {
//         for(size_t i = 0; i < ddims[0]; i++) { dbuf[i] = scale * sbuf[i]; }
//     } else if(ndim == 2) {
//         size_t sz[] = {ddims[0], ddims[1]};
//         size_t i[2], c;
//         for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
//             for(i[1] = 0; i[1] < sz[1]; i[1]++, c++) {
//                 dbuf[c] = scale * sbuf[idx(2, i, sz, perm_to_dest)];
//             }
//         }
//     } else if(ndim == 3) {
//         size_t sz[] = {ddims[0], ddims[1], ddims[2]};
//         size_t i[3], c;
//         for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
//             for(i[1] = 0; i[1] < sz[1]; i[1]++) {
//                 for(i[2] = 0; i[2] < sz[2]; i[2]++, c++) {
//                     dbuf[c] = scale * sbuf[idx(3, i, sz, perm_to_dest)];
//                 }
//             }
//         }
//     } else if(ndim == 4) {
//         size_t sz[] = {ddims[0], ddims[1], ddims[2], ddims[3]};
//         size_t i[4], c;
//         for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
//             for(i[1] = 0; i[1] < sz[1]; i[1]++) {
//                 for(i[2] = 0; i[2] < sz[2]; i[2]++) {
//                     for(i[3] = 0; i[3] < sz[3]; i[3]++, c++) {
//                         dbuf[c] = scale * sbuf[idx(4, i, sz, perm_to_dest)];
//                     }
//                 }
//             }
//         }
//     } else {
//         NOT_IMPLEMENTED();
//     }

// }

// template<typename T>
// inline void index_permute_acc(T* dbuf, const T* sbuf,
//                               const PermVector& perm_to_dest,
//                               const std::vector<size_t>& ddims, T scale) {
//     EXPECTS(dbuf != nullptr && sbuf != nullptr);
//     EXPECTS(perm_to_dest.size() == ddims.size());

//     const size_t ndim = perm_to_dest.size();
//     EXPECTS(ddims.size() == ndim);

//     if(ndim == 0) {
//         dbuf[0] = scale * sbuf[0];
//     } else if(ndim == 1) {
//         for(size_t i = 0; i < ddims[0]; i++) { dbuf[i] += scale * sbuf[i]; }
//     } else if(ndim == 2) {
//         size_t sz[] = {ddims[0], ddims[1]};
//         size_t i[2], c;
//         for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
//             for(i[1] = 0; i[1] < sz[1]; i[1]++, c++) {
//                 dbuf[c] += scale * sbuf[idx(2, i, sz, perm_to_dest)];
//             }
//         }
//     } else if(ndim == 3) {
//         size_t sz[] = {ddims[0], ddims[1], ddims[2]};
//         size_t i[3], c;
//         for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
//             for(i[1] = 0; i[1] < sz[1]; i[1]++) {
//                 for(i[2] = 0; i[2] < sz[2]; i[2]++, c++) {
//                     dbuf[c] += scale * sbuf[idx(3, i, sz, perm_to_dest)];
//                 }
//             }
//         }
//     } else if(ndim == 4) {
//         size_t sz[] = {ddims[0], ddims[1], ddims[2], ddims[3]};
//         size_t i[4], c;
//         for(c = 0, i[0] = 0; i[0] < sz[0]; i[0]++) {
//             for(i[1] = 0; i[1] < sz[1]; i[1]++) {
//                 for(i[2] = 0; i[2] < sz[2]; i[2]++) {
//                     for(i[3] = 0; i[3] < sz[3]; i[3]++, c++) {
//                         dbuf[c] += scale * sbuf[idx(4, i, sz, perm_to_dest)];
//                     }
//                 }
//             }
//         }
//     } else {
//         NOT_IMPLEMENTED();
//     }
// }

// /**
//  * @brief
//  *
//  * @todo add support for triangular arrays
//  *
//  * @tparam T
//  * @param dbuf
//  * @param ddims
//  * @param dlabel
//  * @param sbuf
//  * @param sdims
//  * @param slabel
//  * @param scale
//  * @param update
//  */
// template<typename T>
// inline void block_add(T* dbuf, const std::vector<size_t>& ddims,
//                       const IndexLabelVec& dlabel, T* sbuf,
//                       const std::vector<size_t>& sdims,
//                       const IndexLabelVec& slabel, T scale, bool update) {
//     if(are_permutations(dlabel, slabel)) {
//         EXPECTS(slabel.size() == dlabel.size());
//         EXPECTS(sdims.size() == slabel.size());
//         EXPECTS(ddims.size() == dlabel.size());
//         auto label_perm = perm_compute(dlabel, slabel);
//         for(unsigned i = 0; i < label_perm.size(); i++) {
//             EXPECTS(ddims[i] == sdims[label_perm[i]]);
//         }
//         if(!update) {
//             index_permute(dbuf, sbuf, label_perm, ddims, scale);
//         } else {
//             index_permute_acc(dbuf, sbuf, label_perm, ddims, scale);
//         }
//     } else {
//         IndexLabelVec unique_labels = unique_entries(dlabel);
//         unique_labels               = sort_on_dependence(unique_labels);

//         const auto& dperm_map = perm_map_compute(unique_labels, dlabel);
//         const auto& sperm_map = perm_map_compute(unique_labels, slabel);
//         const auto& dinv_pm   = perm_map_compute(dlabel, unique_labels);

//         auto idx = [](const auto& index_vec, const auto& dims_vec) {
//             size_t ret = 0, ld = 1;
//             EXPECTS(index_vec.size() == dims_vec.size());
//             for(int i = index_vec.size(); i >= 0; i--) {
//                 ret += ld * index_vec[i];
//                 ld *= dims_vec[i];
//             }
//             return ret;
//         };

//         std::vector<size_t> itrv(unique_labels.size(), 0);
//         std::vector<size_t> endv(unique_labels.size());
//         endv = internal::perm_map_apply(ddims, dinv_pm);
//         do {
//             const auto& itval  = itrv;
//             const auto& sindex = perm_map_apply(itval, sperm_map);
//             const auto& dindex = perm_map_apply(itval, dperm_map);
//             if(!update) {
//                 dbuf[idx(dindex, ddims)] = scale * sbuf[idx(sindex, sdims)];
//             } else {
//                 dbuf[idx(dindex, ddims)] += scale * sbuf[idx(sindex, sdims)];
//             }
//         } while(internal::cartesian_iteration(itrv, endv));
//     }
// }

// template<typename T>
// inline void block_mult(T cscale, T* cbuf, const std::vector<size_t>& cdims,
//                        const IndexLabelVec& clabel, T abscale, T* abuf,
//                        const std::vector<size_t>& adims,
//                        const IndexLabelVec& alabel, T* bbuf,
//                        const std::vector<size_t>& bdims,
//                        const IndexLabelVec& blabel) {
//     for(const auto& d : cdims) {
//         if(d == 0) { return; }
//     }
//     for(const auto& d : adims) {
//         if(d == 0) { return; }
//     }
//     for(const auto& d : bdims) {
//         if(d == 0) { return; }
//     }

//     IndexLabelVec all_labels{clabel};
//     all_labels.insert(all_labels.end(), alabel.begin(), alabel.end());
//     all_labels.insert(all_labels.end(), blabel.begin(), blabel.end());

//     IndexLabelVec unique_labels = unique_entries(all_labels);
//     IndexLabelVec sorted_labels = sort_on_dependence(unique_labels);

//     const auto& cperm_map  = perm_map_compute(sorted_labels, clabel);
//     const auto& aperm_map  = perm_map_compute(sorted_labels, alabel);
//     const auto& bperm_map  = perm_map_compute(sorted_labels, blabel);
//     const auto& all_inv_pm = perm_map_compute(all_labels, sorted_labels);

//     auto idx = [](const auto& index_vec, const auto& dims_vec) {
//         size_t ret = 0, ld = 1;
//         EXPECTS(index_vec.size() == dims_vec.size());
//         for(int i = -1 + index_vec.size(); i >= 0; i--) {
//             ret += ld * index_vec[i];
//             ld *= dims_vec[i];
//         }
//         return ret;
//     };

//     std::vector<size_t> itrv(sorted_labels.size(), 0);
//     std::vector<size_t> endv(sorted_labels.size());

//     std::vector<size_t> all_dims{cdims};
//     all_dims.insert(all_dims.end(), adims.begin(), adims.end());
//     all_dims.insert(all_dims.end(), bdims.begin(), bdims.end());
//     endv = internal::perm_map_apply(all_dims, all_inv_pm);

//     if(std::abs(cscale) > 1e-11) { NOT_IMPLEMENTED(); }
//     do {
//         const auto& itval  = itrv;
//         const auto& cindex = perm_map_apply(itval, cperm_map);
//         const auto& aindex = perm_map_apply(itval, aperm_map);
//         const auto& bindex = perm_map_apply(itval, bperm_map);
//         size_t cidx        = idx(cindex, cdims);
//         cbuf[cidx] +=
//           abscale * abuf[idx(aindex, adims)] * bbuf[idx(bindex, bdims)];
//     } while(internal::cartesian_iteration(itrv, endv));
// }

} // namespace tamm::internal
