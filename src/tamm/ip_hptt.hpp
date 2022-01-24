#pragma once

#include <set>

#include "hptt/hptt.h"
#include "tamm/errors.hpp"
#include "tamm/types.hpp"

namespace tamm::blockops::hptt {

///////////////////////////////////////////////////////////////////////////////
//
//                                 IP using HPTT
//
///////////////////////////////////////////////////////////////////////////////

// template <typename T>
// void index_permute_assign_hptt(T* lbuf, const T* rbuf,
//                                const PermVector& perm_to_dest,
//                                const std::vector<size_t>& ddims, bool is_assign,
//                                T scale = 1.0) {
//   const int ndim = ddims.size();
//   int perm[ndim];
//   int size[ndim];
//   T beta = is_assign ? 0 : 1;
//   int numThreads = 1;
//   for (size_t i = 0; i < ddims.size(); i++) {
//     size[i] = ddims[i].value();
//   }

//   for (size_t i = 0; i < perm_to_dest.size(); i++) {
//     perm[i] = perm_to_dest[i].value();
//   }
//   // create a plan (shared_ptr)
//   auto plan = hptt::create_plan(perm, ndim, scale, rbuf, size, NULL, beta, lbuf,
//                                 NULL, hptt::ESTIMATE, numThreads, NULL, true);

//   // execute the transposition
//   plan->execute();
// }

template <typename T>
void index_permute_hptt(T lscale, T* lbuf, T rscale, const T* rbuf,
                        const PermVector& perm_to_dest,
                        const std::vector<size_t>& sdims) {
  const int ndim = sdims.size();
  int perm[ndim];
  int size[ndim];
  int num_threads = 1;
  for (size_t i = 0; i < sdims.size(); i++) {
    size[i] = sdims[i];
  }

  for (size_t i = 0; i < perm_to_dest.size(); i++) {
    perm[i] = perm_to_dest[i];
  }
  // create a plan (shared_ptr)
  auto plan =
      ::hptt::create_plan(perm, ndim, rscale, rbuf, size, NULL, lscale, lbuf,
                          NULL, ::hptt::ESTIMATE, num_threads, NULL, true);

  // execute the transposition
  plan->execute();
}

}  // namespace tamm::blockops::hptt
