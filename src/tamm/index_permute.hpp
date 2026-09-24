#pragma once

#include <cstddef>
#include <vector>

#include "tamm/errors.hpp"
#include "tamm/kernels/cpu_permute.hpp"
#include "tamm/types.hpp"

namespace tamm::blockops::permute {

///////////////////////////////////////////////////////////////////////////////
//
//           Index permute using the in-house CPU permute kernel
//           (replaces the old HPTT-based index_permute_hptt)
//
///////////////////////////////////////////////////////////////////////////////

// Out-of-place permute with HPTT-identical semantics:
//   lbuf = lscale * lbuf + rscale * permute(rbuf)
// perm_to_dest maps each destination axis to its source axis and sdims holds
// the source extents, both in natural order; the output extents follow as
// outDims[k] == sdims[perm_to_dest[k]]. Scale types are independent template
// parameters so call sites may mix e.g. an int literal rscale with a
// complex buffer (as the old single-T HPTT wrapper allowed via conversion).
template<typename BL, typename T1, typename BA, typename T2>
void index_permute(BL lscale, T1* lbuf, BA rscale, const T2* rbuf,
                   const PermVector& perm_to_dest, const std::vector<size_t>& sdims) {
  const size_t ndim = sdims.size();
  EXPECTS(perm_to_dest.size() == ndim);
  EXPECTS(ndim <= static_cast<size_t>(kernels::gpu::permute_maxrank));
  EXPECTS(lbuf != nullptr && (ndim == 0 || rbuf != nullptr));

  size_t outDims[kernels::gpu::permute_maxrank] = {};
  int    perm[kernels::gpu::permute_maxrank]    = {};
  for(size_t i = 0; i < ndim; ++i) {
    perm[i] = static_cast<int>(perm_to_dest[i]);
    EXPECTS(perm[i] >= 0 && static_cast<size_t>(perm[i]) < ndim);
    outDims[i] = sdims[static_cast<size_t>(perm[i])];
  }
  kernels::cpu::permute(lbuf, rbuf, static_cast<int>(ndim), outDims, perm, rscale, lscale);
}

} // namespace tamm::blockops::permute
