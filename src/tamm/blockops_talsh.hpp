#ifndef TAMM_BLOCKOPS_TALSH_HPP_
#define TAMM_BLOCKOPS_TALSH_HPP_

#include <array>
#include <set>
#include <vector>

#include "tamm/block_span.hpp"
#include "tamm/errors.hpp"
#include "tamm/types.hpp"

#include <algorithm>
#include <complex>
#include <numeric>
#include <vector>

#include "tamm/talsh_tamm.hpp"
#include "tamm/cuda_memory_allocator.hpp"
using tensor_handle = talsh_tens_t;

#undef C0
#undef C4
#undef C5
#undef C6
#undef C7
#undef C8
#undef C9
#undef C10
#endif

namespace tamm::blockops::talsh {

///////////////////////////////////////////////////////////////////////////////
//
//                                 talsh routines
//
///////////////////////////////////////////////////////////////////////////////

template <typename T1, typename T2>
void block_mult_talsh(tensor_handle& th_c, tensor_handle& th_a, tensor_handle& th_b, 
      BlockSpan<T1>& lhs, const T1 alpha, BlockSpan<T1>& rhs1, BlockSpan<T1>& rhs2) {
      T2* abufp = const_cast<T2*>(rhs1); 
      T3* bbufp = const_cast<T3*>(rhs2); 

      // if constexpr(std::is_same_v<T1,T2> && std::is_same_v<T1,T3>){
           th_a = gpu_mult.host_block(adims.size(), 
              tal_adims, abufp); 
           th_b = gpu_mult.host_block(bdims.size(), 
              tal_bdims, bbufp); 
          if(copy_ctrl == COPY_TTT)
            th_c = gpu_mult.host_block(cdims.size(), 
               tal_cdims, cbuf); 

          gpu_mult.mult_block(talsh_task, dev_id, th_c, th_a, th_b, talsh_op_string, 
              rscale, copy_ctrl, is_assign); 
}



}  // namespace tamm::blockops::talsh

#endif  // TAMM_BLOCKOPS_TALSH_HPP_