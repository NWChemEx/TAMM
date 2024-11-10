#pragma once

#include "ga/ga_linalg.h"
#include "tamm/block_span.hpp"
//#include "tamm/kernels/multiply.hpp"
#include "tamm/types.hpp"
#include "tamm/utils.hpp"

#include <algorithm>
#include <complex>
#include <numeric>
#include <vector>

namespace tamm::blockops::bops_blas {
template<typename T, typename T1, typename T2, typename T3>
std::tuple<BlockSpan<T>, BlockSpan<T>, BlockSpan<T>>
prep_buffers(T1 lscale, BlockSpan<T1>& lhs, T1 rscale, BlockSpan<T2>& rhs1, BlockSpan<T3>& rhs2) {
  auto         adims = rhs1.block_dims();
  auto         bdims = rhs2.block_dims();
  auto         cdims = lhs.block_dims();
  const size_t asize =
    std::accumulate(adims.begin(), adims.end(), (size_t) 1, std::multiplies<size_t>());
  const size_t bsize =
    std::accumulate(bdims.begin(), bdims.end(), (size_t) 1, std::multiplies<size_t>());
  // const size_t csize =
  //   std::accumulate(cdims.begin(), cdims.end(), (size_t) 1, std::multiplies<size_t>());

  BlockSpan<T> lhs_T;
  BlockSpan<T> rhs1_T;
  BlockSpan<T> rhs2_T;

  // FIXME: {a,b}buf_comp_ptr memory is deallocated

  if constexpr(std::is_same_v<T1, T2>) {
    // T2 (matrix A) is complex, T3 (B) is real
    if constexpr(internal::is_complex_v<T1>) {
      // copy B to complex buffer
      std::vector<T1> bbuf_complex(bsize);
      T3*             bbuf_comp_ptr = reinterpret_cast<T3*>(bbuf_complex.data());
      blas::copy(bsize, rhs2.data(), 1, bbuf_comp_ptr, 2);

      lhs_T  = lhs;
      rhs1_T = rhs1;
      rhs2_T = BlockSpan{bbuf_comp_ptr, rhs2.block_dims()};

    } // is_complex<T1>
    else {
      // T1,T2 (C,A) are real, T3 (B) is complex
      std::vector<T1> bbuf_real(bsize);
      T1*             bbuf_comp_ptr = reinterpret_cast<T1*>(rhs2.data());
      blas::copy(bsize, bbuf_comp_ptr, 2, bbuf_real.data(), 1);

      lhs_T  = lhs;
      rhs1_T = rhs1;
      rhs2_T = BlockSpan{bbuf_comp_ptr, rhs2.block_dims()};

    } // is_real<T1>

  } // is_same_v<T1,T2>
  else if constexpr(std::is_same_v<T1, T3>) {
    // T3 (matrix B) is complex, T2 (A) is real
    if constexpr(internal::is_complex_v<T1>) {
      std::vector<T1> abuf_complex(asize);
      T2*             abuf_comp_ptr = reinterpret_cast<T2*>(abuf_complex.data());
      blas::copy(asize, rhs1.data(), 1, abuf_comp_ptr, 2);

      lhs_T  = lhs;
      rhs1_T = BlockSpan{abuf_comp_ptr, rhs1.block_dims()};
      rhs2_T = rhs2;
    }
    else {
      // T1,T3 (C,B) are real, T2 (A) is complex
      std::vector<T1> abuf_real(asize);
      T1*             abuf_comp_ptr = reinterpret_cast<T1*>(rhs1.data());
      blas::copy(asize, abuf_comp_ptr, 2, abuf_real.data(), 1);

      lhs_T  = lhs;
      rhs1_T = BlockSpan{abuf_comp_ptr, rhs1.block_dims()};
      rhs2_T = rhs2;
    }

  } // is_same_v<T1,T3>

  else if constexpr(internal::is_complex_v<T1> && std::is_same_v<T2, T3>) {
    // T1 is complex, T2, T3 are real
    std::vector<T1> abuf_complex(asize);
    std::vector<T1> bbuf_complex(bsize);
    T2*             abuf_comp_ptr = reinterpret_cast<T2*>(abuf_complex.data());
    T2*             bbuf_comp_ptr = reinterpret_cast<T2*>(bbuf_complex.data());

    blas::copy(asize, rhs1.data(), 1, abuf_comp_ptr, 2);
    blas::copy(bsize, rhs2.data(), 1, bbuf_comp_ptr, 2);

    lhs_T  = lhs;
    rhs1_T = BlockSpan{abuf_comp_ptr, rhs1.block_dims()};
    rhs2_T = BlockSpan{bbuf_comp_ptr, rhs2.block_dims()};
  }

  else NOT_IMPLEMENTED();

  return std::make_tuple(lhs_T, rhs1_T, rhs2_T);
}

template<typename T1, typename T2>
void prep_rhs_buffer(const BlockSpan<T2>& rhs, std::vector<T1>& new_rhs) {
  size_t r_size   = rhs.num_elements();
  T1*    rhs_data = new_rhs.data();
  if constexpr(internal::is_complex_v<T1>) {
    T2* rhs_buf  = const_cast<T2*>(rhs.buf());
    T2* rbuf_ptr = reinterpret_cast<T2*>(rhs_data);
    blas::copy(r_size, rhs_buf, 1, rbuf_ptr, 2);
  }
  // real = complex
  else if constexpr(internal::is_complex_v<T2>) {
    T2* rhs_buf  = const_cast<T2*>(rhs.buf());
    T1* rbuf_ptr = reinterpret_cast<T1*>(rhs_buf);
    blas::copy(r_size, rbuf_ptr + 1, 2, rhs_data, 1);
  }
  else NOT_ALLOWED();
}

} // namespace tamm::blockops::bops_blas
