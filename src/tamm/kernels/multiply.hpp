#ifndef TAMM_MULTIPLY_HPP_
#define TAMM_MULTIPLY_HPP_

#include "tamm/errors.hpp"
#include "tamm/types.hpp"
#include "tamm/kernels/assign.hpp"

#include <algorithm>
#include CBLAS_HEADER
#include <complex>
#include <numeric>
#include <vector>

// #define USE_TALSH
#ifdef USE_TALSH
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

namespace tamm {

namespace internal {

template<typename T>
void gemm_wrapper(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
                  const CBLAS_TRANSPOSE TransB, const int M, const int N,
                  const int K, T alpha, const T* A, const int lda, const T* B,
                  const int ldb, T beta, T* C, const int ldc);

template<>
inline void gemm_wrapper<double>(const CBLAS_ORDER Order,
                                 const CBLAS_TRANSPOSE TransA,
                                 const CBLAS_TRANSPOSE TransB, const int M,
                                 const int N, const int K, double alpha,
                                 const double* A, const int lda,
                                 const double* B, const int ldb, double beta,
                                 double* C, const int ldc) {
    cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
                ldc);
}

template<>
inline void gemm_wrapper<float>(const CBLAS_ORDER Order,
                                const CBLAS_TRANSPOSE TransA,
                                const CBLAS_TRANSPOSE TransB, const int M,
                                const int N, const int K, float alpha,
                                const float* A, const int lda, const float* B,
                                const int ldb, float beta, float* C,
                                const int ldc) {
    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
                ldc);
}

template<>
inline void gemm_wrapper<std::complex<float>>(
  const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
  std::complex<float> alpha, const std::complex<float>* A, const int lda,
  const std::complex<float>* B, const int ldb, std::complex<float> beta,
  std::complex<float>* C, const int ldc) {
    cblas_cgemm(Order, TransA, TransB, M, N, K, (const float*)&alpha, (const float*)A, lda,
               (const float*)B, ldb, (const float*)&beta, (float*)C, ldc);
}

template<>
inline void gemm_wrapper<std::complex<double>>(
  const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
  std::complex<double> alpha, const std::complex<double>* A, const int lda,
  const std::complex<double>* B, const int ldb, std::complex<double> beta,
  std::complex<double>* C, const int ldc) {
    cblas_zgemm(Order, TransA, TransB, M, N, K, (const double*)&alpha, (const double*)A,
                lda, (const double*)B, ldb, (const double*)&beta, (double*)C, ldc);
}
} // namespace internal

namespace kernels {

template<typename T>
void block_multiply(int my_rank, T alpha, const T* abuf, const SizeVec& adims,
                    const IntLabelVec& alabels, const T* bbuf,
                    const SizeVec& bdims, const IntLabelVec& blabels, T beta,
                    T* cbuf, const SizeVec& cdims, const IntLabelVec& clabels) {
    const Size asize = std::accumulate(adims.begin(), adims.end(), Size{1},
                                       std::multiplies<Size>());
    const Size bsize = std::accumulate(bdims.begin(), bdims.end(), Size{1},
                                       std::multiplies<Size>());
    const Size csize = std::accumulate(cdims.begin(), cdims.end(), Size{1},
                                       std::multiplies<Size>());

    EXPECTS(abuf != nullptr && bbuf != nullptr && cbuf != nullptr);

    IntLabelVec asorted_labels{alabels}, bsorted_labels{blabels},
      csorted_labels{clabels};
    std::sort(asorted_labels.begin(), asorted_labels.end());
    std::sort(bsorted_labels.begin(), bsorted_labels.end());
    std::sort(csorted_labels.begin(), csorted_labels.end());

    std::vector<IntLabel> inner_labels, aouter_labels, bouter_labels,
      batch_labels, areduce_labels, breduce_labels;
    std::vector<Size> inner_dims, aouter_dims, bouter_dims, batch_dims,
      areduce_dims, breduce_dims;

    int B = 1, M = 1, N = 1, K = 1, AR = 1, BR = 1;
    for(size_t i = 0; i < cdims.size(); i++) {
        const auto& lbl = clabels[i];
        bool is_in_a =
          std::binary_search(asorted_labels.begin(), asorted_labels.end(), lbl);
        bool is_in_b =
          std::binary_search(bsorted_labels.begin(), bsorted_labels.end(), lbl);
        if(is_in_a && is_in_b) {
            batch_labels.push_back(lbl);
            batch_dims.push_back(cdims[i]);
            B *= static_cast<int>(cdims[i].value());
        } else if(is_in_a) {
            aouter_labels.push_back(lbl);
            aouter_dims.push_back(cdims[i]);
            M *= static_cast<int>(cdims[i].value());
        } else if(is_in_b) {
            bouter_labels.push_back(lbl);
            bouter_dims.push_back(cdims[i]);
            N *= static_cast<int>(cdims[i].value());
        } else {
            UNREACHABLE();
        }
    }

    for(size_t i = 0; i < adims.size(); i++) {
        const auto& lbl = alabels[i];
        bool is_in_b =
          std::binary_search(bsorted_labels.begin(), bsorted_labels.end(), lbl);
        bool is_in_c =
          std::binary_search(csorted_labels.begin(), csorted_labels.end(), lbl);
        if(is_in_b && is_in_c) {
            // no-op -- already added in batch_labels
        } else if(is_in_b) {
            inner_labels.push_back(lbl);
            inner_dims.push_back(adims[i]);
            K *= static_cast<int>(adims[i].value());
        } else if(is_in_c) {
            // no-op -- already added to aouter
        } else {
            AR *= adims[i].value();
            areduce_dims.push_back(adims[i]);
            areduce_labels.push_back(lbl);
        }
    }

    for(size_t i = 0; i < bdims.size(); i++) {
        const auto& lbl = blabels[i];
        bool is_in_a =
          std::binary_search(asorted_labels.begin(), asorted_labels.end(), lbl);
        bool is_in_c =
          std::binary_search(csorted_labels.begin(), csorted_labels.end(), lbl);
        if(is_in_a && is_in_c) {
            // no-op -- already added in batch_labels
        } else if(is_in_a) {
            // no-op -- already in inner_labels
        } else if(is_in_c) {
            // no-op -- already added to bouter
        } else {
            BR *= bdims[i].value();
            breduce_dims.push_back(bdims[i]);
            breduce_labels.push_back(lbl);
        }
    }

    std::vector<IntLabel> ainter_labels{areduce_labels};
    ainter_labels.insert(ainter_labels.end(), batch_labels.begin(),
                         batch_labels.end());
    ainter_labels.insert(ainter_labels.end(), aouter_labels.begin(),
                         aouter_labels.end());
    ainter_labels.insert(ainter_labels.end(), inner_labels.begin(),
                         inner_labels.end());

    std::vector<IntLabel> binter_labels{breduce_labels};
    binter_labels.insert(binter_labels.end(), batch_labels.begin(),
                         batch_labels.end());
    binter_labels.insert(binter_labels.end(), inner_labels.begin(),
                         inner_labels.end());
    binter_labels.insert(binter_labels.end(), bouter_labels.begin(),
                         bouter_labels.end());

    std::vector<IntLabel> cinter_labels{batch_labels};
    cinter_labels.insert(cinter_labels.end(), aouter_labels.begin(),
                         aouter_labels.end());
    cinter_labels.insert(cinter_labels.end(), bouter_labels.begin(),
                         bouter_labels.end());

    SizeVec ainter_dims{areduce_dims};
    ainter_dims.insert(ainter_dims.end(), batch_dims.begin(), batch_dims.end());
    ainter_dims.insert(ainter_dims.end(), aouter_dims.begin(),
                       aouter_dims.end());
    ainter_dims.insert(ainter_dims.end(), inner_dims.begin(), inner_dims.end());

    SizeVec binter_dims{breduce_dims};
    binter_dims.insert(binter_dims.end(), batch_dims.begin(), batch_dims.end());
    binter_dims.insert(binter_dims.end(), inner_dims.begin(), inner_dims.end());
    binter_dims.insert(binter_dims.end(), bouter_dims.begin(),
                       bouter_dims.end());

    SizeVec cinter_dims{batch_dims};
    cinter_dims.insert(cinter_dims.end(), aouter_dims.begin(),
                       aouter_dims.end());
    cinter_dims.insert(cinter_dims.end(), bouter_dims.begin(),
                       bouter_dims.end());

    auto transA    = CblasNoTrans;
    auto transB    = CblasNoTrans;
    int ainter_ld  = K;
    int binter_ld  = N;
    int cinter_ld  = N;
    int cbatch_ld  = M * N;
    int abatch_ld  = M * K;
    int bbatch_ld  = K * N;
    int areduce_ld = B * abatch_ld;
    int breduce_ld = B * bbatch_ld;

    #ifndef USE_TALSH
    std::vector<T> ainter_buf(static_cast<size_t>(asize.value())),
      binter_buf(static_cast<size_t>(bsize.value())),
      cinter_buf(static_cast<size_t>(csize.value()));
    assign(ainter_buf.data(), ainter_dims, ainter_labels, T{1}, abuf, adims,
           alabels, true);
    assign(binter_buf.data(), binter_dims, binter_labels, T{1}, bbuf, bdims,
           blabels, true);

    // dgemm
    for(size_t ari = 0; ari < AR; ari++) {
        for(size_t bri = 0; bri < BR; bri++) {
            for(size_t i = 0; i < B; i++) {
                internal::gemm_wrapper<T>(
                  CblasRowMajor, transA, transB, M, N, K, alpha,
                  ainter_buf.data() + ari * areduce_ld + i * abatch_ld,
                  ainter_ld,
                  binter_buf.data() + bri * breduce_ld + i * bbatch_ld,
                  binter_ld, beta, cinter_buf.data() + i * cbatch_ld,
                  cinter_ld);

            }
        }
    }
    assign(cbuf, cdims, clabels, T{1}, cinter_buf.data(), cinter_dims,
        cinter_labels, true);
    #else

    auto talsh_op_string = internal::talsh_mult_op_string(
        clabels, alabels, blabels); 

    auto aid_size = adims.size();
    auto bid_size = bdims.size();
    auto cid_size = cdims.size();
    int tal_adims[aid_size];
    int tal_bdims[bid_size];
    int tal_cdims[cid_size];
    
    std::vector<int> taid;
    std::vector<int> tbid;
    std::vector<int> tcid;
    std::transform(std::begin(adims), std::end(adims),
                 std::back_inserter(taid),[](tamm::Size i) -> int {return i.value();});
    std::transform(std::begin(bdims), std::end(bdims),
                 std::back_inserter(tbid),[](tamm::Size i) -> int {return i.value();});
    std::transform(std::begin(cdims), std::end(cdims),
                 std::back_inserter(tcid),[](tamm::Size i) -> int {return i.value();});

    std::reverse(taid.begin(),taid.end());
    std::reverse(tbid.begin(),tbid.end());
    std::reverse(tcid.begin(),tcid.end());

    std::copy(taid.begin(),taid.end(),tal_adims);
    std::copy(tbid.begin(),tbid.end(),tal_bdims);
    std::copy(tcid.begin(),tcid.end(),tal_cdims);

    bool hadamard = false;
    for(auto x: cinter_labels) {
      auto r1 =  std::find(std::begin(ainter_labels), std::end(ainter_labels), x) != std::end(ainter_labels);
      auto r2 =  std::find(std::begin(binter_labels), std::end(binter_labels), x) != std::end(binter_labels);
      if (r1 && r2) hadamard = true;
    }

    bool reduction_op = false;
    for(auto x: ainter_labels) {
      auto r1 =  std::find(std::begin(cinter_labels), std::end(cinter_labels), x) == std::end(cinter_labels);
      auto r2 =  std::find(std::begin(binter_labels), std::end(binter_labels), x) == std::end(binter_labels);
      if (r1 && r2) reduction_op = true;
    }
    for(auto x: binter_labels) {
      auto r1 =  std::find(std::begin(cinter_labels), std::end(cinter_labels), x) == std::end(cinter_labels);
      auto r2 =  std::find(std::begin(ainter_labels), std::end(ainter_labels), x) == std::end(ainter_labels);
      if (r1 && r2) reduction_op = true;
    }

    if(hadamard || reduction_op) {
      // std::cout << " hadamard or reduction op: " << talsh_op_string << "\n";
    std::vector<T> ainter_buf(static_cast<size_t>(asize.value())),
      binter_buf(static_cast<size_t>(bsize.value())),
      cinter_buf(static_cast<size_t>(csize.value()));
      assign(ainter_buf.data(), ainter_dims, ainter_labels, T{1}, abuf, adims,
            alabels, true);
      assign(binter_buf.data(), binter_dims, binter_labels, T{1}, bbuf, bdims,
            blabels, true);

      for(size_t ari = 0; ari < AR; ari++) {
        for(size_t bri = 0; bri < BR; bri++) {
            for(size_t i = 0; i < B; i++) {
                internal::gemm_wrapper<T>(
                  CblasRowMajor, transA, transB, M, N, K, alpha,
                  ainter_buf.data() + ari * areduce_ld + i * abatch_ld,
                  ainter_ld,
                  binter_buf.data() + bri * breduce_ld + i * bbatch_ld,
                  binter_ld, beta, cinter_buf.data() + i * cbatch_ld,
                  cinter_ld);

            }
        }
      }

      assign(cbuf, cdims, clabels, T{1}, cinter_buf.data(), cinter_dims,
         cinter_labels, true);
    }

    else {
      // std::cout << " not hadamard\n";
      // std::cout << talsh_op_string << std::endl;
      // std::cout << aid_size << ":" << bid_size << ":" << cid_size << std::endl;

      // adata, bdata, cdata will have to be created 
      // using pinned memory else where for now using 
      // regular memory
      // double *adata = host_pinned_memory(abatch_ld*sizeof(double)); 
      // double *bdata = host_pinned_memory(bbatch_ld*sizeof(double)); 
      // double *cdata = host_pinned_memory(cbatch_ld*sizeof(double)); 

      TALSH gpu_mult;
      T* abufp = const_cast<T*>(abuf); 
      T* bbufp = const_cast<T*>(bbuf); 

      // Create tensor objects 
      tensor_handle T1 = gpu_mult.host_block(adims.size(), 
          tal_adims, abufp); 
      tensor_handle T2 = gpu_mult.host_block(bdims.size(), 
          tal_bdims, bbufp); 
      tensor_handle T3 = gpu_mult.host_block(cdims.size(), 
          tal_cdims, cbuf); 

      gpu_mult.mult_block(my_rank, T3, T1, T2, talsh_op_string, 
          alpha, COPY_TTT); 

      talshTensorDestruct(&T1);
      talshTensorDestruct(&T2);
      talshTensorDestruct(&T3);
      // free_host_pinned_memory(adata);
      // free_host_pinned_memory(bdata);
      // free_host_pinned_memory(cdata);
    }
    #endif
} // block_multiply()

} // namespace kernels

} // namespace tamm

#endif // TAMM_MULTIPLY_HPP_
