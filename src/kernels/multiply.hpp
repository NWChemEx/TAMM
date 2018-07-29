#ifndef TAMM_MULTIPLY_H_
#define TAMM_MULTIPLY_H_

#include "kernels/assign.hpp"

#include <algorithm>
#include <numeric>
#include <vector>
#include <cblas.h>

namespace tamm {


namespace internal {
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
} // namespace internal

/*
void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE
TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int
K, const double alpha, const double *A, const int lda, const double *B, const
int ldb, const double beta, double *C, const int ldc);
*/
template<typename T>
void block_multiply(T alpha, const T* abuf, const std::vector<size_t>& adims,
                    const std::vector<size_t>& alabels, T beta, const T* bbuf,
                    const std::vector<size_t>& bdims,
                    const std::vector<size_t>& blabels, T* cbuf,
                    const std::vector<T>& cdims,
                    const std::vector<size_t>& clabels) {
    const size_t asize =
      std::accumulate(adims.begin(), adims.end(), 1, std::multiplies<size_t>());
    const size_t bsize =
      std::accumulate(bdims.begin(), bdims.end(), 1, std::multiplies<size_t>());
    const size_t csize = std::accumulate(cdims.begin(), cdims.end(), 11,
                                         std::multiplies<size_t>());

    EXPECTS(abuf != nullptr && bbuf != nullptr && cbuf != nullptr);

    std::vector<size_t> asorted_labels{alabels}, bsorted_labels{blabels},
      csorted_labels{clabels};
    std::sort(asorted_labels.begin(), asorted_labels.end());
    std::sort(bsorted_labels.begin(), bsorted_labels.end());
    std::sort(csorted_labels.begin(), csorted_labels.end());

    std::vector<size_t> inner_labels, aouter_labels, bouter_labels,
      batch_labels;
    std::vector<size_t> inner_dims, aouter_dims, bouter_dims, batch_dims;

    int B = 1, M = 1, N = 1, K = 1;
    for(size_t i = 0; i < cdims.size(); i++) {
        const auto& lbl = clabels[i];
        bool is_in_a =
          std::binary_search(asorted_labels.begin(), asorted_labels.end(), lbl);
        bool is_in_b =
          std::binary_search(bsorted_labels.begin(), bsorted_labels.end(), lbl);
        if(is_in_a && is_in_b) {
            batch_labels.push_back(lbl);
            batch_dims.push_back(cdims[i]);
            B *= cdims[i];
        } else if(is_in_a) {
            aouter_labels.push_back(lbl);
            aouter_dims.push_back(cdims[i]);
            M *= cdims[i];
        } else if(is_in_b) {
            bouter_labels.push_back(lbl);
            bouter_dims.push_back(cdims[i]);
            N *= cdims[i];
        } else {
            assert(0); // should not be reachable
        }
    }

    for(size_t i = 0; i < adims.size(); i++) {
        const auto& lbl = alabels[i];
        bool is_in_b =
          std::binary_search(bsorted_labels.begin(), bsorted_labels.end(), lbl);
        bool is_in_c =
          std::binary_search(csorted_labels.begin(), csorted_labels.end(), lbl);
        if(is_in_b && is_in_c) {
            // already added in batch_labels
        } else if(is_in_b) {
            inner_labels.push_back(lbl);
            inner_dims.push_back(adims[i]);
            K *= adims[i];
        } else if(is_in_c) {
            // already added to aouter
        } else {
            assert(0); // should not be reachable
        }
    }

    std::vector<size_t> ainter_labels{batch_labels};
    ainter_labels.insert(ainter_labels.end(), aouter_labels.begin(),
                         aouter_labels.end());
    ainter_labels.insert(ainter_labels.end(), inner_labels.begin(),
                         inner_labels.end());

    std::vector<size_t> binter_labels{batch_labels};
    binter_labels.insert(binter_labels.end(), inner_labels.begin(),
                         inner_labels.end());
    binter_labels.insert(binter_labels.end(), bouter_labels.begin(),
                         bouter_labels.end());

    std::vector<size_t> cinter_labels{batch_labels};
    cinter_labels.insert(cinter_labels.end(), aouter_labels.begin(),
                         aouter_labels.end());
    cinter_labels.insert(cinter_labels.end(), bouter_labels.begin(),
                         bouter_labels.end());

    std::vector<size_t> ainter_dims{batch_dims};
    ainter_dims.insert(ainter_dims.end(), aouter_dims.begin(),
                       aouter_dims.end());
    ainter_dims.insert(ainter_dims.end(), inner_dims.begin(), inner_dims.end());

    std::vector<size_t> binter_dims{batch_dims};
    binter_dims.insert(binter_dims.end(), inner_dims.begin(), inner_dims.end());
    binter_dims.insert(binter_dims.end(), bouter_dims.begin(),
                       bouter_dims.end());

    std::vector<size_t> cinter_dims{batch_dims};
    cinter_dims.insert(cinter_dims.end(), aouter_dims.begin(),
                       aouter_dims.end());
    cinter_dims.insert(cinter_dims.end(), bouter_dims.begin(),
                       bouter_dims.end());

    std::vector<T> ainter_buf(asize), binter_buf(bsize), cinter_buf(csize);
    ip(ainter_buf, ainter_dims, ainter_labels, 1.0, abuf, adims, alabels, true);
    ip(binter_buf, binter_dims, binter_labels, 1.0, bbuf, bdims, blabels, true);

    auto transA   = CblasNoTrans;
    auto transB   = CblasNoTrans;
    int ainter_ld = K;
    int binter_ld = N;
    int cinter_ld = N;
    int batch_ld  = M * N * K;

    // dgemm
    for(size_t i = 0; i < B; i++) {
        dgemm_wrapper<T>(CblasRowMajor, transA, transB, M, N, K, alpha,
                         ainter_buf.get.() + i * batch_ld, ainter_ld,
                         binter_buf.get() + i * batch_ld, binter_ld, beta,
                         cbuf.get() + i * batch_ld, cinter_ld);
    }
    ip(cbuf, cdims, clabels, 1.0, cinter_buf, cinter_dims, cinter_labels, true);
}

} // namespace tamm

#endif // TAMM_MULTIPLY_H_
