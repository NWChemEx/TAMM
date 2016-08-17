#ifndef __tce_capi_h__
#define __tce_capi_h__

#include "fapi.h"
#include "typesf2c.h"
#include "index.h"
#include "variables.h"
#include <vector>
#include <iostream>

namespace ctce {

  /**
   * CXX function that wraps FORTRAN tce_sort2 and tce_sort4
   */
  void tce_sort(double *sbuf, double *dbuf, const std::vector<size_t>& ids,
      std::vector<size_t>& iv, double alpha);

  /**
   * CXX function that wraps FORTRAN fdgemm
   */
  void cdgemm(char transa, char transb, size_t m, size_t n, size_t k,
      double alpha, double *a, size_t lda, double *b, size_t ldb, 
      double beta, double *c, size_t ldc);

  /**
   * CXX function that wraps FORTRAN add_hash_block
   */
  /* void tce_add_hash_block_(size_t *d_c,double *buf_a,size_t size,size_t k_c_offset,  */
  /*     const std::vector<size_t>& is, const std::vector<IndexName>& ns); */

  void cadd_hash_block(size_t d_c, double *buf_a, size_t size, Integer *hash, size_t key);


  /**
   * CXX function that wraps FORTRAN tce_sortacc6
   */
  void sortacc(double *sbuf, double *dbuf, const std::vector<size_t>& ids, 
      std::vector<size_t>& perm, double alpha);

  void ctce_restricted(int dim, int nupper,
                       const std::vector<size_t> &value,
                       std::vector<size_t> &pvalue_r);


}; // namespace ctce

#endif
