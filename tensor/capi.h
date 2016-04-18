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
  void tce_sort(double *sbuf, double *dbuf, const std::vector<Integer>& ids,
      std::vector<Integer>& iv, double alpha);

  /**
   * CXX function that wraps FORTRAN fdgemm
   */
  void cdgemm(char transa, char transb, Integer m, Integer n, Integer k,
      double alpha, double *a, Integer lda, double *b, Integer ldb, 
      double beta, double *c, Integer ldc);

  /**
   * CXX function that wraps FORTRAN add_hash_block
   */
  void tce_add_hash_block_(Integer *d_c,double *buf_a,Integer size,Integer k_c_offset, 
      const std::vector<Integer>& is, const std::vector<IndexName>& ns);

  /**
   * CXX function that wraps FORTRAN tce_sortacc6
   */
  void sortacc(double *sbuf, double *dbuf, const std::vector<Integer>& ids, 
      std::vector<Integer>& perm, double alpha);

}; // namespace ctce

#endif
