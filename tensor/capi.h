#ifndef __tamm_capi_h__
#define __tamm_capi_h__

#include "common.h"
#include <vector>
#include "ga_abstract.h"

/**
 * C/C++ wrappers to Fortran routines invoked by tamm
 */

namespace tamm {

/**
 * C++ function that wraps FORTRAN tce_sort2 and tce_sort4
 */
void
tamm_sort(double *sbuf, double *dbuf, const std::vector<size_t>& ids,
          std::vector<size_t>& iv, double alpha);

/**
 * C++ function that wraps FORTRAN dgemm
 */
void
cdgemm(char transa, char transb, size_t m, size_t n, size_t k,
       double alpha, double *a, size_t lda, double *b, size_t ldb,
       double beta, double *c, size_t ldc);

/**
 * C++ function that wraps FORTRAN add_hash_block
 */
void
cadd_hash_block(gmem::Handle d_c, double *buf_a, size_t size,
                Fint *hash, size_t key);


/**
 * C++ function that wraps FORTRAN tce_sortacc*
 */
void
tamm_sortacc(double *sbuf, double *dbuf, const std::vector<size_t>& ids,
             std::vector<size_t>& perm, double alpha);


/**
 * C++ function that wraps FORTRAN tce_restricted
 */
void
tamm_restricted(int dim, int nupper,
                const std::vector<size_t> &value,
                std::vector<size_t> &pvalue_r);

/**
 * C++ function that wraps FORTRAN get_hash_block_i
 */
void
cget_hash_block_i(gmem::Handle d_a, double *buf, size_t size,
                  size_t d_a_offset,
                  size_t key, std::vector<size_t> &is);

/**
 * C++ function that wraps FORTRAN get_hash_block_ma
 */
void
cget_hash_block_ma(gmem::Handle d_a, double *buf, size_t size,
                   size_t d_a_offset, size_t key);

/**
 * C++ function that wraps FORTRAN get_hash_block
 */
void
cget_hash_block(gmem::Handle d_a, double *buf, size_t size,
                size_t d_a_offset, size_t key);

} /*namespace tamm*/

#endif /*__tamm_capi_h__*/

