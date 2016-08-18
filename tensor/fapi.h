#ifndef __ctce_fapi_h__
#define __ctce_fapi_h__

#include "common.h"

/**
 * All fortran functions used by ctce
 */

EXTERN_C void
FORTRAN_FUNC(tce_restricted_2,TCE_RESTRICTED_2)
(Fint *i1, Fint *i2, Fint *o1, Fint *o2);

EXTERN_C void
FORTRAN_FUNC(tce_restricted_4,TCE_RESTRICTED_4)
(Fint *i1, Fint *i2, Fint *i3, Fint *i4,
 Fint *o1, Fint *o2, Fint *o3, Fint *o4);

EXTERN_C void
FORTRAN_FUNC(get_hash_block_ma,GET_HASH_BLOCK_MA)
(double *big_array, double *array, Fint *size, Fint *hash, Fint *key);

EXTERN_C void
FORTRAN_FUNC(get_hash_block,GET_HASH_BLOCK)
(Fint *darr, double *buf, Fint *size, Fint *hash, Fint *key);

EXTERN_C void
FORTRAN_FUNC(get_hash_block_i,GET_HASH_BLOCK_I)
(Fint *darr, double *buf, Fint *size, Fint *hash, Fint *key,
 Fint *g2b, Fint *g1b, Fint *g4b, Fint *g3b);

EXTERN_C void
FORTRAN_FUNC(tce_sort_2,TCE_SORT_2)
(double *unsorted , double *sorted, Fint *a, Fint *b,
 Fint *i, Fint *j, double *factor);

EXTERN_C void
FORTRAN_FUNC(tce_sort_4,TCE_SORT_4)
(double *unsorted , double *sorted,
 Fint *a, Fint *b, Fint *c, Fint *d,
 Fint *i, Fint *j, Fint *k, Fint *l,
 double *factor);

EXTERN_C void
FORTRAN_FUNC(dgemm,DGEMM)
(char *transa, char *transb, Fint *m, Fint *n, Fint *k,
 double *alpha, double *a, Fint *lda, double *b, Fint *ldb,
 double *beta, double *c, Fint *ldc);

EXTERN_C void
FORTRAN_FUNC(add_hash_block,ADD_HASH_BLOCK)
(Fint *d_file, double *array, Fint *size, Fint *hash, Fint *key);

EXTERN_C void
FORTRAN_FUNC(tce_sortacc_4,TCE_SORTACC_4)
(double *unsorted , double *sorted,
 Fint *a, Fint *b, Fint *c, Fint *d,
 Fint *i, Fint *j, Fint *k, Fint *l,
 double *factor);

EXTERN_C void
FORTRAN_FUNC(tce_sortacc_6,TCE_SORTACC_6)
(double *unsorted , double *sorted,
 Fint *a, Fint *b, Fint *c, Fint *d, Fint *e, Fint *f,
 Fint *i, Fint *j, Fint *k, Fint *l, Fint *m, Fint *n,
 double *factor);

#endif /*__ctce_fapi_h__*/

