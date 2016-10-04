#ifndef __tamm_fapi_h__
#define __tamm_fapi_h__

#include "common.h"

/**
 * All fortran functions used by tamm
 */

#define ftce_restricted_2  FORTRAN_FUNC(tce_restricted_2,TCE_RESTRICTED_2)
#define ftce_restricted_4  FORTRAN_FUNC(tce_restricted_4,TCE_RESTRICTED_4)
#define fget_hash_block_ma FORTRAN_FUNC(get_hash_block_ma,GET_HASH_BLOCK_MA)
#define fget_hash_block    FORTRAN_FUNC(get_hash_block,GET_HASH_BLOCK)
#define fget_hash_block_i  FORTRAN_FUNC(get_hash_block_i,GET_HASH_BLOCK_I)
#define fget_block_ind_i   FORTRAN_FUNC(get_block_ind_i,GET_BLOCK_IND_I)
#define ftce_sort_2        FORTRAN_FUNC(tce_sort_2,TCE_SORT_2)
#define ftce_sort_2        FORTRAN_FUNC(tce_sort_2,TCE_SORT_2)
#define ftce_sort_4        FORTRAN_FUNC(tce_sort_4,TCE_SORT_4)
#define fdgemm             FORTRAN_FUNC(dgemm,DGEMM)
#define fadd_hash_block    FORTRAN_FUNC(add_hash_block,ADD_HASH_BLOCK)
#define fadd_block         FORTRAN_FUNC(add_block,ADD_BLOCK)
#define fget_block         FORTRAN_FUNC(get_block,GET_BLOCK)
#define ftce_hash          FORTRAN_FUNC(tce_hash,TCE_HASH)
#define ftce_sortacc_2     FORTRAN_FUNC(tce_sortacc_2,TCE_SORTACC_2)
#define ftce_sortacc_4     FORTRAN_FUNC(tce_sortacc_4,TCE_SORTACC_4)
#define ftce_sortacc_6     FORTRAN_FUNC(tce_sortacc_6,TCE_SORTACC_6)
#define fname_and_create   FORTRAN_FUNC(name_and_create,NAME_AND_CREATE)
#define fdestroy           FORTRAN_FUNC(destroy,DESTROY)
#define ftce_hash_v2       FORTRAN_FUNC(tce_hash_v2,TCE_HASH_V2)

EXTERN_C void
ftce_restricted_2(Fint *i1, Fint *i2, Fint *o1, Fint *o2);

EXTERN_C void
ftce_restricted_4(Fint *i1, Fint *i2, Fint *i3, Fint *i4,
                  Fint *o1, Fint *o2, Fint *o3, Fint *o4);

EXTERN_C void
fget_hash_block_ma(double *big_array, double *array,
                   Fint *size, Fint *hash, Fint *key);

EXTERN_C void
fget_hash_block(Fint *darr, double *buf,
                Fint *size, Fint *hash, Fint *key);

EXTERN_C void
fget_hash_block_i(Fint *darr, double *buf,
                  Fint *size, Fint *hash, Fint *key,
                  Fint *g2b, Fint *g1b, Fint *g4b, Fint *g3b);

EXTERN_C void
fget_block_ind_i(Fint *ida, double *buf, Fint *isize, Fint *ikey, Fint *indexc,
                 Fint *is3, Fint *is2, Fint *is1, Fint *is0);

EXTERN_C void
ftce_sort_2(double *unsorted , double *sorted, Fint *a, Fint *b,
            Fint *i, Fint *j, double *factor);

EXTERN_C void
ftce_sort_4(double *unsorted , double *sorted,
            Fint *a, Fint *b, Fint *c, Fint *d,
            Fint *i, Fint *j, Fint *k, Fint *l,
            double *factor);

EXTERN_C void
fdgemm(char *transa, char *transb, BlasInt *m, BlasInt *n, BlasInt *k,
       double *alpha, double *a, BlasInt *lda, double *b, BlasInt *ldb,
       double *beta, double *c, BlasInt *ldc);

EXTERN_C void
fadd_hash_block(Fint *d_file, double *array,
                Fint *size, Fint *hash, Fint *key);

EXTERN_C void
ftce_sortacc_2(double *unsorted , double *sorted,
               Fint *a, Fint *b, Fint *i, Fint *j,
               double *factor);

EXTERN_C void
ftce_sortacc_4(double *unsorted , double *sorted,
               Fint *a, Fint *b, Fint *c, Fint *d,
               Fint *i, Fint *j, Fint *k, Fint *l,
               double *factor);

EXTERN_C void
ftce_sortacc_6(double *unsorted , double *sorted,
               Fint *a, Fint *b, Fint *c, Fint *d, Fint *e, Fint *f,
               Fint *i, Fint *j, Fint *k, Fint *l, Fint *m, Fint *n,
               double *factor);

EXTERN_C void
ftce_hash(Fint *hash, Fint *key, Fint *offset);

EXTERN_C void
fadd_block(Fint *d_a, double *buf, Fint *size, Fint *offset);

EXTERN_C void
fget_block(Fint *d_a, double *buf, Fint *size, Fint *offset);

EXTERN_C void
fname_and_create(Fint *da, Fint *size);

EXTERN_C void
fdestroy(Fint *da, Fint *offset);

EXTERN_C void
ftce_hash_v2(Fint *hash, Fint *key, Fint *offset);


#endif /*__tamm_fapi_h__*/

