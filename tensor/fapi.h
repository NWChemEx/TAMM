#ifndef __FAPI_H__
#define __FAPI_H__

#ifdef __cplusplus 
extern "C" {
#endif

#include <stddef.h>

/* #include "macdecls.h" */
#ifdef __cplusplus 
typedef long Integer;
#else
typedef long long Integer;
#endif

  /* all the FORTRAN apis */

  void tce_restricted_2_(Integer *i1,Integer *i2, Integer *o1, Integer *o2);

  void tce_restricted_4_(Integer *i1,Integer *i2, Integer *i3, Integer *i4,
      Integer *o1, Integer *o2, Integer *o3, Integer *o4);

  void get_hash_block_ma_(double *big_array, double *array, Integer *size, Integer *hash, Integer *key);

  void get_hash_block_(Integer *darr, double *buf, Integer *size, Integer *hash, Integer *key);

  void get_hash_block_i_(Integer *darr, double *buf, Integer *size, Integer *hash, Integer *key, 
      Integer *g2b, Integer *g1b, Integer *g4b, Integer *g3b);

  void tce_sort_2_(double *unsorted , double *sorted,
      Integer *a, Integer *b,
      Integer *i, Integer *j,
      double *factor);

  void tce_sort_4_(double *unsorted , double *sorted,
      Integer *a, Integer *b, Integer *c, Integer *d,
      Integer *i, Integer *j, Integer *k, Integer *l,
      double *factor);

  void dgemm_(char *transa, char *transb,
      Integer *m, Integer *n, Integer *k,
      double *alpha,
      double *a, Integer *lda,
      double *b, Integer *ldb,
      double *beta,
      double *c, Integer *ldc);

  void add_hash_block_(Integer *d_file, double *array, Integer *size, Integer *hash, Integer *key);

  void tce_sortacc_4_(double *unsorted , double *sorted,
      Integer *a, Integer *b, Integer *c, Integer *d,
      Integer *i, Integer *j, Integer *k, Integer *l,
      double *factor);

  void tce_sortacc_6_(double *unsorted , double *sorted,
      Integer *a, Integer *b, Integer *c, Integer *d, Integer *e, Integer *f,
      Integer *i, Integer *j, Integer *k, Integer *l, Integer *m, Integer *n,
      double *factor);

  void ccsd_t_singles_l_(double *k_singles, Integer *k_t1_local,
              Integer *d_v2, Integer *k_t1_offset, Integer *k_v2_offset,
              Integer *t_h1b, Integer *t_h2b, Integer *t_h3b, 
              Integer *t_p4b, Integer *t_p5b, Integer *t_p6b, Integer *toggle);

  void ccsd_t_doubles_l_(double *k_doubles, Integer *d_t2,
              Integer *d_v2, Integer *k_t2_offset, Integer *k_v2_offset,
              Integer *t_h1b, Integer *t_h2b, Integer *t_h3b, 
              Integer *t_p4b, Integer *t_p5b, Integer *t_p6b, Integer *toggle);

  void ccsd_t_doubles_(double *k_doubles, Integer *d_t2,
              Integer *d_v2, Integer *k_t2_offset, Integer *k_v2_offset,
              Integer *t_h1b, Integer *t_h2b, Integer *t_h3b, 
              Integer *t_p4b, Integer *t_p5b, Integer *t_p6b, Integer *toggle);


  void fget_hash_block_ma(void *big_array, double *buf, size_t size, void *hash_map, size_t key);

  void fget_hash_block(void *darr, double *buf, size_t size, void *hash_map, size_t key);
 
  void fadd_hash_block(void *darr, double *buf, size_t size, void *hash_map, size_t key);
 
  void fget_hash_block_i(void *darr, double *buf, size_t size, void *hash_map, size_t key, 
      size_t g3b, size_t g2b, size_t g1b, size_t g0b);

#ifdef __cplusplus 
}
#endif


#endif /*__FAPI_H__*/

