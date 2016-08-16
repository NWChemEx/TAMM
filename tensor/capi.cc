#include "capi.h"
#include "index_sort.h"
#if defined(CBLAS)
#include <cblas.h>
#endif

namespace ctce {

#define TIMER 1
  void tce_sort(double *sbuf, double *dbuf, const std::vector<Integer>& ids, std::vector<Integer>& iv, double alpha) {
    Integer *int_mb = Variables::int_mb();
    Integer k_range = Variables::k_range()-1;
#if TIMER
    double start = rtclock();
#endif

    {
      vector<size_t> sizes;
      vector<int> perm;
      for(int i=0; i<ids.size(); i++) {
        sizes.push_back(int_mb[k_range+ids[i]]);
        perm.push_back(iv[i]);
      }
      index_sort(sbuf, dbuf, ids.size(), &sizes[0], &perm[0], alpha);
      return;
    }

    if(ids.size() == 0) {
      dbuf[0] = sbuf[0] * alpha;
    }
    else if(ids.size() == 1) {
      for(int i=0; i< int_mb[k_range+ids[0]]; i++) {
	dbuf[i] = sbuf[i] * alpha;
      }
    }
    else if (ids.size()==2) {
      tce_sort_2_(sbuf, dbuf, &int_mb[k_range+ids[0]], &int_mb[k_range+ids[1]], &iv[0], &iv[1], &alpha);
    }
    else if (ids.size()==3) {
      Integer *rmb = int_mb + k_range;
      // Integer dim1 = 1, dim2 = rmb[ids[0]], dim3 = rmb[ids[1]], dim4 = rmb[ids[2]];
      // Integer perm1 = 1, perm2 = iv[0]+1, perm3=iv[1]+1, perm4=iv[2]+1;

      // tce_sort_4_(sbuf, dbuf, &dim1, &dim2, &dim3, &dim4,
      // 		  &perm1, &perm2, &perm3, &perm4, &alpha);
      Integer dim1 = rmb[ids[0]], dim2 = rmb[ids[1]], dim3 = rmb[ids[2]], dim4 = 1;
      Integer perm1 = iv[0], perm2=iv[1], perm3=iv[2], perm4=4;

      tce_sort_4_(sbuf, dbuf, &dim1, &dim2, &dim3, &dim4,
		  &perm1, &perm2, &perm3, &perm4, &alpha);
    }
    else if (ids.size()==4) {
      tce_sort_4_(sbuf, dbuf, &int_mb[k_range+ids[0]], &int_mb[k_range+ids[1]], &int_mb[k_range+ids[2]], &int_mb[k_range+ids[3]],
          &iv[0], &iv[1], &iv[2], &iv[3], &alpha);
    }
    else {
      assert(0); //not implemented
    }
#if TIMER
    double end = rtclock();
    Timer::so_time += end - start;
    Timer::so_num += 1;
#endif
  }

extern "C" {
  void tce_hash_(Integer *hash, Integer *key, Integer *offset);
  void add_block_(Integer *d_a, double *buf, Integer *size, Integer *offset);
}

void ctce_hash(Integer *hash, Integer key, Integer *offset)
{
#if 0
  tce_hash_(hash, &key, offset);
#else
  Integer length = hash[0];
  Integer *ptr = std::lower_bound(&hash[1], &hash[length+1], key);
  if(ptr == &hash[length+1] || key < *ptr) {
    fprintf(stderr,"ctce_hash: key not found");
    assert(0);
  }
  *offset = *(ptr + length);
#endif
}

void cadd_block(Integer d_a, double *buf, Integer size, Integer offset)
{
#if 0
  add_block_(&d_a, buf, &size, &offset);
#else
  {
    int lo[2] = {0,offset};
    int hi[2] = {0,offset+size-1};
    int ld[1] = {100000};
    double alpha = 1.0;
    NGA_Acc(d_a,lo,hi,buf,ld,&alpha);
  }
#endif
}

void cadd_hash_block(Integer d_c, double *buf_a, Integer size, Integer *hash, Integer key) {
  Integer offset;
#if 1
  ctce_hash(hash, key, &offset);
  cadd_block(d_c, buf_a, size, offset);
#else
  add_hash_block_(&d_c, buf_a, &size, hash, &key);
#endif
}

  void tce_add_hash_block_(Integer *d_c, double *buf_a, Integer size, Integer k_c_offset, const std::vector<Integer>& is, const std::vector<IndexName>& ns) {
    Integer *int_mb = Variables::int_mb();
    Integer noab = Variables::noab();
    Integer nvab = Variables::nvab();
    Integer key=0, offset=1;
    for (int i=is.size()-1; i>=0; i--) {
      bool check = (Table::rangeOf(ns[i])==TO);
      if (check) key += (is[i]-1)*offset;
      else key += (is[i]-noab-1)*offset; // TV
      offset *= (check)?noab:nvab;
    }

//#define TIMER 0
#if TIMER
    double start = rtclock();
#endif
    //    std::cout << "ckey" << key << std::endl;

#if 0
    add_hash_block_(d_c, buf_a, &size, &int_mb[k_c_offset], &key);
#else
    cadd_hash_block(*d_c, buf_a, size, &int_mb[k_c_offset], key);
#endif
#if TIMER
    double end = rtclock();
    Timer::ah_time += end - start;
    Timer::ah_num += 1;
#endif
  }

  void cdgemm(char transa, char transb, Integer m, Integer n, Integer k,
      double alpha, double *a, Integer lda, double *b, Integer ldb, double beta,
      double *c, Integer ldc) {
#if TIMER
    double start = rtclock();
#endif
    
#if defined(CBLAS)
    CBLAS_TRANSPOSE TransA = (transa=='N') ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE TransB = (transb=='N') ? CblasNoTrans : CblasTrans;
    cblas_dgemm(CblasColMajor, TransA, TransB,
                m, n, k, alpha, a, lda, b, ldb,
                beta, c, ldc);
#else
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#endif

#if TIMER
    double end = rtclock();
    Timer::dg_time += end - start;
    Timer::dg_num += 1;
#endif
  }

  void sortacc(double *sbuf, double *dbuf, const std::vector<Integer>& ids, std::vector<Integer>& perm, double alpha) {
    Integer *int_mb = Variables::int_mb();
    Integer k_range = Variables::k_range()-1;
#if TIMER
    double start = rtclock();
#endif

    {
      vector<size_t> sizes;
      vector<int> perms;
      for(int i=0; i<ids.size(); i++) {
        sizes.push_back(int_mb[k_range+ids[i]]);
        perms.push_back(perm[i]);
      }
      index_sortacc(sbuf, dbuf, ids.size(), &sizes[0], &perms[0], alpha);
      return;
    }

    if (ids.size()==4) {
      tce_sortacc_4_(sbuf, dbuf, &int_mb[k_range+ids[0]], &int_mb[k_range+ids[1]], &int_mb[k_range+ids[2]], 
          &int_mb[k_range+ids[3]], &perm[0], &perm[1], &perm[2], &perm[3], &alpha);
    }
    else if (ids.size()==6) {
      tce_sortacc_6_(sbuf, dbuf, &int_mb[k_range+ids[0]], &int_mb[k_range+ids[1]], &int_mb[k_range+ids[2]], 
          &int_mb[k_range+ids[3]], &int_mb[k_range+ids[4]], &int_mb[k_range+ids[5]],
          &perm[0], &perm[1], &perm[2], &perm[3], &perm[4], &perm[5], &alpha);
    }
    else {
      assert(0);
    }
#if TIMER
    double end = rtclock();
    Timer::sa_time += end - start;
    Timer::sa_num += 1;
#endif
  }


}; // namespace ctce
