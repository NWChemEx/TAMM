#include "capi.h"
#include "index_sort.h"
#if defined(CBLAS)
#include <cblas.h>
#endif

namespace ctce {

#define TIMER 1
  void tce_sort(double *sbuf, double *dbuf, const std::vector<size_t>& ids, std::vector<size_t>& iv, double alpha) {
    Integer *int_mb = Variables::int_mb();
    size_t k_range = Variables::k_range()-1;
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
      Integer iv0 = iv[0], iv1=iv[1];
      tce_sort_2_(sbuf, dbuf, &int_mb[k_range+ids[0]], &int_mb[k_range+ids[1]], &iv0, &iv1, &alpha);
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
      Integer iv0 = iv[0], iv1=iv[1], iv2= iv[2], iv3=iv[3];
      tce_sort_4_(sbuf, dbuf, &int_mb[k_range+ids[0]], &int_mb[k_range+ids[1]], &int_mb[k_range+ids[2]], &int_mb[k_range+ids[3]],
          &iv0, &iv1, &iv2, &iv3, &alpha);
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

void ctce_hash(Integer *hash, size_t key, Integer *offset)
{
#if 0
  tce_hash_(hash, &key, offset);
#else
  Integer length = hash[0];
  Integer ikey=key;
  Integer *ptr = std::lower_bound(&hash[1], &hash[length+1], key);
  if(ptr == &hash[length+1] || ikey < *ptr) {
    fprintf(stderr,"ctce_hash: key not found");
    assert(0);
  }
  *offset = *(ptr + length);
#endif
}

void cadd_block(size_t d_a, double *buf, size_t size, size_t offset)
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

void cadd_hash_block(size_t d_c, double *buf_a, size_t size, Integer *hash, size_t key) {
  Integer offset;
  Integer ida = d_c;
  Integer isize = size;
  Integer ikey = key;
#if 1
  ctce_hash(hash, key, &offset);
  cadd_block(d_c, buf_a, size, offset);
#else
  add_hash_block_(&ida, buf_a, &isize, hash, &ikey);
#endif
}

#if 0
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
#endif

  void cdgemm(char transa, char transb, size_t m, size_t n, size_t k,
      double alpha, double *a, size_t lda, double *b, size_t ldb, double beta,
      double *c, size_t ldc) {
#if TIMER
    double start = rtclock();
#endif
    Integer im=m, in=n, ik=k, ilda=lda, ildb=ldb,ildc=ldc;
#if defined(CBLAS)
    CBLAS_TRANSPOSE TransA = (transa=='N') ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE TransB = (transb=='N') ? CblasNoTrans : CblasTrans;
    cblas_dgemm(CblasColMajor, TransA, TransB,
                m, n, k, alpha, a, lda, b, ldb,
                beta, c, ldc);
#else
    dgemm_(&transa, &transb, &im, &in, &ik, &alpha, a, &ilda, b, &ildb, &beta, c, &ildc);
#endif

#if TIMER
    double end = rtclock();
    Timer::dg_time += end - start;
    Timer::dg_num += 1;
#endif
  }

  void sortacc(double *sbuf, double *dbuf, const std::vector<size_t>& ids, std::vector<size_t>& perm, double alpha) {
    Integer *int_mb = Variables::int_mb();
    size_t k_range = Variables::k_range()-1;
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
      Integer p0=perm[0], p1=perm[1], p2=perm[2], p3=perm[3];
      tce_sortacc_4_(sbuf, dbuf, &int_mb[k_range+ids[0]], &int_mb[k_range+ids[1]], &int_mb[k_range+ids[2]], 
          &int_mb[k_range+ids[3]], &p0, &p1, &p2, &p3, &alpha);
    }
    else if (ids.size()==6) {
      Integer p0=perm[0], p1=perm[1], p2=perm[2], p3=perm[3], p4=perm[4], p5=perm[5];
      tce_sortacc_6_(sbuf, dbuf, &int_mb[k_range+ids[0]], &int_mb[k_range+ids[1]], &int_mb[k_range+ids[2]], 
          &int_mb[k_range+ids[3]], &int_mb[k_range+ids[4]], &int_mb[k_range+ids[5]],
          &p0, &p1, &p3, &p3, &p4, &p5, &alpha);
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


  void ctce_restricted(int dim, int nupper,
                       const std::vector<size_t> &value,
                       std::vector<size_t> &pvalue_r) {
    std::vector<Integer> temp(dim);
    std::vector<Integer> ivalue(dim);
    assert(value.size() == dim);
    temp.resize(dim);
    Integer dummy0=1, dummy1=1;
    for(int i=0; i<dim; i++) {
      ivalue[i] = value[i];
    }
    if(dim==1) {
      assert(nupper==0);
      tce_restricted_2_(&dummy0, &ivalue[0],&dummy1,&temp[0]);
    }
    else if (dim==2)   {
      tce_restricted_2_(&ivalue[0],&ivalue[1],&temp[0],&temp[1]);
    }
    else if(dim==3) {
      assert(nupper==1);
      tce_restricted_4_(&dummy0,&ivalue[0],&ivalue[1],&ivalue[2],
			&dummy1,&temp[0],&temp[1],&temp[2]);
    }
    else if (dim==4) {
      tce_restricted_4_(&ivalue[0],&ivalue[1],&ivalue[2],&ivalue[3],
          &temp[0],&temp[1],&temp[2],&temp[3]);
    }
    else {
      assert(0);
    }
    pvalue_r.clear();
    for (int i=0; i<dim; i++) {
      //ids_[i].setValueR(temp[i]);
      //value_r_[i] = temp[i];
      pvalue_r.push_back(temp[i]);
    }
  }

}; // namespace ctce
