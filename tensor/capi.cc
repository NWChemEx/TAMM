#include "common.h"

#include "capi.h"
#include "index_sort.h"

#include CBLAS_HEADER_FILE

#include "fapi.h"
#include "index.h"
#include "variables.h"

namespace ctce {

void
ctce_sort(double *sbuf, double *dbuf, const std::vector<size_t>& ids,
          std::vector<size_t>& iv, double alpha) {
  assert(ids.size() == iv.size());

#if USE_FORTRAN_FUNCTIONS
  vector<Fint> dims(ids.size()), ip(iv.size());
  for(int i=0; i<ids.size(); i++) {
    dims[i] = Variables::k_range(ids[i]);
    ip[i] = iv[i];
  }
  if(ids.size() == 0) {
    dbuf[0] = sbuf[0] * alpha;
  } else if(ids.size() == 1) {
    for(int i=0; i< dims[0]; i++) {
      dbuf[i] = sbuf[i] * alpha;
    }
  } else if (ids.size()==2) {
    ftce_sort_2(sbuf, dbuf, &dims[0], &dims[1], &ip[0], &ip[1], &alpha);
  } else if (ids.size()==3) {
    ftce_sort_4(sbuf, dbuf, &dims[0], &dims[1], &dims[2], &dims[3],
                &ip[0], &ip[1], &ip[2], &ip[3], &alpha);
  } else if (ids.size()==4) {
    ftce_sort_4(sbuf, dbuf, &dims[0], &dims[1], &dims[2], &dims[3],
                &ip[0], &ip[1], &ip[2], &ip[3], &alpha);
  } else {
    assert(0); //not implemented
  }
#else
  vector<size_t> sizes;
  vector<int> perm;
  for(int i=0; i<ids.size(); i++) {
    sizes.push_back(Variables::k_range(ids[i]));
    perm.push_back(iv[i]);
  }
  index_sort(sbuf, dbuf, ids.size(), &sizes[0], &perm[0], alpha);
#endif
}

void
ctce_hash(Fint *hash, size_t key, Fint *offset) {

#if USE_FORTRAN_FUNCTIONS
  Fint ikey = key;
  ftce_hash(hash, &ikey, offset);
#else
  Fint length = hash[0];
  Fint ikey=key;
  Fint *ptr = std::lower_bound(&hash[1], &hash[length+1], key);
  if(ptr == &hash[length+1] || ikey < *ptr) {
    fprintf(stderr,"ctce_hash: key not found");
    assert(0);
  }
  *offset = *(ptr + length);
#endif
}

void
cadd_block(size_t d_a, double *buf, size_t size, size_t offset) {

#if USE_FORTRAN_FUNCTIONS
  Fint ida = d_a;
  Fint isize = size;
  Fint ioffset = offset;
  fadd_block(&ida, buf, &isize, &ioffset);
#else
  int lo[2] = {0,offset};
  int hi[2] = {0,offset+size-1};
  int ld[1] = {100000};
  double alpha = 1.0;
  NGA_Acc(d_a,lo,hi,buf,ld,&alpha);
#endif
}

void
cadd_hash_block(size_t d_c, double *buf_a, size_t size, Fint *hash, size_t key) {

#if USE_FORTRAN_FUNCTIONS
  Fint ida = d_c;
  Fint isize = size;
  Fint ikey = key;
  fadd_hash_block(&ida, buf_a, &isize, hash, &ikey);
#else
  Fint offset;
  ctce_hash(hash, key, &offset);
  cadd_block(d_c, buf_a, size, offset);
#endif
}

void
cget_block(size_t d_a, double *buf, size_t size, size_t offset) {

#if USE_FORTRAN_FUNCTIONS
  Fint ida = d_a;
  Fint isize = size;
  Fint ioffset = offset;
  fget_block(&ida, buf, &isize, &ioffset);
#else
  int lo[2] = {0,offset};
  int hi[2] = {0,offset+size-1};
  int ld[1] = {100000};
  NGA_Get(d_a,lo,hi,buf,ld);
#endif
}

void
cdgemm(char transa, char transb, size_t m, size_t n, size_t k,
       double alpha, double *a, size_t lda, double *b, size_t ldb, double beta,
       double *c, size_t ldc) {
#if USE_FORTRAN_FUNCTIONS
  BlasInt im=m, in=n, ik=k, ilda=lda, ildb=ldb,ildc=ldc;
  fdgemm(&transa, &transb, &im, &in, &ik, &alpha, a, &ilda,
         b, &ildb, &beta, c, &ildc);
#else
  CBLAS_TRANSPOSE TransA = (transa=='N') ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE TransB = (transb=='N') ? CblasNoTrans : CblasTrans;
  cblas_dgemm(CblasColMajor, TransA, TransB,
              m, n, k, alpha, a, lda, b, ldb,
              beta, c, ldc);
#endif
}

void
ctce_sortacc(double *sbuf, double *dbuf, const std::vector<size_t>& ids,
             std::vector<size_t>& perm, double alpha) {
  assert(ids.size() == perm.size());

#if USE_FORTRAN_FUNCTIONS
  vector<Fint> dims, ip;
  for(int i=0; i<ids.size(); i++) {
    dims[i] = Variables::k_range(ids[i]);
    ip[i] = perm[i];
  }
  if (ids.size()==4) {
    ftce_sortacc_4(sbuf, dbuf, &dims[0], &dims[1], &dims[2], &dims[3],
                   &ip[0], &ip[1], &ip[2], &ip[3], &alpha);
  } else if (ids.size()==6) {
    ftce_sortacc_6(sbuf, dbuf,
                   &dims[0], &dims[1], &dims[2], &dims[3], &dims[4], &dims[5],
                   &ip[0], &ip[1], &ip[3], &ip[3], &ip[4], &ip[5], &alpha);
  } else {
    assert(0); //not implemented
  }
#else
  vector<size_t> sizes;
  vector<int> perms;
  for(int i=0; i<ids.size(); i++) {
    sizes.push_back(Variables::k_range(ids[i]));
    perms.push_back(perm[i]);
  }
  index_sortacc(sbuf, dbuf, ids.size(), &sizes[0], &perms[0], alpha);
#endif
}


void
ctce_restricted(int dim, int nupper,
                const std::vector<size_t> &value,
                std::vector<size_t> &pvalue_r) {
#if USE_FORTRAN_FUNCTIONS
  std::vector<Fint> temp(dim);
  std::vector<Fint> ivalue(dim);
  assert(value.size() == dim);
  temp.resize(dim);
  Fint dummy0=1, dummy1=1;
  for(int i=0; i<dim; i++) {
    ivalue[i] = value[i];
  }

  if(dim==1) {
    assert(nupper==0);
    ftce_restricted_2(&dummy0, &ivalue[0],&dummy1,&temp[0]);
  } else if (dim==2)   {
    ftce_restricted_2(&ivalue[0],&ivalue[1],&temp[0],&temp[1]);
  } else if(dim==3) {
    assert(nupper==1);
    ftce_restricted_4(&dummy0,&ivalue[0],&ivalue[1],&ivalue[2],
                      &dummy1,&temp[0],&temp[1],&temp[2]);
  } else if (dim==4) {
    ftce_restricted_4(&ivalue[0],&ivalue[1],&ivalue[2],&ivalue[3],
                      &temp[0],&temp[1],&temp[2],&temp[3]);
  } else {
    assert(0);
  }
  pvalue_r.clear();
  for (int i=0; i<dim; i++) {
    pvalue_r.push_back(temp[i]);
  }
#else
  int lval = 0;
  Fint *int_mb = Variables::int_mb();
  size_t k_spin = Variables::k_spin()-1;
  for (int i=0; i<value.size(); i++) {
    lval += int_mb[k_spin+value[i]];
  }
  if(Variables::restricted() && (dim!=0) && (dim%2==0) && lval==2*dim) {
    pvalue_r.resize(dim);
    Fint k_alpha = Variables::k_alpha()-1;
    for(int i=0; i<dim; i++) {
      pvalue_r[i] = int_mb[value[i] + k_alpha];
    }
  }  else {
    pvalue_r = value;
  }
#endif
}

void
cget_hash_block_i(size_t da, double *buf, size_t size, size_t offset,
                  size_t key, std::vector<size_t> &is) {
  Fint ida = da;
  assert(is.size()==4);
  Fint is0=is[0], is1=is[1], is2=is[2], is3=is[3];
  Fint isize = size;
  Fint *int_mb = Variables::int_mb();
  Fint ikey = key;
  fget_hash_block_i(&ida, buf, &isize, &int_mb[offset], &ikey,
                    &is3, &is2, &is1, &is0);
}

void
cget_hash_block_ma(size_t da, double *buf, size_t size,
                   size_t offset, size_t key) {
#if USE_FORTRAN_FUNCTIONS
  double *dbl_mb = Variables::dbl_mb();
  Fint isize = size;
  Fint *int_mb = Variables::int_mb();
  Fint ikey = key;
  fget_hash_block_ma(&dbl_mb[da], buf, &isize, &int_mb[offset], &ikey);
#else
  Fint *int_mb = Variables::int_mb();
  double *dbl_mb = Variables::dbl_mb();
  Fint *hash = &int_mb[offset];
  Fint ioffset;
  ctce_hash(hash, key, &ioffset);
  memcpy(buf,&dbl_mb[da]+ioffset,size*sizeof(double));
#endif
}

void
cget_hash_block(size_t da, double *buf, size_t size, size_t offset, size_t key) {

#if USE_FORTRAN_FUNCTIONS
  Fint ida = da;
  Fint isize = size;
  Fint *int_mb = Variables::int_mb();
  Fint  ikey = key;
  fget_hash_block(&ida, buf, &isize, &int_mb[offset], &ikey);
#else
  Fint *int_mb = Variables::int_mb();
  Fint *hash = &int_mb[offset];
  Fint ioffset;
  ctce_hash(hash, key, &ioffset);
  cget_block(da, buf, size, ioffset);
#endif
}

} /* namespace ctce*/
