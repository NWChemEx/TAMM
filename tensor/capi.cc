#include "common.h"

#include "capi.h"
#include "index_sort.h"

#ifndef LINUX_BLAS 
#include CBLAS_HEADER_FILE
#endif

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
    Fint dimx = 1, permx = 4;
    ftce_sort_4(sbuf, dbuf, &dims[0], &dims[1], &dims[2], &dimx,
                &permx, &ip[0], &ip[1], &ip[2], &alpha);
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
  //Fint *int_mb = Variables::int_mb();
  //size_t k_range = Variables::k_range()-1;
  vector<Fint> dims(ids.size()), ip(ids.size());
  for(int i=0; i<ids.size(); i++) {
    dims[i] = Variables::k_range(ids[i]);
    //dims[i] = int_mb[k_range+ids[i]];
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
    //assert(nupper==0);
    ftce_restricted_2(&dummy0, &ivalue[0],&dummy1,&temp[0]);
  } else if (dim==2)   {
    ftce_restricted_2(&ivalue[0],&ivalue[1],&temp[0],&temp[1]);
  } else if(dim==3) {
    //assert(nupper==1);
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

static std::vector<size_t>
invert_perm(std::vector<size_t> &perm) {
  std::vector<size_t> perm_out;
  int n = perm.size();
  for(size_t i=n; i>=1; i--) {
    perm_out.push_back(n - (std::find(perm.begin(), perm.end(), i) - perm.begin()));
  }
  return perm_out;
}

static size_t
index_pair(int i, int j) {
  return i*(i-1)/2 + j;
}

static bool
is_spin_nonzero(const vector<size_t> &ids) {
  int lval=0, rval=0;
  Fint *int_mb = Variables::int_mb();
  Fint k_spin = Variables::k_spin()-1;
  int nupper = ids.size()/2;
  int dim = ids.size();
  assert(dim == 2*nupper);
  for(int i=0; i<nupper; i++) lval += int_mb[k_spin+ids[i]];
  for(int i=nupper; i<dim; i++) rval += int_mb[k_spin+ids[i]];
  return (rval - lval == dim - 2*nupper);
}

int
cget_add_ind_i(size_t da, double *buf, size_t size, size_t offset_unused,
               size_t key_unused, const std::vector<size_t> &is_in,
               std::vector<size_t> &is_out, std::vector<size_t>& perm,
               string name, ga_nbhdl_t *nbh) {
  assert(is_in.size() == 4);
  assert(perm.size() == 4);
  vector<size_t> isa(is_in.size());
  // if(!is_spin_nonzero(is_in)) {
  //   cout<<name<<" skip"<<endl;
  //   return false;
  // }

  Fint *int_mb = Variables::int_mb();
  size_t k_b2am = Variables::k_b2am();
  for(int i=0; i<is_in.size(); i++) {
    isa[i] = int_mb[k_b2am + is_in[i] - 1];
    // cout<<"is_in["<<i<<"]="<<is_in[i]<<endl;
    //cout<<"isa_in["<<i<<"]="<<isa[i]<<endl;
  }

  is_out = is_in;

  bool a=false, b=false, c=false;

  if(isa[1] < isa[0]) {
    std::swap(isa[0], isa[1]);
    std::swap(perm[2], perm[3]);
    std::swap(is_out[0], is_out[1]);
    b = true;
  }
  if(isa[3] < isa[2]) {
    std::swap(isa[2], isa[3]);
    std::swap(is_out[2], is_out[3]);
    std::swap(perm[0], perm[1]);
    a = true;
  }
  size_t icol = index_pair(isa[1], isa[0]);
  size_t irow = index_pair(isa[3], isa[2]);
  if(irow < icol) {
    std::swap(isa[0], isa[2]);
    std::swap(perm[0], perm[2]);
    std::swap(is_out[0], is_out[2]);
    std::swap(isa[1], isa[3]);
    std::swap(is_out[1], is_out[3]);
    std::swap(perm[1], perm[3]);
    c = true;
  }

  int x = 0;
  x |= a ? 4 : 0;
  x |= b ? 2 : 0;
  x |= c ? 1 : 0;

  Fint key = 0;
  Fint k_v2_alpha_offset = Variables::k_v2_alpha_offset();
  Fint offset;
  size_t noa = Variables::noa();
  size_t nva = Variables::nva();
  //cout<<"noa="<<noa<<" nva="<<nva<<endl;
#if 0
  for(int i=0; i<=3; i++) {
    //cout<<"isa["<<i<<"]="<<isa[i]<<endl;
    key = key * (noa+nva) + (isa[i]-1);
  }
#else
  size_t nova = noa+nva;
  key = (isa[1]-1) + nova*(isa[0]-1 + nova*(isa[3]-1 + nova*(isa[2]-1)));
#endif
  //cout<<name<<" choice: "<<a<<" "<<b<<" "<<c<<" "<<endl;
  //cout<<name<<" key: "<<key<<endl;

  //cout<<"v2orb alpha key="<<key<<endl;
  /** @bug Works only for ccsd. Needs to be modified for
   * ccsd_act. Check get_block_ind.F */
  ftce_hash_v2(&int_mb[k_v2_alpha_offset],&key,&offset);
  //ctce_hash(&int_mb[k_v2_alpha_offset],key,&offset);

  int lo[2] = {0,offset};
  int hi[2] = {0,offset+size-1};
  int ld[1] = {100000}; //ignored
  int d_v2orb = Variables::d_v2orb();
  NGA_NbGet(d_v2orb,lo,hi,buf,ld,nbh);

  return x;
}

void
cget_block_ind_i(size_t da, double *buf, size_t size, size_t offset,
                 size_t key, std::vector<size_t> &is) {
#if USE_FORTRAN_FUNCTIONS || 1
  Fint ida = da;
  Fint isize = size;
  assert(is.size()==4);
  Fint is0=is[0], is1=is[1], is2=is[2], is3=is[3];
  Fint ikey = key;
  Fint *int_mb = Variables::int_mb();
  Fint *indexc = &int_mb[offset]; /*seems unused*/
  fget_block_ind_i(&ida, buf, &isize, &ikey, indexc,
                   &is3, &is2, &is1, &is0);
#else
  memset(buf, 0, size*sizeof(double));
  Fint *int_mb = Variables::int_mb();
  Fint k_range = Variables::k_range()-1;
  Fint k_spin = Variables::k_spin()-1;

  assert(is.size()==4);
  assert(is_spin_nonzero(is));

  ga_nbhdl_t nbh1, nbh2;
  bool comm1=false, comm2=false;
  double *bufa=NULL, *bufb=NULL;
  vector<size_t> vperma, vpermb;
  vector<size_t> visa_out, visb_out;

  {
    size_t isa[4] = {is[3], is[1], is[2], is[0]};
    // int perma[4] = {2, 4, 1, 3};
    int perma[4] = {4,2,3,1};
    vector<size_t> visa;//, visa_out;
    visa.insert(visa.end(), isa, isa+4);
    vperma.insert(vperma.end(), perma, perma+4);
    //cout<<"--visa =";
    for(int i=visa.size()-1; i>=0; i--) {
      //cout<<" "<<visa[i];
    }
    //cout<<endl;
    if(int_mb[k_spin+is[0]] == int_mb[k_spin+is[2]] &&
       int_mb[k_spin+is[1]] == int_mb[k_spin+is[3]]) {
      bufa = new double[size];
      int x = cget_add_ind_i(da, bufa, size, offset, key, visa, visa_out, vperma, " perma",&nbh1);
      comm1 = true;

      vector<size_t> vpa = vperma;
      size_t p[4] = {};
      switch(x) {
      case 0: //000
        p[0]=4; p[1]=2; p[2]=3; p[3]=1; // 4231
        assert(vpa[0]==4);
        assert(vpa[1]==2);
        assert(vpa[2]==3);
        assert(vpa[3]==1);
        break;
      case 1: //001
        p[0]=2; p[1]=4; p[2]=1; p[3]=3; // 3142
        assert(vpa[0]==3);
        assert(vpa[1]==1);
        assert(vpa[2]==4);
        assert(vpa[3]==2);
        break;
      case 2: //010
        p[0]=4; p[1]=1; p[2]=3; p[3]=2; // 4213
        assert(vpa[0]==4);
        assert(vpa[1]==2);
        assert(vpa[2]==1);
        assert(vpa[3]==3);
        break;
      case 3: //011
        p[0]=2; p[1]=3; p[2]=1; p[3]=4; // 1342
        break;
      case 4: //100
        p[0]=3; p[1]=2; p[2]=4; p[3]=1; // 2431
        break;
      case 5: //101
        p[0]=1; p[1]=4; p[2]=2; p[3]=3; // 3124
        break;
      case 6: //110
        p[0]=3; p[1]=1; p[2]=4; p[3]=2; // 2413
        break;
      case 7: //111
        p[0]=1; p[1]=3; p[2]=2; p[3]=4; // 1324
        break;
      default:
        assert(0);
      }
      //vperma.clear();
      //vperma.insert(vperma.end(),p,p+4);
      vperma = invert_perm(vperma);
      //cout<<"perma perm="<<vperma[0]<<" "<<vperma[1]<<" "<<vperma[2]<<" "<<vperma[3]<<endl;
      //ctce_sortacc(bufa, buf, visa_out, vperma, 1.0);
    }
    else {
      //cout<<" perma skip"<<endl;
    }
    //delete [] bufa;
  }
  {
    size_t isb[4] = {is[2], is[1], is[3], is[0]};
    int permb[4] = {4, 1, 3, 2};
    vector<size_t> visb;//, visb_out;
    //vector<size_t> vpermb;
    visb.insert(visb.end(), isb, isb+4);
    vpermb.insert(vpermb.end(), permb, permb+4);
    //double *bufb = new double[size];
    //cout<<"--visb =";
    for(int i=visb.size()-1; i>=0; i--) {
      //cout<<" "<<visb[i];
    }
    //cout<<endl;
    if(int_mb[k_spin+is[0]] == int_mb[k_spin+is[3]] &&
       int_mb[k_spin+is[1]] == int_mb[k_spin+is[2]]) {
      bufb = new double[size];
      int x = cget_add_ind_i(da, bufb, size, offset, key, visb, visb_out, vpermb, " permb", &nbh2);
      comm2 = true;
      size_t p[4] = {};
      switch(x) {
      case 0: //000
        p[0]=4; p[1]=2; p[2]=1; p[3]=3; // 4132
        break;
      case 1: //001
        p[0]=2; p[1]=4; p[2]=3; p[3]=1; // 3241
        break;
      case 2: //010
        p[0]=4; p[1]=1; p[2]=2; p[3]=3; // 4123
        break;
      case 3: //011
        p[0]=2; p[1]=3; p[2]=4; p[3]=1; // 2341
        break;
      case 4: //100
        p[0]=3; p[1]=2; p[2]=1; p[3]=4; // 1432
        break;
      case 5: //101
        p[0]=1; p[1]=4; p[2]=3; p[3]=2; // 3214
        break;
      case 6: //110
        p[0]=3; p[1]=1; p[2]=2; p[3]=4; // 1423
        break;
      case 7: //111
        p[0]=1; p[1]=3; p[2]=4; p[3]=2; // 2314
        break;
      default:
        assert(0);
      }
      //vpermb.clear();
      //vpermb.insert(vpermb.end(),p,p+4);
      vpermb = invert_perm(vpermb);
      //@bug Should visb_out also be inverted?
      //cout<<"permb perm="<<vpermb[0]<<" "<<vpermb[1]<<" "<<vpermb[2]<<" "<<vpermb[3]<<endl;
      //ctce_sortacc(bufb, buf, visb_out, vpermb, -1.0);
    }
    else {
      //cout<<" permb skip"<<endl;
    }
    //delete [] bufb;
  }

  if(comm1) {
    NGA_NbWait(&nbh1);
    assert(bufa);
    ctce_sortacc(bufa, buf, visa_out, vperma, 1.0);
    delete [] bufa;
  }
  if(comm2) {
    NGA_NbWait(&nbh2);
    assert(bufb);
    ctce_sortacc(bufb, buf, visb_out, vpermb, -1.0);
    delete [] bufb;
  }
#endif
}

void
cget_hash_block_i(size_t da, double *buf, size_t size, size_t offset,
                  size_t key, std::vector<size_t> &is) {
#if USE_FORTRAN_FUNCTIONS
  Fint ida = da;
  assert(is.size()==4);
  Fint is0=is[0], is1=is[1], is2=is[2], is3=is[3];
  Fint isize = size;
  Fint *int_mb = Variables::int_mb();
  Fint ikey = key;
  fget_hash_block_i(&ida, buf, &isize, &int_mb[offset], &ikey,
                    &is3, &is2, &is1, &is0);
#else
  cget_block_ind_i(da, buf, size, offset, key, is);
#endif
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
