/* -*- c-file-offsets: ((inextern-lang . nil)); -*- */

#include <stdio.h>
#include <assert.h>
#include "gt.h"
#include "gti.h"
#include "gtu.h"
#include "gti_error.h"

#include "fapi.h"

static GTI_context g_gtcontext = GTI_CONTEXT_INITIALIZER;

int GT_init(int noa, int noA, int nob, int noB,
            int nva, int nvA, int nvb, int nvB,
            int *tiles,
            int *spins,
            int *syms)
{
  int gt_errno = GT_SUCCESS;
  size_t nblocks = noa+nob+nva+nvb;

  GTI_CHECK(noa>0 || nob>0 || nva>0 || nvb>0, GT_ERR_BLOCKS);
  GTI_CHECK(noA>=0 || noB>=0 || nvA>=0 || nvB>=0, GT_ERR_BLOCKS);
  GTI_CHECK(noA<=noa || noB<=nob || nvA<=nva || nvB<=nvb, GT_ERR_BLOCKS);
  GTI_CHECK(tiles, GT_ERR_TILES_PTR);
  GTI_CHECK(spins, GT_ERR_SPINS_PTR);
  GTI_CHECK(syms,  GT_ERR_SYMS_PTR);

  g_gtcontext.noa = noa;
  g_gtcontext.nob = nob;
  g_gtcontext.nva = nva;
  g_gtcontext.nvb = nvb;
  g_gtcontext.noA = noA;
  g_gtcontext.noB = noB;
  g_gtcontext.nvA = nvA;
  g_gtcontext.nvB = nvB;

  g_gtcontext.tiles = (int*)GTU_malloc(sizeof(int)*nblocks);
  g_gtcontext.spins = (int*)GTU_malloc(sizeof(int)*nblocks);
  g_gtcontext.syms  = (int*)GTU_malloc(sizeof(int)*nblocks);

  GTI_CHECK(g_gtcontext.tiles, GT_ERR_ALLOC);
  GTI_CHECK(g_gtcontext.spins, GT_ERR_ALLOC);
  GTI_CHECK(g_gtcontext.syms, GT_ERR_ALLOC);

  GTU_memcpy(g_gtcontext.tiles, tiles, nblocks*sizeof(int));
  GTU_memcpy(g_gtcontext.spins, spins, nblocks*sizeof(int));
  GTU_memcpy(g_gtcontext.syms,   syms, nblocks*sizeof(int));

 fn_exit:
  return gt_errno;

 fn_fail:
  goto fn_exit;
}

int GT_finalize()
{
  int gt_errno = GT_SUCCESS;
  GTI_context nullcontext = GTI_CONTEXT_INITIALIZER;

  GTU_free(g_gtcontext.tiles);
  GTU_free(g_gtcontext.spins);
  GTU_free(g_gtcontext.syms);
  g_gtcontext = nullcontext;

 fn_exit:
  return gt_errno;

 fn_fail:
  goto fn_exit;
}

/* layout: O[(a-A) A] O[B (b-B)] V[(a-A) A]  V[B (b-B)]
 */
void GTI_set_tensor_index_bounds(GT_tensor *tensor)
{
  size_t noa = g_gtcontext.noa, noA = g_gtcontext.noA;
  size_t nob = g_gtcontext.nob, noB = g_gtcontext.noB;
  size_t nva = g_gtcontext.nva, nvA = g_gtcontext.nvA;
  size_t nvb = g_gtcontext.nvb, nvB = g_gtcontext.nvB;
  size_t noab = noa + nob, nvab = nva + nvb, novab = noab + nvab;
  size_t lbs[3][6] = {
    /* cols:      GT_a,        GT_A,      GT_b,     GT_B, GT_c, GT_C       */
    /*GT_iO*/   {    0,      noa-noA,      noa,      noa,    0, noa-noA     },
    /*GT_iV*/   { noab, noab+nva-nvA, noab+nva, noab+nva, noab, noab+nva-nvA},
    /*GT_iN*/   {    0,      noa-noA,      noa,      noa,    0, noa-noA     }};
  size_t ubs[3][6] = {
    /* cols:       GT_a,     GT_A,  GT_b,        GT_B,  GT_c, GT_C        */
    /*GT_iO*/ {     noa,      noa,  noab,      noa+noB,  noab,  noa+noB    },
    /*GT_iV*/ {noab+nva, noab+nva, novab, noab+nva+nvB, novab, noab+nva+nvB},
    /*GT_iN*/ {noab+nva, noab+nva, novab, noab+nva+nvB, novab, noab+nva+nvB}};
  size_t *lb, *ub;
  unsigned i, ito, ipo;
  GT_index_type it;
  GT_index_part ip;

  assert(tensor);
  lb = tensor->lb;
  ub = tensor->ub;

  for(i=0; i<tensor->ndim; i++) {
    it = tensor->itype[i];
    ip = tensor->ipart[i];
    if(it == GT_iO) {
      ito = 0;
    } else if(it == GT_iV) {
      ito = 1;
    } else if(it == GT_iN) {
      ito = 2;
    } else {
      assert(0); /*unknown type*/
    }
    if(ip == GT_a) {
      ipo = 0;
    } else if(ip == GT_A) {
      ipo = 1;
    } else if(ip == GT_b) {
      ipo = 2;
    } else if(ip == GT_B) {
      ipo = 3;
    } else if(ip == GT_c) {
      ipo = 4;
    } else if(ip == GT_C) {
      ipo = 5;
    } else {
      assert(0); /* unknown index part */
    }
    lb[i] = lbs[ito][ipo];
    ub[i] = ubs[ito][ipo];
  }
}

void GTI_set_tensor_perm(GT_tensor *tensor)
{
  unsigned i, last_o, last_v, last_n;

  assert(tensor);
  last_o = last_v = last_n = -1;
  for(i=0; i<tensor->nupper; i++) {
    switch(tensor->itype[i]) {
    case GT_iO:
      tensor->perm[i] = last_o;
      last_o = i;
      break;
    case GT_iV:
      tensor->perm[i] = last_v;
      last_v = i;
      break;
    case GT_iN:
      tensor->perm[i] = last_n;
      last_n = i;
      break;
    }
  }

  last_o = last_v = last_n = -1;
  for(i=tensor->nupper; i<tensor->ndim; i++) {
    switch(tensor->itype[i]) {
    case GT_iO:
      tensor->perm[i] = last_o;
      last_o = i;
      break;
    case GT_iV:
      tensor->perm[i] = last_v;
      last_v = i;
      break;
    case GT_iN:
      tensor->perm[i] = last_n;
      last_n = i;
      break;
    }
  }
}

int GT_tensor_attach(GT_tensor_flags tensor_flags,
                     GT_index_part_flags iflags,
                     unsigned irrep,
                     GT_procgroup procgroup,
                     GT_distribution_type dtype,
                     GT_distribution_info dinfo,
                     GT_tensor *newtensor)
{
  int gt_errno = GT_SUCCESS;
  GT_tensor nulltensor = GT_TENSOR_INITIALIZER;
  unsigned i, ndim, nupper;
  GT_eltype eltype;
  GT_index_type itype[GT_MAXDIM];
  GT_index_part ipart[GT_MAXDIM];
  GT_spin_type spin_type;

  GTI_CHECK(!GT_PROCGROUP_ISNULL(procgroup), GT_ERR_PROCGROUP);
/*   GTI_CHECK(irrep < 2*g_gtcontext.num_spatial_blocks, GT_ERR_IRREP); */
  ndim = GT_TENSOR_FLAG_NDIM(tensor_flags);
  nupper = GT_TENSOR_FLAG_NUPPER(tensor_flags);
  eltype = GT_TENSOR_FLAG_ELTYPE(tensor_flags);
  spin_type = GT_TENSOR_FLAG_SPIN(tensor_flags);
  GTI_CHECK(ndim<=8, GT_ERR_NDIM);
  GTI_CHECK(nupper<=ndim, GT_ERR_NUPPER);
  GTI_CHECK(eltype>=GT_DOUBLE && eltype<=GT_DCOMPLEX, GT_ERR_ELTYPE);
  GTI_CHECK(spin_type==GT_ORBITAL || spin_type==GT_RESTRICTED,
            GT_ERR_SPIN);

  for(i=0; i<newtensor->ndim; i++) {
    GTI_CHECK(GT_TENSOR_FLAG_ITYPE(tensor_flags,i)>=GT_iO &&
              GT_TENSOR_FLAG_ITYPE(tensor_flags,i)<=GT_iN, GT_ERR_INDEX_TYPE);
    GTI_CHECK(GT_INDEX_PART_IFLAG(tensor_flags,i)>=GT_a &&
              GT_INDEX_PART_IFLAG(tensor_flags,i)<=GT_C,
              GT_ERR_INDEX_PART);
  }

  GTI_CHECK(dtype!=GT_NWDIST && dtype!=GT_NWMA,
            GT_ERR_DIST_TYPE);

  GTI_CHECK(dinfo.nw.map, GT_ERR_DIST_INFO);

  if(spin_type==GT_RESTRICTED) {
    if(tensor_flags & GT_NNNN != GT_NNNN) {
      /*spin restricted only works for NNNN*/
      RETURN_ERROR(GT_ERR_SPIN);
    }
    if(ndim != 4) {
      /*for now spin restricted only works for 4d tensors*/
      RETURN_ERROR(GT_ERR_SPIN);
    }
  }
  GTI_CHECK(newtensor, GT_ERR_TENSOR_PTR);
  *newtensor = nulltensor;

  newtensor->active = true;
  newtensor->irrep = irrep;
  newtensor->ndim = ndim;
  newtensor->nupper = nupper;
  newtensor->eltype = eltype;
  newtensor->spin_type = spin_type;
  newtensor->spin_val = nupper - (ndim - nupper);
  for(i=0; i<newtensor->ndim; i++) {
    newtensor->itype[i] = GT_TENSOR_FLAG_ITYPE(tensor_flags, i);
    newtensor->ipart[i] = GT_INDEX_PART_IFLAG(tensor_flags, i);
  }

  newtensor->procgroup = procgroup;
  newtensor->distribution.dtype = dtype;
  newtensor->distribution.dinfo = dinfo;

  GTI_set_tensor_index_bounds(newtensor);
  GTI_set_tensor_perm(newtensor);
 fn_exit:
  return gt_errno;

 fn_fail:
  goto fn_exit;

}

int GT_tensor_detach(GT_tensor *tensor)
{
  int gt_errno = GT_SUCCESS;
  GT_tensor nulltensor = GT_TENSOR_INITIALIZER;

  GTI_CHECK(tensor, GT_ERR_TENSOR_PTR);
  *tensor = nulltensor;

 fn_exit:
  return gt_errno;

 fn_fail:
  goto fn_exit;
}

bool is_spatial_nonzero(GT_tensor *tensor, GT_index *index)
{
  unsigned i, irrep;
  assert(tensor);
  assert(index);
  assert(tensor->ndim == index->ndim);

  irrep = 0;
  for(i=0; i<index->ndim; i++) {
    irrep ^= g_gtcontext.syms[index->id[i]];
  }
  return irrep == tensor->irrep;
}

bool is_spin_nonzero(GT_tensor *tensor, GT_index *index)
{
  unsigned i, spin;
  assert(tensor);
  assert(index);
  assert(tensor->ndim == index->ndim);

  spin=0;
  for(i=0; i<tensor->nupper; i++) {
    spin += g_gtcontext.spins[index->id[i]];
  }
  for(i=tensor->nupper; i<tensor->ndim; i++) {
    spin -= g_gtcontext.spins[index->id[i]];
  }
  if(spin != tensor->spin_val) {
    return false;
  }
  if(tensor->spin_type == GT_RESTRICTED) {
    spin = 0;
    for(i=0; i<tensor->ndim; i++) {
      spin += g_gtcontext.spins[index->id[i]];
    }
    if(spin == 2*tensor->ndim) {
      return false;
    }
  }
  return true;
}


int is_index_nonzero_unique(GT_tensor *tensor, GT_index *index)
{
  int gt_errno = GT_SUCCESS;
  unsigned i;

  GTI_CHECK(tensor, GT_ERR_TENSOR_PTR);
  GTI_CHECK(index,  GT_ERR_INDEX_PTR);
  GTI_CHECK(tensor->active, GT_ERR_TENSOR);
  GTI_CHECK(tensor->ndim != index->ndim, GT_ERR_INDEX);

  assert(is_spatial_nonzero(tensor, index));
  assert(is_spin_nonzero(tensor, index));

  /*index inside valid bounds*/
  for(i=0; i<tensor->ndim; i++) {
    GTI_CHECK(index->id[i]>=tensor->lb[i] &&
              index->id[i]<tensor->ub[i],
              GT_ERR_INDEX);
  }

  /*index is permutation unique*/
  for(i=0; i<tensor->ndim; i++) {
    if(tensor->perm[i] >= 0) {
      GTI_CHECK(index->id[i] >= index->id[tensor->perm[i]],
                GT_ERR_INDEX);
    }
  }

 fn_exit:
  return gt_errno;

 fn_fail:
  goto fn_exit;
}

int GT_get(GT_tensor *tensor,
           GT_index  *index,
           GT_buf    *buffer,
           GT_handle *handle)
{
  int gt_errno = GT_SUCCESS;
  unsigned i;
  size_t key, size;

  GTI_CHECK((gt_errno=is_index_nonzero_unique(tensor, index)) == GT_SUCCESS,
            gt_errno);

  for(i=0, size=1; i<index->ndim; i++) {
    size *= g_gtcontext.tiles[i];
  }
  /*buffer has space for the tile*/
  GTI_CHECK(buffer, GT_ERR_BUFFER_PTR);
  GTI_CHECK(size <= buffer->buflen, GT_ERR_BUFLEN);
  GTI_CHECK(handle, GT_ERR_HANDLE_PTR);

  buffer->size = size;
  key=0;
  for(i=0; i<index->ndim; i++) {
    key = key * (tensor->ub[i]-tensor->lb[i]) + (index->id[i] -tensor->lb[i]-1);
  }

  if(tensor->spin_type == GT_RESTRICTED) {
    assert(tensor->ndim==4); /*should have been checked during tensor
                               creation. we only handle this case for
                               now.*/
    assert(tensor->distribution.dtype == GT_NWDIST);
    fget_hash_block_i(tensor->distribution.dinfo.nw.array_handle, buffer->buf, size,
                      tensor->distribution.dinfo.nw.map, key,
                      index->id[3], index->id[2], index->id[1], index->id[0]);
  }
  else if (tensor->distribution.dtype == GT_NWMA) {
    fget_hash_block_ma(tensor->distribution.dinfo.nw.array_handle, buffer->buf,
                       size, tensor->distribution.dinfo.nw.map, key);
  }
  else if(tensor->distribution.dtype == GT_NWDIST) {
    /*spin-orbital from nwchem*/
    fget_hash_block(tensor->distribution.dinfo.nw.array_handle, buffer->buf, size,
                    tensor->distribution.dinfo.nw.map, key);
  }
  else {
    GTI_CHECK(0, GT_ERR_NOT_IMPLEMENTED);
  }

 fn_exit:
  return gt_errno;

 fn_fail:
  goto fn_exit;
}

int GT_add(GT_tensor *tensor, GT_index *index,
           GT_buf *buffer, GT_handle *handle)
{
  int gt_errno = GT_SUCCESS;
  unsigned i;
  size_t key, size;

  GTI_CHECK((gt_errno=is_index_nonzero_unique(tensor, index)) == GT_SUCCESS,
            gt_errno);

  for(i=0, size=1; i<index->ndim; i++) {
    size *= g_gtcontext.tiles[i];
  }
  /*buffer size matches tile size*/
  GTI_CHECK(buffer, GT_ERR_BUFFER_PTR);
  GTI_CHECK(size == buffer->size, GT_ERR_BUFLEN);
  GTI_CHECK(handle, GT_ERR_HANDLE_PTR);

  key=0;
  for(i=0; i<index->ndim; i++) {
    key = key * (tensor->ub[i]-tensor->lb[i]) + (index->id[i] -tensor->lb[i]-1);
  }

  GTI_CHECK(tensor->distribution.dtype!=GT_NWMA, GT_ERR_NOT_SUPPORTED);
  GTI_CHECK(tensor->spin_type!=GT_RESTRICTED, GT_ERR_NOT_SUPPORTED);
  GTI_CHECK(tensor->distribution.dtype==GT_NWDIST, GT_ERR_NOT_IMPLEMENTED);
  fadd_hash_block(tensor->distribution.dinfo.nw.array_handle, buffer->buf, size,
                  tensor->distribution.dinfo.nw.map, key);

 fn_exit:
  return gt_errno;

 fn_fail:
  goto fn_exit;
}

int GT_test(GT_handle *handle, int *flag)
{
  int gt_errno = GT_SUCCESS;

#if 0
  GTI_CHECK(handle, GT_ERR_HANDLE_PTR);
  GTI_CHECK(flag,   GT_ERR_HANDLE_FLAG);
  GT_test(handle, flag);
#endif

 fn_exit:
  return gt_errno;

 fn_fail:
  goto fn_exit;
}

int GT_wait(GT_handle *handle)
{
  int gt_errno = GT_SUCCESS;
#if 0
  GT_wait(handle);
#endif

 fn_exit:
  return gt_errno;

 fn_fail:
  goto fn_exit;
}

int GT_elsize(GT_eltype eltype, size_t *elsize)
{
  int gt_errno = GT_SUCCESS;

  GTI_CHECK(elsize, GT_ERR_NULLPTR);

  switch(eltype) {
  case GT_DOUBLE:
    *elsize = sizeof(double);
    break;
  case GT_FLOAT:
    *elsize = sizeof(float);
    break;
  case GT_COMPLEX:
  case GT_DCOMPLEX:
    RETURN_ERROR(GT_ERR_NOT_IMPLEMENTED);
    break;
  default:
    RETURN_ERROR(GT_ERR_ELTYPE);
  }

 fn_exit:
  return gt_errno;

 fn_fail:
  goto fn_exit;
}


int GT_buf_alloc(GT_eltype eltype, size_t nels, GT_buf *buffer)
{
  int gt_errno = GT_SUCCESS;
  GT_buf nullbuf = GT_BUF_INITIALIZER;
  void *ptr;
  size_t elsize;

  GTI_CHECK(buffer, GT_ERR_BUFFER_PTR);
  GTI_CHECK(nels>=0, GT_ERR_BUFLEN);
  GTI_CHECK((gt_errno=GT_elsize(eltype,&elsize))==GT_SUCCESS, gt_errno);

  *buffer = nullbuf;
  ptr = GTU_malloc(elsize * nels);
  GTI_CHECK(ptr, GT_ERR_ALLOC);

  buffer->buflen = nels;
  buffer->size   = 0;
  buffer->eltype = eltype;
  buffer->buf    = ptr;

 fn_exit:
  return gt_errno;

 fn_fail:
  goto fn_exit;
}

int GT_buf_free(GT_buf *buf)
{
  int gt_errno = GT_SUCCESS;
  GT_buf nullbuf = GT_BUF_INITIALIZER;

  GTI_CHECK(buf, GT_ERR_BUFFER_PTR);

  GTU_free(buf->buf);
  *buf = nullbuf;

 fn_exit:
  return gt_errno;

 fn_fail:
  goto fn_exit;
}
