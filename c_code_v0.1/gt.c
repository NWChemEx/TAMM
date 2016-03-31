/* -*- c-file-offsets: ((inextern-lang . nil)); -*- */

#include <stdio.h>
#include "gt.h"
#include "gti.h"
#include "gtu.h"
#include "gti_error.h"

#include "fapi.h"

static GTI_context g_gtcontext;

/* struct GTI_index { */
/*   int ntiles; */
/*   size_t *tiles; */
/*   int *spins; */
/*   int *spatials; */
/* }; */

/* struct GTI_tensor { */
/* }; */

/* struct GTI_group { */
/* }; */

/* struct GTI_info { */
/* }; */

int GT_init(int noa, int nob, int nva, int nvb,
            int *tiles,
            int *spins,
            int *syms) {
  int gt_errno = GT_SUCCESS;
  size_t nb = noa+nob+nva+nvb;

  if(noa<=0 || nob<=0 || nva<=0 || nvb<=0) {
    gt_errno = GT_ERR_BLOCKS;
    goto fn_fail;
  }
  GTI_CHECK(tiles, GT_ERR_TILES_PTR);
  GTI_CHECK(spins, GT_ERR_SPINS_PTR);
  GTI_CHECK(syms,  GT_ERR_SYMS_PTR);

  g_gtcontext.noa = noa;
  g_gtcontext.nob = nob;
  g_gtcontext.nva = nva;
  g_gtcontext.nvb = nvb;

  g_gtcontext.tiles = (int*)GTU_malloc(sizeof(int)*nb);
  g_gtcontext.spins = (int*)GTU_malloc(sizeof(int)*nb);
  g_gtcontext.syms  = (int*)GTU_malloc(sizeof(int)*nb);

  GTI_CHECK(g_gtcontext.tiles, GT_ERR_ALLOC);
  GTI_CHECK(g_gtcontext.spins, GT_ERR_ALLOC);
  GTI_CHECK(g_gtcontext.syms, GT_ERR_ALLOC);
  
  GTU_memcpy(g_gtcontext.tiles, tiles, nb*sizeof(int));
  GTU_memcpy(g_gtcontext.spins, spins, nb*sizeof(int));
  GTU_memcpy(g_gtcontext.syms,   syms, nb*sizeof(int));  

 fn_exit:
  return gt_errno;

 fn_fail:
  goto fn_exit;
}

int GT_finalize() {
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

int GT_tensor_attach(GT_tensor_type_cons tensor_type_cons, 
                     int                 irrep,
                     GT_procgroup        procgroup, 
                     GT_tensor_mapel    *map,
                     GT_tensor_mapel     array_handle,
                     GT_tensor          *newtensor) {
  int gt_errno = GT_SUCCESS; 
  GT_tensor nulltensor = GT_TENSOR_INITIALIZER;
  union GT_tensor_type ttype;
  ttype.ttcons = tensor_type_cons;

  GTI_CHECK(!GT_PROCGROUP_ISNULL(procgroup), GT_ERR_PROCGROUP);
  GTI_CHECK(map, GT_ERR_MAP);
  GTI_CHECK(array_handle, GT_ERR_ARRAY_HANDLE);
  GTI_CHECK(newtensor, GT_ERR_TENSOR_PTR);
  *newtensor = nulltensor;

  if(0/*spin restricted and !NNNN*/) {
    RETURN_ERROR(GT_ERR_SPIN);
  }
  if(!(ttype.info.spintype==0 || ttype.info.ndim+1==4)) {
    /*spin restricted and not 4d*/
    RETURN_ERROR(GT_ERR_SPIN); /*for now. only for 4d tensor*/
  }

  newtensor->active = 1;
  newtensor->irrep = irrep;
  newtensor->procgroup = procgroup;
  newtensor->tensor_type.ttcons = tensor_type_cons;
  newtensor->array_handle = array_handle;
  newtensor->map = map;

  /*TODO: setup lb, ub, ld arrays*/
 fn_exit:
  return gt_errno;
  
 fn_fail:
  goto fn_exit;

}

int GT_tensor_detach(GT_tensor *tensor) {
  int gt_errno = GT_SUCCESS; 
  GT_tensor nulltensor = GT_TENSOR_INITIALIZER;

  GTI_CHECK(tensor, GT_ERR_TENSOR_PTR);
  *tensor = nulltensor;

 fn_exit:
  return gt_errno;
  
 fn_fail:
  goto fn_exit;
}

int GT_get(GT_tensor *tensor, 
           GT_index  *index, 
           GT_buf    *buffer, 
           GT_handle *handle) {
  int gt_errno = GT_SUCCESS; 
  int i;
  Integer is[GT_MAXDIM];
  Integer key, size;
  double *dbl_mb(); /*TODO: handle these correctly*/
  struct tinfo info;

  GTI_CHECK(tensor, GT_ERR_TENSOR_PTR);
  GTI_CHECK(index,  GT_ERR_INDEX_PTR);
  GTI_CHECK(buffer, GT_ERR_BUFFER_PTR);
  GTI_CHECK(handle, GT_ERR_HANDLE_PTR);
  GTI_CHECK(tensor->active, GT_ERR_TENSOR);  

  info = tensor->tensor_type.info;
  if (info.ndim != index->ndim) {
    RETURN_ERROR(GT_ERR_INDEX);
  }
  for(i=0; i<index->ndim; i++) {
    if(index->id[i]<tensor->lb[i] || index->id[i]>tensor->ub[i]) {
      /*index is invalid*/
      RETURN_ERROR(GT_ERR_INDEX);
    }
    /*TODO: check permutation symmetry correctness in tensor*/
  }
  
  key=0;
  for(i=index->ndim-1; i>=0; i++) {
    key += (index->id[i] - tensor->lb[i]-1) * tensor->ld[i];
  }
  
  for(i=0, size=1; i<index->ndim; i++) {
    size *= g_gtcontext.tiles[i];
  }
  GTI_CHECK(size <= buffer->buflen, GT_ERR_BUFLEN);

  if(info.spintype != 0 /*spin restricted*/)  {
    GTI_CHECK(index->ndim = 4, GT_ERR_SPIN); 
    get_hash_block_i_(&tensor->array_handle, buffer->buf, &size, 
                      tensor->map, &key, 
                      &is[3], &is[2], &is[1], &is[0]);
  }
  else if (info.replicated/*replicated*/) {
    get_hash_block_ma_((dbl_mb() + tensor->array_handle), buffer->buf, 
                       &size, tensor->map, &key);
  }
  else { /*regular spin-orbital on ga*/
    get_hash_block_(&tensor->array_handle, buffer->buf, &size, tensor->map, &key);
  }

 fn_exit:
  return gt_errno;
  
 fn_fail:
  goto fn_exit;
}


int GT_add(GT_tensor *tensor, GT_index *index, GT_buf *buffer, GT_handle *handle);

int GT_test(GT_handle *handle, int *flag) {
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

int GT_wait(GT_handle *handle) {
  int gt_errno = GT_SUCCESS; 
#if 0
  GT_wait(handle);
#endif

 fn_exit:
  return gt_errno;
  
 fn_fail:
  goto fn_exit;
}

int GT_tensor_gettype(GT_tensor *tensor, GT_tensor_type *tensor_type) {
  int gt_errno = GT_SUCCESS; 

  GTI_CHECK(tensor, GT_ERR_TENSOR_PTR);
  GTI_CHECK(tensor_type, GT_ERR_TENSOR_TYPE_PTR);

  *tensor_type = tensor->tensor_type;

 fn_exit:
  return gt_errno;
  
 fn_fail:
  goto fn_exit;
}

int GT_buf_alloc(size_t buflen, GT_buf *buffer) {
  int gt_errno = GT_SUCCESS; 
  GT_buf nullbuf = GT_BUF_INITIALIZER;
  void *ptr;

  *buffer = nullbuf;
  GTI_CHECK(buflen>=0, GT_ERR_BUFLEN);
  GTI_CHECK(buffer, GT_ERR_BUFFER_PTR);

  ptr = GTU_malloc(buflen);
  GTI_CHECK(ptr, GT_ERR_ALLOC);

  buffer->buflen = buflen;
  buffer->buf = ptr;

 fn_exit:
  return gt_errno;
  
 fn_fail:
  goto fn_exit;
}

int GT_buf_free(GT_buf *buf) {
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


/* int GT_finalize() { */
/*   return GT_SUCCESS; */
/* } */

/* GTI_index* GTI_index_get_ptr(GT_index *index) { */
/* } */

/* GT_index* GTI_index_get_handle(GTI_index *index) { */
/* } */

/* int GT_index_create(int ntiles, size_t *tiles, int *spins, int *spatials, GT_index *index) { */
/* }; */

/* int GT_index_destroy(GT_index *index) { */
  
/* } */

/* int GT_index_ntiles(GT_index index, int *ntiles); */

/* int GT_index_tiles(GT_index index, size_t **tiles); */

/* int GT_index_spins(GT_index index, int **spins); */

/* int GT_index_spatials(GT_index index, int **spatials); */

/* int GT_info_create(GT_info *newinfo); */

/* int GT_info_destroy(GT_info *info); */

/* /\** TODO: simplify info object management. This can be very tedious to use*\/ */
/* int GT_info_set_indices(GT_info tensor_info, int ndim, GT_index *indices); */

/* int GT_info_set_eltype(GT_info tensor_info, GT_eltype eltype); */

/* int GT_info_set_procgroup(GT_info tensor_info, GT_group *pgroup); */

/* int GT_info_set_restricted(GT_info tensor_info, int restricted); */

/* int GT_info_set_type_nwma(GT_info tensor_info, double *array_data, Integer *map_offset); */

/* int GT_info_set_type_nwi(GT_info tensor_info, Integer array_handle, Integer *map_offset); */

/* int GT_info_set_type_nw(GT_info tensor_info, Integer array_handle, Integer *map_offset); */

/* int GT_info_set_type_none(GT_info tensor_info); */

/* int GT_create(GT_info tensor_info, GT_tensor *newtensor); */

/* int GT_attach(GT_info tensor_info, GT_tensor *newtensor); */

/* int GT_destroy(GT_tensor *tensor); */

/* int GT_detach(GT_tensor *tensor); */

/* int GT_get(GT_tensor tensor, int ndim, size_t *block, void *buffer, size_t buflen, GT_handle_t *handle); */

/* int GT_put(GT_tensor tensor, int ndim, size_t *block, void *buffer, size_t buflen, GT_handle_t *handle); */

/* int GT_add(GT_tensor tensor, int ndim, size_t *block, void *buffer, size_t buflen, GT_handle_t *handle); */

/* int GT_test(GT_handle_t *handle, int *flag); */

/* int GT_wait(GT_handle_t *handle); */

/* int GT_ndim(GT_tensor tensor, int *ndim); */

/* int GT_indices(GT_tensor tensor, int *ndim, GT_index *indices); */

/* int GT_type(GT_tensor tensor, GT_tensor_type *tensor_type); */


/* typedef void * GT_buffer; */

/* int GT_buf_alloc(GT_tensor tensor, GT_buffer *buf); */

/* int GT_buf_buffer(GT_buffer buf, void **ptr); */

/* int GT_buf_size(GT_buffer buf, int *size); */

/* int GT_buf_free(GT_buffer *buf); */

/* int GT_get2(GT_tensor tensor, int i1, int i2, GT_buffer *buf, GT_handle *handle); */
/* int GT_put2(GT_tensor tensor, int i1, int i2, GT_buffer *buf, GT_handle *handle); */
/* int GT_add2(GT_tensor tensor, int i1, int i2, GT_buffer *buf, GT_handle *handle); */

/* int GT_get4(GT_tensor tensor, int i1, int i2, int i3, int i4, GT_buffer *buf, GT_handle *handle); */
/* int GT_put4(GT_tensor tensor, int i1, int i2, int i3, int i4, GT_buffer *buf, GT_handle *handle); */
/* int GT_add4(GT_tensor tensor, int i1, int i2, int i3, int i4, GT_buffer *buf, GT_handle *handle); */

/* #ifdef __cplusplus  */
/* } */
/* #endif */

/* #endif /\* GT_H_INCLUDED *\/ */

