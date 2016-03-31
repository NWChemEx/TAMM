/* -*- c-file-offsets: ((inextern-lang . nil)); -*- */

#ifndef GT_H_INCLUDED
#define GT_H_INCLUDED

#ifdef __cplusplus 
extern "C" {
#endif

#define GT_MPI_PORT 1
#include "gtd.h"

/* Error classes */
#define GT_SUCCESS              0   /* Successful return code */
#define GT_ERR_BUF_SMALL        1   /* Buffer length too small */
#define GT_ERR_NDIM             2   /* incorrect number of dimensions */
#define GT_ERR_INDEX            3   /* invalid tensor index */
#define GT_ERR_ACCTYPE          4   /* invalid accumulate type */
#define GT_ERR_HANDLE           5   /* invalid non-blocking handle*/
#define GT_ERR_BUFFER           6   /* invalid buffer pointer */
#define GT_ERR_TENSOR           7   /* invalid tensor handle */
#define GT_ERR_INFO             8   /* invalid tensor info handle */
#define GT_ERR_ELSIZE           9   /* invalid element size */
#define GT_ERR_GROUP           10   /* invalid process group */
#define GT_ERR_NTILES          11   /* invalid number of tiles */
#define GT_ERR_TILES           12   /* invalid tiles */
#define GT_ERR_BLOCKS          13   /* invalid number of blocks */

#define GT_ERR_BUFFER_PTR      14
#define GT_ERR_TENSOR_PTR 15
#define GT_ERR_TENSOR_TYPE_PTR 16
#define GT_ERR_ALLOC 17
#define GT_ERR_HANDLE_PTR 18
#define GT_ERR_HANDLE_FLAG 19
#define GT_ERR_SPINS_PTR 20
#define GT_ERR_SYMS_PTR 21
#define GT_ERR_TILES_PTR 22
#define GT_ERR_BUFLEN 23
#define GT_ERR_MAP 24
#define GT_ERR_PROCGROUP 25
#define GT_ERR_INDEX_PTR 27
#define GT_ERR_SPIN 28
#define GT_ERR_ARRAY_HANDLE 29

/* Data types */
typedef enum GT_tensor_type_cons GT_tensor_type_cons;
typedef struct GT_index          GT_index;
typedef struct GT_tensor         GT_tensor;
typedef GTD_procgroup            GT_procgroup;
typedef GTD_handle               GT_handle;
typedef struct GT_buf            GT_buf;
typedef union GT_tensor_type     GT_tensor_type;
typedef GTD_tensor_mapel         GT_tensor_mapel;

#define GT_MAXDIM 8

/* Null objects */
/* #define GT_NULL                (NULL) */
#define GT_TENSOR_INITIALIZER  GTI_TENSOR_INITIALIZER
#define GT_INDEX_INITIALIZER   {0}
/* #define GT_INFO_NULL           (NULL) */
#define GT_BUF_INITIALIZER     {0, NULL}
#define GTI_TENSOR_INITIALIZER  {0, 0, NULL}

/*dimensions can be: O, V, Oa, Ob, Va, Vb, N, Oc, Vc.

dimension types: N, O, V
partitions of dimensions: *, a, b, ac, a_ac, b_ac
*/

enum GT_eltype {
  GT_DOUBLE          = 0x00000000,
  GT_FLOAT           = 0x00100000,
  GT_COMPLEX         = 0x00200000,
  GT_DCOMPLEX        = 0x00300000,
};

enum GT_indextype {
  GT_Oxxx            = 0x00001000,
  GT_xOxx            = 0x00000100,
  GT_xxOx            = 0x00000010,
  GT_xxxO            = 0x00000001,

  GT_Vxxx            = 0x00002000,
  GT_xVxx            = 0x00000200,
  GT_xxVx            = 0x00000020,
  GT_xxxV            = 0x00000002,

  GT_Nxxx            = 0x00003000,
  GT_xNxx            = 0x00000300,
  GT_xxNx            = 0x00000030,
  GT_xxxN            = 0x00000003,
};

enum GT_dims {
  GT_DIM1            = 0x00000000,
  GT_DIM2            = 0x10000000,
  GT_DIM3            = 0x20000000,
  GT_DIM4            = 0x30000000,
};

enum GT_distribution {
  GT_DISTRIBUTED     = 0x00000000,
  GT_REPLICATED      = 0x80000000
};

enum GT_spintype {
  GT_SPIN_ORBITAL    = 0x00000000,
  GT_SPIN_RESTRICTED = 0x08000000
};

enum GT_tensor_type_cons {
  OO                = (GT_DIM2 | GT_Oxxx | GT_xOxx),
  OV                = (GT_DIM2 | GT_Oxxx | GT_xOxx),
  VO                = (GT_DIM2 | GT_Oxxx | GT_xOxx),
  VV                = (GT_DIM2 | GT_Oxxx | GT_xOxx),

  NN                = (GT_DIM2 | GT_Nxxx | GT_xNxx),

  OOOO              = (GT_DIM4 | GT_Oxxx | GT_xOxx | GT_xxOx | GT_xxxO),
  OOOV              = (GT_DIM4 | GT_Oxxx | GT_xOxx | GT_xxOx | GT_xxxV),
  OOVO              = (GT_DIM4 | GT_Oxxx | GT_xOxx | GT_xxVx | GT_xxxO),
  OOVV              = (GT_DIM4 | GT_Oxxx | GT_xOxx | GT_xxVx | GT_xxxV),

  OVOO              = (GT_DIM4 | GT_Oxxx | GT_xVxx | GT_xxOx | GT_xxxO),
  OVOV              = (GT_DIM4 | GT_Oxxx | GT_xVxx | GT_xxOx | GT_xxxV),
  OVVO              = (GT_DIM4 | GT_Oxxx | GT_xVxx | GT_xxVx | GT_xxxO),
  OVVV              = (GT_DIM4 | GT_Oxxx | GT_xVxx | GT_xxVx | GT_xxxV),

  VOOO              = (GT_DIM4 | GT_Vxxx | GT_xOxx | GT_xxOx | GT_xxxO),
  VOOV              = (GT_DIM4 | GT_Vxxx | GT_xOxx | GT_xxOx | GT_xxxV),
  VOVO              = (GT_DIM4 | GT_Vxxx | GT_xOxx | GT_xxVx | GT_xxxO),
  VOVV              = (GT_DIM4 | GT_Vxxx | GT_xOxx | GT_xxVx | GT_xxxV),

  VVOO              = (GT_DIM4 | GT_Vxxx | GT_xVxx | GT_xxOx | GT_xxxO),
  VVOV              = (GT_DIM4 | GT_Vxxx | GT_xVxx | GT_xxOx | GT_xxxV),
  VVVO              = (GT_DIM4 | GT_Vxxx | GT_xVxx | GT_xxVx | GT_xxxO),
  VVVV              = (GT_DIM4 | GT_Vxxx | GT_xVxx | GT_xxVx | GT_xxxV),

  NNNN              = (GT_DIM4 | GT_Nxxx | GT_xNxx | GT_xxNx | GT_xxxN),
};

union GT_tensor_type {
  struct tinfo {
    unsigned replicated: 1;
    unsigned ndim      : 3;
    unsigned spintype  : 4;
    unsigned eltype    : 4;
    
    unsigned unused    : 4;
    
    unsigned dim0      : 4;
    unsigned dim1      : 4;
    unsigned dim2      : 4;
    unsigned dim3      : 4;
  } info;
  GT_tensor_type_cons ttcons;
};

  /* Tensor dimension ordering:
     |----|--|----|--|--|----|--|----|
     0    0c Oa  Obc Ob Vac  Va Vbc  Vb
   */

struct GT_tensor {
  int active;
  int irrep;
  GT_tensor_type tensor_type;
  GT_procgroup procgroup;
  GTD_tensor_mapel *map;
  GT_tensor_mapel   array_handle;
  size_t ld[GT_MAXDIM];
  size_t lb[GT_MAXDIM + 9]; 
  size_t ub[GT_MAXDIM + 9]; 
};



struct GT_index {
  int ndim; 
  size_t id[GT_MAXDIM];
};

struct GT_buf {
  size_t buflen; 
  void *buf;
};

int GT_init(int noa, int nob, int nva, int nvb,
            int *tiles,
            int *spins,
            int *syms);

int GT_finalize();

  int GT_tensor_attach(GT_tensor_type_cons tensor_type_cons, int irrep, GT_procgroup procgroup, GT_tensor_mapel  *map, GT_tensor_mapel array_handle, GT_tensor *newtensor);

int GT_tensor_detach(GT_tensor *tensor);

int GT_get(GT_tensor *tensor, GT_index *index, GT_buf *buffer, GT_handle *handle);

int GT_add(GT_tensor *tensor, GT_index *index, GT_buf *buffer, GT_handle *handle);

int GT_test(GT_handle *handle, int *flag);

int GT_wait(GT_handle *handle);

int GT_tensor_gettype(GT_tensor *tensor, GT_tensor_type *tensor_type);

  /*TODO: add a size (active size) to buffer. Set it through API or on
    get. Check size match between buffer and tensor block on put/add.
   */
int GT_buf_alloc(size_t buflen, GT_buf *buffer);

int GT_buf_free(GT_buf *buffer);

/* int GT_get2(GT_tensor tensor, int i1, int i2, GT_buffer *buf, GT_handle *handle); */
/* int GT_put2(GT_tensor tensor, int i1, int i2, GT_buffer *buf, GT_handle *handle); */
/* int GT_add2(GT_tensor tensor, int i1, int i2, GT_buffer *buf, GT_handle *handle); */

/* int GT_get4(GT_tensor tensor, int i1, int i2, int i3, int i4, GT_buffer *buf, GT_handle *handle); */
/* int GT_put4(GT_tensor tensor, int i1, int i2, int i3, int i4, GT_buffer *buf, GT_handle *handle); */
/* int GT_add4(GT_tensor tensor, int i1, int i2, int i3, int i4, GT_buffer *buf, GT_handle *handle); */

#ifdef __cplusplus 
}
#endif

#endif /* GT_H_INCLUDED */

/* /\** TODO: need a way to create OV, VO, etc. in a generic manner *\/ */
/* int GT_index_create(int ntiles, size_t *tiles, int *spins, int *spatials, GT_index *index); */

/* int GT_index_destroy(GT_index *index); */

/* int GT_index_ntiles(GT_index index, int *ntiles); */

/* int GT_index_tiles(GT_index index, size_t **tiles); */

/* int GT_index_spins(GT_index index, int **spins); */

/* int GT_index_spatials(GT_index index, int **spatials); */

/* int GT_info_set_type_nwma(GT_info tensor_info, double *array_data, Integer *map_offset); */

/* int GT_info_set_type_nwi(GT_info tensor_info, Integer array_handle, Integer *map_offset); */

/* int GT_info_set_type_nw(GT_info tensor_info, Integer array_handle, Integer *map_offset); */

/* int GT_info_set_type_none(GT_info tensor_info); */

/* int GT_info_create(GT_info *newinfo); */

/* int GT_info_destroy(GT_info *info); */

/* /\** TODO: simplify info object management. This can be very tedious to use*\/ */
/* int GT_info_set_indices(GT_info tensor_info, int ndim, GT_index *indices); */

/* int GT_info_set_eltype(GT_info tensor_info, GT_eltype eltype); */

/* int GT_info_set_procgroup(GT_info tensor_info, GT_group *pgroup); */

/* int GT_info_set_restricted(GT_info tensor_info, int restricted); */

/* int GT_create(GT_info tensor_info, GT_tensor *newtensor); */

/* int GT_attach(GT_info tensor_info, GT_tensor *newtensor); */

/* int GT_destroy(GT_tensor *tensor); */

/* int GT_ndim(GT_tensor tensor, int *ndim); */

/* int GT_indices(GT_tensor tensor, int *ndim, GT_index *indices); */

/* typedef void *                   GT_index; */
/* typedef enum GT_eltype           GT_eltype; */
/* typedef void *                   GT_info; */
/* typedef long long                MA_handle; */

/* int GT_put(GT_tensor *tensor, GT_index *index, GT_buf *buffer, GT_handle *handle); */

