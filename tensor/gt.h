/* -*- c-file-offsets: ((inextern-lang . nil)); -*- */

#ifndef GT_H_INCLUDED
#define GT_H_INCLUDED

#ifdef __cplusplus 
extern "C" {
#endif

#include <stddef.h>

#define GT_MPI_PORT 1
#include "gtd.h"

#if __STDC_VERSION__ <= 199409L
/*c89 bool*/
typedef enum { false=0, true=1 } bool;
#else
#  include <stdbool.h>
#endif

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
#define GT_ERR_NULLPTR          9   /* invalid element size */
#define GT_ERR_GROUP           10   /* invalid process group */
#define GT_ERR_NTILES          11   /* invalid number of tiles */
#define GT_ERR_TILES           12   /* invalid tiles */
#define GT_ERR_BLOCKS          13   /* invalid number of blocks */

#define GT_ERR_BUFFER_PTR      14
#define GT_ERR_TENSOR_PTR      15
#define GT_ERR_TENSOR_TYPE_PTR 16
#define GT_ERR_ALLOC           17
#define GT_ERR_HANDLE_PTR      18
#define GT_ERR_HANDLE_FLAG     19
#define GT_ERR_SPINS_PTR       20
#define GT_ERR_SYMS_PTR        21
#define GT_ERR_TILES_PTR       22
#define GT_ERR_BUFLEN          23
#define GT_ERR_MAP             24
#define GT_ERR_PROCGROUP       25
#define GT_ERR_INDEX_PTR       27
#define GT_ERR_SPIN            28
#define GT_ERR_ARRAY_HANDLE    29
#define GT_ERR_NUPPER          30
#define GT_ERR_ELTYPE          31
#define GT_ERR_INDEX_PART      32
#define GT_ERR_INDEX_TYPE      33
#define GT_ERR_DIST_TYPE       34
#define GT_ERR_DIST_INFO       35
#define GT_ERR_NOT_IMPLEMENTED 36
#define GT_ERR_NOT_SUPPORTED   37

/* Data types */
typedef enum GT_tensor_type_cons GT_tensor_type_cons;
typedef struct GT_index          GT_index;
typedef struct GT_tensor         GT_tensor;
typedef GTD_procgroup            GT_procgroup;
typedef GTD_handle               GT_handle;
typedef struct GT_buf            GT_buf;
typedef union GT_tensor_type     GT_tensor_type;
typedef union GT_distribution_info GT_distribution_info;
typedef enum GT_distribution_type GT_distribution_type;
typedef struct GT_distribution GT_distribution;
typedef union GT_tensor_type GT_tensor_type;
typedef unsigned GT_tensor_flags;
typedef unsigned GT_index_part_flags;
typedef enum GT_eltype GT_eltype;
typedef enum GT_index_type GT_index_type;
typedef enum GT_index_part GT_index_part;

#define GT_MAXDIM 8

/* Null objects */
/* #define GT_NULL                (NULL) */
#define GT_TENSOR_INITIALIZER  {0}
#define GT_INDEX_INITIALIZER   {0}
#define GT_BUF_INITIALIZER     {0}
/* #define GT_INFO_NULL           (NULL) */

/* 
   dimensions can be: O, V, N

   dim type can be: a, A, b, B, c, C. a - alpha, b - beta, c - both. A
   - alpha active, etc.

   Number of upper dims not necessarily equal to number of lower
   dimensions.

   Num types = 4

   bits per dim = 2 (O,V,N)
   bits per tensor = 4 (4 eltypes) + 4 (ndims) + 4 (n. upper dims)

   index part =  4 bits (a,b,c,A,B,C)  per dim

   Other things not in bit-rep: distribution type, spin-orbital vs spin-restricted
*/

/*dimensions can be: O, V, Oa, Ob, Va, Vb, N, Oc, Vc.

dimension types: N, O, V
partitions of dimensions: *, a, b, ac, a_ac, b_ac
*/

/* 
   #bits in tensor flags
   eltype:  4
   spin  :  4
   itype : 16
   nupper:  4
   ndim  :  4
 */
enum GT_eltype {
  GT_DOUBLE        = 0x10000000,
  GT_FLOAT         = 0x20000000,
  GT_COMPLEX       = 0x30000000,
  GT_DCOMPLEX      = 0x40000000,
};

//#define GT_DIM0          0x00000000
#define GT_DIM1          0x00000001
#define GT_DIM2          0x00000002
#define GT_DIM3          0x00000003
#define GT_DIM4          0x00000004
#define GT_DIM5          0x00000005
#define GT_DIM6          0x00000006
#define GT_DIM7          0x00000007
#define GT_DIM8          0x00000008

#define GT_nupper0       0x00000010
#define GT_nupper1       0x00000020
#define GT_nupper2       0x00000030
#define GT_nupper3       0x00000040
#define GT_nupper4       0x00000050
#define GT_nupper5       0x00000060
#define GT_nupper6       0x00000070
#define GT_nupper7       0x00000080
#define GT_nupper8       0x00000090

#define GT_O0            0x00400000
#define GT_O1            0x00100000
#define GT_O2            0x00040000
#define GT_O3            0x00010000
#define GT_O4            0x00004000
#define GT_O5            0x00001000
#define GT_O6            0x00000400
#define GT_O7            0x00000100

#define GT_V0            0x00500000
#define GT_V1            0x00200000
#define GT_V2            0x00050000
#define GT_V3            0x00020000
#define GT_V4            0x00005000
#define GT_V5            0x00002000
#define GT_V6            0x00000500
#define GT_V7            0x00000200

#define GT_N0            0x00600000
#define GT_N1            0x00300000
#define GT_N2            0x00060000
#define GT_N3            0x00030000
#define GT_N4            0x00006000
#define GT_N5            0x00003000
#define GT_N6            0x00000600
#define GT_N7            0x00000300

enum GT_index_type {
  GT_iO = 1, 
  GT_iV = 2, 
  GT_iN = 3
};


#define GT_TENSOR_FLAG_ELTYPE(x)   ((x) & 0xF0000000)
#define GT_TENSOR_FLAG_NDIM(x)     ((x) & 0x0000000F)
#define GT_TENSOR_FLAG_NUPPER(x)   ((((x) & 0x000000F0)>>4)-1)
#define GT_TENSOR_FLAG_ITYPE(tflag,id) (((tflag)>>((7-(id))*2+8)) & 0x3)
#define GT_INDEX_PART_IFLAG(tflag,id) (((tflag)>>((7-(id))*4+32)) & 0xF)
#define GT_TENSOR_FLAG_SPIN(tflag) (tflag & 0x0F000000)

#define GT_IPx1(a)                                              ((a)<<(3*7+4))
#define GT_IPx2(a,b)                  (GT_IPx1(a)             | ((b)<<(3*6+4)))
#define GT_IPx3(a,b,c)                (GT_IPx2(a,b)           | ((c)<<(3*5+4)))
#define GT_IPx4(a,b,c,d)              (GT_IPx3(a,b,c)         | ((d)<<(3*4+4)))
#define GT_IPx5(a,b,c,d,e)            (GT_IPx4(a,b,c,d)       | ((e)<<(3*3+4)))
#define GT_IPx6(a,b,c,d,e,f)          (GT_IPx5(a,b,c,d,e)     | ((f)<<(3*2+4)))
#define GT_IPx7(a,b,c,d,e,f,g)        (GT_IPx6(a,b,c,d,e,f)   | ((g)<<(3*1+4)))
#define GT_IPx8(a,b,c,d,e,f,g,h)      (GT_IPx7(a,b,c,d,e,f,g) | ((h)<<(3*0+4)))

#define GT_INDEX_PART1(a)               (GT_DIM1 | GT_IPx1(a))
#define GT_INDEX_PART2(a,b)             (GT_DIM2 | GT_IPx2(a,b))
#define GT_INDEX_PART3(a,b,c)           (GT_DIM3 | GT_IPx3(a,b,c))
#define GT_INDEX_PART4(a,b,c,d)         (GT_DIM4 | GT_IPx4(a,b,c,d))
#define GT_INDEX_PART5(a,b,c,d,e)       (GT_DIM5 | GT_IPx5(a,b,c,d,e))
#define GT_INDEX_PART6(a,b,c,d,e,f)     (GT_DIM6 | GT_IPx6(a,b,c,d,e,f))
#define GT_INDEX_PART7(a,b,c,d,e,f,g)   (GT_DIM7 | GT_IPx7(a,b,c,d,e,f,g))
#define GT_INDEX_PART8(a,b,c,d,e,f,g,h) (GT_DIM8 | GT_IPx8(a,b,c,d,e,f,g,h))

enum GT_index_part {
  GT_a = 1,
  GT_A = 2,
  GT_b = 3,
  GT_B = 4,
  GT_c = 5,
  GT_C = 6
};

/* enum GT_dims { */
/*   GT_DIM1            = 0x00000000, */
/*   GT_DIM2            = 0x10000000, */
/*   GT_DIM3            = 0x20000000, */
/*   GT_DIM4            = 0x30000000, */
/* }; */

enum GT_distribution_type {
/*   GT_DISTRIBUTED     = 1, */
/*   GT_REPLICATED      = 2, */
  GT_NWDIST          = 3, /*NWChem distrubuted*/
  GT_NWMA            = 4  /*NWChem MA-based replicated*/
};

union GT_distribution_info {
  struct {
    void *map;
    void *array_handle;
  } nw;
};

struct GT_distribution {
  GT_distribution_type dtype;
  GT_distribution_info dinfo;
};

#define GT_SPIN_ORBITAL    0x00000000
#define GT_SPIN_RESTRICTED 0x08000000

enum GT_spin_type {
  GT_ORBITAL = GT_SPIN_ORBITAL,
  GT_RESTRICTED  = GT_SPIN_RESTRICTED
};

typedef enum GT_spin_type GT_spin_type;

#define GT_O           (GT_DIM1 | GT_O0)
#define GT_V           (GT_DIM1 | GT_V0)

#define GT_OO          (GT_DIM2 | GT_O0 | GT_O1)
#define GT_OV          (GT_DIM2 | GT_O0 | GT_V1)
#define GT_VO          (GT_DIM2 | GT_V0 | GT_O1)
#define GT_VV          (GT_DIM2 | GT_V0 | GT_V1)

#define GT_NN          (GT_DIM2 | GT_N0 | GT_N1)

#define GT_OOO         (GT_DIM3 | GT_O0 | GT_O1 | GT_O2)
#define GT_OOV         (GT_DIM3 | GT_O0 | GT_O1 | GT_V2)
#define GT_OVO         (GT_DIM3 | GT_O0 | GT_V1 | GT_O2)
#define GT_OVV         (GT_DIM3 | GT_O0 | GT_V1 | GT_V2)
#define GT_VOO         (GT_DIM3 | GT_V0 | GT_O1 | GT_O2)
#define GT_VOV         (GT_DIM3 | GT_V0 | GT_O1 | GT_V2)
#define GT_VVO         (GT_DIM3 | GT_V0 | GT_V1 | GT_O2)
#define GT_VVV         (GT_DIM3 | GT_V0 | GT_V1 | GT_V2)

#define GT_NNN         (GT_DIM3 | GT_N0 | GT_N1 | GT_N2)

#define GT_OOOO        (GT_DIM4 | GT_O0 | GT_O1 | GT_O2 | GT_O3)
#define GT_OOOV        (GT_DIM4 | GT_O0 | GT_O1 | GT_O2 | GT_V3)
#define GT_OOVO        (GT_DIM4 | GT_O0 | GT_O1 | GT_V2 | GT_O3)
#define GT_OOVV        (GT_DIM4 | GT_O0 | GT_O1 | GT_V2 | GT_V3)
                                                              
#define GT_OVOO        (GT_DIM4 | GT_O0 | GT_V1 | GT_O2 | GT_O3)
#define GT_OVOV        (GT_DIM4 | GT_O0 | GT_V1 | GT_O2 | GT_V3)
#define GT_OVVO        (GT_DIM4 | GT_O0 | GT_V1 | GT_V2 | GT_O3)
#define GT_OVVV        (GT_DIM4 | GT_O0 | GT_V1 | GT_V2 | GT_V3)
                                                             
#define GT_VOOO        (GT_DIM4 | GT_V0 | GT_O1 | GT_O2 | GT_O3)
#define GT_VOOV        (GT_DIM4 | GT_V0 | GT_O1 | GT_O2 | GT_V3)
#define GT_VOVO        (GT_DIM4 | GT_V0 | GT_O1 | GT_V2 | GT_O3)
#define GT_VOVV        (GT_DIM4 | GT_V0 | GT_O1 | GT_V2 | GT_V3)
                                                             
#define GT_VVOO        (GT_DIM4 | GT_V0 | GT_V1 | GT_O2 | GT_O3)
#define GT_VVOV        (GT_DIM4 | GT_V0 | GT_V1 | GT_O2 | GT_V3)
#define GT_VVVO        (GT_DIM4 | GT_V0 | GT_V1 | GT_V2 | GT_O3)
#define GT_VVVV        (GT_DIM4 | GT_V0 | GT_V1 | GT_V2 | GT_V3)

#define GT_NNNN        (GT_DIM4 | GT_N0 | GT_N1 | GT_N2 | GT_N3)


/* union GT_tensor_type { */
/*   struct tinfo { */
/*     unsigned eltype    : 4; */

/*     unsigned spin_type : 4; */

/*     unsigned itype0    : 2; */
/*     unsigned itype1    : 2; */
/*     unsigned itype2    : 2; */
/*     unsigned itype3    : 2; */
/*     unsigned itype4    : 2; */
/*     unsigned itype5    : 2; */
/*     unsigned itype6    : 2; */
/*     unsigned itype7    : 2; */

/*     unsigned nupper    : 4; */
/*     unsigned ndim      : 4; */
/*   } info; */
/*   GT_tensor_type_cons ttcons; */
/* }; */

  /* Tensor dimension ordering:
     |----|--|----|--|--|----|--|----|
     0    0c Oa  Obc Ob Vac  Va Vbc  Vb
   */

struct GT_tensor {
  bool active;
  unsigned irrep;
  unsigned ndim;
  unsigned nupper;
  GT_eltype eltype;
  GT_spin_type spin_type;
  unsigned spin_val; /*spin_upper - spin_lower must be spin_val*/
  GT_index_type itype[GT_MAXDIM];
  GT_index_part ipart[GT_MAXDIM];
/*   GT_tensor_type tensor_type; */
  GT_procgroup procgroup;
  GT_distribution distribution;

  /*These are used to check and linearize indexing into the
    array. Stored here to avoid recomputation*/

  size_t lb[GT_MAXDIM]; 
  size_t ub[GT_MAXDIM];
  size_t perm[GT_MAXDIM];
};

struct GT_index {
  int ndim; 
  size_t id[GT_MAXDIM];
};

struct GT_buf {
  size_t buflen; /*total allocated buffer in number of elements of type eltype*/
  GT_eltype eltype; /*element type*/
  size_t size; /*active size of buffer*/
  void *buf;
};

int GT_init(int noa, int noA, int nob, int noB,
            int nva, int nvA, int nvb, int nvB, 
            int *tiles,
            int *spins,
            int *syms);

int GT_finalize();

int GT_tensor_attach(GT_tensor_flags tensor_flags,  /*e.g., GT_OOOO | GT_nupper2 */
                     GT_index_part_flags iflags, /*e.g., GT_INDEX_PART4(GT_a,GT_a,GT_a,GT_a) */
                     unsigned irrep, 
                     GT_procgroup procgroup,
                     GT_distribution_type dtype, /*e.g., GT_NWDIST*/
                     GT_distribution_info dinfo,
                     GT_tensor *newtensor);

int GT_tensor_detach(GT_tensor *tensor);

int GT_get(GT_tensor *tensor, GT_index *index, GT_buf *buffer, GT_handle *handle);

int GT_add(GT_tensor *tensor, GT_index *index, GT_buf *buffer, GT_handle *handle);

int GT_test(GT_handle *handle, int *flag);

int GT_wait(GT_handle *handle);

int GT_tensor_gettype(GT_tensor *tensor, GT_tensor_type *tensor_type);

int GT_buf_alloc(GT_eltype eltype, size_t nels, GT_buf *buffer);

int GT_buf_free(GT_buf *buffer);

int GT_elsize(GT_eltype eltype, size_t *elsize);

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

