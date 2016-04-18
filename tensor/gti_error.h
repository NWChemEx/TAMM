/* -*- c-file-offsets: ((inextern-lang . nil)); -*- */

#ifndef GTI_ERROR_H_INCLUDED
#define GTI_ERROR_H_INCLUDED

#ifdef __cplusplus 
extern "C" {
#endif

#define GTI_CHECK(cond,code)        \
  if(!(cond)) {                     \
    gt_errno = code;                \
    goto fn_fail;                   \
  }

#define RETURN_ERROR(code)          \
  {                                 \
    gt_errno = code;                \
    goto fn_fail;                   \
  }

/* #define GTI_CHECK_TILES_PTR(p)       \ */
/*   if((p)==NULL) {                    \ */
/*     gt_errno = GT_ERR_TILE_PTR;      \ */
/*     goto fn_fail;                    \ */
/*   } */

/* #define GTI_CHECK_SPINS_PTR(p)       \ */
/*   if((p)==NULL) {                    \ */
/*     gt_errno = GT_ERR_SPINS_PTR;     \ */
/*     goto fn_fail;                    \ */
/*   } */

/* #define GTI_CHECK_SYMS_PTR(p)        \ */
/*   if((p)==NULL) {                    \ */
/*     gt_errno = GT_ERR_SYMS_PTR;      \ */
/*     goto fn_fail;                    \ */
/*   } */

/* #define GTI_CHECK_ALLOC(p)           \ */
/*   if((p)==NULL) {                    \ */
/*     gt_errno = GT_ERR_ALLOC;         \ */
/*     goto fn_fail;                    \ */
/*   } */

/* #define GTI_CHECK_TENSOR_PTR(p)      \ */
/*   if((p)==NULL) {                    \ */
/*     gt_errno = GT_ERR_TENSOR_PTR;    \ */
/*     goto fn_fail;                    \ */
/*   } */

/* #define GTI_CHECK_HANDLE_PTR(p)      \ */
/*   if((p)==NULL) {                    \ */
/*     gt_errno = GT_ERR_HANDLE_PTR;    \ */
/*     goto fn_fail;                    \ */
/*   } */

/* #define GTI_CHECK_HANDLE_FLAG(p)     \ */
/*   if((p)==NULL) {                    \ */
/*     gt_errno = GT_ERR_HANDLE_FLAG;   \ */
/*     goto fn_fail;                    \ */
/*   } */

/* #define GTI_CHECK_TENSOR_TYPE_PTR(p) \ */
/*   if((p)==NULL) {                    \ */
/*     gt_errno = GT_ERR_TENSOR_TYPE_PTR;\ */
/*     goto fn_fail;                    \ */
/*   } */

/* #define GTI_CHECK_BUFFER_PTR(p)      \ */
/*   if((p)==NULL) {                    \ */
/*     gt_errno = GT_ERR_BUFFER_PTR;    \ */
/*     goto fn_fail;                    \ */
/*   } */


#ifdef __cplusplus 
}
#endif

#endif /* GTI_ERROR_H_INCLUDED */
