/* -*- c-file-offsets: ((inextern-lang . nil)); -*- */

#ifndef GTI_H_INCLUDED
#define GTI_H_INCLUDED

typedef struct GTI_context GTI_context;
/* typedef struct GTI_tensor  GTI_tensor; */

/* #include "gt.h" */

#ifdef __cplusplus 
extern "C" {
#endif

struct GTI_context {
  int noa, nob, nva, nvb;
  int *tiles;
  int *spins;
  int *syms;
};

#define GTI_CONTEXT_INITIALIZER {0,0,0,0,NULL,NULL,NULL,NULL}

//extern struct GTI_context g_gtcontext;

/* GTI_tensor*  GTI_tensor_get_ptr(GT_tensor tensor) { */
/*   return (GTI_tensor*)tensor; */
/* } */

#ifdef __cplusplus 
}
#endif

#endif /* GTI_H_INCLUDED */


