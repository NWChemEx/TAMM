/* -*- c-file-offsets: ((inextern-lang . nil)); -*- */

#ifndef GTU_H_INCLUDED
#define GTU_H_INCLUDED

#include <stdlib.h>
#include <string.h>

#define GTU_malloc(n)       malloc((size_t)(n))
#define GTU_calloc(a,b)     calloc((size_t)(a),(size_t)b)
#define GTU_free(p)         free((void *)(p))
#define GTU_realloc(a,b)    realloc((void *)(a),(size_t)(b))

#define GTU_memcpy(d,s,n)   memcpy((void*)(d),(void*)(s),(size_t)(n))
#define GTU_strcpy(d,s)     strcpy((void*)(d),(void*)(s))
#define GTU_strncpy(d,s,n)  strncpy((void*)(d),(void*)(s),(size_t)(n))

#endif /* GTU_H_INCLUDED */
