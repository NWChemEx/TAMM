/**
 * Defines here abstract things that should come from the build
 * system
 */


#include <stdint.h>
#include <stddef.h>
#include "ctce_headers.h"

typedef int64_t Fint;
typedef Fint Integer; /*for now to get compilation to work*/
typedef Fint BlasInt; /*blas integer size could be different*/
typedef uint32_t Tile;

#define FORTRAN_FUNC(fname,fNAME) fname ## _

#ifdef __cplusplus
#  define EXTERN_C extern "C"
#else
#  define EXTERN_C
#endif

#define CBLAS_HEADER_FILE "cblas.h"

#define USE_FORTRAN_FUNCTIONS 1
