/**
 * Defines here abstract things that should come from the build
 * system
 */


#include <stddef.h>

typedef long Fint;
typedef Fint Integer; /*for now to get compilation to work*/

#define FORTRAN_FUNC(fname,fNAME) fname ## _

#ifdef __cplusplus
#  define EXTERN_C extern "C"
#else
#  define EXTERN_C
#endif

#define CBLAS_HEADER_FILE "cblas.h"

