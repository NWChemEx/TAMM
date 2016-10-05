#ifndef __tamm_sort_h__
#define __tamm_sort_h__

#include <cstddef>

namespace tamm {

void index_sort(const double *sbuf, double *dbuf, int ndim, const size_t *sizes,
                const int *perm, double alpha);
void index_sortacc(const double *sbuf, double *dbuf, int ndim,
                   const size_t *sizes, const int *perm, double alpha);

} /*tamm*/

#endif /* __tamm_sort_h__ */
