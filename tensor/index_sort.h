#ifndef __ctce_sort_h__
#define __ctce_sort_h__

#include <cstddef>

namespace ctce {
  
  void index_sort(const double *sbuf, double *dbuf, int ndim, const size_t *sizes,
                  const int *perm, double alpha);

} /*ctce*/

#endif /* __ctce_sort_h__ */
