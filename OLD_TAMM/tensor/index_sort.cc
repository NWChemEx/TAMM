//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------
#include "tensor/index_sort.h"
#include <algorithm>
#include <cassert>

namespace tamm {

static void copy(const double *sbuf, double *dbuf, size_t sz, double alpha) {
  for (size_t i = 0; i < sz; i++) {
    dbuf[i] = alpha * sbuf[i];
  }
}

void index_sort_2(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                  int p1, int p2, double alpha);
void index_sort_3(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                  size_t sz3, int p1, int p2, int p3, double alpha);
void index_sort_4(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                  size_t sz3, size_t sz4, int p1, int p2, int p3, int p4,
                  double alpha);

void index_sort(const double *sbuf, double *dbuf, int ndim, const size_t *sizes,
                const int *perm, double alpha) {
  assert(sbuf);
  assert(dbuf);
  assert(ndim >= 0);
  assert(ndim == 0 || sizes);
  assert(ndim == 0 || perm);

  if (ndim == 0) {
    *dbuf = *sbuf * alpha;
  } else if (ndim == 1) {
    copy(sbuf, dbuf, sizes[0], alpha);
  } else if (ndim == 2) {
    index_sort_2(sbuf, dbuf, sizes[0], sizes[1], perm[0], perm[1], alpha);
  } else if (ndim == 3) {
    index_sort_3(sbuf, dbuf, sizes[0], sizes[1], sizes[2], perm[0], perm[1],
                 perm[2], alpha);
  } else if (ndim == 4) {
    index_sort_4(sbuf, dbuf, sizes[0], sizes[1], sizes[2], sizes[3], perm[0],
                 perm[1], perm[2], perm[3], alpha);
  } else {
    assert(0);
  }
}

void index_sort_2(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                  int p1, int p2, double alpha) {
  assert(sbuf);
  assert(dbuf);

  if (p1 == 1 && p2 == 2) {
    copy(sbuf, dbuf, sz1 * sz2, alpha);
  } else if (p1 == 2 && p2 == 1) {
    for (size_t i = 0, c = 0; i < sz1; i++) {
      for (size_t j = 0; j < sz2; j++, c++) {
        dbuf[j * sz1 + i] = alpha * sbuf[c];
      }
    }
  } else {
    assert(0);
  }
}

static inline size_t idx(int n, size_t *id, size_t *sz, int *p) {
  size_t idx = 0;
  for (int i = 0; i < n - 1; i++) {
    idx = (idx + id[p[i]]) * sz[p[i + 1]];
  }
  if (n > 0) {
    idx += id[p[n - 1]];
  }
  return idx;
}

void index_sort_3(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                  size_t sz3, int p1, int p2, int p3, double alpha) {
  assert(sbuf);
  assert(dbuf);

  if (p1 == 1 && p2 == 2 && p3 == 3) {
    copy(sbuf, dbuf, sz1 * sz2 * sz3, alpha);
  } else {
    size_t i[3], c;
    size_t sz[3] = {sz1, sz2, sz3};
    int p[3] = {p1 - 1, p2 - 1, p3 - 1};
    for (i[0] = 0, c = 0; i[0] < sz[0]; i[0]++) {
      for (i[1] = 0; i[1] < sz[1]; i[1]++) {
        for (i[2] = 0; i[2] < sz[2]; i[2]++, c++) {
          dbuf[idx(3, i, sz, p)] = alpha * sbuf[c];
        }
      }
    }
  }
}

void index_sort_4(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                  size_t sz3, size_t sz4, int p1, int p2, int p3, int p4,
                  double alpha) {
  assert(sbuf);
  assert(dbuf);

  if (p1 == 1 && p2 == 2 && p3 == 3 && p4 == 4) {
    copy(sbuf, dbuf, sz1 * sz2 * sz3 * sz4, alpha);
  } else {
    size_t i[4], c;
    size_t sz[4] = {sz1, sz2, sz3, sz4};
    int p[4] = {p1 - 1, p2 - 1, p3 - 1, p4 - 1};
    for (i[0] = 0, c = 0; i[0] < sz[0]; i[0]++) {
      for (i[1] = 0; i[1] < sz[1]; i[1]++) {
        for (i[2] = 0; i[2] < sz[2]; i[2]++) {
          for (i[3] = 0; i[3] < sz[3]; i[3]++, c++) {
            dbuf[idx(4, i, sz, p)] = alpha * sbuf[c];
          }
        }
      }
    }
  }
}

/*
 * accumulate variants
 */

static void copy_add(const double *sbuf, double *dbuf, size_t sz,
                     double alpha) {
  for (size_t i = 0; i < sz; i++) {
    dbuf[i] += alpha * sbuf[i];
  }
}

void index_sortacc_2(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                     int p1, int p2, double alpha);
void index_sortacc_3(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                     size_t sz3, int p1, int p2, int p3, double alpha);
void index_sortacc_4(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                     size_t sz3, size_t sz4, int p1, int p2, int p3, int p4,
                     double alpha);

void index_sortacc(const double *sbuf, double *dbuf, int ndim,
                   const size_t *sizes, const int *perm, double alpha) {
  assert(sbuf);
  assert(dbuf);
  assert(ndim >= 0);
  assert(ndim == 0 || sizes);
  assert(ndim == 0 || perm);

  if (ndim == 0) {
    *dbuf += *sbuf * alpha;
  } else if (ndim == 1) {
    copy_add(sbuf, dbuf, sizes[0], alpha);
  } else if (ndim == 2) {
    index_sortacc_2(sbuf, dbuf, sizes[0], sizes[1], perm[0], perm[1], alpha);
  } else if (ndim == 3) {
    index_sortacc_3(sbuf, dbuf, sizes[0], sizes[1], sizes[2], perm[0], perm[1],
                    perm[2], alpha);
  } else if (ndim == 4) {
    index_sortacc_4(sbuf, dbuf, sizes[0], sizes[1], sizes[2], sizes[3], perm[0],
                    perm[1], perm[2], perm[3], alpha);
  } else {
    assert(0);
  }
}

void index_sortacc_2(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                     int p1, int p2, double alpha) {
  assert(sbuf);
  assert(dbuf);

  if (p1 == 1 && p2 == 2) {
    copy_add(sbuf, dbuf, sz1 * sz2, alpha);
  } else if (p1 == 2 && p2 == 1) {
    for (size_t i = 0, c = 0; i < sz1; i++) {
      for (size_t j = 0; j < sz2; j++, c++) {
        dbuf[j * sz1 + i] += alpha * sbuf[c];
      }
    }
  } else {
    assert(0);
  }
}

void index_sortacc_3(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                     size_t sz3, int p1, int p2, int p3, double alpha) {
  assert(sbuf);
  assert(dbuf);

  if (p1 == 1 && p2 == 2 && p3 == 3) {
    copy_add(sbuf, dbuf, sz1 * sz2 * sz3, alpha);
  } else {
    size_t i[3], c;
    size_t sz[3] = {sz1, sz2, sz3};
    int p[4] = {p1 - 1, p2 - 1, p3 - 1};
    for (i[0] = 0, c = 0; i[0] < sz[0]; i[0]++) {
      for (i[1] = 0; i[1] < sz[1]; i[1]++) {
        for (i[2] = 0; i[2] < sz[2]; i[2]++) {
          dbuf[idx(3, i, sz, p)] += alpha * sbuf[c];
        }
      }
    }
  }
}

void index_sortacc_4(const double *sbuf, double *dbuf, size_t sz1, size_t sz2,
                     size_t sz3, size_t sz4, int p1, int p2, int p3, int p4,
                     double alpha) {
  assert(sbuf);
  assert(dbuf);

  if (p1 == 1 && p2 == 2 && p3 == 3 && p4 == 4) {
    copy(sbuf, dbuf, sz1 * sz2 * sz3 * sz4, alpha);
  } else {
    size_t i[4], c;
    size_t sz[4] = {sz1, sz2, sz3, sz4};
    int p[4] = {p1 - 1, p2 - 1, p3 - 1, p4 - 1};
    for (i[0] = 0, c = 0; i[0] < sz[0]; i[0]++) {
      for (i[1] = 0; i[1] < sz[1]; i[1]++) {
        for (i[2] = 0; i[2] < sz[2]; i[2]++) {
          for (i[3] = 0; i[3] < sz[3]; i[3]++, c++) {
            dbuf[idx(4, i, sz, p)] += alpha * sbuf[c];
          }
        }
      }
    }
  }
}
} /*namespace tamm*/
