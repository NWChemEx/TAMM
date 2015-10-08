#ifndef __ctce_func_h__
#define __ctce_func_h__

#include "variables.h"
#include "sys/time.h"
#include <cassert>
#include <algorithm>
using namespace std;

namespace ctce {

  /**
  * Get current time stamp, use to time the computation
  */
  double rtclock();

  /**
  * Return n! currently hard-coded for n<5
  */
  int factorial(int n);

  /**
  * Compute the reduction beta for isuperp stuff
  */
  double computeBeta(const std::vector<IndexName>& sum_ids, const std::vector<Integer>& sum_vec);

  /**
  * Compute the buffer size given a vector of indices value
  */
  Integer compute_size(const std::vector<Integer>& ids);

  /**
  * Check spin restricted and nonzero
  */
  int is_spin_restricted_nonzero(const std::vector<Integer>& ids, const Integer& sval);

  /**
  * Check spin
  */
  int is_spin_nonzero(const std::vector<Integer>& ids);

  /**
  * Check spatial symmetry
  */
  int is_spatial_nonzero(const std::vector<Integer> &ids, const Integer sval);

  /**
  * Return type of the index, either a pIndex or a hIndex
  */
  IndexType getIndexType(const IndexName& name);

  /* only used in ccsd_t.cc */
  int is_spin_restricted_le(const std::vector<Integer>& ids, const Integer& sval);

  /**
  * Compute the reduction factor
  */
  double computeFactor(const std::vector<Integer>& ids);

  /**
  * Compute energy, hard-coded
  */
  void computeEnergy(const std::vector<Integer>& rvec, const std::vector<Integer>& ovec,
      double *energy1, double *energy2, double *buf_single, double *buf_double, const double& factor);
}

#endif /* __ctce_func_h__ */
