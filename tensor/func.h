#ifndef __tamm_func_h__
#define __tamm_func_h__

#include "variables.h"
#include "sys/time.h"
#include <cassert>
#include <algorithm>
using namespace std;

namespace tamm {

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
  * @param[in] sum_ids, sum_vec
  */
  double computeBeta(const std::vector<IndexName>& sum_ids, const std::vector<size_t>& sum_vec);

  /**
  * Compute the buffer size given a vector of indices value
  * @param[in] ids 
  */
  size_t compute_size(const std::vector<size_t>& ids);

  /**
  * Return type of the index, either a pIndex or a hIndex
  */
  IndexType getIndexType(const IndexName& name);

  /* only used in ccsd_t.cc */
  int is_spin_restricted_le(const std::vector<size_t>& ids, const size_t& sval);

  /**
  * Compute the reduction factor
  */
  double computeFactor(const std::vector<size_t>& ids);

  /**
  * Compute energy, hard-coded
  */
  void computeEnergy(const std::vector<size_t>& rvec, const std::vector<size_t>& ovec,
      double *energy1, double *energy2, double *buf_single, double *buf_double, const double& factor);
}

#endif /* __tamm_func_h__ */
