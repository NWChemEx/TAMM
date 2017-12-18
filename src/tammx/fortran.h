#ifndef TAMMX_FORTRAN_H_
#define TAMMX_FORTRAN_H_

//for Integer
#include "ga.h"

namespace tammx {

/**
 * Fortran integer type
 */
using FortranInt = Integer;

/**
 * @brief Class to track and capture Fortran state used by MA.
 *
 * This allows inter-operation with buffers allocated by MA, especially within NWChem.
 */
class MA {
 public:
  /**
   * @brief Get Fortran MA variables.
   *
   * This function is invoked transitively from Fortran. That is:
   * @code
   * call cfunc(int_mb, dbl_mb) // in Fortran code
   * void cfunc_(FortranInt* int_mb, double* dbl_mb) {
   *    MA::init(int_mb, dbl_mb);
   * }
   * @endcode
   *
   * @param int_mb Base pointer to MA integer buffer
   * @param dbl_mb Base pointer to MA double buffer
   */
  static void init(FortranInt* int_mb, double* dbl_mb) {
    int_mb_ = int_mb - 1;
    dbl_mb_ = dbl_mb - 1;
  }

  /**
   * Get MA's base integer pointer
   * @return MA's base integer pointer
   */
  static FortranInt* int_mb() {
    return int_mb_;
  }

  /**
   * Get MA's base double pointer
   * @return MA's base double pointer
   */
  static double* dbl_mb() {
    return dbl_mb_;
  }

 private:
  static FortranInt* int_mb_;
  static double* dbl_mb_;
};

} // namespace tammx

extern "C" {
  /**
   * @brief Signature of a tensor addition operation in NWChem TCE
   */
  typedef void add_fn(Integer *ta, Integer *offseta,
                      Integer *tc, Integer *offsetc);

  /**
   * @brief Signature of a tensor addition operation in NWChem TCE
   */
  typedef void mult_fn(Integer *ta, Integer *offseta,
                       Integer *tb, Integer *offsetb,
                       Integer *tc, Integer *offsetc);
}


#endif // TAMMX_FORTRAN_H_
