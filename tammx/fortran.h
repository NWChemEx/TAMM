#ifndef TAMMX_FORTRAN_H_
#define TAMMX_FORTRAN_H_

//for Integer
#include "ga.h"

//using Fint = Integer;

namespace tammx {

class MA {
 public:
  static void init(Integer* int_mb, double* dbl_mb) {
    int_mb_ = int_mb - 1;
    dbl_mb_ = dbl_mb - 1;
  }

  static Integer* int_mb() {
    return int_mb_;
  }

  static double* dbl_mb() {
    return dbl_mb_;
  }

 private:
  static Integer* int_mb_;
  static double* dbl_mb_;
};

} // namespace tammx

// extern "C" {
//   typedef void add_fn(Integer *ta, Integer *offseta,
//                       Integer *tc, Integer *offsetc);

//   typedef void mult_fn(Integer *ta, Integer *offseta,
//                        Integer *tb, Integer *offsetb,
//                        Integer *tc, Integer *offsetc);
// };


#endif // TAMMX_FORTRAN_H_
