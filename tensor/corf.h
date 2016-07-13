#ifndef __ctce_corf_h__
#define __ctce_corf_h__

#include "variables.h"
#include "expression.h"

namespace ctce {
  typedef void (*add_fn)(Integer*,Integer*,Integer*,Integer*);
  typedef void (*mult_fn)(Integer*,Integer*,Integer*,Integer*,Integer*,Integer*);
  typedef void (*offset_fn)(Integer*,Integer*,Integer*);

  void CorFortran(int use_c, Assignment &as, add_fn fn);
  void CorFortran(int use_c, Multiplication& m, mult_fn fn);
  void CorFortran(int use_c, Tensor *tensor, offset_fn fn);
  void destroy(Tensor *t);

}; /*ctce*/

#endif /*__ctce_corf_h__*/
