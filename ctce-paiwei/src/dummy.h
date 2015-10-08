#ifndef __ctce_dummy_h__
#define __ctce_dummy_h__

#include "copyIter.h"
#include "typesf2c.h"
#include "func.h"

namespace ctce {

  /* this will be replaced soon */
  class Dummy {
    private:
      static CopyIter type1_; /*< perm for (2,1): (0,1,2)(2,0,1)(0,2,1) */
      static CopyIter type2_; /*< perm for (1,2): (0,1,2)(1,0,2)(1,2,0) */
      static CopyIter type3_; /*< swap permutation (0,1)(1,0) */
      static CopyIter type4_; /*< no permutation (1) */
    public:
      static void construct();
      static const CopyIter& type1() { return type1_; }
      static const CopyIter& type2() { return type2_; }
      static const CopyIter& type3() { return type3_; }
      static const CopyIter& type4() { return type4_; }
  };

} // namespace ctce

#endif
