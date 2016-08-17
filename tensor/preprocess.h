#ifndef __ctce_prep_h__
#define __ctce_prep_h__

#include "tensor.h"
#include "triangular.h"
#include "antisymm.h"
#include "iterGroup.h"

namespace ctce {

  extern "C" {

    /**
     * Generate triangluar iterator
     * @param[out] trig_itr
     * @param[in] name, group
     */
    void genTrigIter(IterGroup<triangular>& trig_itr, const std::vector<IndexName>& name, const std::vector<int>& group);

    /**
     * Generate anti-symmetry iterator
     */
    void genAntiIter(const std::vector<size_t> &vtab, IterGroup<antisymm>& ext_itr, const Tensor& tC, const Tensor& tA, const Tensor& tB);


  } // extern C

} // namespace ctce

#endif
