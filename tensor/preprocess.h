#ifndef __tamm_preprocess_h__
#define __tamm_preprocess_h__

#include "antisymm.h"
#include "iterGroup.h"
#include "tensor.h"
#include "triangular.h"

namespace tamm {

/**
 * Generate triangluar iterator
 * @param[out] trig_itr
 * @param[in] name, group
 */
void genTrigIter(IterGroup<triangular>& trig_itr,
                 const std::vector<IndexName>& name,
                 const std::vector<int>& group);

/**
 * Generate anti-symmetry iterator
 */
void genAntiIter(const std::vector<size_t>& vtab, IterGroup<antisymm>& ext_itr,
                 const Tensor& tC, const Tensor& tA, const Tensor& tB);

} /* namespace tamm */

#endif /* __tamm_preprocess_h__ */
