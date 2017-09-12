#ifndef TAMM_TENSOR_TAMMX_H_
#define TAMM_TENSOR_TAMMX_H_

#include <array>
#include <vector>
#include <cassert>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <map>
#include <iostream>
#include <type_traits>
#include "tammx/boundvec.h"
#include "tammx/types.h"
#include "tammx/index_sort.h"
#include "tammx/util.h"
#include "tammx/tce.h"
#include "tammx/work.h"
#include "tammx/combination.h"
#include "tammx/product_iterator.h"
#include "tammx/triangle_loop.h"
#include "tammx/copy_symmetrizer.h"

/**
 * @todo Check pass-by-value, reference, or pointer, especially for
 * Block and Tensor
 *
 * @todo Parallelize parallel_work
 *
 * @todo Implement TCE::init() and TCE::finalize()
 *
 * @todo should TCE be a "singleton" or an object? Multiple distinct
 * CC calculations in one run?
 *
 * @todo Make everything process-group aware
 *
 * @todo BoundVec should properly destroy objects
 *
 * @todo A general expression template formulation
 *
 * @todo Move/copy semantics for Tensor and Block
 *
 * @todo Scoped allocation for Tensor & Block
 *
 */

#include "tammx/block.h"
#include "tammx/labeled-block.h"
#include "tammx/labeled-tensor.h"
#include "tammx/work.h"
#include "tammx/ops.h"


#endif  // TAMM_TENSOR_TAMMX_H_

