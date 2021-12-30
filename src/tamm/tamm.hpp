#ifndef TAMM_TAMM_HPP_
#define TAMM_TAMM_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tamm/execution_context.hpp"
#include "tamm/index_space.hpp"
#include "tamm/labeled_tensor.hpp"
#include "tamm/ops.hpp"
#include "tamm/scheduler.hpp"
#include "tamm/tensor.hpp"
// #include "tamm/spin_tensor.hpp"
#include "tamm/dag_impl.hpp"
#include "tamm/tamm_utils.hpp"
#include "tamm/lru_cache.hpp"
#include "tamm/blockops_blis.hpp"
#include "tamm/op_dag.hpp"
//#ifdef USE_TALSH
// #include "tamm/blockops_talsh.hpp"
//#endif
#include <nlohmann/json.hpp>

namespace tamm {
void initialize(int argc, char *argv[]);
void finalize();
void tamm_terminate(std::string msg);
}

#endif // TAMM_TAMM_H_
