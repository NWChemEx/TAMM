#ifdef USE_DPCPP
#include <CL/sycl.hpp>

#ifdef TAMM_INTEL_ATS
#include "mkl_sycl.hpp"
#else
#include "oneapi/mkl.hpp"
#endif

using namespace cl::sycl::ONEAPI;
#endif
