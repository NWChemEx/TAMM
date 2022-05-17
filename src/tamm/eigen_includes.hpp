#pragma once

#if defined(USE_CUDA) || defined(USE_DPCPP)
#pragma push_macro("__CUDA_ARCH__")
#pragma push_macro("__NVCC__")
#pragma push_macro("__CUDACC__")
#undef __CUDA_ARCH__
#undef __NVCC__
#undef __CUDACC__
#define __NVCC__REDEFINE__
#endif
#define EIGEN_NO_CUDA

#if defined(USE_DPCPP)
#pragma push_macro("SYCL_DEVICE_ONLY")
#pragma push_macro("__SYCL_DEVICE_ONLY__")
#undef SYCL_DEVICE_ONLY
#undef __SYCL_DEVICE_ONLY__
#undef EIGEN_USE_SYCL
// #define EIGEN_DONT_VECTORIZE
#define __SYCL__REDEFINE__
#endif

#if defined(USE_HIP) || defined(USE_DPCPP)
#pragma push_macro("__HIP_DEVICE_COMPILE__")
#endif
#define EIGEN_NO_HIP

// Eigen matrix algebra library
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#undef I

/* NVCC restore */
#if defined(__NVCC__REDEFINE__)
#pragma pop_macro("__CUDACC__")
#pragma pop_macro("__NVCC__")
#pragma pop_macro("__CUDA_ARCH__")
#endif

/*SYCL restore*/
#if defined(__SYCL__REDEFINE__)
#pragma pop_macro("SYCL_DEVICE_ONLY")
#pragma pop_macro("__SYCL_DEVICE_ONLY__")
#endif

/*HIP restore*/
#if defined(__HIP__REDEFINE__)
#pragma pop_macro("__HIP_DEVICE_COMPILE__")
#endif
