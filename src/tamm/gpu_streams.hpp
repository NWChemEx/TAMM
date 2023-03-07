#pragma once

#include "tamm/errors.hpp"
#include <map>

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#elif defined(USE_HIP)
#include <hip/hip_runtime.h>
#include <rocblas.h>
#elif defined(USE_DPCPP)
#include "sycl_device.hpp"
#endif

namespace tamm {

#if defined(USE_HIP)
using gpuStream_t           = hipStream_t;
using gpuEvent_t            = hipEvent_t;
using gpuBlasHandle_t       = rocblas_handle;
using gpuMemcpyKind         = hipMemcpyKind;
#define gpuMemcpyHostToDevice  hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost  hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice  hipMemcpyDeviceToDevice    

#define HIP_CHECK(err)                                                                      \
  do {                                                                                      \
    hipError_t err_ = (err);                                                                \
    if(err_ != hipSuccess) {                                                                \
      std::printf("HIP Exception code: %s at %s : %d\n", hipGetErrorString(err_), __FILE__, \
                  __LINE__);                                                                \
      throw std::runtime_error("hip runtime error");                                        \
    }                                                                                       \
  } while(0)
    
#elif defined(USE_CUDA)
using gpuStream_t           = cudaStream_t;
using gpuEvent_t            = cudaEvent_t;
using gpuBlasHandle_t       = cublasHandle_t;
using gpuMemcpyKind         = cudaMemcpyKind;
#define gpuMemcpyHostToDevice  cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost  cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice  cudaMemcpyDeviceToDevice    

#define CUDA_CHECK(err)                                                                       \
  do {                                                                                        \
    cudaError_t err_ = (err);                                                                 \
    if(err_ != cudaSuccess) {                                                                 \
      std::printf("CUDA Exception code: %s at %s : %d\n", cudaGetErrorString(err_), __FILE__, \
                  __LINE__);                                                                  \
      throw std::runtime_error("cuda runtime error");                                         \
    }                                                                                         \
  } while(0)
    
#elif defined(USE_DPCPP)
using gpuStream_t   = sycl::queue;
using gpuEvent_t    = sycl::event;
using gpuMemcpyKind = int;
#define gpuMemcpyHostToDevice 0
#define gpuMemcpyDeviceToHost 1
#define gpuMemcpyDeviceToDevice 2

auto sycl_asynchandler = [](sycl::exception_list exceptions) {
  for(std::exception_ptr const& e: exceptions) {
    try {
      std::rethrow_exception(e);
    } catch(sycl::exception const& ex) {
      std::cout << "Caught asynchronous SYCL exception:" << std::endl
                << ex.what() << ", SYCL code: " << ex.code() << std::endl;
    }
  }
};
#endif

static inline void getDeviceCount(int* id) {
#if defined(USE_CUDA)
    CUDA_CHECK(cudaGetDeviceCount(id));
#elif defined(USE_HIP)
    HIP_CHECK(hipGetDeviceCount(id));
#elif defined(USE_DPCPP)
  syclGetDeviceCount(id);
#endif
}

static inline void gpuSetDevice(int active_device) {
#ifdef USE_CUDA
    CUDA_CHECK(cudaSetDevice(active_device));
#elif defined(USE_HIP)
    HIP_CHECK(hipSetDevice(active_device));
#elif defined(USE_DPCPP)
  syclSetDevice(active_device);
#endif
}

template<typename T>
static void gpuMemcpyAsync(T* dst, const T* src, size_t count, gpuMemcpyKind kind,
                           gpuStream_t& stream) {
#if defined(USE_DPCPP)
  if(kind == gpuMemcpyDeviceToDevice) { stream.copy(src, dst, count); }
  else { stream.memcpy(dst, src, count * sizeof(T)); }
#elif defined(USE_CUDA)
  CUDA_CHECK(cudaMemcpyAsync(dst, src, count * sizeof(T), kind, stream));
#elif defined(USE_HIP)
  HIP_CHECK(hipMemcpyAsync(dst, src, count * sizeof(T), kind, stream));
#endif
}

class GPUStreamPool {
protected:
  bool _initialized{false};
  // Active GPU set by a given MPI-rank from execution context ctor
  int _active_device;
  // total number of GPUs on node
  int _ngpus{0};

  // Map of GPU-IDs and stream
  std::map<int, gpuStream_t*> _devID2Stream;

#if defined(USE_CUDA) || defined(USE_HIP)
  // Map of GPU-IDs and blashandle
  std::map<int, gpuBlasHandle_t*> _devID2Handle;
#endif

private:
  GPUStreamPool() {
    getDeviceCount(&_ngpus);

    for(int devID = 0; devID < _ngpus; devID++) { // # of GPUs per node
      _active_device = devID;
      gpuSetDevice(devID);

      // 1. populate gpu-streams, gpu-blas handles per GPU
      gpuStream_t* stream = nullptr;
#if defined(USE_CUDA)
      stream = new cudaStream_t;
      CUDA_CHECK(cudaStreamCreate(stream));

      gpuBlasHandle_t* handle = new gpuBlasHandle_t;
      cublasCreate(handle);
      cublasSetStream(*handle, *stream);
      _devID2Handle[devID] = handle;
#elif defined(USE_HIP)
      stream = new hipStream_t;
      HIP_CHECK(hipStreamCreate(stream));

      gpuBlasHandle_t* handle = new gpuBlasHandle_t;
      rocblas_create_handle(handle);
      rocblas_set_stream(*handle, *stream);
      _devID2Handle[devID] = handle;
#elif defined(USE_DPCPP)
      stream = new sycl::queue(*sycl_get_context(devID), *sycl_get_device(devID), sycl_asynchandler,
                               sycl::property_list{sycl::property::queue::in_order{}});
#endif

      _devID2Stream[devID] = stream;
    } // devID

    _initialized = false;
  }

  ~GPUStreamPool() {
    _initialized = false;

    for(int devID = 0; devID < _ngpus; devID++) { // # of GPUs per node
      gpuSetDevice(devID);

      // 1. destroy gpu-streams, gpu-blas handles per GPU
      gpuStream_t* stream = _devID2Stream[devID];
#if defined(USE_CUDA)
      CUDA_CHECK(cudaStreamDestroy(*stream));

      gpuBlasHandle_t* handle = _devID2Handle[devID];
      cublasDestroy(*handle);
      handle = nullptr;
#elif defined(USE_HIP)
      HIP_CHECK(hipStreamDestroy(*stream));

      gpuBlasHandle_t* handle = _devID2Handle[devID];
      rocblas_destroy_handle(*handle);
      handle = nullptr;
#elif defined(USE_DPCPP)
      delete stream;
#endif

      stream = nullptr;
    } // devID
  }

  void check_device() {
    if(!_initialized) {
      EXPECTS_STR(false, "Error: active GPU-device not set! call set_device()!");
    }
  }

public:
  /// sets an active device for getting streams and blas handles
  void set_device(int device) {
    if(!_initialized) {
      _active_device = device;
      gpuSetDevice(_active_device);
      _initialized = true;
    }
  }

  /// Returns a GPU stream in a round-robin fashion
  gpuStream_t& getStream() {
    check_device();
    return *(_devID2Stream[_active_device]);
  }

#if !defined(USE_DPCPP)
  /// Returns a GPU BLAS handle that is valid only for the CUDA and HIP builds
  gpuBlasHandle_t& getBlasHandle() {
    check_device();
    return *(_devID2Handle[_active_device]);
  }
#endif

  /// Returns the instance of device manager singleton.
  inline static GPUStreamPool& getInstance() {
    static GPUStreamPool d_m{};
    return d_m;
  }

  GPUStreamPool(const GPUStreamPool&)            = delete;
  GPUStreamPool& operator=(const GPUStreamPool&) = delete;
  GPUStreamPool(GPUStreamPool&&)                 = delete;
  GPUStreamPool& operator=(GPUStreamPool&&)      = delete;
};

} // namespace tamm
